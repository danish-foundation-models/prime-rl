"""Version pinned offline manifests for the exact scripts used by an RL config."""
import argparse
import hashlib
import json
import os
import tomllib
from pathlib import Path

from verifiers.v1.acp import ACP_SOURCE
from verifiers.v1.harnesses.utils.launch import CHAT_PROGRAM_SOURCE

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--config', type=Path, required=True)
p.add_argument('--output-config', type=Path)
p.add_argument('--output-dir', type=Path)
p.add_argument('--check', action='store_true')
a = p.parse_args()
if not a.check:
    assert a.output_config is not None and a.output_dir is not None
    assert not a.output_config.exists(), 'Refuse to overwrite a config'
text = a.config.read_text()
config = tomllib.loads(text)
sources = {'bash': CHAT_PROGRAM_SOURCE, 'null': CHAT_PROGRAM_SOURCE,
           'pi': ACP_SOURCE, 'prime-agent': ACP_SOURCE, 'opencode': ACP_SOURCE,
           }
bundles = {}
def visit(value):
    if isinstance(value, dict):
        agent = value.get('agent')
        if isinstance(agent, dict):
            runtime = agent.get('runtime', {})
            if runtime.get('offline_python_bundle'):
                harness = agent['harness']['id']
                if harness == 'text2sql':
                    from text2sql.harness import PROGRAM_SOURCE
                    source = PROGRAM_SOURCE
                else:
                    source = sources[harness]
                digest = hashlib.sha256(source.encode()).hexdigest()
                compile(source, harness, 'exec')
                bundles.setdefault(runtime['offline_python_bundle'], {})[harness] = (digest, source)
        for child in value.values(): visit(child)
    elif isinstance(value, list):
        for child in value: visit(child)
visit(config)
if a.check:
    for old, scripts in bundles.items():
        manifest = json.loads(Path(old + '.json').read_text())
        assert hashlib.sha256(Path(old).read_bytes()).hexdigest() == manifest['sha256'], old
        for harness, (digest, _) in scripts.items():
            if digest not in manifest['scripts']:
                raise SystemExit(f'Offline script manifest mismatch: {harness} {digest} in {old}')
    print(f'Offline script preflight passed ({len(bundles)} bundles)')
    raise SystemExit(0)
a.output_dir.mkdir(parents=True, exist_ok=False)
report = []
for index, (old, scripts) in enumerate(sorted(bundles.items())):
    archive = Path(old)
    manifest = json.loads(Path(old + '.json').read_text())
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest['sha256']
    new = a.output_dir / f'python-{index}.tar.gz'
    # Package bytes and dependencies are unchanged; version the script allowlist.
    os.symlink(archive, new)
    manifest['source_manifest'] = old + '.json'
    manifest['scripts'] = sorted(set(manifest['scripts']) | {v[0] for v in scripts.values()})
    manifest['validated_scripts'] = {h: d for h, (d, _) in scripts.items()}
    for harness, (digest, source) in scripts.items():
        (a.output_dir / f'{index}-{harness}-{digest}.py').write_text(source)
    Path(str(new) + '.json').write_text(json.dumps(manifest, indent=2) + '\n')
    text = text.replace(old, str(new))
    report.append({'archive': str(new), 'scripts': manifest['validated_scripts']})
a.output_config.write_text(text)
(a.output_dir / 'preflight.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
