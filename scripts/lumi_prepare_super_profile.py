"""Consolidate the validated Super serving sources into one immutable profile."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--helpers', type=Path, required=True)
    p.add_argument('--reload', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    files = {}
    origins = {}
    for source, prefix in ((args.base / 'source', 'source'), (args.helpers / 'dfm_evals', 'source/dfm_evals')):
        for path in sorted(source.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                name = str(Path(prefix) / path.relative_to(source))
                if name in files:
                    raise ValueError(f'Duplicate source: {name}')
                files[name] = path.read_bytes()
                origins[name] = str(path.resolve())
    reload_path = args.reload / 'layerwise.py'
    reload_name = 'source/vllm/model_executor/model_loader/reload/layerwise.py'
    files[reload_name] = reload_path.read_bytes()
    origins[reload_name] = str(reload_path.resolve())
    bootstrap = (args.base / 'sitecustomize.py').read_text()
    tree = ast.parse(bootstrap)
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'overrides' for t in n.targets))
    mapping = ast.literal_eval(assignment.value)
    mapping['vllm.model_executor.model_loader.reload.layerwise'] = reload_name.removeprefix('source/')
    for name in ('nemotron_moe', 'nemotron_prefill', 'nemotron_worker'):
        mapping[f'dfm_evals.{name}'] = f'dfm_evals/{name}.py'
    lines = bootstrap.splitlines(keepends=True)
    lines[assignment.lineno - 1:assignment.end_lineno] = ['overrides = ' + repr(mapping) + '\n']
    files['sitecustomize.py'] = ''.join(lines).encode()
    manifest = {name: {'sha256': hashlib.sha256(data).hexdigest(), 'origin': origins.get(name, 'generated')} for name, data in sorted(files.items())}
    encoded = json.dumps(manifest, indent=2, sort_keys=True).encode() + b'\n'
    out = args.output / hashlib.sha256(encoded).hexdigest()[:16]
    files['manifest.json'] = encoded
    for name, data in files.items():
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != data:
                raise ValueError(f'Refusing to replace profile: {target}')
        else:
            target.write_bytes(data)
    print(out)


if __name__ == '__main__':
    main()
