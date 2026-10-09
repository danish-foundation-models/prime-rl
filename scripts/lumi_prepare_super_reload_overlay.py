"""Build a private vLLM reload overlay; never modify the installed dependency."""

import argparse
import hashlib
import json
from pathlib import Path


def main(args):
    original = args.source.read_text()
    old = """        # Buffer loaded weights, track loading progress
        info.loaded_weights.append((param_name, bound_args))
        num_loaded, ret = get_numel_loaded(original_loader, bound_args)
        info.load_numel += num_loaded
"""
    new = """        # Expert-parallel loaders explicitly decline nonlocal experts. They
        # must not retain those device tensors until local weights are complete.
        num_loaded, ret = get_numel_loaded(original_loader, bound_args)
        if num_loaded == 0 and ret is False:
            return ret
        info.loaded_weights.append((param_name, bound_args))
        info.load_numel += num_loaded
"""
    if original.count(old) != 1:
        raise ValueError("Unexpected vLLM layerwise source; review patch before rebuilding")
    patched = original.replace(old, new)
    marker = "logger = init_logger(__name__)"
    if patched.count(marker) != 1:
        raise ValueError("Unexpected vLLM logger declaration")
    patched = patched.replace(
        marker, marker + '\nlogger.info("Reload overlay enabled: discard explicitly rejected nonlocal expert buffers")'
    )
    compile(patched, str(args.source), "exec")
    digest = hashlib.sha256(patched.encode()).hexdigest()
    out = args.output / digest[:16]
    out.mkdir(parents=True, exist_ok=True)
    contents = {
        "layerwise.py": patched,
        "sitecustomize.py": '''"""Compose the frozen Super bootstrap with a private reload fix."""
import importlib.abc
import importlib.util
import os
from pathlib import Path
import runpy
import sys

class _ReloadOverride(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "vllm.model_executor.model_loader.reload.layerwise":
            return importlib.util.spec_from_file_location(fullname, Path(__file__).with_name("layerwise.py"))
        return None

sys.meta_path.insert(0, _ReloadOverride())
runpy.run_path(os.environ["PRIME_RL_BASE_SITECUSTOMIZE"])
''',
        "manifest.json": json.dumps(
            {
                "source": str(args.source.resolve()),
                "source_sha256": hashlib.sha256(original.encode()).hexdigest(),
                "patched_sha256": digest,
            },
            indent=2,
        )
        + "\n",
    }
    for name, content in contents.items():
        p = out / name
        if p.exists():
            if p.read_text() != content:
                raise ValueError(f"Refusing to replace existing overlay: {p}")
        else:
            p.write_text(content)
    print(out.resolve())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
