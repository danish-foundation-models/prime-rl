"""Env-selected vLLM patches for Nemotron-3 Super inference on H100 (vLLM 0.28).

INFER_OPT is a comma-separated list of patch names, e.g. ``INFER_OPT=nvtx,ep_compact_moe``.
Patches install lazily: each registers an import hook on the vLLM module it touches, so loading
this package costs nothing in processes that never import that module.
"""
import importlib
import os
import sys

_PATCHES = {
    "nvtx": "infer_opt.nvtx",
    "ep_compact_moe": "infer_opt.ep_moe",
    "fast_generate": "infer_opt.fast_generate",
    "fi_moe_chunk": "infer_opt.fi_moe_chunk",
    "dg_jit_per_proc": "infer_opt.dg_jit_per_proc",
    "dp_affinity": "infer_opt.dp_affinity",
    "mtp_keep_last_block": "infer_opt.mtp_keep_last_block",
    "offload_fix": "infer_opt.offload_fix",
}
_applied: set[str] = set()


def _log(msg: str) -> None:
    print(f"[infer_opt pid={os.getpid()}] {msg}", file=sys.stderr, flush=True)


def apply(names: str | None = None) -> None:
    names = names if names is not None else os.environ.get("INFER_OPT", "")
    for name in filter(None, (n.strip() for n in names.split(","))):
        if name in _applied:
            continue
        if name not in _PATCHES:
            _log(f"unknown patch {name!r}; known: {sorted(_PATCHES)}")
            continue
        importlib.import_module(_PATCHES[name]).install()
        _applied.add(name)


def when_imported(module: str, fn) -> None:
    """Run fn(module) now if `module` is imported, else right after its first import."""
    if module in sys.modules:
        fn(sys.modules[module])
        return

    class _Finder:
        def find_spec(self, fullname, path=None, target=None):
            if fullname != module:
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(fullname)
            if spec is None or spec.loader is None:
                return spec
            orig_exec = spec.loader.exec_module

            def exec_module(mod):
                orig_exec(mod)
                fn(mod)

            spec.loader.exec_module = exec_module
            return spec

    import importlib.util  # noqa: F401  (needed by the finder)

    sys.meta_path.insert(0, _Finder())


def patch_source(module: str, edits: list[tuple[str, str]], tag: str) -> None:
    """Load `module` from its source with exact text edits applied (each old string must occur once).

    For backports into long vLLM functions that cannot be wrapped. Must run before the module's first
    import; any edit that does not match aborts the patch and the stock module loads.
    """
    if module in sys.modules:
        _log(f"{tag}: {module} already imported; source patch not applied")
        return

    class _Finder:
        def find_spec(self, fullname, path=None, target=None):
            if fullname != module:
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(fullname)
            if spec is None or spec.origin is None:
                return spec
            src = open(spec.origin).read()
            for old, new in edits:
                if src.count(old) != 1:
                    _log(f"{tag}: edit anchor found {src.count(old)}x, not patching {module}: {old[:80]!r}")
                    return spec
                src = src.replace(old, new)
            code = compile(src, spec.origin, "exec")

            def exec_module(mod, _code=code):
                exec(_code, mod.__dict__)

            spec.loader.exec_module = exec_module
            _log(f"{tag}: {len(edits)} source edits applied to {module}")
            return spec

    import importlib.util  # noqa: F401

    sys.meta_path.insert(0, _Finder())
