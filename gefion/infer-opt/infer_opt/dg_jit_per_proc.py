"""Per-process cache directory for FlashInfer's TRT-LLM DeepGEMM JIT (FP8 MoE on H100).

FlashInfer's CUTLASS FP8 MoE JIT-compiles DeepGEMM kernels lazily, per new token count, and caches
the cubins in TRTLLM_DG_CACHE_DIR, default $HOME/.tensorrt_llm. On Gefion $HOME is shared WEKA,
so all 8 DP ranks (and every job) compile into one directory. A rank can see a kernel's cache
directory before another rank has renamed its cubin into it, and then asserts
`!cubin.empty() || isPathValid(path_)` (deep_gemm/runtime.cuh:63). Seen with FP8 + MTP, at startup
and after ~10 minutes of serving (jobs 531190), because MTP's verify batches reach new token counts.

This patch gives every process its own cache directory under INFER_OPT_DG_CACHE_BASE (default
/tmp/infer-opt-dg-cache). It must be set before the first DeepGEMM call, which happens long after
import, so setting it at sitecustomize time is early enough.

Enable with INFER_OPT=dg_jit_per_proc.
"""
import os

from infer_opt import _log


def install() -> None:
    base = os.environ.get("INFER_OPT_DG_CACHE_BASE", "/tmp/infer-opt-dg-cache")
    path = os.path.join(base, f"pid{os.getpid()}")
    os.makedirs(path, exist_ok=True)
    os.environ["TRTLLM_DG_CACHE_DIR"] = path  # read by the C++ side via getenv at first JIT
    _log(f"dg_jit_per_proc: TRTLLM_DG_CACHE_DIR={path}")
