"""NVTX ranges around the eager (non-CUDA-graph) parts of a vLLM step, for nsys breakdowns."""
import functools

import torch

from infer_opt import _log, when_imported


def _wrap(cls, name: str, label: str) -> None:
    is_static = isinstance(cls.__dict__.get(name), staticmethod)
    orig = getattr(cls, name)

    @functools.wraps(orig)
    def wrapped(*a, **k):
        torch.cuda.nvtx.range_push(label)
        try:
            return orig(*a, **k)
        finally:
            torch.cuda.nvtx.range_pop()

    setattr(cls, name, staticmethod(wrapped) if is_static else wrapped)


def install() -> None:
    def runner(mod):
        R = mod.GPUModelRunner
        for name, label in [("execute_model", "step:execute_model"), ("sample_tokens", "step:sample_tokens"),
                            ("_prepare_inputs", "step:prepare_inputs"), ("_model_forward", "step:model_forward"),
                            ("_bookkeeping_sync", "step:bookkeeping"), ("_update_states", "step:update_states")]:
            if hasattr(R, name):
                _wrap(R, name, label)
        _log("nvtx: GPUModelRunner wrapped")

    def sampler(mod):
        S = mod.Sampler
        _wrap(S, "forward", "sampler")
        _wrap(S, "gather_logprobs", "sampler:gather_logprobs")
        _wrap(S, "sample", "sampler:sample")

    def logits(mod):
        _wrap(mod.LogitsProcessor, "forward", "lm_head")

    when_imported("vllm.v1.worker.gpu_model_runner", runner)
    when_imported("vllm.v1.sample.sampler", sampler)
    when_imported("vllm.model_executor.layers.logits_processor", logits)
