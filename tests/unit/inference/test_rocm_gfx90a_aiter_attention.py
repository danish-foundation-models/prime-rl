from types import SimpleNamespace

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from prime_rl.inference.vllm.kernels.rocm_gfx90a_aiter_attention import (
    _can_use_batch_prefill,
)


def make_inputs(block_size=2048):
    impl = SimpleNamespace(
        num_heads=4,
        num_kv_heads=1,
        head_size=128,
        kv_cache_dtype="auto",
        alibi_slopes=None,
        sliding_window=(-1, -1),
        logits_soft_cap=0.0,
        sinks=None,
    )
    metadata = SimpleNamespace(causal=True)
    with FakeTensorMode():
        query = torch.empty(
            (32768, 4, 128),
            device="cuda",
            dtype=torch.bfloat16,
        )
        output = torch.empty_like(query)
        key_cache = torch.empty(
            (512, block_size, 1, 128),
            device="cuda",
            dtype=torch.bfloat16,
        )
        value_cache = torch.empty_like(key_cache)
        cu_seqlens_q = torch.empty((9,), device="cuda", dtype=torch.int32)
        block_table = torch.empty(
            (8, 64),
            device="cuda",
            dtype=torch.int32,
        )
    return (
        impl,
        metadata,
        query,
        key_cache,
        value_cache,
        output,
        cu_seqlens_q,
        4096,
        block_table,
    )


def test_accepts_exact_nemotron_tp8_batch_prefill_cell():
    assert _can_use_batch_prefill(*make_inputs())


def test_rejects_unvalidated_page_size():
    assert not _can_use_batch_prefill(*make_inputs(block_size=1024))


def test_rejects_noncausal_attention():
    inputs = list(make_inputs())
    inputs[1].causal = False
    assert not _can_use_batch_prefill(*inputs)
