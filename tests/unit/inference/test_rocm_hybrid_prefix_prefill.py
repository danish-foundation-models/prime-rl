import inspect

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from prime_rl.inference.vllm.kernels import rocm_hybrid_prefix_prefill
from prime_rl.inference.vllm.kernels.rocm_hybrid_prefix_prefill import (
    can_dispatch_hybrid_prefix_prefill,
    wrap_hybrid_prefix_prefill,
)


def make_fake_inputs(*, physical_block_size=2048, context_len=98304):
    with FakeTensorMode():
        q = torch.empty((32768, 4, 128), device="cuda", dtype=torch.bfloat16)
        k = torch.empty((32768, 1, 128), device="cuda", dtype=torch.bfloat16)
        v = torch.empty_like(k)
        o = torch.empty_like(q)
        blocks = (context_len + 32768) // physical_block_size
        k_cache = torch.empty(
            (blocks, 1, 16, physical_block_size, 8),
            device="cuda",
            dtype=torch.bfloat16,
        )
        v_cache = torch.empty(
            (blocks, 1, 128, physical_block_size),
            device="cuda",
            dtype=torch.bfloat16,
        )
        block_table = torch.empty((1, blocks), device="cuda", dtype=torch.int32)
        query_start = torch.empty((2,), device="cuda", dtype=torch.int32)
        seq_lens = torch.empty((1,), device="cuda", dtype=torch.int32)
        scale = torch.ones((), device="cuda", dtype=torch.float32)
    return (
        q,
        k,
        v,
        o,
        "auto",
        k_cache,
        v_cache,
        block_table,
        query_start,
        seq_lens,
        context_len + 32768,
        32768,
        scale,
        scale,
        None,
        -1,
        128**-0.5,
        True,
        None,
        None,
        False,
        True,
    )


def stock_prefill(
    q,
    k,
    v,
    o,
    kv_cache_dtype,
    k_cache,
    v_cache,
    b_loc,
    b_start_loc,
    b_seq_len,
    max_seq_len,
    max_input_len,
    k_scale,
    v_scale,
    alibi_slopes=None,
    sliding_window=None,
    sm_scale=None,
    skip_decode=False,
    fp8_out_scale=None,
    sinks=None,
    is_block_table_ptr=False,
    causal=True,
):
    return "stock"


def test_hybrid_prefix_prefill_accepts_measured_nemotron_tp8_cell():
    assert can_dispatch_hybrid_prefix_prefill(*make_fake_inputs())


def test_hybrid_prefix_prefill_leaves_no_prefix_chunk_on_stock():
    assert not can_dispatch_hybrid_prefix_prefill(*make_fake_inputs(context_len=0))


def test_hybrid_prefix_prefill_rejects_default_non_power_of_two_page():
    assert not can_dispatch_hybrid_prefix_prefill(*make_fake_inputs(physical_block_size=1152))


def test_hybrid_prefix_prefill_wrapper_delegates_unsupported_cell(monkeypatch):
    wrapped = wrap_hybrid_prefix_prefill(stock_prefill)
    inputs = make_fake_inputs(physical_block_size=1152)
    monkeypatch.setattr(
        rocm_hybrid_prefix_prefill,
        "hybrid_prefix_prefill_gfx90a",
        lambda **_kwargs: raise_unexpected_launch(),
    )

    assert wrapped(*inputs) == "stock"
    assert inspect.signature(wrapped) == inspect.signature(stock_prefill)


def test_hybrid_prefix_prefill_wrapper_dispatches_supported_cell(monkeypatch):
    calls = []
    wrapped = wrap_hybrid_prefix_prefill(stock_prefill)
    inputs = make_fake_inputs()
    monkeypatch.setattr(
        rocm_hybrid_prefix_prefill,
        "hybrid_prefix_prefill_gfx90a",
        lambda **kwargs: calls.append(kwargs),
    )

    assert wrapped(*inputs) is None
    assert calls[0]["q"] is inputs[0]
    assert calls[0]["o"] is inputs[3]
    assert calls[0]["max_seq_len"] == 131072
    assert wrap_hybrid_prefix_prefill(wrapped) is wrapped


def raise_unexpected_launch():
    raise AssertionError("unsupported input reached the custom kernel")
