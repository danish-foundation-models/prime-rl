import inspect

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from prime_rl.inference.vllm.kernels import rocm_hybrid_paged_attention
from prime_rl.inference.vllm.kernels.rocm_hybrid_paged_attention import (
    can_dispatch_hybrid_paged_attention,
    wrap_hybrid_paged_attention,
)


def make_fake_inputs(
    *,
    physical_block_size=2048,
    max_seq_len=131072,
    local_query_heads=4,
    local_kv_heads=1,
):
    with FakeTensorMode():
        query = torch.empty(
            (8, local_query_heads, 128),
            device="cuda",
            dtype=torch.bfloat16,
        )
        output = torch.empty_like(query)
        key_cache = torch.empty(
            (512, local_kv_heads, 16, physical_block_size, 8),
            device="cuda",
            dtype=torch.bfloat16,
        )
        value_cache = torch.empty(
            (512, local_kv_heads, 128, physical_block_size),
            device="cuda",
            dtype=torch.bfloat16,
        )
        block_table = torch.empty((8, 64), device="cuda", dtype=torch.int32)
        query_start_loc = torch.empty((9,), device="cuda", dtype=torch.int32)
        seq_lens = torch.empty((8,), device="cuda", dtype=torch.int32)
    return (
        query,
        query,
        query,
        output,
        "auto",
        key_cache,
        value_cache,
        block_table,
        query_start_loc,
        seq_lens,
        max_seq_len,
        1,
        None,
        None,
        None,
        -1,
        128**-0.5,
        None,
        None,
        False,
        True,
    )


def stock_attention(
    query,
    key,
    value,
    output,
    kv_cache_dtype,
    key_cache,
    value_cache,
    block_table,
    query_start_loc,
    seq_lens,
    max_seq_len,
    max_query_len,
    k_scale,
    v_scale,
    alibi_slopes=None,
    sliding_window=None,
    sm_scale=None,
    output_scale=None,
    sinks=None,
    is_block_table_ptr=False,
    causal=True,
):
    return "stock"


def test_hybrid_paged_attention_dispatch_accepts_exact_nemotron_tp8_cell():
    assert can_dispatch_hybrid_paged_attention(*make_fake_inputs())


def test_hybrid_paged_attention_dispatch_accepts_exact_nemotron_tp4_cell():
    assert can_dispatch_hybrid_paged_attention(
        *make_fake_inputs(
            local_query_heads=8,
            physical_block_size=4096,
        )
    )


def test_hybrid_paged_attention_dispatch_accepts_exact_nemotron_tp1_cell():
    assert can_dispatch_hybrid_paged_attention(
        *make_fake_inputs(
            local_query_heads=32,
            local_kv_heads=2,
            physical_block_size=8192,
        )
    )


def test_hybrid_paged_attention_dispatch_rejects_other_query_head_counts():
    assert not can_dispatch_hybrid_paged_attention(*make_fake_inputs(local_query_heads=2))


def test_hybrid_paged_attention_dispatch_leaves_sub_4k_decode_on_stock():
    assert not can_dispatch_hybrid_paged_attention(*make_fake_inputs(max_seq_len=2048))


def test_hybrid_paged_attention_dispatch_rejects_default_non_power_of_two_page():
    assert not can_dispatch_hybrid_paged_attention(*make_fake_inputs(physical_block_size=1152))


def test_hybrid_paged_attention_wrapper_delegates_unsupported_cell(monkeypatch):
    wrapped = wrap_hybrid_paged_attention(stock_attention)
    inputs = make_fake_inputs(physical_block_size=1152)

    monkeypatch.setattr(
        rocm_hybrid_paged_attention,
        "hybrid_paged_attention_gfx90a",
        lambda **_kwargs: raise_unexpected_launch(),
    )

    assert wrapped(*inputs) == "stock"
    assert inspect.signature(wrapped) == inspect.signature(stock_attention)


def test_hybrid_paged_attention_wrapper_dispatches_supported_cell(monkeypatch):
    calls = []
    wrapped = wrap_hybrid_paged_attention(stock_attention)
    inputs = make_fake_inputs()

    monkeypatch.setattr(
        rocm_hybrid_paged_attention,
        "hybrid_paged_attention_gfx90a",
        lambda **kwargs: calls.append(kwargs),
    )

    assert wrapped(*inputs) is None
    assert calls[0]["query"] is inputs[0]
    assert calls[0]["output"] is inputs[3]
    assert calls[0]["max_seq_len"] == 131072
    assert wrap_hybrid_paged_attention(wrapped) is wrapped


def raise_unexpected_launch():
    raise AssertionError("unsupported input reached the custom kernel")
