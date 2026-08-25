from types import SimpleNamespace

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from prime_rl.inference.vllm.kernels.rocm_nemotron_int8_paged_attention import (
    can_dispatch_int8_per_token_head,
    can_dispatch_int8_prefix_prefill,
)


def make_impl(**overrides):
    values = {
        "kv_cache_dtype": "int8_per_token_head",
        "num_heads": 4,
        "num_kv_heads": 1,
        "head_size": 128,
        "alibi_slopes": None,
        "sinks": None,
        "logits_soft_cap": 0,
        "sliding_window": (-1, -1),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def make_inputs(*, query_tokens=8, max_query_len=1, max_seq_len=131072):
    batch_size = query_tokens if max_query_len == 1 else 1
    with FakeTensorMode():
        query = torch.empty(
            (query_tokens, 4, 128),
            device="cuda",
            dtype=torch.bfloat16,
        )
        key = torch.empty(
            (query_tokens, 1, 128),
            device="cuda",
            dtype=torch.bfloat16,
        )
        value = torch.empty_like(key)
        output = torch.empty_like(query)
        kv_cache = torch.empty(
            (128, 2, 2048, 1, 132),
            device="cuda",
            dtype=torch.int8,
        )
        metadata = SimpleNamespace(
            num_actual_tokens=query_tokens,
            max_query_len=max_query_len,
            query_start_loc=torch.arange(
                batch_size + 1,
                device="cuda",
                dtype=torch.int32,
            )
            * max_query_len,
            max_seq_len=max_seq_len,
            seq_lens=torch.full(
                (batch_size,),
                max_seq_len,
                device="cuda",
                dtype=torch.int32,
            ),
            block_table=torch.empty(
                (batch_size, 128),
                device="cuda",
                dtype=torch.int32,
            ),
            causal=True,
            use_cascade=False,
        )
    return query, key, value, kv_cache, metadata, output


def test_int8_decode_dispatch_accepts_exact_long_context_contract():
    query, _, _, kv_cache, metadata, output = make_inputs()

    assert can_dispatch_int8_per_token_head(
        make_impl(),
        query,
        kv_cache,
        metadata,
        output,
    )


def test_int8_prefix_dispatch_accepts_exact_chunked_prefill_contract():
    inputs = make_inputs(
        query_tokens=32768,
        max_query_len=32768,
        max_seq_len=131072,
    )
    query, key, value, kv_cache, metadata, output = inputs

    assert can_dispatch_int8_prefix_prefill(
        make_impl(),
        query,
        key,
        value,
        kv_cache,
        metadata,
        output,
    )


def test_int8_dispatch_rejects_other_model_and_cache_contracts():
    query, key, value, kv_cache, metadata, output = make_inputs()

    assert not can_dispatch_int8_per_token_head(
        make_impl(num_heads=8),
        query,
        kv_cache,
        metadata,
        output,
    )
    assert not can_dispatch_int8_prefix_prefill(
        make_impl(),
        query,
        key,
        value,
        kv_cache.to(torch.bfloat16),
        metadata,
        output,
    )
