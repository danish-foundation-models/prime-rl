import inspect
from types import SimpleNamespace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from prime_rl.inference.vllm.kernels import rocm_fp8_paged_mqa
from prime_rl.inference.vllm.kernels.rocm_fp8_paged_mqa import (
    _validate_fp8_paged_mqa_inputs,
    can_dispatch_fp8_paged_mqa_gfx90a,
    fp8_paged_mqa_logits_gfx90a,
    wrap_fp8_paged_mqa_logits_gfx90a,
)


def make_fake_inputs(
    *,
    q_shape=(2, 1, 32, 128),
    q_dtype=torch.float8_e4m3fn,
    cache_shape=(32, 64, 1, 132),
    table_width=None,
):
    batch_size = q_shape[0]
    block_size = cache_shape[1]
    if table_width is None:
        table_width = 32768 // block_size
    with FakeTensorMode():
        q = torch.empty(q_shape, device="cuda", dtype=q_dtype)
        kv_cache = torch.empty(cache_shape, device="cuda", dtype=torch.uint8)
        weights = torch.empty(
            (batch_size, 32),
            device="cuda",
            dtype=torch.float32,
        )
        context_lens = torch.empty(
            (batch_size,),
            device="cuda",
            dtype=torch.int32,
        )
        block_tables = torch.empty(
            (batch_size, table_width),
            device="cuda",
            dtype=torch.int32,
        )
    return q, kv_cache, weights, context_lens, block_tables


def test_validate_fp8_paged_mqa_accepts_glm_decode_contract():
    inputs = make_fake_inputs()

    batch_size, heads, head_dim, block_size, context_lens = _validate_fp8_paged_mqa_inputs(
        *inputs,
        32768,
    )

    assert (batch_size, heads, head_dim) == (2, 32, 128)
    assert block_size == 64
    assert context_lens.shape == (2,)


@pytest.mark.parametrize("block_size", (1, 16, 32, 64))
def test_validate_fp8_paged_mqa_accepts_runtime_page_sizes(block_size):
    inputs = make_fake_inputs(cache_shape=(32, block_size, 132))

    *_, actual_block_size, _ = _validate_fp8_paged_mqa_inputs(*inputs, 32768)

    assert actual_block_size == block_size


def test_validate_fp8_paged_mqa_accepts_aiter_singleton_head_view():
    inputs = make_fake_inputs(cache_shape=(32, 16, 1, 132))

    *_, actual_block_size, _ = _validate_fp8_paged_mqa_inputs(*inputs, 32768)

    assert actual_block_size == 16


@pytest.mark.parametrize(
    ("input_overrides", "message"),
    (
        ({"q_shape": (2, 2, 32, 128)}, "only next_n == 1"),
        ({"q_shape": (2, 1, 16, 128)}, "only 32 indexer heads"),
        ({"q_shape": (2, 1, 32, 64)}, "head dimension 128"),
        ({"q_dtype": torch.float8_e4m3fnuz}, "OCP torch.float8_e4m3fn"),
        (
            {"cache_shape": (32, 8, 1, 132)},
            r"block_size in \{1, 16, 32, 64\}",
        ),
    ),
)
def test_validate_fp8_paged_mqa_rejects_unsupported_contracts(
    input_overrides,
    message,
):
    inputs = make_fake_inputs(**input_overrides)

    with pytest.raises(ValueError, match=message):
        _validate_fp8_paged_mqa_inputs(*inputs, 32768)


def test_validate_fp8_paged_mqa_rejects_short_block_table():
    inputs = make_fake_inputs(table_width=511)

    with pytest.raises(ValueError, match="do not cover max_model_len"):
        _validate_fp8_paged_mqa_inputs(*inputs, 32768)


def test_validate_fp8_paged_mqa_requires_rocm_device():
    q = torch.empty((1, 1, 32, 128), dtype=torch.float8_e4m3fn)

    with pytest.raises(ValueError, match="q must be on a ROCm GPU"):
        _validate_fp8_paged_mqa_inputs(q, q, q, q, q, 32768)


def test_fp8_paged_mqa_direct_call_rejects_non_gfx90a(monkeypatch):
    inputs = make_fake_inputs()
    monkeypatch.setattr(torch.version, "hip", "7.1")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(gcnArchName="gfx942:sramecc+:xnack-"),
    )
    rocm_fp8_paged_mqa._require_gfx90a.cache_clear()

    with pytest.raises(RuntimeError, match="validated only on gfx90a"):
        fp8_paged_mqa_logits_gfx90a(*inputs, 32768)


def test_fp8_paged_mqa_wrapper_dispatches_exact_cell_and_preserves_abi(monkeypatch):
    calls = []
    custom_result = object()

    def stock(
        q_fp8,
        kv_cache_fp8,
        weights,
        context_lens,
        block_tables,
        schedule_metadata,
        max_model_len,
    ):
        calls.append(
            (
                q_fp8,
                kv_cache_fp8,
                weights,
                context_lens,
                block_tables,
                schedule_metadata,
                max_model_len,
            )
        )
        return "stock"

    def custom(
        q,
        kv_cache,
        weights,
        context_lens,
        block_tables,
        max_model_len,
        *,
        max_context_len,
    ):
        calls.append(
            (
                q,
                kv_cache,
                weights,
                context_lens,
                block_tables,
                max_model_len,
                max_context_len,
            )
        )
        return custom_result

    monkeypatch.setattr(
        rocm_fp8_paged_mqa,
        "fp8_paged_mqa_logits_gfx90a",
        custom,
    )
    wrapped = wrap_fp8_paged_mqa_logits_gfx90a(stock)
    inputs = make_fake_inputs()
    schedule_metadata = object()
    monkeypatch.setattr(
        "vllm.forward_context.get_forward_context",
        lambda: SimpleNamespace(attn_metadata={}),
    )

    assert can_dispatch_fp8_paged_mqa_gfx90a(*inputs, 32768)
    assert wrapped(*inputs, schedule_metadata, 32768) is custom_result
    assert calls == [(*inputs, 32768, 32768)]
    assert inspect.signature(wrapped) == inspect.signature(stock)
    assert wrap_fp8_paged_mqa_logits_gfx90a(wrapped) is wrapped


def test_fp8_paged_mqa_wrapper_delegates_every_unsupported_cell(monkeypatch):
    stock_calls = []

    def stock(
        q_fp8,
        kv_cache_fp8,
        weights,
        context_lens,
        block_tables,
        schedule_metadata,
        max_model_len,
    ):
        stock_calls.append(
            (
                q_fp8,
                kv_cache_fp8,
                weights,
                context_lens,
                block_tables,
                schedule_metadata,
                max_model_len,
            )
        )
        return "stock"

    monkeypatch.setattr(
        rocm_fp8_paged_mqa,
        "fp8_paged_mqa_logits_gfx90a",
        lambda *_args: pytest.fail("unsupported input reached the custom kernel"),
    )
    wrapped = wrap_fp8_paged_mqa_logits_gfx90a(stock)
    unsupported = make_fake_inputs(q_shape=(2, 2, 32, 128))
    schedule_metadata = object()

    assert not can_dispatch_fp8_paged_mqa_gfx90a(*unsupported, 32768)
    assert wrapped(*unsupported, schedule_metadata, 32768) == "stock"
    assert stock_calls == [(*unsupported, schedule_metadata, 32768)]


def _is_gfx90a() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return getattr(properties, "gcnArchName", "").split(":")[0] == "gfx90a"


@pytest.mark.skipif(not _is_gfx90a(), reason="requires a gfx90a ROCm GPU")
@pytest.mark.parametrize("block_size", (1, 16, 32, 64))
def test_fp8_paged_mqa_reads_vllm_preshuffled_cache(block_size):
    """Exercise the real writer and decode its physical cache layout."""

    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        indexer_k_quant_and_cache_triton,
    )

    torch.manual_seed(83 + block_size)
    num_blocks = 4
    batch_size = 2
    head_dim = 128
    max_model_len = 2 * block_size
    num_physical_tokens = num_blocks * block_size

    source_k = torch.randn(
        (num_physical_tokens, head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    cache = torch.empty(
        (num_blocks, block_size, 1, head_dim + 4),
        device="cuda",
        dtype=torch.uint8,
    )
    slot_mapping = torch.arange(
        num_physical_tokens,
        device="cuda",
        dtype=torch.int64,
    )
    indexer_k_quant_and_cache_triton(
        source_k,
        cache,
        slot_mapping,
        quant_block_size=128,
        scale_fmt="float32",
    )
    normal_cache = torch.empty(
        (num_physical_tokens, 1, 1, head_dim + 4),
        device="cuda",
        dtype=torch.uint8,
    )
    indexer_k_quant_and_cache_triton(
        source_k,
        normal_cache,
        slot_mapping,
        quant_block_size=128,
        scale_fmt="float32",
    )

    normal_packed = normal_cache.view(num_physical_tokens, -1)
    quantized_k = normal_packed[:, :head_dim].view(torch.float8_e4m3fn)
    scale = normal_packed[:, head_dim:].view(torch.float32).flatten()
    packed_cache = cache.view(num_blocks, -1)
    written_values = packed_cache[:, : block_size * head_dim].view(torch.float8_e4m3fn)
    token_offsets = torch.arange(
        block_size,
        device="cuda",
        dtype=torch.int64,
    )[:, None]
    dimension_offsets = torch.arange(
        head_dim,
        device="cuda",
        dtype=torch.int64,
    )[None, :]
    if block_size == 1:
        writer_offsets = token_offsets * head_dim + dimension_offsets
    else:
        writer_offsets = (
            (token_offsets // 16) * 16 * head_dim
            + (dimension_offsets // 16) * 16 * 16
            + (token_offsets % 16) * 16
            + dimension_offsets % 16
        )
    decoded_values = written_values[:, writer_offsets].reshape(
        num_physical_tokens,
        head_dim,
    )
    written_scales = packed_cache[:, block_size * head_dim :].view(torch.float32)
    torch.testing.assert_close(decoded_values, quantized_k, rtol=0, atol=0)
    torch.testing.assert_close(written_scales.flatten(), scale, rtol=0, atol=0)

    q = (
        torch.randn(
            (batch_size, 1, 32, head_dim),
            device="cuda",
            dtype=torch.bfloat16,
        )
        .to(torch.float8_e4m3fn)
        .contiguous()
    )
    weights = torch.randn(
        (batch_size, 32),
        device="cuda",
        dtype=torch.float32,
    )
    context_lens = torch.tensor(
        [max(1, max_model_len - 3), block_size + 1],
        device="cuda",
        dtype=torch.int32,
    )
    block_tables = torch.tensor(
        [[2, 0], [3, 1]],
        device="cuda",
        dtype=torch.int32,
    )

    actual = fp8_paged_mqa_logits_gfx90a(
        q,
        cache,
        weights,
        context_lens,
        block_tables,
        max_model_len,
    )

    expected = torch.full_like(actual, float("-inf"))
    for batch_idx in range(batch_size):
        logical_physical_tokens = torch.cat(
            [
                torch.arange(
                    int(block_tables[batch_idx, page].item()) * block_size,
                    (int(block_tables[batch_idx, page].item()) + 1) * block_size,
                    device="cuda",
                )
                for page in range(block_tables.shape[1])
            ]
        )
        context_len = int(context_lens[batch_idx].item())
        logical_physical_tokens = logical_physical_tokens[:context_len]
        scores = torch.relu(q[batch_idx, 0].float() @ quantized_k[logical_physical_tokens].float().T)
        scores = (scores * weights[batch_idx, :, None]).sum(dim=0)
        scores *= scale[logical_physical_tokens]
        expected[batch_idx, :context_len] = scores

    torch.testing.assert_close(actual, expected, rtol=5.0e-5, atol=5.0e-5)
