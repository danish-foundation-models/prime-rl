# SPDX-License-Identifier: MIT
#
# The query/key tiling and FP8-cache view follow ROCm/AITER's MIT-licensed
# pa_mqa_logits kernel:
# https://github.com/ROCm/aiter/blob/main/aiter/ops/triton/_triton_kernels/attention/pa_mqa_logits.py
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# This gfx90a variant is independently specialized for next_n=1, OCP E4M3FN,
# paged block sizes 1/16/32/64, 32 heads, and head dimension 128. vLLM may
# allocate the sparse-indexer cache independently of the attention KV cache:
# runtime decode has used token pages [blocks, 1, 1, 132], while profiling and
# isolated paths also use larger [blocks, block_size, 132] pages. The AITER
# public contract documents a singleton-head
# [blocks, block_size, 1, 132] view, so the wrapper accepts both zero-copy
# views. vLLM writes block sizes above one in its 16-token x 16-byte SHUFFLE
# layout, while block size one remains linear. Token pages are coalesced 16 at
# a time so the matrix multiply does not degenerate to a 32x128 by 128x1 dot.
# The exact E4M3 values are converted to FP16 before tl.dot because CDNA2 has
# no native FP8 matrix instruction. FP16 is exact for E4M3, accumulates into
# FP32, and is faster than BF16 for this kernel on gfx90a.

import functools

import torch
from vllm.triton_utils import tl, triton

_PAGED_MQA_DISPATCH_MARKER = "_prime_rl_gfx90a_fp8_paged_mqa_dispatch"


@functools.lru_cache(maxsize=None)
def _require_gfx90a(device_index: int) -> None:
    if torch.version.hip is None:
        raise RuntimeError("FP8 paged MQA requires a ROCm PyTorch build")
    properties = torch.cuda.get_device_properties(device_index)
    architecture = getattr(properties, "gcnArchName", "").split(":")[0]
    if architecture != "gfx90a":
        raise RuntimeError(
            f"FP8 paged MQA is validated only on gfx90a; device {device_index} reports {properties.gcnArchName}"
        )


@triton.jit
def _fp8_paged_mqa_logits_gfx90a_kernel(
    q_ptr,
    q_stride_batch: tl.int64,
    q_stride_head: tl.int64,
    q_stride_dim: tl.int64,
    cache_values_ptr,
    cache_values_stride_block: tl.int64,
    cache_values_stride_element: tl.int64,
    cache_scales_ptr,
    cache_scales_stride_block: tl.int64,
    cache_scales_stride_token: tl.int64,
    weights_ptr,
    weights_stride_batch: tl.int64,
    weights_stride_head: tl.int64,
    context_lens_ptr,
    block_tables_ptr,
    block_tables_stride_batch: tl.int64,
    block_tables_stride_page: tl.int64,
    logits_ptr,
    logits_stride_batch: tl.int64,
    logits_stride_token: tl.int64,
    num_cache_blocks,
    MAX_MODEL_LEN: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    logical_page = tl.program_id(0)
    batch_idx = tl.program_id(1)

    head_offsets = tl.arange(0, HEADS)
    dim_offsets = tl.arange(0, HEAD_DIM)
    token_offsets = tl.arange(0, BLOCK_SIZE)
    logical_tokens = logical_page * BLOCK_SIZE + token_offsets
    context_len = tl.load(context_lens_ptr + batch_idx)
    if logical_page * BLOCK_SIZE >= context_len:
        return
    valid_tokens = logical_tokens < context_len

    physical_block = tl.load(
        block_tables_ptr + batch_idx * block_tables_stride_batch + logical_page * block_tables_stride_page,
        mask=logical_page * BLOCK_SIZE < context_len,
        other=0,
    ).to(tl.int64)
    valid_block = (physical_block >= 0) & (physical_block < num_cache_blocks)
    cache_mask = valid_tokens[:, None] & valid_block

    q = tl.load(
        q_ptr + batch_idx * q_stride_batch + head_offsets[:, None] * q_stride_head + dim_offsets[None, :] * q_stride_dim
    ).to(tl.float16)
    # vLLM's indexer cache writer stores block sizes above one as
    # [token_tile, head_tile, token_in_tile, byte_in_head_tile], with both
    # tiles fixed at 16. Keep this address transform paired with
    # indexer_k_quant_and_cache_triton's SHUFFLE layout.
    token_tile_size: tl.constexpr = 16
    head_tile_size: tl.constexpr = 16
    shuffled_offsets = (
        (token_offsets[:, None] // token_tile_size) * token_tile_size * HEAD_DIM
        + (dim_offsets[None, :] // head_tile_size) * token_tile_size * head_tile_size
        + (token_offsets[:, None] % token_tile_size) * head_tile_size
        + dim_offsets[None, :] % head_tile_size
    )
    cache_values = tl.load(
        cache_values_ptr + physical_block * cache_values_stride_block + shuffled_offsets * cache_values_stride_element,
        mask=cache_mask,
        other=0.0,
    ).to(tl.float16)

    # FP8 values are exactly representable as FP16. The dot product uses FP16
    # MFMA inputs and an FP32 accumulator on gfx90a.
    per_head_scores = tl.dot(q, tl.trans(cache_values), out_dtype=tl.float32)
    per_head_scores = tl.maximum(per_head_scores, 0.0)
    weights = tl.load(weights_ptr + batch_idx * weights_stride_batch + head_offsets * weights_stride_head)
    logits = tl.sum(per_head_scores * weights[:, None], axis=0)
    scales = tl.load(
        cache_scales_ptr + physical_block * cache_scales_stride_block + token_offsets * cache_scales_stride_token,
        mask=valid_tokens & valid_block,
        other=0.0,
    )
    logits *= scales
    logits = tl.where(valid_tokens & valid_block, logits, float("-inf"))
    tl.store(
        logits_ptr + batch_idx * logits_stride_batch + logical_tokens * logits_stride_token,
        logits,
        mask=logical_tokens < MAX_MODEL_LEN,
    )


@triton.jit
def _fp8_token_paged_mqa_logits_gfx90a_kernel(
    q_ptr,
    q_stride_batch: tl.int64,
    q_stride_head: tl.int64,
    q_stride_dim: tl.int64,
    cache_values_ptr,
    cache_values_stride_block: tl.int64,
    cache_values_stride_element: tl.int64,
    cache_scales_ptr,
    cache_scales_stride_block: tl.int64,
    weights_ptr,
    weights_stride_batch: tl.int64,
    weights_stride_head: tl.int64,
    context_lens_ptr,
    block_tables_ptr,
    block_tables_stride_batch: tl.int64,
    block_tables_stride_page: tl.int64,
    logits_ptr,
    logits_stride_batch: tl.int64,
    logits_stride_token: tl.int64,
    num_cache_blocks,
    MAX_MODEL_LEN: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    TOKENS_PER_PROGRAM: tl.constexpr,
):
    token_tile = tl.program_id(0)
    batch_idx = tl.program_id(1)

    head_offsets = tl.arange(0, HEADS)
    dim_offsets = tl.arange(0, HEAD_DIM)
    token_offsets = tl.arange(0, TOKENS_PER_PROGRAM)
    logical_tokens = token_tile * TOKENS_PER_PROGRAM + token_offsets
    context_len = tl.load(context_lens_ptr + batch_idx)
    if token_tile * TOKENS_PER_PROGRAM >= context_len:
        return
    valid_tokens = logical_tokens < context_len

    # With one token per physical page, adjacent logical tokens can map to
    # arbitrary cache blocks. Gather 16 pages before the FP16 matrix multiply
    # so gfx90a still receives a useful 32x128 by 128x16 tile.
    physical_blocks = tl.load(
        block_tables_ptr + batch_idx * block_tables_stride_batch + logical_tokens * block_tables_stride_page,
        mask=valid_tokens,
        other=0,
    ).to(tl.int64)
    valid_blocks = (physical_blocks >= 0) & (physical_blocks < num_cache_blocks)
    cache_mask = valid_tokens[:, None] & valid_blocks[:, None]

    q = tl.load(
        q_ptr + batch_idx * q_stride_batch + head_offsets[:, None] * q_stride_head + dim_offsets[None, :] * q_stride_dim
    ).to(tl.float16)
    cache_values = tl.load(
        cache_values_ptr
        + physical_blocks[:, None] * cache_values_stride_block
        + dim_offsets[None, :] * cache_values_stride_element,
        mask=cache_mask,
        other=0.0,
    ).to(tl.float16)

    per_head_scores = tl.dot(q, tl.trans(cache_values), out_dtype=tl.float32)
    per_head_scores = tl.maximum(per_head_scores, 0.0)
    weights = tl.load(weights_ptr + batch_idx * weights_stride_batch + head_offsets * weights_stride_head)
    logits = tl.sum(per_head_scores * weights[:, None], axis=0)
    scales = tl.load(
        cache_scales_ptr + physical_blocks * cache_scales_stride_block,
        mask=valid_tokens & valid_blocks,
        other=0.0,
    )
    logits *= scales
    logits = tl.where(valid_tokens & valid_blocks, logits, float("-inf"))
    tl.store(
        logits_ptr + batch_idx * logits_stride_batch + logical_tokens * logits_stride_token,
        logits,
        mask=logical_tokens < MAX_MODEL_LEN,
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def can_dispatch_fp8_paged_mqa_gfx90a(
    q: object,
    kv_cache: object,
    weights: object,
    context_lens: object,
    block_tables: object,
    max_model_len: object,
) -> bool:
    """Return whether inputs match the complete measured decode cell.

    Device architecture is checked once by the installer. This predicate keeps
    the runtime wrapper total: malformed or unsupported calls select the saved
    vLLM implementation instead of entering the strict direct kernel.
    """

    tensors = (q, kv_cache, weights, context_lens, block_tables)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        return False
    assert isinstance(q, torch.Tensor)
    assert isinstance(kv_cache, torch.Tensor)
    assert isinstance(weights, torch.Tensor)
    assert isinstance(context_lens, torch.Tensor)
    assert isinstance(block_tables, torch.Tensor)

    if q.device.type != "cuda" or q.dtype != torch.float8_e4m3fn or q.ndim != 4:
        return False
    batch_size, next_n, heads, head_dim = q.shape
    if (next_n, heads, head_dim) != (1, 32, 128) or not q.is_contiguous():
        return False

    if kv_cache.device != q.device or kv_cache.dtype != torch.uint8 or kv_cache.ndim not in (3, 4):
        return False
    block_size = kv_cache.shape[1]
    if block_size not in (1, 16, 32, 64):
        return False
    if tuple(kv_cache.shape[2:]) not in ((head_dim + 4,), (1, head_dim + 4)):
        return False
    if not kv_cache.is_contiguous():
        return False

    if (
        weights.device != q.device
        or weights.dtype != torch.float32
        or tuple(weights.shape) != (batch_size, heads)
        or not weights.is_contiguous()
    ):
        return False
    if (
        context_lens.device != q.device
        or context_lens.dtype != torch.int32
        or context_lens.numel() != batch_size
        or not context_lens.is_contiguous()
    ):
        return False
    if (
        block_tables.device != q.device
        or block_tables.dtype != torch.int32
        or block_tables.ndim != 2
        or block_tables.shape[0] != batch_size
        or not block_tables.is_contiguous()
    ):
        return False
    return (
        type(max_model_len) is int
        and max_model_len > 0
        and max_model_len <= block_tables.shape[1] * block_size
    )


def wrap_fp8_paged_mqa_logits_gfx90a(original):
    """Dispatch one exact gfx90a cell and preserve stock behavior otherwise."""

    if getattr(original, _PAGED_MQA_DISPATCH_MARKER, None) is original:
        return original

    @functools.wraps(original)
    def _dispatch(
        q_fp8,
        kv_cache_fp8,
        weights,
        context_lens,
        block_tables,
        schedule_metadata,
        max_model_len,
    ):
        if can_dispatch_fp8_paged_mqa_gfx90a(
            q_fp8,
            kv_cache_fp8,
            weights,
            context_lens,
            block_tables,
            max_model_len,
        ):
            max_context_len = max_model_len
            from vllm.forward_context import get_forward_context
            from vllm.v1.attention.backends.mla.indexer import (
                DeepseekV32IndexerMetadata,
            )

            attn_metadata = get_forward_context().attn_metadata
            if isinstance(attn_metadata, dict):
                active_bounds = [
                    value.max_seq_len
                    for value in attn_metadata.values()
                    if isinstance(value, DeepseekV32IndexerMetadata)
                    and value.decode is not None
                ]
                if active_bounds:
                    max_context_len = min(max(active_bounds), max_model_len)
            return fp8_paged_mqa_logits_gfx90a(
                q_fp8,
                kv_cache_fp8,
                weights,
                context_lens,
                block_tables,
                max_model_len,
                max_context_len=max_context_len,
            )
        return original(
            q_fp8,
            kv_cache_fp8,
            weights,
            context_lens,
            block_tables,
            schedule_metadata,
            max_model_len,
        )

    setattr(_dispatch, _PAGED_MQA_DISPATCH_MARKER, _dispatch)
    return _dispatch


def _validate_fp8_paged_mqa_inputs(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
) -> tuple[int, int, int, int, torch.Tensor]:
    _require(q.device.type == "cuda", "q must be on a ROCm GPU")
    _require(q.dtype == torch.float8_e4m3fn, "q must use OCP torch.float8_e4m3fn")
    _require(q.ndim == 4, "q must have shape [batch, next_n, heads, head_dim]")
    batch_size, next_n, heads, head_dim = q.shape
    _require(next_n == 1, "only next_n == 1 decode is supported")
    _require(heads == 32, "only 32 indexer heads are supported")
    _require(head_dim == 128, "only indexer head dimension 128 is supported")
    _require(q.is_contiguous(), "q must be contiguous")

    _require(kv_cache.device == q.device, "kv_cache must be on the same device as q")
    _require(kv_cache.dtype == torch.uint8, "kv_cache must use packed uint8 storage")
    block_size = kv_cache.shape[1] if kv_cache.ndim in (3, 4) else 0
    packed_row_shape = tuple(kv_cache.shape[2:])
    _require(
        block_size in (1, 16, 32, 64) and packed_row_shape in ((head_dim + 4,), (1, head_dim + 4)),
        "kv_cache must have shape [num_blocks, block_size, 132] or "
        "[num_blocks, block_size, 1, 132] with block_size in {1, 16, 32, 64}; "
        f"got {tuple(kv_cache.shape)}",
    )
    _require(kv_cache.is_contiguous(), "kv_cache must be contiguous")

    _require(weights.device == q.device, "weights must be on the same device as q")
    _require(weights.dtype == torch.float32, "weights must use torch.float32")
    _require(
        tuple(weights.shape) == (batch_size, heads),
        "weights must have shape [batch, 32]",
    )
    _require(weights.is_contiguous(), "weights must be contiguous")

    _require(
        context_lens.device == q.device,
        "context_lens must remain on the same GPU as q",
    )
    _require(context_lens.dtype == torch.int32, "context_lens must use torch.int32")
    _require(
        context_lens.numel() == batch_size,
        "context_lens must contain one value per batch row",
    )
    _require(context_lens.is_contiguous(), "context_lens must be contiguous")

    _require(
        block_tables.device == q.device,
        "block_tables must be on the same device as q",
    )
    _require(block_tables.dtype == torch.int32, "block_tables must use torch.int32")
    _require(
        block_tables.ndim == 2 and block_tables.shape[0] == batch_size,
        "block_tables must have shape [batch, max_pages]",
    )
    _require(block_tables.is_contiguous(), "block_tables must be contiguous")
    _require(isinstance(max_model_len, int), "max_model_len must be an integer")
    _require(max_model_len > 0, "max_model_len must be positive")
    _require(
        max_model_len <= block_tables.shape[1] * block_size,
        "block_tables do not cover max_model_len",
    )
    return batch_size, heads, head_dim, block_size, context_lens.reshape(batch_size)


def fp8_paged_mqa_logits_gfx90a(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    max_context_len: int | None = None,
) -> torch.Tensor:
    """Compute GLM sparse-indexer decode logits without host synchronization.

    The only allocated workspace is the returned FP32 logits tensor. Inputs are
    read-only. Context lengths and page-table entries remain on the GPU.

    The upstream AITER target is
    ``aiter/ops/triton/_triton_kernels/attention/pa_mqa_logits.py`` with its
    public wrapper in ``aiter/ops/triton/attention/pa_mqa_logits.py``. vLLM
    should dispatch it from
    ``vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`` in
    ``rocm_fp8_paged_mqa_logits`` when AITER is unavailable on gfx90a.
    """

    batch_size, heads, head_dim, block_size, context_lens = _validate_fp8_paged_mqa_inputs(
        q,
        kv_cache,
        weights,
        context_lens,
        block_tables,
        max_model_len,
    )
    if max_context_len is None:
        max_context_len = max_model_len
    _require(
        type(max_context_len) is int,
        "max_context_len must be an integer",
    )
    _require(max_context_len > 0, "max_context_len must be positive")
    _require(
        max_context_len <= max_model_len,
        "max_context_len must not exceed max_model_len",
    )
    device_index = q.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    _require_gfx90a(device_index)

    packed_cache = kv_cache.view(kv_cache.shape[0], -1)
    value_width = block_size * head_dim
    cache_values = packed_cache[:, :value_width].view(torch.float8_e4m3fn)
    cache_scales = packed_cache[:, value_width:].view(torch.float32)
    q_decode = q[:, 0]
    logits = torch.full(
        (batch_size, max_model_len),
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    if block_size == 1:
        tokens_per_program = 16
        grid = (
            triton.cdiv(max_context_len, tokens_per_program),
            batch_size,
        )
        _fp8_token_paged_mqa_logits_gfx90a_kernel[grid](
            q_decode,
            q_decode.stride(0),
            q_decode.stride(1),
            q_decode.stride(2),
            cache_values,
            cache_values.stride(0),
            cache_values.stride(1),
            cache_scales,
            cache_scales.stride(0),
            weights,
            weights.stride(0),
            weights.stride(1),
            context_lens,
            block_tables,
            block_tables.stride(0),
            block_tables.stride(1),
            logits,
            logits.stride(0),
            logits.stride(1),
            kv_cache.shape[0],
            MAX_MODEL_LEN=max_model_len,
            HEADS=heads,
            HEAD_DIM=head_dim,
            TOKENS_PER_PROGRAM=tokens_per_program,
            num_warps=4,
            waves_per_eu=2,
        )
    else:
        grid = (triton.cdiv(max_context_len, block_size), batch_size)
        _fp8_paged_mqa_logits_gfx90a_kernel[grid](
            q_decode,
            q_decode.stride(0),
            q_decode.stride(1),
            q_decode.stride(2),
            cache_values,
            cache_values.stride(0),
            cache_values.stride(1),
            cache_scales,
            cache_scales.stride(0),
            cache_scales.stride(1),
            weights,
            weights.stride(0),
            weights.stride(1),
            context_lens,
            block_tables,
            block_tables.stride(0),
            block_tables.stride(1),
            logits,
            logits.stride(0),
            logits.stride(1),
            kv_cache.shape[0],
            MAX_MODEL_LEN=max_model_len,
            HEADS=heads,
            HEAD_DIM=head_dim,
            BLOCK_SIZE=block_size,
            num_warps=4,
            waves_per_eu=2,
        )
    return logits
