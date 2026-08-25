# SPDX-License-Identifier: Apache-2.0

import functools

import torch
from vllm.triton_utils import tl, triton

from prime_rl.inference.vllm.kernels.rocm_hybrid_paged_attention import (
    _hybrid_paged_attention_reduce_kernel,
)

_HEAD_SIZE = 128
_LOCAL_QUERY_HEADS = 4
_LOCAL_KV_HEADS = 1
_TOKEN_TILE = 32
_PARTITION_SIZE = 1024
_MAX_SEQ_LEN = 262144
_WRAPPER_MARKER = "_prime_rl_nemotron_int8_paged_attention"

@functools.lru_cache(maxsize=None)
def _require_gfx90a(device_index: int) -> None:
    if torch.version.hip is None:
        raise RuntimeError("INT8 paged attention requires ROCm")
    properties = torch.cuda.get_device_properties(device_index)
    architecture = getattr(properties, "gcnArchName", "").split(":")[0]
    if architecture != "gfx90a":
        raise RuntimeError(
            f"INT8 paged attention is validated only on gfx90a; device {device_index} reports {properties.gcnArchName}"
        )


@triton.jit
def _int8_paged_attention_partials_kernel(
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    key_scale_ptr,
    value_scale_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    partial_acc_ptr,
    scale,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    key_stride_0: tl.int64,
    key_stride_1: tl.int64,
    key_stride_2: tl.int64,
    key_stride_3: tl.int64,
    value_stride_0: tl.int64,
    value_stride_1: tl.int64,
    value_stride_2: tl.int64,
    value_stride_3: tl.int64,
    key_scale_stride_0: tl.int64,
    key_scale_stride_1: tl.int64,
    key_scale_stride_2: tl.int64,
    value_scale_stride_0: tl.int64,
    value_scale_stride_1: tl.int64,
    value_scale_stride_2: tl.int64,
    block_table_stride: tl.int64,
    partial_scalar_stride_seq: tl.int64,
    partial_scalar_stride_head: tl.int64,
    partial_scalar_stride_partition: tl.int64,
    partial_acc_stride_seq: tl.int64,
    partial_acc_stride_head: tl.int64,
    partial_acc_stride_partition: tl.int64,
    partial_acc_stride_query: tl.int64,
    HEAD_SIZE: tl.constexpr,
    LOCAL_QUERY_HEADS: tl.constexpr,
    PHYSICAL_BLOCK_SIZE: tl.constexpr,
    TOKEN_TILE: tl.constexpr,
    PARTITION_SIZE: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    partition_idx = tl.program_id(2)
    query_head_offsets = tl.arange(0, 16)
    dim_offsets = tl.arange(0, HEAD_SIZE)
    token_offsets = tl.arange(0, TOKEN_TILE)
    query_head_mask = query_head_offsets < LOCAL_QUERY_HEADS

    query = tl.load(
        query_ptr + seq_idx * query_stride_0 + query_head_offsets[:, None] * query_stride_1 + dim_offsets[None, :],
        mask=query_head_mask[:, None],
        other=0.0,
    ).to(tl.float16)
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    partition_start = partition_idx * PARTITION_SIZE
    running_max = tl.full([16], float("-inf"), dtype=tl.float32)
    running_sum = tl.zeros([16], dtype=tl.float32)
    accumulator = tl.zeros([16, HEAD_SIZE], dtype=tl.float32)

    for tile_idx in tl.range(0, PARTITION_SIZE // TOKEN_TILE):
        logical_tokens = partition_start + tile_idx * TOKEN_TILE + token_offsets
        valid_tokens = logical_tokens < seq_len
        physical_blocks = tl.load(
            block_tables_ptr + seq_idx * block_table_stride + logical_tokens // PHYSICAL_BLOCK_SIZE,
            mask=valid_tokens,
            other=0,
        ).to(tl.int64)
        offsets_in_block = logical_tokens % PHYSICAL_BLOCK_SIZE
        key_offsets = (
            physical_blocks[:, None] * key_stride_0
            + offsets_in_block[:, None] * key_stride_1
            + kv_head_idx * key_stride_2
            + dim_offsets[None, :] * key_stride_3
        )
        value_offsets = (
            physical_blocks[:, None] * value_stride_0
            + offsets_in_block[:, None] * value_stride_1
            + kv_head_idx * value_stride_2
            + dim_offsets[None, :] * value_stride_3
        )
        key_scale_offsets = (
            physical_blocks * key_scale_stride_0
            + offsets_in_block * key_scale_stride_1
            + kv_head_idx * key_scale_stride_2
        )
        value_scale_offsets = (
            physical_blocks * value_scale_stride_0
            + offsets_in_block * value_scale_stride_1
            + kv_head_idx * value_scale_stride_2
        )
        key_scales = tl.load(
            key_scale_ptr + key_scale_offsets,
            mask=valid_tokens,
            other=0.0,
        )
        value_scales = tl.load(
            value_scale_ptr + value_scale_offsets,
            mask=valid_tokens,
            other=0.0,
        )
        keys = tl.load(
            key_cache_ptr + key_offsets,
            mask=valid_tokens[:, None],
            other=0,
            eviction_policy="evict_last",
        ).to(tl.float16)
        values = tl.load(
            value_cache_ptr + value_offsets,
            mask=valid_tokens[:, None],
            other=0,
            eviction_policy="evict_last",
        ).to(tl.float16)
        keys *= key_scales[:, None].to(tl.float16)
        values *= value_scales[:, None].to(tl.float16)

        scores = scale * tl.dot(query, tl.trans(keys))
        scores = tl.where(
            query_head_mask[:, None] & valid_tokens[None, :],
            scores,
            float("-inf"),
        )
        tile_max = tl.max(scores, axis=1)
        next_max = tl.maximum(running_max, tile_max)
        probabilities = tl.exp(scores - next_max[:, None])
        probabilities = tl.where(
            next_max[:, None] == float("-inf"),
            0.0,
            probabilities,
        )
        correction = tl.exp(running_max - next_max)
        correction = tl.where(
            running_max == float("-inf"),
            0.0,
            correction,
        )
        running_sum = running_sum * correction + tl.sum(probabilities, axis=1)
        accumulator = accumulator * correction[:, None]
        accumulator += tl.dot(probabilities.to(values.dtype), values)
        running_max = next_max

    scalar_offsets = (
        seq_idx * partial_scalar_stride_seq
        + kv_head_idx * partial_scalar_stride_head
        + partition_idx * partial_scalar_stride_partition
        + query_head_offsets
    )
    tl.store(
        partial_max_ptr + scalar_offsets,
        running_max,
        mask=query_head_mask,
    )
    tl.store(
        partial_sum_ptr + scalar_offsets,
        running_sum,
        mask=query_head_mask,
    )
    acc_offsets = (
        seq_idx * partial_acc_stride_seq
        + kv_head_idx * partial_acc_stride_head
        + partition_idx * partial_acc_stride_partition
        + query_head_offsets[:, None] * partial_acc_stride_query
        + dim_offsets[None, :]
    )
    tl.store(
        partial_acc_ptr + acc_offsets,
        accumulator,
        mask=query_head_mask[:, None],
    )


def int8_per_token_head_paged_attention_gfx90a(
    query: torch.Tensor,
    output: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    key_scale_cache: torch.Tensor,
    value_scale_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    sm_scale: float,
) -> None:
    _require_gfx90a(query.device.index)
    if (
        query.dtype != torch.bfloat16
        or key_cache.dtype != torch.int8
        or value_cache.dtype != torch.int8
        or key_scale_cache.dtype != torch.float32
        or value_scale_cache.dtype != torch.float32
    ):
        raise ValueError("unsupported INT8 paged-attention dtypes")
    num_sequences = query.shape[0]
    num_partitions = (max_seq_len + _PARTITION_SIZE - 1) // _PARTITION_SIZE
    partial_shape = (
        num_sequences,
        _LOCAL_KV_HEADS,
        num_partitions,
        16,
    )
    partial_max = torch.empty(
        partial_shape,
        device=query.device,
        dtype=torch.float32,
    )
    partial_sum = torch.empty_like(partial_max)
    partial_acc = torch.empty(
        (*partial_shape, _HEAD_SIZE),
        device=query.device,
        dtype=torch.float32,
    )
    physical_block_size = key_cache.shape[1]
    grid = (num_sequences, _LOCAL_KV_HEADS, num_partitions)
    _int8_paged_attention_partials_kernel[grid](
        query,
        key_cache,
        value_cache,
        key_scale_cache,
        value_scale_cache,
        block_table,
        seq_lens,
        partial_max,
        partial_sum,
        partial_acc,
        sm_scale,
        query.stride(0),
        query.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        key_scale_cache.stride(0),
        key_scale_cache.stride(1),
        key_scale_cache.stride(2),
        value_scale_cache.stride(0),
        value_scale_cache.stride(1),
        value_scale_cache.stride(2),
        block_table.stride(0),
        partial_max.stride(0),
        partial_max.stride(1),
        partial_max.stride(2),
        partial_acc.stride(0),
        partial_acc.stride(1),
        partial_acc.stride(2),
        partial_acc.stride(3),
        HEAD_SIZE=_HEAD_SIZE,
        LOCAL_QUERY_HEADS=_LOCAL_QUERY_HEADS,
        PHYSICAL_BLOCK_SIZE=physical_block_size,
        TOKEN_TILE=_TOKEN_TILE,
        PARTITION_SIZE=_PARTITION_SIZE,
        num_warps=4,
        num_stages=1,
    )
    _hybrid_paged_attention_reduce_kernel[(num_sequences, _LOCAL_KV_HEADS)](
        output,
        seq_lens,
        partial_max,
        partial_sum,
        partial_acc,
        num_partitions,
        output.stride(0),
        output.stride(1),
        partial_max.stride(0),
        partial_max.stride(1),
        partial_max.stride(2),
        partial_acc.stride(0),
        partial_acc.stride(1),
        partial_acc.stride(2),
        partial_acc.stride(3),
        HEAD_SIZE=_HEAD_SIZE,
        LOCAL_QUERY_HEADS=_LOCAL_QUERY_HEADS,
        PARTITION_SIZE=_PARTITION_SIZE,
        num_warps=4,
    )


def can_dispatch_int8_per_token_head(
    impl,
    query,
    kv_cache,
    attn_metadata,
    output,
) -> bool:
    if not all(isinstance(tensor, torch.Tensor) for tensor in (query, kv_cache, output)):
        return False
    if attn_metadata is None:
        return False
    num_actual_tokens = getattr(attn_metadata, "num_actual_tokens", None)
    max_query_len = getattr(attn_metadata, "max_query_len", None)
    max_seq_len = getattr(attn_metadata, "max_seq_len", None)
    seq_lens = getattr(attn_metadata, "seq_lens", None)
    block_table = getattr(attn_metadata, "block_table", None)
    return (
        getattr(impl, "kv_cache_dtype", None) == "int8_per_token_head"
        and getattr(impl, "num_heads", None) == _LOCAL_QUERY_HEADS
        and getattr(impl, "num_kv_heads", None) == _LOCAL_KV_HEADS
        and getattr(impl, "head_size", None) == _HEAD_SIZE
        and getattr(impl, "alibi_slopes", None) is None
        and getattr(impl, "sinks", None) is None
        and getattr(impl, "logits_soft_cap", None) == 0
        and getattr(attn_metadata, "causal", None) is True
        and max_query_len == 1
        and type(max_seq_len) is int
        and 4096 <= max_seq_len <= _MAX_SEQ_LEN
        and type(num_actual_tokens) is int
        and 1 <= num_actual_tokens <= 64
        and query.dtype == torch.bfloat16
        and query.device.type == "cuda"
        and output.device == query.device
        and kv_cache.device == query.device
        and query.ndim == 3
        and tuple(query.shape[1:])
        == (
            _LOCAL_QUERY_HEADS,
            _HEAD_SIZE,
        )
        and output.shape == query.shape
        and kv_cache.dtype == torch.int8
        and kv_cache.ndim == 5
        and kv_cache.shape[1] == 2
        and kv_cache.shape[3] == _LOCAL_KV_HEADS
        and kv_cache.shape[4] == _HEAD_SIZE + 4
        and isinstance(seq_lens, torch.Tensor)
        and isinstance(block_table, torch.Tensor)
        and seq_lens.device == query.device
        and block_table.device == query.device
        and seq_lens.dtype == torch.int32
        and block_table.dtype == torch.int32
        and seq_lens.ndim == 1
        and block_table.ndim == 2
        and seq_lens.numel() >= num_actual_tokens
        and block_table.shape[0] >= num_actual_tokens
    )


def can_dispatch_int8_prefix_prefill(
    impl,
    query,
    key,
    value,
    kv_cache,
    attn_metadata,
    output,
) -> bool:
    tensors = (query, key, value, kv_cache, output)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        return False
    if attn_metadata is None:
        return False
    query_start_loc = getattr(attn_metadata, "query_start_loc", None)
    seq_lens = getattr(attn_metadata, "seq_lens", None)
    block_table = getattr(attn_metadata, "block_table", None)
    metadata_tensors = (query_start_loc, seq_lens, block_table)
    if not all(isinstance(tensor, torch.Tensor) for tensor in metadata_tensors):
        return False
    num_actual_tokens = getattr(attn_metadata, "num_actual_tokens", None)
    max_query_len = getattr(attn_metadata, "max_query_len", None)
    max_seq_len = getattr(attn_metadata, "max_seq_len", None)
    batch_size = query_start_loc.numel() - 1
    return (
        getattr(impl, "kv_cache_dtype", None) == "int8_per_token_head"
        and getattr(impl, "num_heads", None) == _LOCAL_QUERY_HEADS
        and getattr(impl, "num_kv_heads", None) == _LOCAL_KV_HEADS
        and getattr(impl, "head_size", None) == _HEAD_SIZE
        and getattr(impl, "alibi_slopes", None) is None
        and getattr(impl, "sinks", None) is None
        and getattr(impl, "logits_soft_cap", None) == 0
        and getattr(impl, "sliding_window", None) == (-1, -1)
        and getattr(attn_metadata, "causal", None) is True
        and getattr(attn_metadata, "use_cascade", None) is False
        and type(num_actual_tokens) is int
        and 2 <= num_actual_tokens <= 32768
        and type(max_query_len) is int
        and 2 <= max_query_len <= 32768
        and type(max_seq_len) is int
        and max_query_len <= max_seq_len <= _MAX_SEQ_LEN
        and 1 <= batch_size <= 64
        and query.dtype == torch.bfloat16
        and key.dtype == torch.bfloat16
        and value.dtype == torch.bfloat16
        and output.dtype == torch.bfloat16
        and query.device.type == "cuda"
        and all(tensor.device == query.device for tensor in tensors[1:])
        and all(tensor.device == query.device for tensor in metadata_tensors)
        and query.ndim == 3
        and tuple(query.shape[1:]) == (_LOCAL_QUERY_HEADS, _HEAD_SIZE)
        and key.ndim == 3
        and tuple(key.shape[1:]) == (_LOCAL_KV_HEADS, _HEAD_SIZE)
        and value.shape == key.shape
        and output.shape == query.shape
        and num_actual_tokens <= query.shape[0]
        and num_actual_tokens <= key.shape[0]
        and kv_cache.dtype == torch.int8
        and kv_cache.ndim == 5
        and kv_cache.shape[1] == 2
        and kv_cache.shape[2] == 2048
        and kv_cache.shape[3] == _LOCAL_KV_HEADS
        and kv_cache.shape[4] == _HEAD_SIZE + 4
        and query_start_loc.dtype == torch.int32
        and query_start_loc.ndim == 1
        and seq_lens.dtype == torch.int32
        and seq_lens.ndim == 1
        and seq_lens.numel() >= batch_size
        and block_table.dtype == torch.int32
        and block_table.ndim == 2
        and block_table.shape[0] >= batch_size
        and block_table.shape[1] * 2048 >= max_seq_len
    )


def int8_prefix_prefill_gfx90a(
    query,
    key,
    value,
    output,
    key_cache,
    value_cache,
    key_scale_cache,
    value_scale_cache,
    block_table,
    query_start_loc,
    seq_lens,
    max_query_len,
    sm_scale,
) -> None:
    from vllm.v1.attention.ops import prefix_prefill

    _require_gfx90a(query.device.index)
    # Measured at both the first 32k chunk and a 96k cached prefix. Keep
    # BLOCK_N=64 for the exact-output reduction order.
    block_m = 256
    num_warps = 8
    batch_size = query_start_loc.numel() - 1
    grid = (
        batch_size,
        _LOCAL_QUERY_HEADS,
        triton.cdiv(max_query_len, block_m),
    )
    prefix_prefill._fwd_kernel[grid](
        query,
        key,
        value,
        key_cache,
        value_cache,
        None,
        block_table,
        sm_scale,
        key_scale_cache,
        value_scale_cache,
        1.0,
        query_start_loc,
        seq_lens,
        1,
        output,
        block_table.stride(0),
        block_table.stride(1),
        query.stride(0),
        query.stride(1),
        query.stride(2),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        stride_k_cache_bs=key_cache.stride(0),
        stride_k_cache_h=key_cache.stride(2),
        stride_k_cache_d=key_cache.stride(3),
        stride_k_cache_bl=key_cache.stride(1),
        stride_k_cache_x=0,
        stride_v_cache_bs=value_cache.stride(0),
        stride_v_cache_h=value_cache.stride(2),
        stride_v_cache_d=value_cache.stride(3),
        stride_v_cache_bl=value_cache.stride(1),
        num_queries_per_kv=_LOCAL_QUERY_HEADS,
        IN_PRECISION=None,
        BLOCK_M=block_m,
        BLOCK_DMODEL=_HEAD_SIZE,
        BLOCK_DMODEL_PADDED=_HEAD_SIZE,
        BLOCK_SIZE=128,
        PHYSICAL_BLOCK_SIZE=2048,
        BLOCK_N=64,
        SLIDING_WINDOW=0,
        num_unroll_cache=1,
        num_unroll_request=1,
        SKIP_DECODE=True,
        USE_SINKS=False,
        USE_FP8=False,
        CAUSAL=True,
        num_warps=num_warps,
        num_stages=1,
    )


def wrap_triton_int8_attention_forward(original):
    if getattr(original, _WRAPPER_MARKER, False):
        return original

    @functools.wraps(original)
    def wrapped(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        if (
            output_scale is None
            and output_block_scale is None
            and can_dispatch_int8_per_token_head(
                self,
                query,
                kv_cache,
                attn_metadata,
                output,
            )
        ):
            self._ensure_scale_caches(kv_cache)
            key_cache, value_cache = kv_cache.unbind(1)
            num_actual_tokens = attn_metadata.num_actual_tokens
            int8_per_token_head_paged_attention_gfx90a(
                query[:num_actual_tokens],
                output[:num_actual_tokens],
                key_cache,
                value_cache,
                self._k_scale_cache,
                self._v_scale_cache,
                attn_metadata.block_table,
                attn_metadata.seq_lens,
                attn_metadata.max_seq_len,
                self.scale,
            )
            return output
        if (
            output_scale is None
            and output_block_scale is None
            and can_dispatch_int8_prefix_prefill(
                self,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
            )
        ):
            self._ensure_scale_caches(kv_cache)
            key_cache, value_cache = kv_cache.unbind(1)
            num_actual_tokens = attn_metadata.num_actual_tokens
            int8_prefix_prefill_gfx90a(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                key_cache,
                value_cache,
                self._k_scale_cache,
                self._v_scale_cache,
                attn_metadata.block_table,
                attn_metadata.query_start_loc,
                attn_metadata.seq_lens,
                attn_metadata.max_query_len,
                self.scale,
            )
            return output

        return original(
            self,
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        )

    setattr(wrapped, _WRAPPER_MARKER, True)
    return wrapped
