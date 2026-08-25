# SPDX-License-Identifier: Apache-2.0

import functools
import inspect
import os

import torch
from vllm.triton_utils import tl, triton

_DISPATCH_MARKER = "_prime_rl_gfx90a_hybrid_paged_attention"
_HEAD_SIZE = 128
_SUPPORTED_LOCAL_QUERY_HEADS = (4, 8, 16, 32)
_SUPPORTED_LOCAL_HEAD_PAIRS = ((4, 1), (8, 1), (16, 1), (32, 2))
_SUPPORTED_PHYSICAL_BLOCK_SIZES = (2048, 4096, 8192)
_TOKEN_TILE = 64
_PARTITION_SIZE = 1024
_MIN_SEQ_LEN = 4096
_MAX_SEQ_LEN = 262144


@functools.lru_cache(maxsize=None)
def _require_gfx90a(device_index: int) -> None:
    if torch.version.hip is None:
        raise RuntimeError("hybrid paged attention requires a ROCm PyTorch build")
    properties = torch.cuda.get_device_properties(device_index)
    architecture = getattr(properties, "gcnArchName", "").split(":")[0]
    if architecture != "gfx90a":
        raise RuntimeError(
            "hybrid paged attention is validated only on gfx90a; "
            f"device {device_index} reports {properties.gcnArchName}"
        )


@triton.jit
def _hybrid_paged_attention_partials_kernel(
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    partial_acc_ptr,
    scale,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    stride_k_cache_0: tl.int64,
    stride_k_cache_1: tl.int64,
    stride_k_cache_2: tl.int64,
    stride_k_cache_3: tl.int64,
    stride_k_cache_4: tl.int64,
    stride_v_cache_0: tl.int64,
    stride_v_cache_1: tl.int64,
    stride_v_cache_2: tl.int64,
    stride_v_cache_3: tl.int64,
    partial_scalar_stride_seq: tl.int64,
    partial_scalar_stride_head: tl.int64,
    partial_scalar_stride_partition: tl.int64,
    partial_acc_stride_seq: tl.int64,
    partial_acc_stride_head: tl.int64,
    partial_acc_stride_partition: tl.int64,
    partial_acc_stride_query: tl.int64,
    X_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    QUERY_HEADS_PER_KV: tl.constexpr,
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
    query_head_mask = query_head_offsets < QUERY_HEADS_PER_KV
    query_head_indices = kv_head_idx * QUERY_HEADS_PER_KV + query_head_offsets

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    partition_start = partition_idx * PARTITION_SIZE
    running_max = tl.full([16], float("-inf"), dtype=tl.float32)
    running_sum = tl.zeros([16], dtype=tl.float32)
    accumulator = tl.zeros([16, HEAD_SIZE], dtype=tl.float32)
    block_table_offset = seq_idx * block_table_stride

    for tile_idx in tl.range(0, PARTITION_SIZE // TOKEN_TILE):
        logical_tokens = partition_start + tile_idx * TOKEN_TILE + token_offsets
        valid_tokens = logical_tokens < seq_len
        logical_blocks = logical_tokens // PHYSICAL_BLOCK_SIZE
        physical_blocks = tl.load(
            block_tables_ptr + block_table_offset + logical_blocks,
            mask=valid_tokens,
            other=0,
        ).to(tl.int64)
        offsets_in_block = logical_tokens % PHYSICAL_BLOCK_SIZE

        value_offsets = (
            physical_blocks[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_1
            + dim_offsets[None, :] * stride_v_cache_2
            + offsets_in_block[:, None] * stride_v_cache_3
        )
        values = tl.load(
            value_cache_ptr + value_offsets,
            mask=valid_tokens[:, None],
            other=0.0,
            eviction_policy="evict_last",
        )
        scores = tl.zeros([16, TOKEN_TILE], dtype=tl.float32)
        x_offsets = tl.arange(0, X_SIZE)
        for head_chunk_idx in tl.static_range(0, HEAD_SIZE // X_SIZE):
            chunk_dims = head_chunk_idx * X_SIZE + x_offsets
            query_chunk = tl.load(
                query_ptr
                + seq_idx * query_stride_0
                + query_head_indices[:, None] * query_stride_1
                + chunk_dims[None, :],
                mask=query_head_mask[:, None],
                other=0.0,
            )
            key_chunk_offsets = (
                physical_blocks[:, None] * stride_k_cache_0
                + kv_head_idx * stride_k_cache_1
                + head_chunk_idx * stride_k_cache_2
                + offsets_in_block[:, None] * stride_k_cache_3
                + x_offsets[None, :] * stride_k_cache_4
            )
            key_chunk = tl.load(
                key_cache_ptr + key_chunk_offsets,
                mask=valid_tokens[:, None],
                other=0.0,
                eviction_policy="evict_last",
            )
            scores += tl.dot(query_chunk, tl.trans(key_chunk))
        scores *= scale
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
        correction = tl.where(running_max == float("-inf"), 0.0, correction)
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


@triton.jit
def _hybrid_paged_attention_reduce_kernel(
    output_ptr,
    seq_lens_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    partial_acc_ptr,
    max_partitions: tl.int32,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    partial_scalar_stride_seq: tl.int64,
    partial_scalar_stride_head: tl.int64,
    partial_scalar_stride_partition: tl.int64,
    partial_acc_stride_seq: tl.int64,
    partial_acc_stride_head: tl.int64,
    partial_acc_stride_partition: tl.int64,
    partial_acc_stride_query: tl.int64,
    HEAD_SIZE: tl.constexpr,
    QUERY_HEADS_PER_KV: tl.constexpr,
    PARTITION_SIZE: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    query_head_offsets = tl.arange(0, 16)
    dim_offsets = tl.arange(0, HEAD_SIZE)
    query_head_mask = query_head_offsets < QUERY_HEADS_PER_KV
    query_head_indices = kv_head_idx * QUERY_HEADS_PER_KV + query_head_offsets
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    num_partitions = (seq_len + PARTITION_SIZE - 1) // PARTITION_SIZE

    running_max = tl.full([16], float("-inf"), dtype=tl.float32)
    running_sum = tl.zeros([16], dtype=tl.float32)
    accumulator = tl.zeros([16, HEAD_SIZE], dtype=tl.float32)

    for partition_idx in tl.range(0, max_partitions):
        valid_partition = partition_idx < num_partitions
        scalar_offsets = (
            seq_idx * partial_scalar_stride_seq
            + kv_head_idx * partial_scalar_stride_head
            + partition_idx * partial_scalar_stride_partition
            + query_head_offsets
        )
        partition_max = tl.load(
            partial_max_ptr + scalar_offsets,
            mask=query_head_mask & valid_partition,
            other=float("-inf"),
        )
        partition_sum = tl.load(
            partial_sum_ptr + scalar_offsets,
            mask=query_head_mask & valid_partition,
            other=0.0,
        )
        acc_offsets = (
            seq_idx * partial_acc_stride_seq
            + kv_head_idx * partial_acc_stride_head
            + partition_idx * partial_acc_stride_partition
            + query_head_offsets[:, None] * partial_acc_stride_query
            + dim_offsets[None, :]
        )
        partition_acc = tl.load(
            partial_acc_ptr + acc_offsets,
            mask=query_head_mask[:, None] & valid_partition,
            other=0.0,
        )

        next_max = tl.maximum(running_max, partition_max)
        running_correction = tl.exp(running_max - next_max)
        running_correction = tl.where(
            running_max == float("-inf"),
            0.0,
            running_correction,
        )
        partition_correction = tl.exp(partition_max - next_max)
        partition_correction = tl.where(
            partition_max == float("-inf"),
            0.0,
            partition_correction,
        )
        running_sum = running_sum * running_correction + partition_sum * partition_correction
        accumulator = accumulator * running_correction[:, None] + partition_acc * partition_correction[:, None]
        running_max = next_max

    result = accumulator / (running_sum[:, None] + 1e-10)
    output_offsets = seq_idx * output_stride_0 + query_head_indices[:, None] * output_stride_1 + dim_offsets[None, :]
    tl.store(
        output_ptr + output_offsets,
        result,
        mask=query_head_mask[:, None],
    )


def can_dispatch_hybrid_paged_attention(
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
) -> bool:
    del key, value, k_scale, v_scale
    tensors = (
        query,
        output,
        key_cache,
        value_cache,
        block_table,
        query_start_loc,
        seq_lens,
    )
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        return False
    if (
        query.device.type != "cuda"
        or output.device != query.device
        or key_cache.device != query.device
        or value_cache.device != query.device
        or block_table.device != query.device
        or query_start_loc.device != query.device
        or seq_lens.device != query.device
    ):
        return False
    if (
        query.dtype != torch.bfloat16
        or output.dtype != torch.bfloat16
        or key_cache.dtype != torch.bfloat16
        or value_cache.dtype != torch.bfloat16
    ):
        return False
    if (
        query.ndim != 3
        or query.shape[1] not in _SUPPORTED_LOCAL_QUERY_HEADS
        or query.shape[2] != _HEAD_SIZE
        or output.shape != query.shape
        or query.stride(2) != 1
        or output.stride(2) != 1
    ):
        return False
    num_sequences = query.shape[0]
    if not 1 <= num_sequences <= 64:
        return False
    if (
        key_cache.ndim != 5
        or (query.shape[1], key_cache.shape[1]) not in _SUPPORTED_LOCAL_HEAD_PAIRS
        or key_cache.shape[2] * key_cache.shape[4] != _HEAD_SIZE
        or key_cache.shape[3] not in _SUPPORTED_PHYSICAL_BLOCK_SIZES
        or value_cache.ndim != 4
        or value_cache.shape[1] != key_cache.shape[1]
        or value_cache.shape[2] != _HEAD_SIZE
        or value_cache.shape[3] != key_cache.shape[3]
    ):
        return False
    if (
        type(max_seq_len) is not int
        or type(max_query_len) is not int
        or type(is_block_table_ptr) is not bool
        or type(causal) is not bool
    ):
        return False
    if (
        block_table.dtype != torch.int32
        or block_table.ndim != 2
        or block_table.shape[0] < num_sequences
        or block_table.shape[1] * key_cache.shape[3] < max_seq_len
        or query_start_loc.dtype != torch.int32
        or query_start_loc.numel() != num_sequences + 1
        or seq_lens.dtype != torch.int32
        or seq_lens.numel() != num_sequences
    ):
        return False
    if (
        kv_cache_dtype != "auto"
        or not _MIN_SEQ_LEN <= max_seq_len <= _MAX_SEQ_LEN
        or max_query_len != 1
        or alibi_slopes is not None
        or sliding_window not in (None, -1, 0)
        or output_scale is not None
        or sinks is not None
        or is_block_table_ptr
        or not causal
        or (sm_scale is not None and not isinstance(sm_scale, float))
    ):
        return False
    return True


def hybrid_paged_attention_gfx90a(
    query,
    output,
    key_cache,
    value_cache,
    block_table,
    seq_lens,
    max_seq_len,
    sm_scale,
    *,
    partition_size=_PARTITION_SIZE,
    token_tile=_TOKEN_TILE,
    num_warps=4,
    num_stages=2,
    waves_per_eu=0,
) -> None:
    _require_gfx90a(query.device.index)
    native_value = os.environ.get("PRIME_ROCM_GFX90A_AITER_PAGED_ATTN")
    if native_value not in (None, "0", "1"):
        raise ValueError(f"PRIME_ROCM_GFX90A_AITER_PAGED_ATTN must be unset, '0', or '1'; got {native_value!r}")
    if native_value == "1":
        from prime_rl.inference.vllm.kernels.rocm_aiter_native_paged_attention import (
            aiter_native_paged_attention_gfx90a,
        )

        aiter_native_paged_attention_gfx90a(
            query=query,
            output=output,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            sm_scale=sm_scale,
        )
        return
    if partition_size not in (256, 512, 1024, 2048, 4096):
        raise ValueError(f"unsupported partition_size={partition_size}")
    if token_tile not in (32, 64, 128) or partition_size % token_tile:
        raise ValueError(f"unsupported token_tile={token_tile}")
    if num_warps not in (1, 2, 4, 8):
        raise ValueError(f"unsupported num_warps={num_warps}")
    if num_stages not in (1, 2, 3):
        raise ValueError(f"unsupported num_stages={num_stages}")
    if waves_per_eu not in (0, 1, 2, 3, 4):
        raise ValueError(f"unsupported waves_per_eu={waves_per_eu}")
    if key_cache.dtype != torch.bfloat16 or value_cache.dtype != torch.bfloat16:
        raise ValueError("key_cache and value_cache must both use bfloat16")
    num_sequences = query.shape[0]
    local_kv_heads = key_cache.shape[1]
    query_heads_per_kv = query.shape[1] // local_kv_heads
    num_partitions = (max_seq_len + partition_size - 1) // partition_size
    partial_shape = (
        num_sequences,
        local_kv_heads,
        num_partitions,
        16,
    )
    partial_max = torch.empty(partial_shape, device=query.device, dtype=torch.float32)
    partial_sum = torch.empty_like(partial_max)
    partial_acc = torch.empty(
        (*partial_shape, _HEAD_SIZE),
        device=query.device,
        dtype=torch.float32,
    )
    scale = sm_scale if sm_scale is not None else _HEAD_SIZE**-0.5
    _hybrid_paged_attention_partials_kernel[(num_sequences, local_kv_heads, num_partitions)](
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        partial_max,
        partial_sum,
        partial_acc,
        scale,
        block_table.stride(0),
        query.stride(0),
        query.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        key_cache.stride(4),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        partial_max.stride(0),
        partial_max.stride(1),
        partial_max.stride(2),
        partial_acc.stride(0),
        partial_acc.stride(1),
        partial_acc.stride(2),
        partial_acc.stride(3),
        X_SIZE=key_cache.shape[4],
        HEAD_SIZE=_HEAD_SIZE,
        QUERY_HEADS_PER_KV=query_heads_per_kv,
        PHYSICAL_BLOCK_SIZE=key_cache.shape[3],
        TOKEN_TILE=token_tile,
        PARTITION_SIZE=partition_size,
        num_warps=num_warps,
        num_stages=num_stages,
        waves_per_eu=waves_per_eu,
    )
    _hybrid_paged_attention_reduce_kernel[(num_sequences, local_kv_heads)](
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
        QUERY_HEADS_PER_KV=query_heads_per_kv,
        PARTITION_SIZE=partition_size,
        num_warps=num_warps,
        num_stages=num_stages,
        waves_per_eu=waves_per_eu,
    )


def wrap_hybrid_paged_attention(original):
    if getattr(original, _DISPATCH_MARKER, False):
        return original

    signature = inspect.signature(original)

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        if can_dispatch_hybrid_paged_attention(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            hybrid_paged_attention_gfx90a(
                query=bound.arguments["query"],
                output=bound.arguments["output"],
                key_cache=bound.arguments["key_cache"],
                value_cache=bound.arguments["value_cache"],
                block_table=bound.arguments["block_table"],
                seq_lens=bound.arguments["seq_lens"],
                max_seq_len=bound.arguments["max_seq_len"],
                sm_scale=bound.arguments["sm_scale"],
            )
            return None
        return original(*args, **kwargs)

    setattr(wrapped, _DISPATCH_MARKER, True)
    return wrapped
