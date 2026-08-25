# SPDX-License-Identifier: Apache-2.0

import functools
import inspect

import torch
from vllm.triton_utils import triton
from vllm.v1.attention.ops import prefix_prefill

from prime_rl.inference.vllm.kernels.rocm_hybrid_paged_attention import (
    _require_gfx90a,
)

_DISPATCH_MARKER = "_prime_rl_gfx90a_hybrid_prefix_prefill"
_HEAD_SIZE = 128
_LOCAL_QUERY_HEADS = 4
_LOCAL_KV_HEADS = 1
_PHYSICAL_BLOCK_SIZE = 2048
_MAX_INPUT_LEN = 32768
_MAX_SEQ_LEN = 262144


def can_dispatch_hybrid_prefix_prefill(
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
) -> bool:
    tensors = (
        q,
        k,
        v,
        o,
        k_cache,
        v_cache,
        b_loc,
        b_start_loc,
        b_seq_len,
        k_scale,
        v_scale,
    )
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        return False
    if q.device.type != "cuda" or not all(tensor.device == q.device for tensor in tensors[1:]):
        return False
    if (
        q.dtype != torch.bfloat16
        or k.dtype != torch.bfloat16
        or v.dtype != torch.bfloat16
        or o.dtype != torch.bfloat16
        or k_cache.dtype != torch.bfloat16
        or v_cache.dtype != torch.bfloat16
    ):
        return False
    if (
        q.ndim != 3
        or tuple(q.shape[1:]) != (_LOCAL_QUERY_HEADS, _HEAD_SIZE)
        or k.shape != (q.shape[0], _LOCAL_KV_HEADS, _HEAD_SIZE)
        or v.shape != k.shape
        or o.shape != q.shape
        or q.stride(2) != 1
        or k.stride(2) != 1
        or v.stride(2) != 1
        or o.stride(2) != 1
    ):
        return False
    if (
        k_cache.ndim != 5
        or k_cache.shape[1] != _LOCAL_KV_HEADS
        or k_cache.shape[2] * k_cache.shape[4] != _HEAD_SIZE
        or k_cache.shape[3] != _PHYSICAL_BLOCK_SIZE
        or v_cache.ndim != 4
        or tuple(v_cache.shape[1:]) != (_LOCAL_KV_HEADS, _HEAD_SIZE, _PHYSICAL_BLOCK_SIZE)
    ):
        return False
    batch_size = b_seq_len.numel()
    if not 1 <= batch_size <= 64:
        return False
    if (
        b_loc.dtype != torch.int32
        or b_loc.ndim != 2
        or b_loc.shape[0] < batch_size
        or b_loc.shape[1] * _PHYSICAL_BLOCK_SIZE < max_seq_len
        or b_start_loc.dtype != torch.int32
        or b_start_loc.numel() != batch_size + 1
        or b_seq_len.dtype != torch.int32
        or b_seq_len.ndim != 1
    ):
        return False
    if (
        type(max_seq_len) is not int
        or type(max_input_len) is not int
        or type(is_block_table_ptr) is not bool
        or type(causal) is not bool
    ):
        return False
    if (
        kv_cache_dtype != "auto"
        or not 2 <= max_input_len <= _MAX_INPUT_LEN
        or not max_input_len <= max_seq_len <= _MAX_SEQ_LEN
        or max_seq_len == max_input_len
        or alibi_slopes is not None
        or sliding_window not in (None, -1, 0)
        or fp8_out_scale is not None
        or sinks is not None
        or is_block_table_ptr
        or not causal
        or not skip_decode
        or (sm_scale is not None and not isinstance(sm_scale, float))
    ):
        return False
    return True


def hybrid_prefix_prefill_gfx90a(
    q,
    k,
    v,
    o,
    k_cache,
    v_cache,
    b_loc,
    b_start_loc,
    b_seq_len,
    max_seq_len,
    max_input_len,
    k_scale,
    v_scale,
    sm_scale,
) -> None:
    _require_gfx90a(q.device.index)
    grid = (
        b_seq_len.numel(),
        _LOCAL_QUERY_HEADS,
        triton.cdiv(max_input_len, 128),
    )
    prefix_prefill._fwd_kernel[grid](
        q,
        k,
        v,
        k_cache,
        v_cache,
        None,
        b_loc,
        sm_scale if sm_scale is not None else _HEAD_SIZE**-0.5,
        k_scale,
        v_scale,
        1.0,
        b_start_loc,
        b_seq_len,
        k_cache.shape[4],
        o,
        b_loc.stride(0),
        b_loc.stride(1),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        stride_k_cache_bs=k_cache.stride(0),
        stride_k_cache_h=k_cache.stride(1),
        stride_k_cache_d=k_cache.stride(2),
        stride_k_cache_bl=k_cache.stride(3),
        stride_k_cache_x=k_cache.stride(4),
        stride_v_cache_bs=v_cache.stride(0),
        stride_v_cache_h=v_cache.stride(1),
        stride_v_cache_d=v_cache.stride(2),
        stride_v_cache_bl=v_cache.stride(3),
        BLOCK_SIZE=128,
        PHYSICAL_BLOCK_SIZE=_PHYSICAL_BLOCK_SIZE,
        num_queries_per_kv=_LOCAL_QUERY_HEADS,
        IN_PRECISION=None,
        BLOCK_DMODEL=_HEAD_SIZE,
        BLOCK_DMODEL_PADDED=_HEAD_SIZE,
        SLIDING_WINDOW=0,
        SKIP_DECODE=True,
        USE_FP8=False,
        BLOCK_M=128,
        BLOCK_N=64,
        num_unroll_cache=1,
        num_unroll_request=1,
        num_warps=4,
        num_stages=1,
        USE_SINKS=False,
        CAUSAL=True,
    )


def wrap_hybrid_prefix_prefill(original):
    if getattr(original, _DISPATCH_MARKER, False):
        return original

    signature = inspect.signature(original)

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        if can_dispatch_hybrid_prefix_prefill(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            hybrid_prefix_prefill_gfx90a(
                **{
                    name: bound.arguments[name]
                    for name in (
                        "q",
                        "k",
                        "v",
                        "o",
                        "k_cache",
                        "v_cache",
                        "b_loc",
                        "b_start_loc",
                        "b_seq_len",
                        "max_seq_len",
                        "max_input_len",
                        "k_scale",
                        "v_scale",
                        "sm_scale",
                    )
                }
            )
            return None
        return original(*args, **kwargs)

    setattr(wrapped, _DISPATCH_MARKER, True)
    return wrapped
