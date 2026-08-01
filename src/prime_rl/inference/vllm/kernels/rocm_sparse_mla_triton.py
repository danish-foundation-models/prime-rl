# SPDX-License-Identifier: Apache-2.0

"""Triton implementation details for GLM BF16 sparse MLA on ROCm.

The attention kernel is adapted from vLLM's Apache-2.0
``vllm.v1.attention.ops.xpu_mla_sparse`` implementation at commit
``ee0da84ab9e04ac7610e28580af62c365e898389``. It specializes the static
top-k loop that performs best on gfx90a and accepts vLLM's ragged metadata
through a bounded int32 packing workspace.
"""

import functools

import torch
from vllm.triton_utils import LOG2E, tl, triton

_GLM_HEAD_DIM = 576
_GLM_NOPE_DIM = 512
_GLM_ROPE_DIM = 64
_GLM_VALUE_DIM = 512
_GLM_INDEX_TOPK = 2048


@functools.lru_cache(maxsize=None)
def _require_gfx90a(device_index: int) -> None:
    if torch.version.hip is None:
        raise RuntimeError("GLM sparse MLA requires a ROCm PyTorch build")
    properties = torch.cuda.get_device_properties(device_index)
    architecture = getattr(properties, "gcnArchName", "").split(":")[0]
    if architecture != "gfx90a":
        raise RuntimeError(
            f"GLM sparse MLA is validated only on gfx90a; device {device_index} reports {properties.gcnArchName}"
        )


@triton.jit
def _pack_ragged_indices_kernel(
    ragged_indices_ptr,
    ragged_indptr_ptr,
    dense_indices_ptr,
    dense_stride,
    num_ragged_indices,
    BLOCK_SIZE: tl.constexpr,
    INDEX_TOPK: tl.constexpr,
):
    row_id = tl.program_id(0)
    block_id = tl.program_id(1)
    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_start = tl.load(ragged_indptr_ptr + row_id)
    row_end = tl.load(ragged_indptr_ptr + row_id + 1)
    row_length = row_end - row_start
    # Metadata is produced internally by vLLM, but keep every pointer
    # dereference in-bounds if a malformed row pointer reaches this boundary.
    # An invalid row becomes all -1 and therefore contributes a zero output.
    valid_row = (row_start >= 0) & (row_end >= row_start) & (row_end <= num_ragged_indices) & (row_length <= INDEX_TOPK)
    valid = valid_row & (offsets < row_length) & (offsets < INDEX_TOPK)
    safe_row_start = tl.where(valid_row, row_start, 0)
    selected = tl.load(
        ragged_indices_ptr + safe_row_start + offsets,
        mask=valid,
        other=-1,
    )
    tl.store(
        dense_indices_ptr + row_id * dense_stride + offsets,
        tl.where(valid, selected, -1),
        mask=offsets < INDEX_TOPK,
    )


@triton.jit
def _glm_sparse_mla_bf16_kernel(
    q_ptr,
    kv_ptr,
    indices_ptr,
    output_ptr,
    q_stride_token,
    q_stride_head,
    kv_stride_token,
    indices_stride_token,
    output_stride_token,
    output_stride_head,
    num_heads,
    num_kv_tokens,
    sm_scale_log2,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    INDEX_TOPK: tl.constexpr,
):
    query_id = tl.program_id(0)
    head_block_id = tl.program_id(1)

    head_offsets = head_block_id * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = head_offsets < num_heads
    nope_offsets = tl.arange(0, NOPE_DIM)
    rope_offsets = tl.arange(0, ROPE_DIM)
    value_offsets = tl.arange(0, VALUE_DIM)

    q_base = q_ptr + query_id * q_stride_token + head_offsets[:, None] * q_stride_head
    q_nope = tl.load(
        q_base + nope_offsets[None, :],
        mask=head_mask[:, None],
        other=0.0,
    )
    q_rope = tl.load(
        q_base + NOPE_DIM + rope_offsets[None, :],
        mask=head_mask[:, None],
        other=0.0,
    )

    negative_large = -3.4028234663852886e38
    running_max = tl.full((BLOCK_H,), negative_large, dtype=tl.float32)
    running_sum = tl.zeros((BLOCK_H,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_H, VALUE_DIM), dtype=tl.float32)
    k_offsets = tl.arange(0, BLOCK_K)

    for k_start in range(0, INDEX_TOPK, BLOCK_K):
        selected_offsets = k_start + k_offsets
        selected = tl.load(
            indices_ptr + query_id * indices_stride_token + selected_offsets,
        )
        valid = (selected >= 0) & (selected < num_kv_tokens)
        safe_selected = tl.where(valid, selected, 0)

        kv_nope = tl.load(
            kv_ptr + safe_selected[:, None] * kv_stride_token + nope_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        kv_rope = tl.load(
            kv_ptr + safe_selected[:, None] * kv_stride_token + NOPE_DIM + rope_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        scores = tl.dot(q_nope, tl.trans(kv_nope))
        scores += tl.dot(q_rope, tl.trans(kv_rope))
        scores *= sm_scale_log2
        scores = tl.where(head_mask[:, None] & valid[None, :], scores, negative_large)

        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, block_max)
        previous_scale = tl.exp2(running_max - new_max)
        probabilities = tl.exp2(scores - new_max[:, None])
        probabilities = tl.where(
            head_mask[:, None] & valid[None, :],
            probabilities,
            0.0,
        )
        new_sum = running_sum * previous_scale + tl.sum(probabilities, axis=1)

        values = tl.load(
            kv_ptr + safe_selected[:, None] * kv_stride_token + value_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        accumulator *= previous_scale[:, None]
        accumulator += tl.dot(probabilities.to(values.dtype), values)
        running_max = new_max
        running_sum = new_sum

    denominator = tl.maximum(running_sum, 1.0e-30)
    output = tl.where(
        running_sum[:, None] > 0.0,
        accumulator / denominator[:, None],
        0.0,
    )
    tl.store(
        output_ptr
        + query_id * output_stride_token
        + head_offsets[:, None] * output_stride_head
        + value_offsets[None, :],
        output.to(tl.bfloat16),
        mask=head_mask[:, None],
    )


def launch_sparse_mla_bf16(
    q: torch.Tensor,
    flat_kv: torch.Tensor,
    selected_indices: torch.Tensor,
    selected_indptr: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """Pack ragged indices and launch the gfx90a static-top-k kernel."""
    device_index = q.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    _require_gfx90a(device_index)

    num_queries, num_heads, _ = q.shape
    dense_indices = torch.empty(
        (num_queries, _GLM_INDEX_TOPK),
        dtype=torch.int32,
        device=q.device,
    )
    pack_block_size = 256
    _pack_ragged_indices_kernel[(num_queries, triton.cdiv(_GLM_INDEX_TOPK, pack_block_size))](
        selected_indices,
        selected_indptr,
        dense_indices,
        dense_indices.stride(0),
        selected_indices.numel(),
        BLOCK_SIZE=pack_block_size,
        INDEX_TOPK=_GLM_INDEX_TOPK,
        num_warps=4,
    )

    output = torch.empty(
        (num_queries, num_heads, _GLM_VALUE_DIM),
        dtype=q.dtype,
        device=q.device,
    )
    block_h = 16
    block_k = 16
    _glm_sparse_mla_bf16_kernel[(num_queries, triton.cdiv(num_heads, block_h))](
        q,
        flat_kv,
        dense_indices,
        output,
        q.stride(0),
        q.stride(1),
        flat_kv.stride(0),
        dense_indices.stride(0),
        output.stride(0),
        output.stride(1),
        num_heads,
        flat_kv.shape[0],
        float(sm_scale) * LOG2E,
        BLOCK_H=block_h,
        BLOCK_K=block_k,
        NOPE_DIM=_GLM_NOPE_DIM,
        ROPE_DIM=_GLM_ROPE_DIM,
        VALUE_DIM=_GLM_VALUE_DIM,
        INDEX_TOPK=_GLM_INDEX_TOPK,
        num_warps=4,
        num_stages=1,
    )
    return output
