# SPDX-License-Identifier: Apache-2.0

"""Triton implementation of GLM-5.2's two absorbed-MLA BMMs."""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton

from prime_rl.inference.vllm.kernels.rocm_gfx90a_mla_bmm import (
    Gfx90aMlaBmmConfig,
)


@triton.jit
def _gfx90a_mla_bmm_kernel(
    left_ptr,
    weight_ptr,
    output_ptr,
    m,
    left_stride_batch,
    left_stride_m,
    left_stride_k,
    weight_stride_batch,
    weight_stride_k,
    weight_stride_n,
    output_stride_batch,
    output_stride_m,
    output_stride_n,
    BATCH: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    token_block = tl.program_id(0)
    output_block = tl.program_id(1)
    batch = tl.program_id(2)
    token_offsets = token_block * BLOCK_M + tl.arange(0, BLOCK_M)
    output_offsets = output_block * BLOCK_N + tl.arange(0, BLOCK_N)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        left = tl.load(
            left_ptr
            + batch * left_stride_batch
            + token_offsets[:, None] * left_stride_m
            + k_offsets[None, :] * left_stride_k,
            mask=token_offsets[:, None] < m,
            other=0.0,
        )
        weight = tl.load(
            weight_ptr
            + batch * weight_stride_batch
            + k_offsets[:, None] * weight_stride_k
            + output_offsets[None, :] * weight_stride_n,
        )
        accumulator += tl.dot(left, weight, out_dtype=tl.float32)

    tl.store(
        output_ptr
        + batch * output_stride_batch
        + token_offsets[:, None] * output_stride_m
        + output_offsets[None, :] * output_stride_n,
        accumulator.to(tl.bfloat16),
        mask=token_offsets[:, None] < m,
    )


def launch_gfx90a_mla_bmm(
    left: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    config: Gfx90aMlaBmmConfig,
) -> None:
    """Launch a strided batched GEMM without materializing contiguous views."""

    batch, m, k = left.shape
    n = weight.shape[2]
    grid = (
        triton.cdiv(m, config.block_m),
        triton.cdiv(n, config.block_n),
        batch,
    )
    _gfx90a_mla_bmm_kernel[grid](
        left,
        weight,
        output,
        m,
        left.stride(0),
        left.stride(1),
        left.stride(2),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        BATCH=batch,
        K=k,
        N=n,
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        BLOCK_K=config.block_k,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
        waves_per_eu=config.waves_per_eu,
    )
