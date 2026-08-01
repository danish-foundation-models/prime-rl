# SPDX-License-Identifier: Apache-2.0

"""Triton kernels for the gfx90a GLM-5.2 FP32-output gate GEMM."""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton

from prime_rl.inference.vllm.kernels.rocm_gfx90a_gate_linear import (
    GLM52_GATE_HIDDEN_SIZE,
    GLM52_GATE_NUM_EXPERTS,
    Gfx90aGateLinearConfig,
)


@triton.jit
def _gate_linear_split_k_kernel(
    hidden_ptr,
    weight_ptr,
    partial_ptr,
    m,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    SPLIT_K: tl.constexpr,
    CACHE_CG: tl.constexpr,
):
    expert_block = tl.program_id(0)
    token_block = tl.program_id(1)
    split_id = tl.program_id(2)

    token_offsets = token_block * BLOCK_M + tl.arange(0, BLOCK_M)
    expert_offsets = expert_block * BLOCK_N + tl.arange(0, BLOCK_N)
    split_extent = HIDDEN_SIZE // SPLIT_K
    split_start = split_id * split_extent
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for split_offset in range(0, split_extent, BLOCK_K):
        k_offsets = split_start + split_offset + tl.arange(0, BLOCK_K)
        hidden = tl.load(
            hidden_ptr + token_offsets[:, None] * HIDDEN_SIZE + k_offsets[None, :],
            mask=token_offsets[:, None] < m,
            other=0.0,
        )
        if CACHE_CG:
            weight = tl.load(
                weight_ptr
                + expert_offsets[:, None] * HIDDEN_SIZE
                + k_offsets[None, :],
                cache_modifier=".cg",
            )
        else:
            weight = tl.load(
                weight_ptr
                + expert_offsets[:, None] * HIDDEN_SIZE
                + k_offsets[None, :],
            )
        accumulator += tl.dot(hidden, tl.trans(weight), out_dtype=tl.float32)

    partial_offsets = split_id * m * NUM_EXPERTS + token_offsets[:, None] * NUM_EXPERTS + expert_offsets[None, :]
    tl.store(
        partial_ptr + partial_offsets,
        accumulator,
        mask=token_offsets[:, None] < m,
    )


@triton.jit
def _gate_linear_reduce_split_k_kernel(
    partial_ptr,
    output_ptr,
    m,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    expert_block = tl.program_id(0)
    token_block = tl.program_id(1)
    token_offsets = token_block * BLOCK_M + tl.arange(0, BLOCK_M)
    expert_offsets = expert_block * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets = token_offsets[:, None] * NUM_EXPERTS + expert_offsets[None, :]
    mask = token_offsets[:, None] < m

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for split_id in range(0, SPLIT_K):
        accumulator += tl.load(
            partial_ptr + split_id * m * NUM_EXPERTS + offsets,
            mask=mask,
            other=0.0,
        )
    tl.store(output_ptr + offsets, accumulator, mask=mask)


@triton.jit
def _gate_linear_no_split_kernel(
    hidden_ptr,
    weight_ptr,
    output_ptr,
    m,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    GROUP_M: tl.constexpr,
    CACHE_CG: tl.constexpr,
):
    program_id = tl.program_id(0)
    token_block_count = tl.cdiv(m, BLOCK_M)
    expert_block_count = tl.cdiv(NUM_EXPERTS, BLOCK_N)
    programs_per_group = GROUP_M * expert_block_count
    group_id = program_id // programs_per_group
    first_token_block = group_id * GROUP_M
    token_blocks_in_group = tl.minimum(token_block_count - first_token_block, GROUP_M)
    token_block = first_token_block + (program_id % programs_per_group) % token_blocks_in_group
    expert_block = (program_id % programs_per_group) // token_blocks_in_group
    token_offsets = token_block * BLOCK_M + tl.arange(0, BLOCK_M)
    expert_offsets = expert_block * BLOCK_N + tl.arange(0, BLOCK_N)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, HIDDEN_SIZE, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        hidden = tl.load(
            hidden_ptr + token_offsets[:, None] * HIDDEN_SIZE + k_offsets[None, :],
            mask=token_offsets[:, None] < m,
            other=0.0,
        )
        if CACHE_CG:
            weight = tl.load(
                weight_ptr
                + expert_offsets[:, None] * HIDDEN_SIZE
                + k_offsets[None, :],
                cache_modifier=".cg",
            )
        else:
            weight = tl.load(
                weight_ptr
                + expert_offsets[:, None] * HIDDEN_SIZE
                + k_offsets[None, :],
            )
        accumulator += tl.dot(hidden, tl.trans(weight), out_dtype=tl.float32)

    offsets = token_offsets[:, None] * NUM_EXPERTS + expert_offsets[None, :]
    tl.store(
        output_ptr + offsets,
        accumulator,
        mask=token_offsets[:, None] < m,
    )


def launch_gfx90a_gate_linear(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    config: Gfx90aGateLinearConfig,
) -> torch.Tensor:
    """Launch the selected split-K kernel and deterministic FP32 reduction."""

    m = hidden_states.shape[0]
    output = torch.empty(
        (m, GLM52_GATE_NUM_EXPERTS),
        dtype=torch.float32,
        device=hidden_states.device,
    )
    grid = (
        triton.cdiv(GLM52_GATE_NUM_EXPERTS, config.block_n),
        triton.cdiv(m, config.block_m),
    )
    launch_options = {
        "num_warps": config.num_warps,
        "num_stages": config.num_stages,
        "waves_per_eu": config.waves_per_eu,
        "matrix_instr_nonkdim": config.matrix_instr_nonkdim,
        "kpack": config.kpack,
    }
    if config.split_k == 1:
        grouped_grid = (triton.cdiv(m, config.block_m) * triton.cdiv(GLM52_GATE_NUM_EXPERTS, config.block_n),)
        _gate_linear_no_split_kernel[grouped_grid](
            hidden_states,
            weight,
            output,
            m,
            BLOCK_M=config.block_m,
            BLOCK_N=config.block_n,
            BLOCK_K=config.block_k,
            HIDDEN_SIZE=GLM52_GATE_HIDDEN_SIZE,
            NUM_EXPERTS=GLM52_GATE_NUM_EXPERTS,
            GROUP_M=config.group_m,
            CACHE_CG=config.cache_modifier == ".cg",
            **launch_options,
        )
        return output

    partial = torch.empty(
        (config.split_k, m, GLM52_GATE_NUM_EXPERTS),
        dtype=torch.float32,
        device=hidden_states.device,
    )
    _gate_linear_split_k_kernel[(*grid, config.split_k)](
        hidden_states,
        weight,
        partial,
        m,
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        BLOCK_K=config.block_k,
        HIDDEN_SIZE=GLM52_GATE_HIDDEN_SIZE,
        NUM_EXPERTS=GLM52_GATE_NUM_EXPERTS,
        SPLIT_K=config.split_k,
        CACHE_CG=config.cache_modifier == ".cg",
        **launch_options,
    )
    reduce_block_m = config.reduce_block_m or config.block_m
    reduce_block_n = config.reduce_block_n or config.block_n
    reduce_num_warps = config.reduce_num_warps or config.num_warps
    reduce_grid = (
        triton.cdiv(GLM52_GATE_NUM_EXPERTS, reduce_block_n),
        triton.cdiv(m, reduce_block_m),
    )
    _gate_linear_reduce_split_k_kernel[reduce_grid](
        partial,
        output,
        m,
        BLOCK_M=reduce_block_m,
        BLOCK_N=reduce_block_n,
        NUM_EXPERTS=GLM52_GATE_NUM_EXPERTS,
        SPLIT_K=config.split_k,
        num_warps=reduce_num_warps,
        num_stages=1,
        waves_per_eu=1,
        matrix_instr_nonkdim=16,
        kpack=1,
    )
    return output
