# SPDX-License-Identifier: Apache-2.0

"""Exact-shape skinny block-INT8 linear kernel for gfx90a.

This module is deliberately separate from vLLM's generic block-INT8 kernel.
It covers only the GLM-5.2 c128 shapes for which a whole M tile can share one
K32-by-N32 weight tile. Unsupported token counts and projection shapes return
``None`` from the selector so the caller can retain the stock vLLM path.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from vllm.triton_utils import tl, triton

ENABLE_ENV = "PRIME_ROCM_GFX90A_SKINNY_INT8"
GROUP_K = 32


@dataclass(frozen=True)
class Gfx90aSkinnyInt8Config:
    block_m: int
    block_n: int = 32
    block_k: int = GROUP_K
    num_warps: int = 4
    num_stages: int = 1


_M8_CONFIG = Gfx90aSkinnyInt8Config(
    block_m=16,
    num_warps=4,
    num_stages=2,
)
_M64_CONFIG = Gfx90aSkinnyInt8Config(
    block_m=64,
    num_warps=8,
    num_stages=1,
)

# M=8 is useful only for the shared-expert gate/up projection. The shared
# down projection already launches 96 N64 programs and does not suffer the
# underfill this N32 kernel is intended to fix.
GFX90A_SKINNY_INT8_SHAPES: dict[
    int,
    frozenset[tuple[int, int]],
] = {
    8: frozenset({(4096, 6144)}),
    64: frozenset(
        {
            (2048, 2048),
            (2624, 6144),
            (3072, 6144),
            (4096, 2048),
            (4096, 6144),
            (6144, 1536),
            (6144, 2048),
        }
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def gfx90a_skinny_int8_requested(
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Read the independent experimental-kernel opt-in."""

    value = (os.environ if environ is None else environ).get(ENABLE_ENV)
    if value in (None, "0"):
        return False
    if value == "1":
        return True
    raise ValueError(f"{ENABLE_ENV} must be unset, '0', or '1'; got {value!r}")


def select_gfx90a_skinny_int8_config(
    m: int,
    n: int,
    k: int,
) -> Gfx90aSkinnyInt8Config | None:
    """Return a candidate only for the exact profiled c128 shapes."""

    shapes = GFX90A_SKINNY_INT8_SHAPES.get(m)
    if shapes is None or (n, k) not in shapes:
        return None
    return _M8_CONFIG if m == 8 else _M64_CONFIG


@triton.jit
def _gfx90a_k32_skinny_int8_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    m,
    n,
    k,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bsk,
    stride_bsn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Accumulate one complete M tile against one N32 output tile.

    K32 is both the MFMA reduction tile and the quantization group, so every
    integer dot product receives exactly one activation-scale and one
    per-output-row weight-scale product before FP32 accumulation.
    """

    pid_n = tl.program_id(axis=0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < m
    mask_n = offs_n < n

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    a_scale_ptrs = a_scale_ptr + offs_m * stride_asm
    b_scale_ptrs = b_scale_ptr + offs_n * stride_bsn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for group_idx in range(0, tl.cdiv(k, BLOCK_K)):
        k_remaining = k - group_idx * BLOCK_K
        a = tl.load(
            a_ptrs,
            mask=mask_m[:, None] & (offs_k[None, :] < k_remaining),
            other=0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < k_remaining) & mask_n[None, :],
            other=0,
        )
        a_scale = tl.load(
            a_scale_ptrs + group_idx * stride_ask,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
        b_scale = tl.load(
            b_scale_ptrs + group_idx * stride_bsk,
            mask=mask_n,
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.dot(a, b).to(tl.float32) * a_scale[:, None] * b_scale[None, :]
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs,
        accumulator.to(tl.bfloat16),
        mask=mask_m[:, None] & mask_n[None, :],
    )


def invoke_gfx90a_skinny_int8_linear(
    quantized_input: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    config: Gfx90aSkinnyInt8Config | None = None,
) -> None:
    """Launch the exact-shape candidate after the caller validates tensors."""

    _require(
        quantized_input.ndim == 2 and weight.ndim == 2 and output.ndim == 2,
        "skinny INT8 matrices must be two-dimensional",
    )
    m, k = quantized_input.shape
    n, weight_k = weight.shape
    _require(weight_k == k, "skinny INT8 A and B K dimensions must match")
    selected = select_gfx90a_skinny_int8_config(m, n, k)
    _require(selected is not None, f"unsupported skinny INT8 shape {(m, n, k)}")
    if config is None:
        config = selected
    _require(
        config == selected,
        "skinny INT8 launch config must match the exact-shape selector",
    )
    _require(config.block_k == GROUP_K, "skinny INT8 block K must equal 32")
    _require(n % config.block_n == 0, "skinny INT8 N must be divisible by block N")
    _require(k % config.block_k == 0, "skinny INT8 K must be divisible by block K")

    grid = (triton.cdiv(n, config.block_n),)
    _gfx90a_k32_skinny_int8_kernel[grid](
        quantized_input,
        weight,
        output,
        input_scale,
        weight_scale,
        m,
        n,
        k,
        quantized_input.stride(0),
        quantized_input.stride(1),
        weight.stride(1),
        weight.stride(0),
        output.stride(0),
        output.stride(1),
        input_scale.stride(0),
        input_scale.stride(1),
        weight_scale.stride(1),
        weight_scale.stride(0),
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        BLOCK_K=config.block_k,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
