# SPDX-License-Identifier: Apache-2.0

"""CDNA2 linear path for OCP-FP8 checkpoints.

The checkpoint remains OCP E4M3FN with 128x128 block scales. At model load,
``transcode_ocp_fp8_weight`` converts the dequantized checkpoint values to
INT8 with per-output-row, 32-K scales. Inference dynamically quantizes BF16
activations with the same 32-K grouping and executes native gfx90a INT8 MFMA.

The matrix kernel comes from vLLM's Apache-2.0-licensed block INT8 utility at
tag v0.24.0 (commit ee0da84ab9e04ac7610e28580af62c365e898389):
``vllm/model_executor/layers/quantization/utils/int8_utils.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from vllm.triton_utils import tl, triton

from prime_rl.inference.vllm.kernels.rocm_gfx90a_skinny_int8_linear import (
    invoke_gfx90a_skinny_int8_linear,
    select_gfx90a_skinny_int8_config,
)

SOURCE_GROUP_N = 128
SOURCE_GROUP_K = 128
TARGET_GROUP_N = 1
TARGET_GROUP_K = 32
RUNTIME_WEIGHT_SCALE_DTYPE = torch.float16
RUNTIME_WEIGHT_SCALE_MIN = 2.0**-24
TRANSCODE_BLOCK_ROWS = 16
TRANSCODE_BLOCK_GROUPS = 8
_VLLM_DENSE_INT8_KERNEL = None
_VLLM_ACTIVATION_QUANTIZER = None


@dataclass(frozen=True)
class Gfx90aInt8LinearConfig:
    block_m: int = 16
    block_n: int = 64
    block_k: int = TARGET_GROUP_K
    group_size_m: int = 1
    num_warps: int = 4
    num_stages: int = 1


DEFAULT_GFX90A_INT8_LINEAR_CONFIG = Gfx90aInt8LinearConfig()

PREFILL_MAX_TOKENS = 2048
GFX90A_INT8_TOKEN_BUCKETS = (1, 8, 32, 64, 128, 256, 512, 1024, 2048)

_M16_N64_G1_W4_S2 = Gfx90aInt8LinearConfig(num_stages=2)
_M16_N64_G8_W4_S2 = Gfx90aInt8LinearConfig(
    group_size_m=8,
    num_stages=2,
)
_M64_N64_G8_W4_S1 = Gfx90aInt8LinearConfig(
    block_m=64,
    block_n=64,
    group_size_m=8,
)
_M64_N64_G1_W4_S2 = Gfx90aInt8LinearConfig(
    block_m=64,
    block_n=64,
    group_size_m=1,
    num_warps=4,
    num_stages=2,
)
_M64_N64_G8_W4_S2 = Gfx90aInt8LinearConfig(
    block_m=64,
    block_n=64,
    group_size_m=8,
    num_warps=4,
    num_stages=2,
)
_M64_N128_G1_W4_S2 = Gfx90aInt8LinearConfig(
    block_m=64,
    block_n=128,
    group_size_m=1,
    num_warps=4,
    num_stages=2,
)
_M64_N128_G8_W4_S2 = Gfx90aInt8LinearConfig(
    block_m=64,
    block_n=128,
    group_size_m=8,
    num_warps=4,
    num_stages=2,
)
_M64_N128_G8_W8_S2 = Gfx90aInt8LinearConfig(
    block_m=64,
    block_n=128,
    group_size_m=8,
    num_warps=8,
    num_stages=2,
)
_M128_N64_G1_W8_S2 = Gfx90aInt8LinearConfig(
    block_m=128,
    block_n=64,
    group_size_m=1,
    num_warps=8,
    num_stages=2,
)
_M128_N64_G8_W8_S2 = Gfx90aInt8LinearConfig(
    block_m=128,
    block_n=64,
    group_size_m=8,
    num_warps=8,
    num_stages=2,
)

# See ``gfx90a_int8_tuning.json`` for the portable runtime identity, sweep
# protocol, candidate space, and selected winners. The outer key is the exact
# GLM-5.2 (N, K) projection shape. Runtime M uses vLLM's
# nearest-measured-token convention, capped at the largest gated M.
GFX90A_INT8_LINEAR_CONFIGS: dict[
    tuple[int, int],
    dict[int, Gfx90aInt8LinearConfig],
] = {
    (2048, 2048): {
        1: _M16_N64_G8_W4_S2,
        8: _M16_N64_G1_W4_S2,
        32: _M16_N64_G1_W4_S2,
        64: _M16_N64_G1_W4_S2,
        128: _M64_N64_G8_W4_S1,
        256: _M64_N64_G1_W4_S2,
        512: _M64_N128_G1_W4_S2,
        1024: _M64_N128_G1_W4_S2,
        2048: _M64_N128_G8_W8_S2,
    },
    (2624, 6144): {
        1: _M16_N64_G8_W4_S2,
        8: _M16_N64_G1_W4_S2,
        32: _M16_N64_G1_W4_S2,
        64: _M16_N64_G1_W4_S2,
        128: _M64_N64_G8_W4_S1,
        256: _M64_N64_G1_W4_S2,
        512: _M64_N64_G8_W4_S2,
        1024: _M128_N64_G8_W8_S2,
        2048: _M128_N64_G1_W8_S2,
    },
    (3072, 6144): {
        1: _M16_N64_G8_W4_S2,
        8: _M16_N64_G1_W4_S2,
        32: _M16_N64_G1_W4_S2,
        64: _M16_N64_G1_W4_S2,
        128: _M64_N64_G8_W4_S1,
        256: _M64_N64_G1_W4_S2,
        512: _M64_N128_G1_W4_S2,
        1024: _M64_N128_G8_W8_S2,
        2048: _M64_N128_G8_W8_S2,
    },
    (4096, 2048): {
        1: _M16_N64_G8_W4_S2,
        8: _M16_N64_G1_W4_S2,
        32: _M16_N64_G1_W4_S2,
        64: _M64_N64_G8_W4_S1,
        128: _M64_N64_G1_W4_S2,
        256: _M64_N128_G1_W4_S2,
        512: _M64_N128_G1_W4_S2,
        1024: _M64_N128_G8_W4_S2,
        2048: _M64_N128_G8_W4_S2,
    },
    (4096, 6144): {
        1: _M16_N64_G8_W4_S2,
        8: _M16_N64_G1_W4_S2,
        32: _M16_N64_G1_W4_S2,
        64: _M64_N64_G8_W4_S1,
        128: _M64_N64_G1_W4_S2,
        256: _M64_N64_G8_W4_S2,
        512: _M64_N128_G8_W4_S2,
        1024: _M64_N128_G8_W4_S2,
        2048: _M64_N128_G8_W8_S2,
    },
    (6144, 1536): {
        1: _M16_N64_G8_W4_S2,
        8: _M16_N64_G1_W4_S2,
        32: _M16_N64_G1_W4_S2,
        64: _M64_N64_G8_W4_S1,
        128: _M64_N64_G1_W4_S2,
        256: _M64_N128_G1_W4_S2,
        512: _M64_N128_G8_W4_S2,
        1024: _M64_N128_G8_W4_S2,
        2048: _M64_N128_G8_W4_S2,
    },
    (6144, 2048): {
        1: _M16_N64_G8_W4_S2,
        8: _M16_N64_G1_W4_S2,
        32: _M16_N64_G1_W4_S2,
        64: _M64_N64_G8_W4_S1,
        128: _M64_N64_G8_W4_S2,
        256: _M64_N128_G1_W4_S2,
        512: _M64_N128_G8_W4_S2,
        1024: _M64_N128_G8_W4_S2,
        2048: _M64_N128_G8_W8_S2,
    },
}


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def bind_vllm_dense_int8_symbols(dense_kernel, activation_quantizer) -> None:
    """Bind ABI-validated vLLM symbols before model execution."""

    global _VLLM_DENSE_INT8_KERNEL, _VLLM_ACTIVATION_QUANTIZER

    _require(callable(dense_kernel), "dense_kernel must be callable")
    _require(
        callable(activation_quantizer),
        "activation_quantizer must be callable",
    )
    _require(
        _VLLM_DENSE_INT8_KERNEL is None or _VLLM_DENSE_INT8_KERNEL is dense_kernel,
        "a different vLLM dense INT8 kernel is already bound",
    )
    _require(
        _VLLM_ACTIVATION_QUANTIZER is None or _VLLM_ACTIVATION_QUANTIZER is activation_quantizer,
        "a different vLLM activation quantizer is already bound",
    )
    _VLLM_DENSE_INT8_KERNEL = dense_kernel
    _VLLM_ACTIVATION_QUANTIZER = activation_quantizer


def select_gfx90a_int8_linear_config(
    m: int,
    n: int,
    k: int,
) -> Gfx90aInt8LinearConfig:
    """Select only launch configs measured for the GLM-5.2 gfx90a path."""

    _require(m > 0, "M must be positive")
    _require(
        m <= PREFILL_MAX_TOKENS,
        f"CDNA2 INT8 linear is gated through M={PREFILL_MAX_TOKENS}; got {m}",
    )
    shape_configs = GFX90A_INT8_LINEAR_CONFIGS.get((n, k))
    _require(
        shape_configs is not None,
        f"CDNA2 INT8 linear has no measured config for (N, K)=({n}, {k})",
    )
    token_bucket = min(
        GFX90A_INT8_TOKEN_BUCKETS,
        key=lambda value: abs(value - m),
    )
    return shape_configs[token_bucket]


def validate_gfx90a_device(device: torch.device) -> str:
    """Validate the ROCm architecture once, outside the inference hot path."""

    _require(device.type == "cuda", "device must be a ROCm GPU")
    _require(torch.version.hip is not None, "a ROCm PyTorch build is required")
    device_index = device.index
    properties = torch.cuda.get_device_properties(torch.cuda.current_device() if device_index is None else device_index)
    architecture = getattr(properties, "gcnArchName", "").split(":")[0]
    _require(architecture == "gfx90a", f"expected gfx90a, got {properties.gcnArchName}")
    return architecture


def _validate_transcode_inputs(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    chunk_rows: int,
    eps: float,
) -> tuple[int, int]:
    _require(
        weight.dtype == torch.float8_e4m3fn,
        "weight must use OCP torch.float8_e4m3fn",
    )
    _require(weight.ndim == 2, "weight must have shape [out_features, in_features]")
    _require(weight.is_contiguous(), "weight must be contiguous")
    _require(
        weight_scale.device == weight.device,
        "weight_scale must be on the same device as weight",
    )
    _require(
        weight_scale.dtype in (torch.float16, torch.bfloat16, torch.float32),
        "weight_scale must use a floating-point dtype",
    )
    _require(weight_scale.ndim == 2, "weight_scale must be two-dimensional")
    _require(weight_scale.is_contiguous(), "weight_scale must be contiguous")
    if weight.device.type == "cpu":
        _require(
            bool(torch.all(torch.isfinite(weight_scale) & (weight_scale > 0))),
            "weight_scale values must be finite and positive",
        )
    _require(
        isinstance(chunk_rows, int) and chunk_rows > 0,
        "chunk_rows must be a positive integer",
    )
    _require(eps > 0.0, "eps must be positive")

    out_features, in_features = weight.shape
    _require(out_features > 0 and in_features > 0, "weight dimensions must be positive")
    expected_scale_shape = (
        _ceil_div(out_features, SOURCE_GROUP_N),
        _ceil_div(in_features, SOURCE_GROUP_K),
    )
    _require(
        tuple(weight_scale.shape) == expected_scale_shape,
        f"weight_scale must have shape {expected_scale_shape} for weight shape {tuple(weight.shape)}",
    )
    return out_features, in_features


def _validate_batched_transcode_inputs(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    eps: float,
) -> tuple[int, int, int]:
    _require(
        weight.ndim == 3,
        "batched weight must have shape [batches, out_features, in_features]",
    )
    batches, out_features, in_features = weight.shape
    _require(batches > 0, "batched weight must contain at least one matrix")
    _require(
        weight_scale.ndim == 3 and weight_scale.is_contiguous(),
        "batched weight_scale must be contiguous and three-dimensional",
    )
    _require(
        weight_scale.shape[0] == batches,
        "batched weight and weight_scale counts must match",
    )
    _validate_transcode_inputs(
        weight[0],
        weight_scale[0],
        chunk_rows=SOURCE_GROUP_N,
        eps=eps,
    )
    _require(weight.is_contiguous(), "batched weight must be contiguous")
    expected_scale_shape = (
        batches,
        _ceil_div(out_features, SOURCE_GROUP_N),
        _ceil_div(in_features, SOURCE_GROUP_K),
    )
    _require(
        tuple(weight_scale.shape) == expected_scale_shape,
        f"batched weight_scale must have shape {expected_scale_shape}",
    )
    if weight.device.type == "cpu":
        _require(
            bool(torch.all(torch.isfinite(weight_scale) & (weight_scale > 0))),
            "batched weight_scale values must be finite and positive",
        )
    return batches, out_features, in_features


def dequantize_block_weight(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    group_n: int,
    group_k: int,
) -> torch.Tensor:
    """Dequantize a logical 2D weight with ceil-shaped block scales."""

    _require(weight.ndim == 2, "weight must be two-dimensional")
    _require(weight_scale.ndim == 2, "weight_scale must be two-dimensional")
    _require(group_n > 0 and group_k > 0, "group sizes must be positive")
    out_features, in_features = weight.shape
    expected_shape = (
        _ceil_div(out_features, group_n),
        _ceil_div(in_features, group_k),
    )
    _require(
        tuple(weight_scale.shape) == expected_shape,
        f"weight_scale must have shape {expected_shape}",
    )
    expanded_scale = weight_scale.float().repeat_interleave(group_n, dim=0).repeat_interleave(group_k, dim=1)
    return weight.float() * expanded_scale[:out_features, :in_features]


@torch.no_grad()
def _transcode_ocp_fp8_weight_reference(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    chunk_rows: int = SOURCE_GROUP_N,
    eps: float = 1.0e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference OCP-FP8-to-INT8 conversion for CPU validation.

    The returned weight has the same logical shape and INT8 storage. Its FP16
    scale tensor has shape ``[out_features, ceil(in_features / 32)]``. FP16
    keeps scale storage to 6.25% of the INT8 weight bytes, while the Triton
    kernel promotes products into its FP32 accumulator. Work is chunked over
    output rows, so temporary storage is bounded by
    ``O(chunk_rows * padded_in_features)`` rather than the full matrix.
    """

    out_features, in_features = _validate_transcode_inputs(weight, weight_scale, chunk_rows, eps)
    target_k_tiles = _ceil_div(in_features, TARGET_GROUP_K)
    padded_in_features = target_k_tiles * TARGET_GROUP_K
    quantized = torch.empty_like(weight, dtype=torch.int8)
    target_scale = torch.empty(
        (out_features, target_k_tiles),
        device=weight.device,
        dtype=RUNTIME_WEIGHT_SCALE_DTYPE,
    )

    for row_start in range(0, out_features, chunk_rows):
        row_end = min(row_start + chunk_rows, out_features)
        rows = row_end - row_start
        source_row_indices = torch.arange(row_start, row_end, device=weight.device) // SOURCE_GROUP_N
        source_scale = weight_scale.index_select(0, source_row_indices).float()
        source_scale = source_scale.repeat_interleave(SOURCE_GROUP_K, dim=1)
        decoded = weight[row_start:row_end].float() * source_scale[:, :in_features]

        if padded_in_features != in_features:
            padded = torch.zeros(
                (rows, padded_in_features),
                device=weight.device,
                dtype=torch.float32,
            )
            padded[:, :in_features] = decoded
            decoded = padded

        groups = decoded.reshape(rows, target_k_tiles, TARGET_GROUP_K)
        ideal_scales = (groups.abs().amax(dim=2).clamp_min(eps) / 127.0).clamp_min(RUNTIME_WEIGHT_SCALE_MIN)
        _require(
            bool(
                torch.all(torch.isfinite(ideal_scales) & (ideal_scales <= torch.finfo(RUNTIME_WEIGHT_SCALE_DTYPE).max))
            ),
            "runtime weight scales overflow FP16",
        )
        stored_scales = ideal_scales.to(RUNTIME_WEIGHT_SCALE_DTYPE)
        _require(
            bool(torch.all(torch.isfinite(stored_scales) & (stored_scales > 0))),
            "runtime weight scales must remain finite and positive in FP16",
        )
        quantization_scales = stored_scales.float()
        transformed = (
            (torch.sign(groups) * torch.floor(torch.abs(groups / quantization_scales[:, :, None]) + 0.5))
            .clamp(-127, 127)
            .to(torch.int8)
            .reshape(rows, padded_in_features)
        )
        quantized[row_start:row_end] = transformed[:, :in_features]
        target_scale[row_start:row_end] = stored_scales

    return quantized, target_scale


@triton.jit
def _transcode_ocp_fp8_to_int8_kernel(
    weight_ptr,
    source_scale_ptr,
    output_ptr,
    target_scale_ptr,
    invalid_ptr,
    out_features,
    in_features,
    target_k_tiles,
    total_rows,
    eps,
    weight_stride_batch: tl.int64,
    weight_stride_n: tl.int64,
    source_scale_stride_batch: tl.int64,
    source_scale_stride_n: tl.int64,
    output_stride_batch: tl.int64,
    output_stride_n: tl.int64,
    target_scale_stride_batch: tl.int64,
    target_scale_stride_n: tl.int64,
    RUNTIME_SCALE_MIN: tl.constexpr,
    RUNTIME_SCALE_MAX: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    SOURCE_GROUP_N: tl.constexpr,
    SOURCE_GROUP_K: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
):
    row_block = tl.program_id(0)
    group_block = tl.program_id(1)
    flat_rows = row_block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = flat_rows < total_rows
    batch = flat_rows // out_features
    row = flat_rows % out_features
    groups = group_block * BLOCK_GROUPS + tl.arange(0, BLOCK_GROUPS)
    elements = tl.arange(0, GROUP_SIZE)
    columns = groups[:, None] * GROUP_SIZE + elements[None, :]
    group_mask = groups < target_k_tiles
    element_mask = row_mask[:, None, None] & group_mask[None, :, None] & (columns[None, :, :] < in_features)

    source = tl.load(
        weight_ptr
        + batch[:, None, None] * weight_stride_batch
        + row[:, None, None] * weight_stride_n
        + columns[None, :, :],
        mask=element_mask,
        other=0.0,
    ).to(tl.float32)
    source_scales = tl.load(
        source_scale_ptr
        + batch[:, None] * source_scale_stride_batch
        + (row[:, None] // SOURCE_GROUP_N) * source_scale_stride_n
        + (groups[None, :] * GROUP_SIZE // SOURCE_GROUP_K),
        mask=row_mask[:, None] & group_mask[None, :],
        other=1.0,
    ).to(tl.float32)
    source_is_finite = source == source
    decoded = tl.where(source_is_finite, source, 0.0) * source_scales[:, :, None]
    absmax = tl.max(tl.abs(decoded), axis=2)
    ideal_scales = tl.maximum(
        tl.maximum(absmax, eps) / 127.0,
        RUNTIME_SCALE_MIN,
    )
    stored_scales = ideal_scales.to(tl.float16)
    valid = (
        row_mask[:, None]
        & group_mask[None, :]
        & (source_scales > 0.0)
        & (source_scales <= 3.4028234663852886e38)
        & (ideal_scales > 0.0)
        & (ideal_scales <= RUNTIME_SCALE_MAX)
        & (stored_scales > 0.0)
        & (stored_scales <= RUNTIME_SCALE_MAX)
    )
    scale_mask = row_mask[:, None] & group_mask[None, :]
    invalid_rows = tl.max(tl.where(scale_mask & ~valid, 1, 0), axis=1)
    invalid_source_groups = tl.max(
        tl.where(element_mask & ~source_is_finite, 1, 0),
        axis=2,
    )
    invalid_source_rows = tl.max(invalid_source_groups, axis=1)
    program_invalid = tl.maximum(
        tl.max(invalid_rows, axis=0),
        tl.max(invalid_source_rows, axis=0),
    )
    tl.atomic_or(
        invalid_ptr,
        program_invalid,
        mask=program_invalid != 0,
    )

    quantization_scales = tl.where(valid, stored_scales.to(tl.float32), 1.0)
    scaled = tl.clamp(
        decoded / quantization_scales[:, :, None],
        -127.0,
        127.0,
    )
    quantized = tl.extra.hip.libdevice.round(scaled).to(tl.int8)
    tl.store(
        output_ptr
        + batch[:, None, None] * output_stride_batch
        + row[:, None, None] * output_stride_n
        + columns[None, :, :],
        quantized,
        mask=element_mask,
    )
    tl.store(
        target_scale_ptr
        + batch[:, None] * target_scale_stride_batch
        + row[:, None] * target_scale_stride_n
        + groups[None, :],
        tl.where(valid, stored_scales, 1.0),
        mask=scale_mask,
    )


def _transcode_ocp_fp8_tensors_triton(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    device_is_validated: bool,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch one fused conversion over contiguous ``[batch, N, K]`` tensors."""

    if not device_is_validated:
        validate_gfx90a_device(weight.device)
    batches, out_features, in_features = weight.shape
    target_k_tiles = _ceil_div(in_features, TARGET_GROUP_K)
    quantized = torch.empty_like(weight, dtype=torch.int8)
    target_scale = torch.empty(
        (batches, out_features, target_k_tiles),
        device=weight.device,
        dtype=RUNTIME_WEIGHT_SCALE_DTYPE,
    )
    invalid = torch.zeros((), device=weight.device, dtype=torch.int32)
    grid = (
        triton.cdiv(batches * out_features, TRANSCODE_BLOCK_ROWS),
        triton.cdiv(target_k_tiles, TRANSCODE_BLOCK_GROUPS),
    )
    _transcode_ocp_fp8_to_int8_kernel[grid](
        weight,
        weight_scale,
        quantized,
        target_scale,
        invalid,
        out_features,
        in_features,
        target_k_tiles,
        batches * out_features,
        eps,
        weight.stride(0),
        weight.stride(1),
        weight_scale.stride(0),
        weight_scale.stride(1),
        quantized.stride(0),
        quantized.stride(1),
        target_scale.stride(0),
        target_scale.stride(1),
        RUNTIME_SCALE_MIN=RUNTIME_WEIGHT_SCALE_MIN,
        RUNTIME_SCALE_MAX=torch.finfo(RUNTIME_WEIGHT_SCALE_DTYPE).max,
        GROUP_SIZE=TARGET_GROUP_K,
        SOURCE_GROUP_N=SOURCE_GROUP_N,
        SOURCE_GROUP_K=SOURCE_GROUP_K,
        BLOCK_ROWS=TRANSCODE_BLOCK_ROWS,
        BLOCK_GROUPS=TRANSCODE_BLOCK_GROUPS,
        num_warps=8,
        num_stages=1,
    )
    _require(
        invalid.item() == 0,
        "source scales must be finite and positive and runtime scales must fit FP16",
    )
    return quantized, target_scale


@torch.no_grad()
def transcode_ocp_fp8_weight(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    chunk_rows: int = SOURCE_GROUP_N,
    eps: float = 1.0e-12,
    device_is_validated: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert one OCP-FP8 matrix to INT8 with FP16 per-row/K32 scales."""

    _validate_transcode_inputs(weight, weight_scale, chunk_rows, eps)
    if weight.device.type == "cpu":
        return _transcode_ocp_fp8_weight_reference(
            weight,
            weight_scale,
            chunk_rows=chunk_rows,
            eps=eps,
        )
    weight_batch = weight.unsqueeze(0)
    scale_batch = weight_scale.unsqueeze(0)
    quantized, target_scale = _transcode_ocp_fp8_tensors_triton(
        weight_batch,
        scale_batch,
        device_is_validated=device_is_validated,
        eps=eps,
    )
    return quantized[0], target_scale[0]


@torch.no_grad()
def transcode_ocp_fp8_weight_batch(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    eps: float = 1.0e-12,
    device_is_validated: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert contiguous ``[batch, N, K]`` weights in one GPU launch."""

    batches, out_features, in_features = _validate_batched_transcode_inputs(
        weight,
        weight_scale,
        eps,
    )
    if weight.device.type != "cpu":
        return _transcode_ocp_fp8_tensors_triton(
            weight,
            weight_scale,
            device_is_validated=device_is_validated,
            eps=eps,
        )

    quantized = torch.empty_like(weight, dtype=torch.int8)
    target_scale = torch.empty(
        (batches, out_features, _ceil_div(in_features, TARGET_GROUP_K)),
        device=weight.device,
        dtype=RUNTIME_WEIGHT_SCALE_DTYPE,
    )
    for batch in range(batches):
        batch_weight, batch_scale = _transcode_ocp_fp8_weight_reference(
            weight[batch],
            weight_scale[batch],
            eps=eps,
        )
        quantized[batch].copy_(batch_weight)
        target_scale[batch].copy_(batch_scale)
    return quantized, target_scale


def _validate_linear_inputs(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    config: Gfx90aInt8LinearConfig,
) -> tuple[int, int, int, tuple[int, ...]]:
    _require(
        hidden_states.dtype == torch.bfloat16,
        "the measured CDNA2 INT8 linear path requires BF16 activations",
    )
    _require(hidden_states.ndim >= 2, "hidden_states must have at least two dimensions")
    _require(hidden_states.is_contiguous(), "hidden_states must be contiguous")
    _require(hidden_states.device.type == "cuda", "hidden_states must be on a ROCm GPU")
    _require(torch.version.hip is not None, "a ROCm PyTorch build is required")
    _require(weight.device == hidden_states.device, "weight must be on the same device")
    _require(weight.dtype == torch.int8, "weight must use torch.int8")
    _require(weight.ndim == 2 and weight.is_contiguous(), "weight must be contiguous 2D")
    _require(
        weight_scale.device == hidden_states.device,
        "weight_scale must be on the same device",
    )
    _require(
        weight_scale.dtype == RUNTIME_WEIGHT_SCALE_DTYPE,
        "weight_scale must use torch.float16",
    )
    _require(weight_scale.is_contiguous(), "weight_scale must be contiguous")

    output_shape = (*hidden_states.shape[:-1], weight.shape[0])
    hidden_2d = hidden_states.view(-1, hidden_states.shape[-1])
    m, k = hidden_2d.shape
    n, weight_k = weight.shape
    _require(k == weight_k, "hidden_states and weight K dimensions must match")
    _require(k % TARGET_GROUP_K == 0, "input K must be divisible by 32")
    _require(
        tuple(weight_scale.shape) == (n, k // TARGET_GROUP_K),
        "weight_scale must have shape [N, K / 32]",
    )
    if bias is not None:
        _require(bias.device == hidden_states.device, "bias must be on the same device")
        _require(bias.dtype == hidden_states.dtype, "bias dtype must match hidden_states")
        _require(tuple(bias.shape) == (n,), "bias must have shape [N]")
    _require(config.block_k == TARGET_GROUP_K, "block_k must equal target group K")
    _require(config.block_m > 0 and config.block_n > 0, "block sizes must be positive")
    return m, n, k, output_shape


def cdna2_int8_linear(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    config: Gfx90aInt8LinearConfig = DEFAULT_GFX90A_INT8_LINEAR_CONFIG,
    device_is_validated: bool = False,
    use_gfx90a_skinny_int8: bool = False,
) -> torch.Tensor:
    """Run the registered dynamic-W8A8 linear path on validated gfx90a shapes.

    The custom vLLM quantization method installs the one-way INT8 runtime
    representation during model loading. This function owns its measured
    launch dispatch and invokes vLLM's block-INT8 Triton kernel directly.
    """

    if not device_is_validated:
        validate_gfx90a_device(hidden_states.device)
    m, n, k, output_shape = _validate_linear_inputs(hidden_states, weight, weight_scale, bias, config)
    from vllm.triton_utils import triton

    _require(
        _VLLM_DENSE_INT8_KERNEL is not None and _VLLM_ACTIVATION_QUANTIZER is not None,
        "vLLM kernel ABI is not installed",
    )
    hidden_2d = hidden_states.view(m, k)
    quantized_input, input_scale = _VLLM_ACTIVATION_QUANTIZER(
        hidden_2d,
        TARGET_GROUP_K,
    )
    output = torch.empty((m, n), device=hidden_states.device, dtype=hidden_states.dtype)
    skinny_config = select_gfx90a_skinny_int8_config(m, n, k) if use_gfx90a_skinny_int8 else None
    if skinny_config is not None:
        invoke_gfx90a_skinny_int8_linear(
            quantized_input,
            weight,
            output,
            input_scale,
            weight_scale,
            config=skinny_config,
        )
    else:
        grid = (triton.cdiv(m, config.block_m) * triton.cdiv(n, config.block_n),)
        _VLLM_DENSE_INT8_KERNEL[grid](
            quantized_input,
            weight,
            output,
            input_scale,
            weight_scale,
            m,
            n,
            k,
            TARGET_GROUP_N,
            TARGET_GROUP_K,
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
            BLOCK_SIZE_M=config.block_m,
            BLOCK_SIZE_N=config.block_n,
            BLOCK_SIZE_K=config.block_k,
            GROUP_SIZE_M=config.group_size_m,
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )
    if bias is not None:
        output = output + bias
    return output.view(output_shape)
