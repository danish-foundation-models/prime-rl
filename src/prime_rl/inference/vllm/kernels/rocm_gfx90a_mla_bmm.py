# SPDX-License-Identifier: Apache-2.0

"""BF16 batched GEMMs for GLM-5.2 absorbed MLA on AMD CDNA2.

The two operations preserve vLLM's existing strided views:

* Q absorption: ``[8, M, 192] @ [8, 192, 512] -> [8, M, 512]``.
* V up-projection: ``[8, M, 512] @ [8, 512, 256] -> [8, M, 256]``.

Both accumulate in FP32 and store BF16. Production dispatch is deliberately
limited to the exact GLM-5.2 TP8 layouts and token counts measured on gfx90a.
The wider token bound is retained only for explicit tuning configurations.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Literal

import torch

MlaBmmKind = Literal["q_absorb", "v_up"]

GLM52_MLA_LOCAL_HEADS = 8
GLM52_MLA_QK_NOPE_HEAD_DIM = 192
GLM52_MLA_KV_LORA_RANK = 512
GLM52_MLA_VALUE_HEAD_DIM = 256
GLM52_MLA_BMM_MAX_TOKENS = 2048
GLM52_MLA_WEIGHT_BATCH_STRIDE = 229376


@dataclass(frozen=True)
class Gfx90aMlaBmmConfig:
    """One BF16 MFMA launch configuration."""

    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int = 1
    waves_per_eu: int = 1


_M16_N64_K32_W4 = Gfx90aMlaBmmConfig(16, 64, 32, 4)
_M32_N64_K32_W4 = Gfx90aMlaBmmConfig(32, 64, 32, 4)
_M64_N64_K32_W8 = Gfx90aMlaBmmConfig(64, 64, 32, 8)

# These seeds keep the bounded tuning harness usable outside the measured
# production cells. The M=16 and M=32 entries are the measured winners from
# LUMI job 20191714; other entries remain tuning seeds only.
GFX90A_MLA_BMM_TOKEN_BUCKETS = (
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
)
GFX90A_MLA_BMM_SEED_CONFIGS: dict[
    MlaBmmKind,
    dict[int, Gfx90aMlaBmmConfig],
] = {
    "q_absorb": {
        1: _M16_N64_K32_W4,
        2: _M16_N64_K32_W4,
        4: _M16_N64_K32_W4,
        8: _M16_N64_K32_W4,
        16: Gfx90aMlaBmmConfig(16, 64, 64, 4, num_stages=2, waves_per_eu=1),
        32: Gfx90aMlaBmmConfig(16, 64, 64, 4, num_stages=1, waves_per_eu=2),
        64: _M32_N64_K32_W4,
        128: _M64_N64_K32_W8,
        256: _M64_N64_K32_W8,
        512: _M64_N64_K32_W8,
        1024: _M64_N64_K32_W8,
        2048: _M64_N64_K32_W8,
    },
    "v_up": {
        1: _M16_N64_K32_W4,
        2: _M16_N64_K32_W4,
        4: _M16_N64_K32_W4,
        8: _M16_N64_K32_W4,
        16: Gfx90aMlaBmmConfig(16, 64, 64, 4, num_stages=2, waves_per_eu=2),
        32: Gfx90aMlaBmmConfig(16, 64, 64, 4, num_stages=2, waves_per_eu=1),
        64: _M32_N64_K32_W4,
        128: _M64_N64_K32_W8,
        256: _M64_N64_K32_W8,
        512: _M64_N64_K32_W8,
        1024: _M64_N64_K32_W8,
        2048: _M64_N64_K32_W8,
    },
}
GFX90A_MLA_BMM_TUNED_TOKENS = (16, 32)
GFX90A_MLA_BMM_TUNED_CONFIGS: dict[
    MlaBmmKind,
    dict[int, Gfx90aMlaBmmConfig],
] = {
    kind: {m: configs[m] for m in GFX90A_MLA_BMM_TUNED_TOKENS} for kind, configs in GFX90A_MLA_BMM_SEED_CONFIGS.items()
}

_KIND_SHAPES: dict[MlaBmmKind, tuple[int, int]] = {
    "q_absorb": (
        GLM52_MLA_QK_NOPE_HEAD_DIM,
        GLM52_MLA_KV_LORA_RANK,
    ),
    "v_up": (
        GLM52_MLA_KV_LORA_RANK,
        GLM52_MLA_VALUE_HEAD_DIM,
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_power_of_two(value: int) -> bool:
    return type(value) is int and value > 0 and value & (value - 1) == 0


def _require_kind(kind: str) -> MlaBmmKind:
    _require(kind in _KIND_SHAPES, f"kind must be one of {tuple(_KIND_SHAPES)}, got {kind!r}")
    return kind  # type: ignore[return-value]


def validate_mla_bmm_config(
    kind: MlaBmmKind,
    config: Gfx90aMlaBmmConfig,
) -> None:
    """Validate matrix-core tile divisibility for one fixed operation."""

    kind = _require_kind(kind)
    k, n = _KIND_SHAPES[kind]
    _require(
        config.block_m >= 16 and _is_power_of_two(config.block_m),
        "block_m must be a power of two of at least 16",
    )
    _require(
        config.block_n >= 16 and _is_power_of_two(config.block_n),
        "block_n must be a power of two of at least 16",
    )
    _require(
        config.block_k >= 16 and _is_power_of_two(config.block_k),
        "block_k must be a power of two of at least 16",
    )
    _require(n % config.block_n == 0, f"block_n must divide {n}")
    _require(k % config.block_k == 0, f"block_k must divide {k}")
    _require(config.num_warps in (4, 8), "num_warps must be 4 or 8")
    _require(config.num_stages in (1, 2), "num_stages must be 1 or 2")
    _require(config.waves_per_eu in (1, 2, 3, 4), "waves_per_eu must be in [1, 4]")


def select_gfx90a_mla_bmm_tuning_seed(
    kind: MlaBmmKind,
    m: int,
) -> Gfx90aMlaBmmConfig:
    """Select the nearest functional seed inside the bounded tuning range."""

    kind = _require_kind(kind)
    _require(type(m) is int and m > 0, "M must be a positive integer")
    _require(
        m <= GLM52_MLA_BMM_MAX_TOKENS,
        f"GLM-5.2 MLA BMM is gated through M={GLM52_MLA_BMM_MAX_TOKENS}; got {m}",
    )
    token_bucket = min(
        GFX90A_MLA_BMM_TOKEN_BUCKETS,
        key=lambda value: (abs(value - m), value),
    )
    return GFX90A_MLA_BMM_SEED_CONFIGS[kind][token_bucket]


def get_gfx90a_mla_bmm_config(
    kind: MlaBmmKind,
    m: int,
) -> Gfx90aMlaBmmConfig | None:
    """Return a measured production config, or ``None`` for stock fallback."""

    kind = _require_kind(kind)
    if type(m) is not int:
        return None
    return GFX90A_MLA_BMM_TUNED_CONFIGS[kind].get(m)


def select_gfx90a_mla_bmm_config(
    kind: MlaBmmKind,
    m: int,
) -> Gfx90aMlaBmmConfig:
    """Select an exact measured production config and reject extrapolation."""

    config = get_gfx90a_mla_bmm_config(kind, m)
    _require(
        config is not None,
        f"GLM-5.2 MLA BMM has measured production configs only for M={GFX90A_MLA_BMM_TUNED_TOKENS}; got {m!r}",
    )
    return config


def matches_gfx90a_mla_bmm_layout(
    kind: MlaBmmKind,
    left: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
) -> bool:
    """Return whether tensors match one measured GLM-5.2 call cell."""

    kind = _require_kind(kind)
    if not all(isinstance(tensor, torch.Tensor) for tensor in (left, weight, output)):
        return False
    if left.ndim != 3 or weight.ndim != 3 or output.ndim != 3:
        return False

    m = left.shape[1]
    if get_gfx90a_mla_bmm_config(kind, m) is None:
        return False
    if not (
        left.dtype == weight.dtype == output.dtype == torch.bfloat16 and left.device == weight.device == output.device
    ):
        return False

    if kind == "q_absorb":
        return (
            tuple(left.shape)
            == (
                GLM52_MLA_LOCAL_HEADS,
                m,
                GLM52_MLA_QK_NOPE_HEAD_DIM,
            )
            and left.stride() == (256, 2048, 1)
            and tuple(weight.shape)
            == (
                GLM52_MLA_LOCAL_HEADS,
                GLM52_MLA_QK_NOPE_HEAD_DIM,
                GLM52_MLA_KV_LORA_RANK,
            )
            and weight.stride() == (GLM52_MLA_WEIGHT_BATCH_STRIDE, 512, 1)
            and tuple(output.shape)
            == (
                GLM52_MLA_LOCAL_HEADS,
                m,
                GLM52_MLA_KV_LORA_RANK,
            )
            and output.stride() == (m * GLM52_MLA_KV_LORA_RANK, GLM52_MLA_KV_LORA_RANK, 1)
        )

    return (
        tuple(left.shape)
        == (
            GLM52_MLA_LOCAL_HEADS,
            m,
            GLM52_MLA_KV_LORA_RANK,
        )
        and left.stride() == (512, 4096, 1)
        and tuple(weight.shape)
        == (
            GLM52_MLA_LOCAL_HEADS,
            GLM52_MLA_KV_LORA_RANK,
            GLM52_MLA_VALUE_HEAD_DIM,
        )
        and weight.stride() == (GLM52_MLA_WEIGHT_BATCH_STRIDE, 1, 512)
        and tuple(output.shape)
        == (
            GLM52_MLA_LOCAL_HEADS,
            m,
            GLM52_MLA_VALUE_HEAD_DIM,
        )
        and output.stride() == (256, 2048, 1)
    )


def can_dispatch_gfx90a_mla_bmm(
    kind: MlaBmmKind,
    left: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
) -> bool:
    """Return whether the measured kernel may replace stock ``torch.bmm``."""

    return (
        matches_gfx90a_mla_bmm_layout(kind, left, weight, output)
        and torch.version.hip is not None
        and left.device.type == "cuda"
    )


@functools.lru_cache(maxsize=None)
def validate_gfx90a_mla_device(device_index: int) -> str:
    """Validate ROCm and gfx90a outside the kernel hot path."""

    _require(torch.version.hip is not None, "a ROCm PyTorch build is required")
    properties = torch.cuda.get_device_properties(device_index)
    architecture = getattr(properties, "gcnArchName", "").split(":")[0]
    _require(architecture == "gfx90a", f"expected gfx90a, got {getattr(properties, 'gcnArchName', None)}")
    return architecture


def _validate_common(
    kind: MlaBmmKind,
    left: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    config: Gfx90aMlaBmmConfig,
) -> int:
    validate_mla_bmm_config(kind, config)
    _require(left.ndim == 3, "left operand must be three-dimensional")
    _require(weight.ndim == 3, "weight must be three-dimensional")
    _require(output.ndim == 3, "output must be three-dimensional")
    m = left.shape[1]
    _require(0 < m <= GLM52_MLA_BMM_MAX_TOKENS, f"M must be in [1, {GLM52_MLA_BMM_MAX_TOKENS}], got {m}")
    _require(left.dtype == torch.bfloat16, "left operand must use torch.bfloat16")
    _require(weight.dtype == torch.bfloat16, "weight must use torch.bfloat16")
    _require(output.dtype == torch.bfloat16, "output must use torch.bfloat16")
    _require(weight.device == left.device, "weight and left operand must share a device")
    _require(output.device == left.device, "output and left operand must share a device")
    return m


def _validate_q_absorb_inputs(
    q_nope: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    config: Gfx90aMlaBmmConfig,
) -> int:
    m = _validate_common("q_absorb", q_nope, weight, output, config)
    _require(
        tuple(q_nope.shape) == (GLM52_MLA_LOCAL_HEADS, m, GLM52_MLA_QK_NOPE_HEAD_DIM),
        "q_nope must have shape [8, M, 192]",
    )
    _require(
        q_nope.stride() == (256, 2048, 1),
        f"q_nope must have stride (256, 2048, 1), got {q_nope.stride()}",
    )
    _require(
        tuple(weight.shape)
        == (
            GLM52_MLA_LOCAL_HEADS,
            GLM52_MLA_QK_NOPE_HEAD_DIM,
            GLM52_MLA_KV_LORA_RANK,
        ),
        "Q-absorb weight must have shape [8, 192, 512]",
    )
    _require(
        weight.stride() == (GLM52_MLA_WEIGHT_BATCH_STRIDE, 512, 1),
        "Q-absorb weight must have stride (229376, 512, 1)",
    )
    _require(
        tuple(output.shape)
        == (
            GLM52_MLA_LOCAL_HEADS,
            m,
            GLM52_MLA_KV_LORA_RANK,
        ),
        "Q-absorb output must have shape [8, M, 512]",
    )
    _require(
        output.stride() == (m * GLM52_MLA_KV_LORA_RANK, GLM52_MLA_KV_LORA_RANK, 1),
        f"Q-absorb output has an unsupported stride {output.stride()}",
    )
    _require(q_nope.device.type == "cuda", "Q-absorb tensors must be on a ROCm GPU")
    _require(torch.version.hip is not None, "a ROCm PyTorch build is required")
    return m


def _validate_v_up_inputs(
    attention_output: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    config: Gfx90aMlaBmmConfig,
) -> int:
    m = _validate_common("v_up", attention_output, weight, output, config)
    _require(
        tuple(attention_output.shape)
        == (
            GLM52_MLA_LOCAL_HEADS,
            m,
            GLM52_MLA_KV_LORA_RANK,
        ),
        "attention_output must have shape [8, M, 512]",
    )
    _require(
        attention_output.stride() == (512, 4096, 1),
        f"attention_output must have stride (512, 4096, 1), got {attention_output.stride()}",
    )
    _require(
        tuple(weight.shape)
        == (
            GLM52_MLA_LOCAL_HEADS,
            GLM52_MLA_KV_LORA_RANK,
            GLM52_MLA_VALUE_HEAD_DIM,
        ),
        "V-up weight must have shape [8, 512, 256]",
    )
    _require(
        weight.stride() == (GLM52_MLA_WEIGHT_BATCH_STRIDE, 1, 512),
        "V-up weight must have stride (229376, 1, 512)",
    )
    _require(
        tuple(output.shape)
        == (
            GLM52_MLA_LOCAL_HEADS,
            m,
            GLM52_MLA_VALUE_HEAD_DIM,
        ),
        "V-up output must have shape [8, M, 256]",
    )
    _require(
        output.stride() == (256, 2048, 1),
        f"V-up output must have stride (256, 2048, 1), got {output.stride()}",
    )
    _require(attention_output.device.type == "cuda", "V-up tensors must be on a ROCm GPU")
    _require(torch.version.hip is not None, "a ROCm PyTorch build is required")
    return m


def gfx90a_glm52_mla_q_absorb(
    q_nope: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor | None = None,
    *,
    config: Gfx90aMlaBmmConfig | None = None,
    device_is_validated: bool = False,
) -> torch.Tensor:
    """Run GLM-5.2's absorbed-query BMM into its native output layout."""

    m = q_nope.shape[1] if q_nope.ndim == 3 else 0
    selected = select_gfx90a_mla_bmm_config("q_absorb", m) if config is None else config
    if output is None:
        output = torch.empty(
            (GLM52_MLA_LOCAL_HEADS, m, GLM52_MLA_KV_LORA_RANK),
            dtype=q_nope.dtype,
            device=q_nope.device,
        )
    _validate_q_absorb_inputs(q_nope, weight, output, selected)
    if not device_is_validated:
        device_index = q_nope.device.index
        validate_gfx90a_mla_device(torch.cuda.current_device() if device_index is None else device_index)
    from prime_rl.inference.vllm.kernels.rocm_gfx90a_mla_bmm_triton import (
        launch_gfx90a_mla_bmm,
    )

    launch_gfx90a_mla_bmm(q_nope, weight, output, selected)
    return output


def gfx90a_glm52_mla_v_up(
    attention_output: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor | None = None,
    *,
    config: Gfx90aMlaBmmConfig | None = None,
    device_is_validated: bool = False,
) -> torch.Tensor:
    """Run GLM-5.2's value up-projection into its transposed output view."""

    m = attention_output.shape[1] if attention_output.ndim == 3 else 0
    selected = select_gfx90a_mla_bmm_config("v_up", m) if config is None else config
    if output is None:
        output_base = torch.empty(
            (m, GLM52_MLA_LOCAL_HEADS, GLM52_MLA_VALUE_HEAD_DIM),
            dtype=attention_output.dtype,
            device=attention_output.device,
        )
        output = output_base.transpose(0, 1)
    _validate_v_up_inputs(attention_output, weight, output, selected)
    if not device_is_validated:
        device_index = attention_output.device.index
        validate_gfx90a_mla_device(torch.cuda.current_device() if device_index is None else device_index)
    from prime_rl.inference.vllm.kernels.rocm_gfx90a_mla_bmm_triton import (
        launch_gfx90a_mla_bmm,
    )

    launch_gfx90a_mla_bmm(attention_output, weight, output, selected)
    return output


__all__ = [
    "GFX90A_MLA_BMM_SEED_CONFIGS",
    "GFX90A_MLA_BMM_TOKEN_BUCKETS",
    "GFX90A_MLA_BMM_TUNED_CONFIGS",
    "GFX90A_MLA_BMM_TUNED_TOKENS",
    "GLM52_MLA_BMM_MAX_TOKENS",
    "GLM52_MLA_KV_LORA_RANK",
    "GLM52_MLA_LOCAL_HEADS",
    "GLM52_MLA_QK_NOPE_HEAD_DIM",
    "GLM52_MLA_VALUE_HEAD_DIM",
    "Gfx90aMlaBmmConfig",
    "MlaBmmKind",
    "can_dispatch_gfx90a_mla_bmm",
    "gfx90a_glm52_mla_q_absorb",
    "gfx90a_glm52_mla_v_up",
    "get_gfx90a_mla_bmm_config",
    "matches_gfx90a_mla_bmm_layout",
    "select_gfx90a_mla_bmm_config",
    "select_gfx90a_mla_bmm_tuning_seed",
    "validate_gfx90a_mla_device",
    "validate_mla_bmm_config",
]
