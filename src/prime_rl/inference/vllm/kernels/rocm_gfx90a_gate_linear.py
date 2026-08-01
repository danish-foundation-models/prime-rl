# SPDX-License-Identifier: Apache-2.0

"""GLM-5.2 BF16-to-FP32 router GEMM for AMD CDNA2.

The production contract is deliberately narrow: a contiguous BF16
``[M, 6144]`` activation, a contiguous BF16 ``[256, 6144]`` gate weight,
``1 <= M <= 2048``, and a gfx90a ROCm device. The output is accumulated and
stored in FP32, avoiding vLLM's ROCm skinny-GEMM fallback, which stores BF16
before ``GateLinear`` casts the router logits to FP32. Dispatch uses the
smallest measured token bucket that contains the runtime token count.
"""

from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any

import torch

GLM52_GATE_HIDDEN_SIZE = 6144
GLM52_GATE_NUM_EXPERTS = 256
GLM52_GATE_MAX_TOKENS = 2048
_EXPECTED_VLLM_DISTRIBUTION = "0.24.0+lumi_aif_gfx90a_ee0da84"
_GATE_MARKER = "_prime_rl_glm52_gfx90a_gate_linear"
_PATCH_MARKER = "_prime_rl_glm52_gfx90a_gate_linear_patch"
_CUSTOM_OP_NAME = "prime_glm52_gfx90a_gate_linear"
_CUSTOM_OP_REGISTERED = False


@dataclass(frozen=True)
class Gfx90aGateLinearConfig:
    """One launch configuration for the exact-shape router GEMM."""

    block_m: int
    block_n: int
    block_k: int
    split_k: int
    num_warps: int
    num_stages: int = 1
    waves_per_eu: int = 1
    group_m: int = 1
    matrix_instr_nonkdim: int = 16
    kpack: int = 1
    cache_modifier: str | None = None
    reduce_block_m: int | None = None
    reduce_block_n: int | None = None
    reduce_num_warps: int | None = None


_M16_N64_K32_S16_W4_WAVE2 = Gfx90aGateLinearConfig(
    16,
    64,
    32,
    16,
    4,
    waves_per_eu=2,
)
_M16_N64_K64_S16_W4 = Gfx90aGateLinearConfig(16, 64, 64, 16, 4)
_M16_N64_K32_S4_W4 = Gfx90aGateLinearConfig(16, 64, 32, 4, 4)
_M32_N32_K128_S12_W4_CG = Gfx90aGateLinearConfig(
    32,
    32,
    128,
    12,
    4,
    num_stages=2,
    waves_per_eu=0,
    kpack=2,
    cache_modifier=".cg",
    reduce_block_m=16,
    reduce_block_n=32,
    reduce_num_warps=8,
)
_M32_N32_K128_S12_W4_WAVE2 = Gfx90aGateLinearConfig(
    32,
    32,
    128,
    12,
    4,
    num_stages=2,
    waves_per_eu=2,
    kpack=2,
    reduce_block_m=16,
    reduce_block_n=32,
    reduce_num_warps=8,
)
_M64_N32_K128_S12_W4_CG = Gfx90aGateLinearConfig(
    64,
    32,
    128,
    12,
    4,
    num_stages=2,
    waves_per_eu=0,
    kpack=2,
    cache_modifier=".cg",
    reduce_block_m=16,
    reduce_block_n=32,
    reduce_num_warps=4,
)
_M32_N64_K128_S12_W4_CG = Gfx90aGateLinearConfig(
    32,
    64,
    128,
    12,
    4,
    waves_per_eu=0,
    kpack=2,
    cache_modifier=".cg",
    reduce_block_m=16,
    reduce_block_n=32,
    reduce_num_warps=4,
)
_M64_N64_K128_S3_W8 = Gfx90aGateLinearConfig(
    64,
    64,
    128,
    3,
    8,
    reduce_block_m=32,
    reduce_block_n=64,
    reduce_num_warps=4,
)
_M64_N128_K64_S2_W8_KPACK2 = Gfx90aGateLinearConfig(
    64,
    128,
    64,
    2,
    8,
    num_stages=2,
    kpack=2,
)
_M64_N64_K64_S2_W8 = Gfx90aGateLinearConfig(64, 64, 64, 2, 8)

# Measured on one MI250X GCD with the pinned LUMI vLLM 0.24 runtime. The
# M=32/64/128/256 cells were selected by exhaustive and long-remeasured search
# in LUMI job 20192276. Keep the bucket keys explicit: the selector uses upper
# bounds, never nearest-neighbor dispatch into a smaller measured shape.
GFX90A_GATE_LINEAR_CONFIGS: dict[int, Gfx90aGateLinearConfig] = {
    1: _M16_N64_K32_S16_W4_WAVE2,
    2: _M16_N64_K32_S16_W4_WAVE2,
    4: _M16_N64_K64_S16_W4,
    8: _M16_N64_K64_S16_W4,
    16: _M16_N64_K64_S16_W4,
    32: _M32_N32_K128_S12_W4_CG,
    64: _M32_N32_K128_S12_W4_WAVE2,
    128: _M64_N32_K128_S12_W4_CG,
    256: _M32_N64_K128_S12_W4_CG,
    512: _M64_N64_K128_S3_W8,
    1024: _M64_N128_K64_S2_W8_KPACK2,
    2048: _M64_N64_K64_S2_W8,
}
GFX90A_GATE_LINEAR_TOKEN_BUCKETS = tuple(GFX90A_GATE_LINEAR_CONFIGS)

_GLM52_CONFIG_FIELDS: dict[str, object] = {
    "model_type": "glm_moe_dsa",
    "hidden_size": GLM52_GATE_HIDDEN_SIZE,
    "n_routed_experts": GLM52_GATE_NUM_EXPERTS,
    "n_shared_experts": 1,
    "num_experts_per_tok": 8,
    "topk_method": "noaux_tc",
    "scoring_func": "sigmoid",
    "n_group": 1,
    "topk_group": 1,
    "moe_intermediate_size": 2048,
    "hidden_act": "silu",
    "norm_topk_prob": True,
    "routed_scaling_factor": 2.5,
    "num_hidden_layers": 78,
    "first_k_dense_replace": 3,
    "moe_layer_freq": 1,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_power_of_two(value: int) -> bool:
    return type(value) is int and value > 0 and value & (value - 1) == 0


def glm52_gate_config_mismatches(config: object) -> tuple[str, ...]:
    """Describe every field that differs from the measured GLM-5.2 contract."""

    mismatches = []
    architectures = tuple(getattr(config, "architectures", ()) or ())
    if architectures != ("GlmMoeDsaForCausalLM",):
        mismatches.append(f"architectures={architectures!r} (expected ('GlmMoeDsaForCausalLM',))")
    for field, expected in _GLM52_CONFIG_FIELDS.items():
        actual = getattr(config, field, None)
        if actual != expected:
            mismatches.append(f"{field}={actual!r} (expected {expected!r})")
    return tuple(mismatches)


def is_glm52_gate_config(config: object) -> bool:
    """Return whether ``config`` exactly matches the gated model contract."""

    return not glm52_gate_config_mismatches(config)


def validate_gate_linear_config(config: Gfx90aGateLinearConfig) -> None:
    """Reject launch shapes that cannot map to gfx90a BF16 MFMA tiles."""

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
    _require(GLM52_GATE_NUM_EXPERTS % config.block_n == 0, "block_n must divide 256")
    _require(config.split_k > 0 and GLM52_GATE_HIDDEN_SIZE % config.split_k == 0, "split_k must divide 6144")
    _require(
        (GLM52_GATE_HIDDEN_SIZE // config.split_k) % config.block_k == 0,
        "block_k must divide the K extent of each split",
    )
    _require(config.num_warps in (4, 8), "num_warps must be 4 or 8")
    _require(config.num_stages in (1, 2), "num_stages must be 1 or 2")
    _require(config.waves_per_eu in (0, 1, 2, 3, 4), "waves_per_eu must be in [0, 4]")
    _require(config.group_m in (1, 2, 4, 8), "group_m must be 1, 2, 4, or 8")
    _require(
        config.matrix_instr_nonkdim == 16,
        "matrix_instr_nonkdim must select gfx90a's 16-wide MFMA layout",
    )
    _require(config.kpack in (1, 2), "kpack must be 1 or 2")
    _require(
        config.cache_modifier in (None, ".cg"),
        "cache_modifier must be None or '.cg'",
    )
    reduction_fields = (
        config.reduce_block_m,
        config.reduce_block_n,
        config.reduce_num_warps,
    )
    _require(
        all(value is None for value in reduction_fields)
        or all(value is not None for value in reduction_fields),
        "reduction overrides must be all unset or all set",
    )
    if config.reduce_block_m is not None:
        _require(
            config.reduce_block_m >= 16
            and _is_power_of_two(config.reduce_block_m),
            "reduce_block_m must be a power of two of at least 16",
        )
        _require(
            config.reduce_block_n is not None
            and config.reduce_block_n >= 16
            and _is_power_of_two(config.reduce_block_n)
            and GLM52_GATE_NUM_EXPERTS % config.reduce_block_n == 0,
            "reduce_block_n must be a power of two of at least 16 that divides 256",
        )
        _require(
            config.reduce_num_warps in (4, 8),
            "reduce_num_warps must be 4 or 8",
        )


def select_gfx90a_gate_linear_config(m: int) -> Gfx90aGateLinearConfig:
    """Select the smallest measured bucket containing ``m`` tokens."""

    _require(type(m) is int and m > 0, "M must be a positive integer")
    _require(m <= GLM52_GATE_MAX_TOKENS, f"GLM-5.2 gate linear is gated through M={GLM52_GATE_MAX_TOKENS}; got {m}")
    token_bucket = next(value for value in GFX90A_GATE_LINEAR_TOKEN_BUCKETS if m <= value)
    return GFX90A_GATE_LINEAR_CONFIGS[token_bucket]


def validate_glm52_gate_scheduler_config(scheduler_config: object) -> int:
    """Fail during model construction if traffic can exceed tuned shapes."""

    max_num_batched_tokens = getattr(
        scheduler_config,
        "max_num_batched_tokens",
        None,
    )
    _require(
        type(max_num_batched_tokens) is int
        and 0 < max_num_batched_tokens <= GLM52_GATE_MAX_TOKENS,
        "gfx90a gate linear requires scheduler max_num_batched_tokens to be "
        f"a positive integer <= {GLM52_GATE_MAX_TOKENS}; got "
        f"{max_num_batched_tokens!r}",
    )
    return max_num_batched_tokens


@functools.lru_cache(maxsize=None)
def validate_gfx90a_device(device_index: int) -> str:
    """Validate the ROCm architecture outside the inference hot path."""

    _require(torch.version.hip is not None, "a ROCm PyTorch build is required")
    properties = torch.cuda.get_device_properties(device_index)
    architecture = getattr(properties, "gcnArchName", "").split(":")[0]
    _require(architecture == "gfx90a", f"expected gfx90a, got {getattr(properties, 'gcnArchName', None)}")
    return architecture


def _validate_gate_linear_inputs(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    config: Gfx90aGateLinearConfig,
) -> int:
    validate_gate_linear_config(config)
    _require(hidden_states.ndim == 2, "hidden_states must have shape [M, 6144]")
    _require(
        hidden_states.shape[1] == GLM52_GATE_HIDDEN_SIZE,
        f"hidden_states K must be {GLM52_GATE_HIDDEN_SIZE}",
    )
    m = hidden_states.shape[0]
    _require(0 < m <= GLM52_GATE_MAX_TOKENS, f"M must be in [1, {GLM52_GATE_MAX_TOKENS}], got {m}")
    _require(hidden_states.dtype == torch.bfloat16, "hidden_states must use torch.bfloat16")
    _require(hidden_states.is_contiguous(), "hidden_states must be contiguous")
    _require(
        tuple(weight.shape) == (GLM52_GATE_NUM_EXPERTS, GLM52_GATE_HIDDEN_SIZE),
        f"weight must have shape ({GLM52_GATE_NUM_EXPERTS}, {GLM52_GATE_HIDDEN_SIZE})",
    )
    _require(weight.dtype == torch.bfloat16, "weight must use torch.bfloat16")
    _require(weight.is_contiguous(), "weight must be contiguous")
    _require(weight.device == hidden_states.device, "weight must be on the same device as hidden_states")
    _require(hidden_states.device.type == "cuda", "hidden_states must be on a ROCm GPU")
    _require(torch.version.hip is not None, "a ROCm PyTorch build is required")
    return m


def gfx90a_glm52_gate_linear(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    *,
    config: Gfx90aGateLinearConfig | None = None,
    device_is_validated: bool = False,
) -> torch.Tensor:
    """Compute the exact GLM-5.2 gate shape with an FP32 output."""

    selected = select_gfx90a_gate_linear_config(hidden_states.shape[0]) if config is None else config
    m = _validate_gate_linear_inputs(hidden_states, weight, selected)
    if not device_is_validated:
        device_index = hidden_states.device.index
        validate_gfx90a_device(torch.cuda.current_device() if device_index is None else device_index)
    from prime_rl.inference.vllm.kernels.rocm_gfx90a_gate_linear_triton import (
        launch_gfx90a_gate_linear,
    )

    output = launch_gfx90a_gate_linear(hidden_states, weight, selected)
    _require(tuple(output.shape) == (m, GLM52_GATE_NUM_EXPERTS), "kernel returned an invalid output shape")
    _require(output.dtype == torch.float32, "kernel must return torch.float32")
    return output


def _custom_op_impl(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return gfx90a_glm52_gate_linear(hidden_states, weight, device_is_validated=True)


def _custom_op_fake(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return hidden_states.new_empty(
        (hidden_states.shape[0], weight.shape[0]),
        dtype=torch.float32,
    )


def _require_exact_vllm_gate_abi(gate_type: type[Any]) -> None:
    installed = version("vllm")
    if installed != _EXPECTED_VLLM_DISTRIBUTION:
        raise RuntimeError(f"gfx90a gate linear requires vLLM {_EXPECTED_VLLM_DISTRIBUTION!r}; found {installed!r}")
    parameters = tuple(inspect.signature(gate_type.forward).parameters.values())
    actual = tuple(parameter.name for parameter in parameters)
    if actual != ("self", "x"):
        raise RuntimeError(f"gfx90a gate linear requires GateLinear.forward(self, x); found {actual}")
    if any(parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD for parameter in parameters):
        raise RuntimeError("gfx90a gate linear requires positional-or-keyword GateLinear.forward parameters")


def _register_custom_op() -> None:
    global _CUSTOM_OP_REGISTERED

    if _CUSTOM_OP_REGISTERED:
        return
    from vllm.platforms import current_platform
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name=_CUSTOM_OP_NAME,
        op_func=_custom_op_impl,
        fake_impl=_custom_op_fake,
        dispatch_key=current_platform.dispatch_key,
    )
    _CUSTOM_OP_REGISTERED = True


def install_glm52_gfx90a_gate_linear(gate: object, model_config: object) -> None:
    """Attach the custom gate only to an exact GLM-5.2 gfx90a module.

    The caller owns opt-in and platform selection. Once selected, every
    contract mismatch is an error; unsupported models continue to use the
    untouched vLLM ``GateLinear`` path.
    """

    mismatches = glm52_gate_config_mismatches(model_config)
    if mismatches:
        raise ValueError("gfx90a gate linear requires the exact GLM-5.2 config: " + "; ".join(mismatches))

    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

    _require_exact_vllm_gate_abi(GateLinear)
    device = gate.weight.device
    _require(device.type == "cuda", "gate weight must be on a ROCm GPU")
    device_index = device.index
    validate_gfx90a_device(torch.cuda.current_device() if device_index is None else device_index)
    _require(type(gate) is GateLinear, f"expected GateLinear, got {type(gate).__name__}")
    _require(
        tuple(gate.weight.shape) == (GLM52_GATE_NUM_EXPERTS, GLM52_GATE_HIDDEN_SIZE),
        "GateLinear weight has an unsupported shape",
    )
    _require(gate.weight.dtype == torch.bfloat16, "GateLinear weight must use torch.bfloat16")
    _require(gate.weight.is_contiguous(), "GateLinear weight must be contiguous")
    _require(getattr(gate, "bias", None) is None, "GateLinear bias is unsupported")
    _require(
        gate.out_dtype in (None, torch.float32),
        "GateLinear out_dtype must be unset or torch.float32, "
        f"got {gate.out_dtype}",
    )

    _register_custom_op()
    original_forward = GateLinear.forward
    if not getattr(original_forward, _PATCH_MARKER, False):

        @functools.wraps(original_forward)
        def _patched_forward(self, x):
            if getattr(self, _GATE_MARKER, False):
                output = torch.ops.vllm.prime_glm52_gfx90a_gate_linear(x, self.weight)
                return output, None
            return original_forward(self, x)

        setattr(_patched_forward, _PATCH_MARKER, True)
        GateLinear.forward = _patched_forward

    if gate.out_dtype is None:
        gate.set_out_dtype(torch.float32)
    setattr(gate, _GATE_MARKER, True)


__all__ = [
    "GFX90A_GATE_LINEAR_CONFIGS",
    "GFX90A_GATE_LINEAR_TOKEN_BUCKETS",
    "GLM52_GATE_HIDDEN_SIZE",
    "GLM52_GATE_MAX_TOKENS",
    "GLM52_GATE_NUM_EXPERTS",
    "Gfx90aGateLinearConfig",
    "gfx90a_glm52_gate_linear",
    "glm52_gate_config_mismatches",
    "install_glm52_gfx90a_gate_linear",
    "is_glm52_gate_config",
    "select_gfx90a_gate_linear_config",
    "validate_gate_linear_config",
    "validate_glm52_gate_scheduler_config",
    "validate_gfx90a_device",
]
