# SPDX-License-Identifier: Apache-2.0

"""BF16-to-FP32 Nemotron-H router GEMM for AMD CDNA2.

vLLM stores Nemotron-H router weights in FP32 on ROCm because its native
BF16-input, FP32-output matrix multiply falls back to a very slow path.  This
keeps that representation for small decode batches and adds a BF16 copy for
the MFMA prefill path.  The checkpoint weights are BF16, so both copies
represent the checkpoint exactly and both paths return FP32 router logits.

This integration is opt-in and limited to the validated Nemotron-H gate family:
contiguous ``[M, 4096]`` BF16 activations, contiguous ``[512, 4096]`` router
weights, FP32 output, and ``1 <= M <= 32768`` on gfx90a.
"""

from __future__ import annotations

import functools
import inspect
from importlib.metadata import version
from typing import Any

import torch

NEMOTRON_ROUTER_HIDDEN_SIZE = 4096
NEMOTRON_ROUTER_NUM_EXPERTS = 512
NEMOTRON_ROUTER_MAX_TOKENS = 32768
_EXPECTED_VLLM_DISTRIBUTION = "0.24.0+lumi_aif_gfx90a_ee0da84"
_CUSTOM_OP_NAME = "prime_nemotron_router_linear"
_CUSTOM_OP_REGISTERED = False
_GATE_MARKER = "_prime_nemotron_gfx90a_router_linear"
_PATCH_MARKER = "_prime_nemotron_gfx90a_router_linear_patch"

_SMALL_CONFIG = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 1,
    "num_warps": 4,
    "num_stages": 2,
    "waves_per_eu": 2,
    "matrix_instr_nonkdim": 16,
    "cache_modifier": None,
    "NUM_KSPLIT": 1,
    "kpack": 1,
    "SPLITK_BLOCK_SIZE": NEMOTRON_ROUTER_HIDDEN_SIZE,
}
_LARGE_CONFIG = {
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 1,
    "num_warps": 8,
    "num_stages": 2,
    "waves_per_eu": 1,
    "matrix_instr_nonkdim": 16,
    "cache_modifier": None,
    "NUM_KSPLIT": 1,
    "kpack": 1,
    "SPLITK_BLOCK_SIZE": NEMOTRON_ROUTER_HIDDEN_SIZE,
}

_NEMOTRON_CONFIG_FIELDS: dict[str, object] = {
    "model_type": "nemotron_h",
    "hidden_size": NEMOTRON_ROUTER_HIDDEN_SIZE,
    "n_routed_experts": NEMOTRON_ROUTER_NUM_EXPERTS,
    "num_experts_per_tok": 22,
    "moe_intermediate_size": 2688,
    "moe_latent_size": 1024,
    "mlp_hidden_act": "relu2",
    "num_hidden_layers": 88,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def nemotron_router_config_mismatches(config: object) -> tuple[str, ...]:
    """Describe deviations from the model family used for measurement."""

    mismatches = []
    architectures = tuple(getattr(config, "architectures", ()) or ())
    if architectures != ("NemotronHForCausalLM",):
        mismatches.append(f"architectures={architectures!r} (expected ('NemotronHForCausalLM',))")
    for field, expected in _NEMOTRON_CONFIG_FIELDS.items():
        actual = getattr(config, field, None)
        if actual != expected:
            mismatches.append(f"{field}={actual!r} (expected {expected!r})")
    return tuple(mismatches)


def select_nemotron_router_config(m: int) -> dict[str, object]:
    """Select a token-range configuration rather than an exact-M dispatch."""

    _require(type(m) is int and m > 0, "M must be a positive integer")
    _require(
        m <= NEMOTRON_ROUTER_MAX_TOKENS,
        f"Nemotron router is gated through M={NEMOTRON_ROUTER_MAX_TOKENS}; got {m}",
    )
    return _SMALL_CONFIG if m <= 512 else _LARGE_CONFIG


def validate_nemotron_router_scheduler(scheduler_config: object) -> int:
    max_tokens = getattr(scheduler_config, "max_num_batched_tokens", None)
    _require(
        type(max_tokens) is int and 0 < max_tokens <= NEMOTRON_ROUTER_MAX_TOKENS,
        "gfx90a Nemotron router requires scheduler max_num_batched_tokens "
        f"in [1, {NEMOTRON_ROUTER_MAX_TOKENS}]; got {max_tokens!r}",
    )
    return max_tokens


def _validate_inputs(
    hidden_states: torch.Tensor,
    weight_fp32: torch.Tensor,
    weight_bf16: torch.Tensor,
) -> int:
    _require(
        hidden_states.ndim == 2,
        "hidden_states must have shape [M, 4096]",
    )
    m, hidden_size = hidden_states.shape
    select_nemotron_router_config(m)
    _require(
        hidden_size == NEMOTRON_ROUTER_HIDDEN_SIZE,
        f"hidden_states K must be {NEMOTRON_ROUTER_HIDDEN_SIZE}",
    )
    _require(
        hidden_states.dtype == torch.bfloat16,
        "hidden_states must use torch.bfloat16",
    )
    _require(hidden_states.is_contiguous(), "hidden_states must be contiguous")
    _require(
        tuple(weight_fp32.shape) == (NEMOTRON_ROUTER_NUM_EXPERTS, NEMOTRON_ROUTER_HIDDEN_SIZE),
        f"FP32 weight must have shape ({NEMOTRON_ROUTER_NUM_EXPERTS}, {NEMOTRON_ROUTER_HIDDEN_SIZE})",
    )
    _require(weight_fp32.dtype == torch.float32, "primary weight must use torch.float32")
    _require(weight_fp32.is_contiguous(), "FP32 weight must be contiguous")
    _require(
        tuple(weight_bf16.shape) == (NEMOTRON_ROUTER_NUM_EXPERTS, NEMOTRON_ROUTER_HIDDEN_SIZE),
        f"BF16 weight must have shape ({NEMOTRON_ROUTER_NUM_EXPERTS}, {NEMOTRON_ROUTER_HIDDEN_SIZE})",
    )
    _require(weight_bf16.dtype == torch.bfloat16, "prefill weight must use torch.bfloat16")
    _require(weight_bf16.is_contiguous(), "BF16 weight must be contiguous")
    _require(
        weight_fp32.device == hidden_states.device and weight_bf16.device == hidden_states.device,
        "weights and activations must be on the same device",
    )
    _require(
        hidden_states.device.type == "cuda" and torch.version.hip is not None,
        "a ROCm GPU tensor is required",
    )
    return m


def nemotron_router_linear(
    hidden_states: torch.Tensor,
    weight_fp32: torch.Tensor,
    weight_bf16: torch.Tensor,
) -> torch.Tensor:
    """Run the validated decode/prefill gate projection."""

    m = _validate_inputs(hidden_states, weight_fp32, weight_bf16)
    if m <= 512:
        return torch.nn.functional.linear(hidden_states.float(), weight_fp32)

    from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16

    output = gemm_a16w16(
        hidden_states,
        weight_bf16,
        dtype=torch.float32,
        config=select_nemotron_router_config(m),
        backend="triton",
    )
    _require(
        tuple(output.shape) == (m, NEMOTRON_ROUTER_NUM_EXPERTS),
        "AITER returned an invalid router output shape",
    )
    _require(
        output.dtype == torch.float32,
        "AITER router output must use torch.float32",
    )
    return output


def _custom_op_fake(
    hidden_states: torch.Tensor,
    weight_fp32: torch.Tensor,
    weight_bf16: torch.Tensor,
) -> torch.Tensor:
    return hidden_states.new_empty(
        (hidden_states.shape[0], weight_fp32.shape[0]),
        dtype=torch.float32,
    )


def _register_custom_op() -> None:
    global _CUSTOM_OP_REGISTERED
    if _CUSTOM_OP_REGISTERED:
        return

    from vllm.platforms import current_platform
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name=_CUSTOM_OP_NAME,
        op_func=nemotron_router_linear,
        fake_impl=_custom_op_fake,
        dispatch_key=current_platform.dispatch_key,
    )
    _CUSTOM_OP_REGISTERED = True


def _require_gfx90a() -> None:
    _require(torch.version.hip is not None, "a ROCm PyTorch build is required")
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    architecture = getattr(properties, "gcnArchName", "").split(":")[0]
    _require(
        architecture == "gfx90a",
        f"expected gfx90a, got {getattr(properties, 'gcnArchName', None)}",
    )


def _require_gate_abi(gate_type: type[Any]) -> None:
    installed = version("vllm")
    if installed != _EXPECTED_VLLM_DISTRIBUTION:
        raise RuntimeError(f"gfx90a Nemotron router requires vLLM {_EXPECTED_VLLM_DISTRIBUTION!r}; found {installed!r}")
    parameters = tuple(inspect.signature(gate_type.__init__).parameters)
    expected = (
        "self",
        "input_size",
        "output_size",
        "bias",
        "out_dtype",
        "params_dtype",
        "force_fp32_compute",
        "prefix",
    )
    if parameters != expected:
        raise RuntimeError(f"gfx90a Nemotron router requires GateLinear.__init__{expected}; found {parameters}")


def install_nemotron_gfx90a_router_linear() -> None:
    """Replace only Nemotron-H's GateLinear construction under explicit opt-in."""

    from vllm.model_executor.layers.fused_moe.router.gate_linear import (
        GateLinear,
    )
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.models import nemotron_h

    if getattr(nemotron_h.NemotronHMoE.__init__, _PATCH_MARKER, False):
        return

    _require_gfx90a()
    _require_gate_abi(GateLinear)
    _register_custom_op()

    class Gfx90aNemotronRouterLinearMethod(UnquantizedLinearMethod):
        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            super().process_weights_after_loading(layer)
            layer.weight_bf16.copy_(layer.weight)

    class Gfx90aNemotronGateLinear(GateLinear):
        def __init__(
            self,
            input_size: int,
            output_size: int,
            bias: bool = False,
            out_dtype: torch.dtype | None = None,
            params_dtype: torch.dtype | None = None,
            force_fp32_compute: bool = False,
            prefix: str = "",
        ):
            eligible = (
                input_size == NEMOTRON_ROUTER_HIDDEN_SIZE
                and output_size == NEMOTRON_ROUTER_NUM_EXPERTS
                and bias is False
                and out_dtype == torch.float32
                and force_fp32_compute is True
            )
            super().__init__(
                input_size=input_size,
                output_size=output_size,
                bias=bias,
                out_dtype=out_dtype,
                params_dtype=params_dtype,
                force_fp32_compute=force_fp32_compute,
                prefix=prefix,
            )
            if eligible:
                self.register_buffer(
                    "weight_bf16",
                    torch.empty(
                        (output_size, input_size),
                        dtype=torch.bfloat16,
                        device=self.weight.device,
                    ),
                    persistent=False,
                )
                self.quant_method = Gfx90aNemotronRouterLinearMethod()
                setattr(self, _GATE_MARKER, True)

        def forward(self, x: torch.Tensor):
            if getattr(self, _GATE_MARKER, False):
                output = torch.ops.vllm.prime_nemotron_router_linear(
                    x,
                    self.weight,
                    self.weight_bf16,
                )
                return output, None
            return super().forward(x)

    original_init = nemotron_h.NemotronHMoE.__init__
    nemotron_h.GateLinear = Gfx90aNemotronGateLinear

    @functools.wraps(original_init)
    def _patched_init(self, config, *args, **kwargs):
        from vllm.config import get_current_vllm_config

        mismatches = nemotron_router_config_mismatches(config)
        if mismatches:
            raise ValueError(
                "gfx90a Nemotron router requires the measured Nemotron-H model family: " + "; ".join(mismatches)
            )
        validate_nemotron_router_scheduler(get_current_vllm_config().scheduler_config)
        original_init(self, config, *args, **kwargs)
        if not getattr(self.gate, _GATE_MARKER, False):
            raise RuntimeError("Nemotron-H gate did not select the gfx90a BF16-to-FP32 path")
        _require(
            self.gate.weight.dtype == torch.float32,
            "Nemotron-H primary gate weight must remain FP32",
        )
        _require(
            self.gate.weight_bf16.dtype == torch.bfloat16,
            "Nemotron-H prefill gate weight must use BF16",
        )
        _require(
            self.gate.out_dtype == torch.float32,
            "Nemotron-H gate output must remain FP32",
        )

    setattr(_patched_init, _PATCH_MARKER, True)
    nemotron_h.NemotronHMoE.__init__ = _patched_init


__all__ = [
    "NEMOTRON_ROUTER_HIDDEN_SIZE",
    "NEMOTRON_ROUTER_MAX_TOKENS",
    "NEMOTRON_ROUTER_NUM_EXPERTS",
    "install_nemotron_gfx90a_router_linear",
    "nemotron_router_config_mismatches",
    "nemotron_router_linear",
    "select_nemotron_router_config",
    "validate_nemotron_router_scheduler",
]
