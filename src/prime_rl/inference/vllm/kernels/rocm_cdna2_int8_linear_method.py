# SPDX-License-Identifier: Apache-2.0

"""Opt-in vLLM integration for the gfx90a fine-group INT8 linear path.

The integration deliberately leaves vLLM's checkpoint representation intact
until ``process_weights_after_loading``. It wraps only serialized, dynamic,
OCP E4M3FN block-FP8 linears with checkpoint block shape ``[128, 128]``.
Weights are then transcoded to INT8 ``[1, 32]`` blocks once, while ``apply``
uses the architecture decision cached by the selector.

Set ``PRIME_ROCM_CDNA2_INT8=1`` and select
``quantization=prime_cdna2_int8``. PRIME-RL's vLLM general plugin registers
the custom quantization config before model construction.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from functools import lru_cache
from typing import Literal

import torch
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.utils import replace_parameter

from prime_rl.inference.vllm.kernels.rocm_cdna2_int8_audit import (
    log_transcode,
)
from prime_rl.inference.vllm.kernels.rocm_cdna2_int8_custom_ops import (
    cdna2_int8_linear_opaque,
)
from prime_rl.inference.vllm.kernels.rocm_cdna2_int8_linear import (
    RUNTIME_WEIGHT_SCALE_DTYPE,
    SOURCE_GROUP_K,
    SOURCE_GROUP_N,
    TARGET_GROUP_K,
    TARGET_GROUP_N,
    select_gfx90a_int8_linear_config,
    transcode_ocp_fp8_weight,
    validate_gfx90a_device,
)
from prime_rl.inference.vllm.kernels.rocm_gfx90a_skinny_int8_linear import (
    gfx90a_skinny_int8_requested,
)

ENABLE_ENV = "PRIME_ROCM_CDNA2_INT8"
SOURCE_BLOCK_SHAPE = (SOURCE_GROUP_N, SOURCE_GROUP_K)
TARGET_BLOCK_SHAPE = (TARGET_GROUP_N, TARGET_GROUP_K)
ReloadMode = Literal["none", "checkpoint", "kernel"]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def cdna2_int8_requested(
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Read the model-wide INT8 opt-in without ambiguous spellings."""

    value = (os.environ if environ is None else environ).get(ENABLE_ENV)
    if value in (None, "0"):
        return False
    if value == "1":
        return True
    raise ValueError(f"{ENABLE_ENV} must be unset, '0', or '1'; got {value!r}")


def _get_fp8_linear_method_type() -> type:
    from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod

    return Fp8LinearMethod


def _validate_reload_mode(reload_mode: ReloadMode) -> None:
    _require(
        reload_mode in ("none", "checkpoint", "kernel"),
        f"unknown CDNA2 INT8 reload mode {reload_mode!r}",
    )
    _require(
        reload_mode == "none",
        "CDNA2 INT8 live reload is not validated; use reload_mode='none'",
    )


def _validate_method_contract(method: object) -> None:
    fp8_method_type = _get_fp8_linear_method_type()
    _require(
        isinstance(method, fp8_method_type),
        "CDNA2 INT8 requires vLLM Fp8LinearMethod",
    )
    quant_config = getattr(method, "quant_config", None)
    _require(quant_config is not None, "Fp8LinearMethod has no quant_config")
    _require(
        getattr(quant_config, "is_checkpoint_fp8_serialized", False) is True,
        "CDNA2 INT8 requires a serialized FP8 checkpoint",
    )
    _require(
        getattr(quant_config, "activation_scheme", None) == "dynamic",
        "CDNA2 INT8 requires dynamic activation quantization",
    )
    block_shape = getattr(quant_config, "weight_block_size", None)
    _require(
        block_shape is not None and tuple(block_shape) == SOURCE_BLOCK_SHAPE,
        "CDNA2 INT8 requires checkpoint weight_block_size=[128, 128]",
    )
    _require(
        getattr(method, "block_quant", False) is True,
        "CDNA2 INT8 requires block-quantized Fp8LinearMethod",
    )
    _require(
        getattr(method, "act_q_static", True) is False,
        "CDNA2 INT8 does not support static activation scales",
    )
    _require(
        getattr(method, "is_scale_e8m0", False) is False,
        "CDNA2 INT8 requires floating-point checkpoint block scales",
    )


@lru_cache
def _resolve_gfx90a_architecture_for_device(device_index: int) -> str:
    return validate_gfx90a_device(torch.device("cuda", device_index))


def _resolve_gfx90a_architecture() -> str:
    return _resolve_gfx90a_architecture_for_device(torch.cuda.current_device())


class Cdna2Int8Fp8LinearMethod(LinearMethodBase):
    """vLLM linear method that owns the OCP-FP8-to-INT8 representation change."""

    uses_meta_device = False

    def __init__(
        self,
        source_method: object,
        *,
        gcn_arch_name: str,
        reload_mode: ReloadMode,
        prefix: str = "",
        audit: bool = False,
        use_gfx90a_skinny_int8: bool | None = None,
    ) -> None:
        _validate_reload_mode(reload_mode)
        _validate_method_contract(source_method)
        _require(
            gcn_arch_name.split(":")[0] == "gfx90a",
            f"expected gfx90a, got {gcn_arch_name}",
        )
        self.source_method = source_method
        self.quant_config = source_method.quant_config
        self.gcn_arch_name = gcn_arch_name
        self.reload_mode = reload_mode
        self.prefix = prefix
        self.audit = audit
        self.use_gfx90a_skinny_int8 = (
            gfx90a_skinny_int8_requested() if use_gfx90a_skinny_int8 is None else use_gfx90a_skinny_int8
        )
        _require(
            type(self.use_gfx90a_skinny_int8) is bool,
            "use_gfx90a_skinny_int8 must be a bool",
        )
        self.block_quant = True
        self.weight_block_size = list(TARGET_BLOCK_SHAPE)
        self.use_marlin = False
        self.use_deep_gemm = False

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        """Create the original checkpoint tensors through vLLM's FP8 method."""

        self.source_method.create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )
        self._validate_source_layer(layer)

    @staticmethod
    def _validate_source_layer(layer: torch.nn.Module) -> None:
        _require(hasattr(layer, "weight"), "linear layer has no weight")
        _require(
            hasattr(layer, "weight_scale_inv"),
            "CDNA2 INT8 requires block scale parameter weight_scale_inv",
        )
        _require(
            tuple(getattr(layer, "weight_block_size", ())) == SOURCE_BLOCK_SHAPE,
            "linear layer checkpoint block shape must be [128, 128]",
        )
        weight = layer.weight
        weight_scale = layer.weight_scale_inv
        _require(
            weight.dtype == torch.float8_e4m3fn,
            "linear checkpoint weight must use OCP torch.float8_e4m3fn",
        )
        _require(
            weight.ndim == 2 and weight.is_contiguous(),
            "linear checkpoint weight must be contiguous [N, K]",
        )
        _require(
            weight.shape[1] % TARGET_GROUP_K == 0,
            "linear checkpoint K must be divisible by 32",
        )
        select_gfx90a_int8_linear_config(
            1,
            weight.shape[0],
            weight.shape[1],
        )
        expected_scale_shape = (
            (weight.shape[0] + SOURCE_GROUP_N - 1) // SOURCE_GROUP_N,
            (weight.shape[1] + SOURCE_GROUP_K - 1) // SOURCE_GROUP_K,
        )
        _require(
            weight_scale.device == weight.device,
            "linear checkpoint scales must be on the weight device",
        )
        _require(
            weight_scale.dtype in (torch.float16, torch.bfloat16, torch.float32),
            "linear checkpoint scales must use a floating-point dtype",
        )
        _require(
            weight_scale.ndim == 2
            and weight_scale.is_contiguous()
            and tuple(weight_scale.shape) == expected_scale_shape,
            f"linear checkpoint scales must have shape {expected_scale_shape}",
        )

    @staticmethod
    def _validate_runtime_layer(layer: torch.nn.Module) -> None:
        _require(
            getattr(layer, "_prime_cdna2_int8_linear_ready", False) is True,
            "CDNA2 INT8 linear weight has not been processed",
        )
        _require(layer.weight.dtype == torch.int8, "CDNA2 runtime weight must use torch.int8")
        _require(
            layer.weight.ndim == 2 and layer.weight.is_contiguous(),
            "CDNA2 runtime weight must be contiguous [N, K]",
        )
        _require(
            tuple(getattr(layer, "weight_block_size", ())) == TARGET_BLOCK_SHAPE,
            "CDNA2 runtime layer block shape must be [1, 32]",
        )
        _require(
            hasattr(layer, "weight_scale_inv"),
            "CDNA2 runtime scale weight_scale_inv is missing",
        )
        expected_scale_shape = (
            layer.weight.shape[0],
            layer.weight.shape[1] // TARGET_GROUP_K,
        )
        _require(
            layer.weight_scale_inv.dtype == RUNTIME_WEIGHT_SCALE_DTYPE
            and layer.weight_scale_inv.is_contiguous()
            and tuple(layer.weight_scale_inv.shape) == expected_scale_shape,
            f"CDNA2 runtime scales must be contiguous FP16 {expected_scale_shape}",
        )

    @torch.no_grad()
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Transcode initial weights or a restored checkpoint-format reload."""

        already_processed = getattr(layer, "_prime_cdna2_int8_linear_ready", False)
        if already_processed and self.reload_mode == "none":
            raise RuntimeError("CDNA2 INT8 linear is static-only; live weight reload was not enabled")
        if already_processed and layer.weight.dtype != torch.float8_e4m3fn:
            raise RuntimeError(
                "CDNA2 INT8 checkpoint reload must restore OCP E4M3FN weights before post-load processing"
            )

        # The stock block-FP8 post-load hook selects/shuffles an FP8 execution
        # layout. This path instead consumes the loader-materialized, TP-sharded
        # OCP tensors directly and owns their one-way INT8 layout conversion.
        self._validate_source_layer(layer)
        source_weight = layer.weight
        source_scale = layer.weight_scale_inv
        quantized_weight, target_scale = transcode_ocp_fp8_weight(
            source_weight,
            source_scale,
            device_is_validated=True,
        )
        replace_parameter(layer, "weight", quantized_weight)
        replace_parameter(layer, "weight_scale_inv", target_scale)
        layer.weight_block_size = list(TARGET_BLOCK_SHAPE)
        layer.input_scale = None
        layer._prime_cdna2_int8_linear_ready = True
        layer._prime_cdna2_int8_source_block_shape = SOURCE_BLOCK_SHAPE
        layer._prime_cdna2_int8_runtime_block_shape = TARGET_BLOCK_SHAPE
        layer._prime_cdna2_int8_reload_mode = self.reload_mode
        self._validate_runtime_layer(layer)
        log_transcode(
            enabled=self.audit,
            kind="linear",
            prefix=self.prefix,
            source_weight=source_weight,
            runtime_weight=layer.weight,
            source_scale=source_scale,
            runtime_scale=layer.weight_scale_inv,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply without device-property queries or scalar reads in the hot path."""

        self._validate_runtime_layer(layer)
        _require(x.ndim >= 2, "CDNA2 runtime input must have at least two dimensions")
        return cdna2_int8_linear_opaque(
            x,
            layer.weight,
            layer.weight_scale_inv,
            bias,
            self.use_gfx90a_skinny_int8,
        )


def select_cdna2_int8_linear_method(
    method: object,
    *,
    environ: Mapping[str, str] | None = None,
    gcn_arch_name: str | None = None,
    reload_mode: ReloadMode = "none",
) -> object:
    """Wrap an eligible vLLM FP8 linear method when the explicit opt-in is set."""

    if not cdna2_int8_requested(environ):
        return method
    _validate_reload_mode(reload_mode)
    _validate_method_contract(method)
    if gcn_arch_name is None:
        gcn_arch_name = _resolve_gfx90a_architecture()
    _require(gcn_arch_name.split(":")[0] == "gfx90a", f"expected gfx90a, got {gcn_arch_name}")
    return Cdna2Int8Fp8LinearMethod(
        method,
        gcn_arch_name=gcn_arch_name,
        reload_mode=reload_mode,
        use_gfx90a_skinny_int8=gfx90a_skinny_int8_requested(environ),
    )


def install_cdna2_int8_linear_method(
    *,
    reload_mode: ReloadMode = "none",
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Compatibility alias for the registered, non-monkeypatched boundary."""

    from prime_rl.inference.vllm.kernels.rocm_cdna2_int8_quant_config import (
        install_cdna2_int8_quantization,
    )

    return install_cdna2_int8_quantization(
        reload_mode=reload_mode,
        environ=environ,
    )
