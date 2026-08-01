# SPDX-License-Identifier: Apache-2.0

"""Registered vLLM quantization boundary for the opt-in gfx90a INT8 path."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, cast

import torch
from vllm.model_executor.layers.quantization.fp8 import Fp8Config

from prime_rl.inference.vllm.kernels.rocm_cdna2_int8_audit import (
    cdna2_int8_audit_requested,
    log_exemption,
)
from prime_rl.inference.vllm.kernels.rocm_cdna2_int8_custom_ops import (
    register_cdna2_int8_custom_ops,
)
from prime_rl.inference.vllm.kernels.rocm_cdna2_int8_linear import (
    PREFILL_MAX_TOKENS,
    SOURCE_GROUP_K,
    SOURCE_GROUP_N,
)
from prime_rl.inference.vllm.kernels.rocm_cdna2_int8_linear_method import (
    Cdna2Int8Fp8LinearMethod,
    ReloadMode,
    _resolve_gfx90a_architecture,
    _validate_reload_mode,
    cdna2_int8_requested,
)
from prime_rl.inference.vllm.kernels.rocm_cdna2_int8_moe_method import (
    Cdna2Int8Fp8MoEMethod,
)
from prime_rl.inference.vllm.kernels.rocm_cdna2_int8_vllm_abi import (
    install_vllm_kernel_abi,
)
from prime_rl.inference.vllm.kernels.rocm_gfx90a_skinny_int8_linear import (
    gfx90a_skinny_int8_requested,
)

QUANTIZATION_NAME = "prime_cdna2_int8"
RELOAD_MODE_ENV = "PRIME_ROCM_CDNA2_INT8_RELOAD_MODE"
_REGISTERED = False
SOURCE_FP8_POST_LOAD_SUFFIXES = ("self_attn.kv_b_proj",)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _requires_source_fp8_post_load(prefix: str) -> bool:
    """Keep layers whose consumers explicitly recognize stock FP8 methods."""

    return prefix.endswith(SOURCE_FP8_POST_LOAD_SUFFIXES)


def _validate_scheduler_contract() -> int:
    """Reject unsupported traffic shapes before discarding FP8 weights."""

    from vllm.config import get_current_vllm_config_or_none

    vllm_config = get_current_vllm_config_or_none()
    _require(
        vllm_config is not None,
        "prime_cdna2_int8 requires a current vLLM configuration during quantized-method construction",
    )
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    max_num_batched_tokens = getattr(
        scheduler_config,
        "max_num_batched_tokens",
        None,
    )
    _require(
        type(max_num_batched_tokens) is int and 0 < max_num_batched_tokens <= PREFILL_MAX_TOKENS,
        "prime_cdna2_int8 requires scheduler max_num_batched_tokens to be "
        f"a positive integer <= {PREFILL_MAX_TOKENS}; got "
        f"{max_num_batched_tokens!r}",
    )
    return max_num_batched_tokens


class Cdna2Int8Fp8Config(Fp8Config):
    """Load an OCP-FP8 checkpoint once, then own its INT8 runtime form."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        reload_mode = os.environ.get(RELOAD_MODE_ENV, "none")
        _validate_reload_mode(reload_mode)
        self.cdna2_reload_mode = cast(ReloadMode, reload_mode)
        self.cdna2_audit = cdna2_int8_audit_requested()
        self.gfx90a_skinny_int8 = gfx90a_skinny_int8_requested()

    @classmethod
    def get_name(cls) -> str:
        return QUANTIZATION_NAME

    @classmethod
    def get_min_capability(cls) -> int:
        # CUDA capability is not meaningful for this gfx90a-only config.
        return 0

    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg: dict[str, Any],
        user_quant: str | None,
        hf_config: Any = None,
    ) -> str | None:
        """Allow an explicit custom runtime for an otherwise FP8 checkpoint."""

        if user_quant != QUANTIZATION_NAME:
            return None
        _require(
            "fp8" in str(hf_quant_cfg.get("quant_method", "")),
            "prime_cdna2_int8 requires an FP8 checkpoint",
        )
        _require(
            hf_quant_cfg.get("activation_scheme") == "dynamic",
            "prime_cdna2_int8 requires dynamic activation quantization",
        )
        _require(
            tuple(hf_quant_cfg.get("weight_block_size", ())) == (SOURCE_GROUP_N, SOURCE_GROUP_K),
            "prime_cdna2_int8 requires weight_block_size=[128, 128]",
        )
        return QUANTIZATION_NAME

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Cdna2Int8Fp8Config:
        result = super().from_config(config)
        _require(
            result.is_checkpoint_fp8_serialized,
            "prime_cdna2_int8 requires a serialized FP8 checkpoint",
        )
        _require(
            result.activation_scheme == "dynamic",
            "prime_cdna2_int8 requires dynamic activation quantization",
        )
        _require(
            tuple(result.weight_block_size or ()) == (SOURCE_GROUP_N, SOURCE_GROUP_K),
            "prime_cdna2_int8 requires weight_block_size=[128, 128]",
        )
        _require(result.store_dtype in (None, "fp8"), "unsupported FP8 store_dtype")
        return result

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ):
        method = super().get_quant_method(layer, prefix)

        from vllm.model_executor.layers.quantization.fp8 import (
            Fp8LinearMethod,
            Fp8MoEMethod,
        )

        if isinstance(method, Fp8LinearMethod):
            if _requires_source_fp8_post_load(prefix):
                log_exemption(
                    enabled=self.cdna2_audit,
                    kind="linear",
                    prefix=prefix,
                    reason="stock_fp8_post_load_required",
                )
                return method
            _validate_scheduler_contract()
            architecture = _resolve_gfx90a_architecture()
            return Cdna2Int8Fp8LinearMethod(
                method,
                gcn_arch_name=architecture,
                reload_mode=self.cdna2_reload_mode,
                prefix=prefix,
                audit=self.cdna2_audit,
                use_gfx90a_skinny_int8=self.gfx90a_skinny_int8,
            )
        if isinstance(method, Fp8MoEMethod):
            _validate_scheduler_contract()
            _resolve_gfx90a_architecture()
            return Cdna2Int8Fp8MoEMethod(
                method,
                reload_mode=self.cdna2_reload_mode,
                prefix=prefix,
                audit=self.cdna2_audit,
            )
        return method


def install_cdna2_int8_quantization(
    *,
    reload_mode: ReloadMode = "none",
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Register the custom quantization config before vLLM parses model config.

    The caller must also select ``quantization=prime_cdna2_int8`` in vLLM.
    Registration uses vLLM's public custom-config seam and does not patch FP8
    methods or retain an FP8 fallback representation.
    """

    global _REGISTERED

    if not cdna2_int8_requested(environ):
        return False
    _validate_reload_mode(reload_mode)
    cdna2_int8_audit_requested(environ)
    gfx90a_skinny_int8_requested(environ)
    install_vllm_kernel_abi()
    register_cdna2_int8_custom_ops()
    existing_reload_mode = os.environ.get(RELOAD_MODE_ENV)
    _require(
        existing_reload_mode in (None, reload_mode),
        "CDNA2 INT8 reload mode changed after registration",
    )
    os.environ[RELOAD_MODE_ENV] = reload_mode
    if _REGISTERED:
        return True

    from vllm.model_executor.layers.quantization import register_quantization_config

    register_quantization_config(QUANTIZATION_NAME)(Cdna2Int8Fp8Config)
    _REGISTERED = True
    return True
