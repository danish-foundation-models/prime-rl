# SPDX-License-Identifier: Apache-2.0

"""vLLM method boundary for the measured GLM-5.2 gfx90a INT8 MoE path."""

from __future__ import annotations

from typing import Literal

import torch
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.utils import replace_parameter

from prime_rl.inference.vllm.kernels.rocm_cdna2_int8_audit import (
    log_transcode,
)
from prime_rl.inference.vllm.kernels.rocm_cdna2_int8_custom_ops import (
    cdna2_int8_moe_opaque,
)
from prime_rl.inference.vllm.kernels.rocm_cdna2_int8_linear import (
    PREFILL_MAX_TOKENS,
    RUNTIME_WEIGHT_SCALE_DTYPE,
    SOURCE_GROUP_K,
    SOURCE_GROUP_N,
    TARGET_GROUP_K,
)
from prime_rl.inference.vllm.kernels.rocm_cdna2_int8_moe import (
    GLOBAL_EP_MAX_TOKENS,
    transcode_ocp_fp8_expert_weight,
)

ReloadMode = Literal["none", "checkpoint", "kernel"]
SOURCE_BLOCK_SHAPE = (SOURCE_GROUP_N, SOURCE_GROUP_K)
TARGET_BLOCK_SHAPE = (1, TARGET_GROUP_K)

GLM52_GLOBAL_EXPERTS = 256
# Keep the original name as the default/test fixture for the proven DP2 cell.
GLM52_LOCAL_EXPERTS = 16
GLM52_EP32_LOCAL_EXPERTS = 8
GLM52_TOP_K = 8
GLM52_HIDDEN_SIZE = 6144
GLM52_INTERMEDIATE_SIZE = 2048
# With EP enabled vLLM folds launch TP8 x DP x PCP1 into EP, resets the
# MoE-local TP factor to one, and retains the launch TP factor as SP8.
GLM52_MOE_TP_SIZE = 1
GLM52_DP_SIZE = 2
GLM52_PCP_SIZE = 1
GLM52_SP_SIZE = 8
GLM52_SUPPORTED_EP_LAYOUTS = frozenset(
    {
        (2, 16, GLM52_LOCAL_EXPERTS),
        (4, 32, GLM52_EP32_LOCAL_EXPERTS),
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_reload_mode(reload_mode: ReloadMode) -> None:
    _require(
        reload_mode in ("none", "checkpoint", "kernel"),
        f"unknown CDNA2 INT8 reload mode {reload_mode!r}",
    )
    _require(
        reload_mode == "none",
        "CDNA2 INT8 live reload is not validated; use reload_mode='none'",
    )


def _get_fp8_moe_method_type() -> type:
    from vllm.model_executor.layers.quantization.fp8 import Fp8MoEMethod

    return Fp8MoEMethod


def _validate_source_method(method: object) -> None:
    _require(
        isinstance(method, _get_fp8_moe_method_type()),
        "CDNA2 INT8 MoE requires vLLM Fp8MoEMethod",
    )
    quant_config = getattr(method, "quant_config", None)
    _require(quant_config is not None, "Fp8MoEMethod has no quant_config")
    _require(
        getattr(quant_config, "is_checkpoint_fp8_serialized", False) is True,
        "CDNA2 INT8 MoE requires a serialized FP8 checkpoint",
    )
    _require(
        getattr(quant_config, "activation_scheme", None) == "dynamic",
        "CDNA2 INT8 MoE requires dynamic activation quantization",
    )
    _require(
        tuple(getattr(quant_config, "weight_block_size", ())) == SOURCE_BLOCK_SHAPE,
        "CDNA2 INT8 MoE requires checkpoint weight_block_size=[128, 128]",
    )
    _require(
        getattr(method, "block_quant", False) is True,
        "CDNA2 INT8 MoE requires block-quantized Fp8MoEMethod",
    )
    _require(
        getattr(method, "weight_scale_name", None) == "weight_scale_inv",
        "CDNA2 INT8 MoE requires weight_scale_inv checkpoint scales",
    )


class Cdna2Int8Fp8MoEMethod(FusedMoEMethodBase):
    """Own the one-way OCP-FP8-to-INT8 expert representation change."""

    uses_meta_device = False

    def __init__(
        self,
        source_method: object,
        *,
        reload_mode: ReloadMode,
        prefix: str = "",
        audit: bool = False,
    ) -> None:
        _validate_reload_mode(reload_mode)
        _validate_source_method(source_method)
        super().__init__(source_method.moe)
        self.source_method = source_method
        self.quant_config = source_method.quant_config
        self.reload_mode = reload_mode
        self.prefix = prefix
        self.audit = audit
        self.weight_scale_name = "weight_scale_inv"

    def _validate_moe_contract(self) -> None:
        moe = self.moe
        parallel = moe.moe_parallel_config
        activation = getattr(moe.activation, "value", moe.activation)
        _require(moe.num_experts == GLM52_GLOBAL_EXPERTS, "expected 256 global experts")
        _require(parallel.use_ep is True, "expert parallelism is required")
        layout = (
            parallel.dp_size,
            parallel.ep_size,
            moe.num_local_experts,
        )
        _require(
            layout in GLM52_SUPPORTED_EP_LAYOUTS,
            "the measured path requires exactly DP2/EP16/local16 or "
            f"DP4/EP32/local8; got DP{layout[0]}/EP{layout[1]}/local{layout[2]}",
        )
        _require(moe.experts_per_token == GLM52_TOP_K, "the measured path requires top_k=8")
        _require(moe.hidden_dim == GLM52_HIDDEN_SIZE, "expected hidden_size=6144")
        _require(
            moe.intermediate_size == GLM52_INTERMEDIATE_SIZE,
            "expected MoE intermediate_size=2048",
        )
        _require(activation == "silu", "the measured path requires gated SiLU")
        _require(moe.in_dtype == torch.bfloat16, "the measured path requires BF16")
        _require(moe.has_bias is False, "biased experts are not supported")
        _require(
            moe.swiglu_limit is None and moe.swiglu_alpha is None and moe.swiglu_beta is None,
            "non-default SwiGLU limits or coefficients are not supported",
        )
        _require(
            moe.max_num_tokens <= PREFILL_MAX_TOKENS,
            f"scheduler max_num_tokens must be <= {PREFILL_MAX_TOKENS}",
        )
        _require(
            parallel.tp_size == GLM52_MOE_TP_SIZE,
            "the measured EP path requires MoE-local TP1",
        )
        _require(
            parallel.pcp_size == GLM52_PCP_SIZE,
            "the measured EP path requires PCP1",
        )
        _require(
            parallel.sp_size == GLM52_SP_SIZE,
            "the measured TP8 launch requires SP8 in the MoE parallel config",
        )
        _require(
            moe.max_num_tokens * parallel.dp_size <= GLOBAL_EP_MAX_TOKENS,
            f"global EP dispatch must be <= {GLOBAL_EP_MAX_TOKENS} tokens",
        )
        _require(
            parallel.all2all_backend == "allgather_reducescatter",
            "the measured path requires legacy allgather_reducescatter dispatch",
        )
        _require(parallel.enable_eplb is False, "EPLB is not validated")

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        self._validate_moe_contract()
        _require(
            num_experts == self.moe.num_local_experts,
            f"expected {self.moe.num_local_experts} local expert tensors",
        )
        _require(hidden_size == GLM52_HIDDEN_SIZE, "expected hidden_size=6144")
        _require(
            intermediate_size_per_partition == GLM52_INTERMEDIATE_SIZE,
            "expected local intermediate_size=2048",
        )
        self.source_method.create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )

    def _validate_source_layer(self, layer: torch.nn.Module) -> None:
        local_experts = self.moe.num_local_experts
        for name in ("w13_weight", "w2_weight", "w13_weight_scale_inv", "w2_weight_scale_inv"):
            _require(hasattr(layer, name), f"MoE layer is missing {name}")
        _require(
            layer.w13_weight.dtype == torch.float8_e4m3fn and layer.w2_weight.dtype == torch.float8_e4m3fn,
            "MoE checkpoint weights must use OCP torch.float8_e4m3fn",
        )
        _require(
            tuple(layer.w13_weight.shape)
            == (
                local_experts,
                2 * GLM52_INTERMEDIATE_SIZE,
                GLM52_HIDDEN_SIZE,
            ),
            "unexpected w13 checkpoint shape",
        )
        _require(
            tuple(layer.w2_weight.shape)
            == (
                local_experts,
                GLM52_HIDDEN_SIZE,
                GLM52_INTERMEDIATE_SIZE,
            ),
            "unexpected w2 checkpoint shape",
        )
        _require(
            tuple(getattr(layer, "weight_block_size", ())) == SOURCE_BLOCK_SHAPE,
            "MoE checkpoint block shape must be [128, 128]",
        )

    def _validate_runtime_layer(self, layer: torch.nn.Module) -> None:
        local_experts = self.moe.num_local_experts
        _require(
            getattr(layer, "_prime_cdna2_int8_moe_ready", False) is True,
            "CDNA2 INT8 MoE weights have not been processed",
        )
        _require(
            layer.w13_weight.dtype == torch.int8 and layer.w2_weight.dtype == torch.int8,
            "CDNA2 runtime expert weights must use INT8",
        )
        _require(
            layer.w13_weight_scale_inv.dtype == RUNTIME_WEIGHT_SCALE_DTYPE
            and layer.w2_weight_scale_inv.dtype == RUNTIME_WEIGHT_SCALE_DTYPE,
            "CDNA2 runtime expert weight scales must use FP16",
        )
        _require(
            tuple(layer.w13_weight_scale_inv.shape)
            == (
                local_experts,
                2 * GLM52_INTERMEDIATE_SIZE,
                GLM52_HIDDEN_SIZE // TARGET_GROUP_K,
            ),
            "unexpected w13 runtime scale shape",
        )
        _require(
            tuple(layer.w2_weight_scale_inv.shape)
            == (
                local_experts,
                GLM52_HIDDEN_SIZE,
                GLM52_INTERMEDIATE_SIZE // TARGET_GROUP_K,
            ),
            "unexpected w2 runtime scale shape",
        )

    @torch.no_grad()
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        already_processed = getattr(layer, "_prime_cdna2_int8_moe_ready", False)
        if already_processed and self.reload_mode == "none":
            raise RuntimeError("CDNA2 INT8 MoE is static-only; live reload was not enabled")
        if already_processed and layer.w13_weight.dtype != torch.float8_e4m3fn:
            raise RuntimeError("checkpoint reload must restore OCP E4M3FN experts before processing")
        # Do not invoke the source FP8 post-load hook: it normalizes/shuffles
        # experts for an FP8 backend. On validated gfx90a the loader tensors are
        # OCP E4M3 in standard contiguous EP-sharded layout, which this method
        # consumes directly before installing its distinct INT8 representation.
        self._validate_source_layer(layer)
        source_w13 = layer.w13_weight
        source_w13_scale = layer.w13_weight_scale_inv
        w13, w13_scale = transcode_ocp_fp8_expert_weight(
            source_w13,
            source_w13_scale,
            device_is_validated=True,
        )
        source_w2 = layer.w2_weight
        source_w2_scale = layer.w2_weight_scale_inv
        w2, w2_scale = transcode_ocp_fp8_expert_weight(
            source_w2,
            source_w2_scale,
            device_is_validated=True,
        )
        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w2_weight", w2)
        replace_parameter(layer, "w13_weight_scale_inv", w13_scale)
        replace_parameter(layer, "w2_weight_scale_inv", w2_scale)
        layer.weight_block_size = list(TARGET_BLOCK_SHAPE)
        layer.w13_input_scale = None
        layer.w2_input_scale = None
        layer._prime_cdna2_int8_moe_ready = True
        layer._prime_cdna2_int8_moe_reload_mode = self.reload_mode
        self._validate_runtime_layer(layer)
        log_transcode(
            enabled=self.audit,
            kind="moe_w13",
            prefix=self.prefix,
            source_weight=source_w13,
            runtime_weight=layer.w13_weight,
            source_scale=source_w13_scale,
            runtime_scale=layer.w13_weight_scale_inv,
        )
        log_transcode(
            enabled=self.audit,
            kind="moe_w2",
            prefix=self.prefix,
            source_weight=source_w2,
            runtime_weight=layer.w2_weight,
            source_scale=source_w2_scale,
            runtime_scale=layer.w2_weight_scale_inv,
        )

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> None:
        return None

    def maybe_make_prepare_finalize(self, routing_tables=None) -> None:
        return None

    @property
    def supports_eplb(self) -> bool:
        return False

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        self._validate_runtime_layer(layer)
        # GLM-5.2 has one shared expert. With supports_internal_mk=False,
        # vLLM's MoERunner executes it outside this routed-expert method and
        # adds its output separately.
        _require(
            (shared_experts is None) == (shared_experts_input is None),
            "shared_experts and shared_experts_input must be provided together",
        )
        _require(layer.expert_map is not None, "the EP path requires an expert_map")
        _require(
            layer.apply_router_weight_on_input is False,
            "apply_router_weight_on_input is not supported",
        )
        return cdna2_int8_moe_opaque(
            x,
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale_inv,
            layer.w2_weight_scale_inv,
            topk_weights,
            topk_ids,
            layer.expert_map,
            layer.global_num_experts,
        )
