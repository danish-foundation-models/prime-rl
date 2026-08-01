# SPDX-License-Identifier: Apache-2.0

"""Opaque vLLM operator boundaries for the gfx90a CDNA2 INT8 runtime."""

from __future__ import annotations

import torch

from prime_rl.inference.vllm.kernels.rocm_cdna2_int8_linear import (
    cdna2_int8_linear,
    select_gfx90a_int8_linear_config,
)
from prime_rl.inference.vllm.kernels.rocm_cdna2_int8_moe import (
    cdna2_int8_fused_experts,
)

LINEAR_OP_NAME = "prime_rocm_gfx90a_cdna2_int8_linear"
MOE_OP_NAME = "prime_rocm_gfx90a_cdna2_int8_moe"
_REGISTERED = False


def _linear_impl(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    use_gfx90a_skinny_int8: bool,
) -> torch.Tensor:
    flattened_tokens = hidden_states.numel() // hidden_states.shape[-1]
    config = select_gfx90a_int8_linear_config(
        flattened_tokens,
        weight.shape[0],
        weight.shape[1],
    )
    return cdna2_int8_linear(
        hidden_states,
        weight,
        weight_scale,
        bias,
        config=config,
        device_is_validated=True,
        use_gfx90a_skinny_int8=use_gfx90a_skinny_int8,
    )


def _linear_fake(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    use_gfx90a_skinny_int8: bool,
) -> torch.Tensor:
    del weight_scale, bias, use_gfx90a_skinny_int8
    return hidden_states.new_empty(
        (*hidden_states.shape[:-1], weight.shape[0]),
    )


def _moe_impl(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor,
    global_num_experts: int,
) -> torch.Tensor:
    return cdna2_int8_fused_experts(
        hidden_states,
        w13_weight,
        w2_weight,
        w13_weight_scale,
        w2_weight_scale,
        topk_weights,
        topk_ids,
        expert_map,
        global_num_experts=global_num_experts,
        device_is_validated=True,
    )


def _moe_fake(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor,
    global_num_experts: int,
) -> torch.Tensor:
    del (
        w13_weight,
        w2_weight,
        w13_weight_scale,
        w2_weight_scale,
        topk_weights,
        topk_ids,
        expert_map,
        global_num_experts,
    )
    return torch.empty_like(hidden_states)


def register_cdna2_int8_custom_ops() -> None:
    """Register graph-opaque operators only for the opt-in ROCm runtime."""

    global _REGISTERED

    if _REGISTERED:
        return

    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        raise RuntimeError("CDNA2 INT8 custom operators require a ROCm vLLM platform")
    if hasattr(torch.ops.vllm, LINEAR_OP_NAME) or hasattr(
        torch.ops.vllm,
        MOE_OP_NAME,
    ):
        raise RuntimeError("CDNA2 INT8 custom operators were registered outside PRIME-RL")

    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name=LINEAR_OP_NAME,
        op_func=_linear_impl,
        fake_impl=_linear_fake,
        dispatch_key=current_platform.dispatch_key,
    )
    direct_register_custom_op(
        op_name=MOE_OP_NAME,
        op_func=_moe_impl,
        fake_impl=_moe_fake,
        dispatch_key=current_platform.dispatch_key,
    )
    _REGISTERED = True


def cdna2_int8_linear_opaque(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    use_gfx90a_skinny_int8: bool,
) -> torch.Tensor:
    """Call the graph-opaque gfx90a dense operator."""

    return torch.ops.vllm.prime_rocm_gfx90a_cdna2_int8_linear(
        hidden_states,
        weight,
        weight_scale,
        bias,
        use_gfx90a_skinny_int8,
    )


def cdna2_int8_moe_opaque(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor,
    global_num_experts: int,
) -> torch.Tensor:
    """Call the graph-opaque gfx90a routed-expert operator."""

    return torch.ops.vllm.prime_rocm_gfx90a_cdna2_int8_moe(
        hidden_states,
        w13_weight,
        w2_weight,
        w13_weight_scale,
        w2_weight_scale,
        topk_weights,
        topk_ids,
        expert_map,
        global_num_experts,
    )
