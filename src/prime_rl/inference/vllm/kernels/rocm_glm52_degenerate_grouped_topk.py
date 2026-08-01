# SPDX-License-Identifier: Apache-2.0

"""Fast path for GLM-5.2's degenerate one-group expert router.

GLM-5.2 configures one expert group and selects that one group. Group
selection is therefore an identity operation, but vLLM's generic grouped
router still launches group-score top-k, group top-k, scatter/mask, and final
expert top-k operations. This integration delegates the exact one-group case
to vLLM's existing biased sigmoid top-k kernel. vLLM's batch-invariant mode is
outside this measured contract because that mode changes the stock grouped
router's sorting semantics.
"""

from __future__ import annotations

import functools
import inspect
from importlib.metadata import version

import torch

from prime_rl.inference.vllm.kernels.rocm_gfx90a_gate_linear import (
    GLM52_GATE_HIDDEN_SIZE,
    GLM52_GATE_MAX_TOKENS,
    GLM52_GATE_NUM_EXPERTS,
    glm52_gate_config_mismatches,
    validate_gfx90a_device,
)

_EXPECTED_VLLM_DISTRIBUTION = "0.24.0+lumi_aif_gfx90a_ee0da84"
_ROUTER_MARKER = "_prime_rl_glm52_degenerate_grouped_topk"
_PATCH_MARKER = "_prime_rl_glm52_degenerate_grouped_topk_patch"

# vLLM folds launch TP8 x DP2 x PCP1 into EP16, resets MoE-local TP to
# one, and retains launch TP as SP8. Routing can run after DP/PCP dispatch.
GLM52_ROUTER_MOE_TP_SIZE = 1
GLM52_ROUTER_DP_SIZE = 2
GLM52_ROUTER_PCP_SIZE = 1
GLM52_ROUTER_EP_SIZE = 16
GLM52_ROUTER_SP_SIZE = 8
GLM52_ROUTER_LOCAL_MAX_TOKENS = GLM52_GATE_MAX_TOKENS
GLM52_ROUTER_GLOBAL_MAX_TOKENS = GLM52_ROUTER_LOCAL_MAX_TOKENS * GLM52_ROUTER_DP_SIZE * GLM52_ROUTER_PCP_SIZE


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def degenerate_grouped_topk_contract_mismatches(
    router: object,
    runner: object,
) -> tuple[str, ...]:
    """Return deviations from GLM-5.2's measured routing contract."""

    expected_router_fields = {
        "top_k": 8,
        "global_num_experts": GLM52_GATE_NUM_EXPERTS,
        "num_expert_group": 1,
        "topk_group": 1,
        "renormalize": True,
        "scoring_func": "sigmoid",
        # GLM applies its 2.5 factor to the routed output in MoERunner.
        "routed_scaling_factor": 1.0,
        "num_fused_shared_experts": 0,
        "eplb_state": None,
    }
    mismatches = []
    for field, expected in expected_router_fields.items():
        actual = getattr(router, field, None)
        if actual != expected:
            mismatches.append(f"router.{field}={actual!r} (expected {expected!r})")
    runner_scale = getattr(runner, "routed_scaling_factor", None)
    if runner_scale != 2.5:
        mismatches.append(f"runner.routed_scaling_factor={runner_scale!r} (expected 2.5)")

    moe_config = getattr(runner, "moe_config", None)
    if moe_config is None:
        mismatches.append("runner.moe_config is required")
    else:
        local_max_tokens = getattr(moe_config, "max_num_tokens", None)
        if local_max_tokens != GLM52_ROUTER_LOCAL_MAX_TOKENS:
            mismatches.append(
                f"runner.moe_config.max_num_tokens={local_max_tokens!r} (expected {GLM52_ROUTER_LOCAL_MAX_TOKENS!r})"
            )

        parallel = getattr(moe_config, "moe_parallel_config", None)
        if parallel is None:
            mismatches.append("runner.moe_config.moe_parallel_config is required")
        else:
            expected_parallel_fields = {
                "tp_size": GLM52_ROUTER_MOE_TP_SIZE,
                "dp_size": GLM52_ROUTER_DP_SIZE,
                "pcp_size": GLM52_ROUTER_PCP_SIZE,
                "ep_size": GLM52_ROUTER_EP_SIZE,
                "sp_size": GLM52_ROUTER_SP_SIZE,
                "use_ep": True,
                "all2all_backend": "allgather_reducescatter",
                "enable_eplb": False,
            }
            for field, expected in expected_parallel_fields.items():
                actual = getattr(parallel, field, None)
                if actual != expected:
                    mismatches.append(
                        f"runner.moe_config.moe_parallel_config.{field}={actual!r} (expected {expected!r})"
                    )

    bias = getattr(router, "e_score_correction_bias", None)
    if not isinstance(bias, torch.Tensor):
        mismatches.append("router.e_score_correction_bias must be a torch.Tensor")
    else:
        if tuple(bias.shape) != (GLM52_GATE_NUM_EXPERTS,):
            mismatches.append(
                f"router.e_score_correction_bias has shape {tuple(bias.shape)!r} (expected ({GLM52_GATE_NUM_EXPERTS},))"
            )
        if bias.dtype != torch.float32:
            mismatches.append(f"router.e_score_correction_bias has dtype {bias.dtype} (expected torch.float32)")
        if not bias.is_contiguous():
            mismatches.append("router.e_score_correction_bias must be contiguous")
    return tuple(mismatches)


def is_degenerate_glm52_grouped_topk(router: object, runner: object) -> bool:
    """Return whether a runner exactly matches the optimized router contract."""

    return not degenerate_grouped_topk_contract_mismatches(router, runner)


def _require_exact_vllm_router_abi(router_type: type, fused_topk_bias: object) -> None:
    installed = version("vllm")
    if installed != _EXPECTED_VLLM_DISTRIBUTION:
        raise RuntimeError(
            f"GLM-5.2 degenerate grouped top-k requires vLLM {_EXPECTED_VLLM_DISTRIBUTION!r}; found {installed!r}"
        )

    method_parameters = tuple(inspect.signature(router_type._compute_routing).parameters.values())
    method_names = tuple(parameter.name for parameter in method_parameters)
    expected_method_names = (
        "self",
        "hidden_states",
        "router_logits",
        "indices_type",
        "input_ids",
    )
    if method_names != expected_method_names:
        raise RuntimeError(
            "GLM-5.2 degenerate grouped top-k requires "
            f"GroupedTopKRouter._compute_routing{expected_method_names}; found {method_names}"
        )
    if any(parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD for parameter in method_parameters[:-1]):
        raise RuntimeError("GroupedTopKRouter._compute_routing positional ABI changed")
    if method_parameters[-1].kind is not inspect.Parameter.KEYWORD_ONLY:
        raise RuntimeError("GroupedTopKRouter._compute_routing.input_ids must be keyword-only")
    if method_parameters[-1].default is not None:
        raise RuntimeError("GroupedTopKRouter._compute_routing.input_ids must default to None")

    fused_parameters = tuple(inspect.signature(fused_topk_bias).parameters.values())
    fused_names = tuple(parameter.name for parameter in fused_parameters)
    expected_fused_names = (
        "hidden_states",
        "gating_output",
        "scoring_func",
        "e_score_correction_bias",
        "topk",
        "renormalize",
        "indices_type",
        "input_tokens",
        "hash_indices_table",
        "routed_scaling_factor",
    )
    if fused_names != expected_fused_names:
        raise RuntimeError(f"fused_topk_bias ABI changed: expected {expected_fused_names}, found {fused_names}")
    if any(parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD for parameter in fused_parameters):
        raise RuntimeError("fused_topk_bias parameters must be positional-or-keyword")
    expected_defaults = {
        "indices_type": None,
        "input_tokens": None,
        "hash_indices_table": None,
        "routed_scaling_factor": 1.0,
    }
    for parameter in fused_parameters:
        expected_default = expected_defaults.get(
            parameter.name,
            inspect.Parameter.empty,
        )
        if parameter.default != expected_default:
            raise RuntimeError(
                "fused_topk_bias default changed: "
                f"{parameter.name}={parameter.default!r}, "
                f"expected {expected_default!r}"
            )


def _validate_router_token_count(num_tokens: int) -> None:
    _require(type(num_tokens) is int, "M must be an integer")
    _require(
        0 < num_tokens <= GLM52_ROUTER_GLOBAL_MAX_TOKENS,
        f"M must be in [1, {GLM52_ROUTER_GLOBAL_MAX_TOKENS}], got {num_tokens}",
    )


def _validate_batch_invariant_contract(enabled: bool) -> None:
    _require(
        not enabled,
        "VLLM_BATCH_INVARIANT=1 is outside the measured GLM-5.2 "
        "degenerate grouped top-k contract",
    )


def _validate_runtime_inputs(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    indices_type: torch.dtype | None,
) -> None:
    _require(
        hidden_states.ndim == 2 and hidden_states.shape[1] == GLM52_GATE_HIDDEN_SIZE,
        f"hidden_states must have shape [M, {GLM52_GATE_HIDDEN_SIZE}]",
    )
    num_tokens = hidden_states.shape[0]
    _validate_router_token_count(num_tokens)
    _require(
        hidden_states.dtype == torch.bfloat16,
        "hidden_states must use torch.bfloat16",
    )
    _require(
        hidden_states.is_contiguous(),
        "hidden_states must be contiguous",
    )
    _require(
        router_logits.ndim == 2,
        "router_logits must be two-dimensional",
    )
    _require(
        num_tokens == router_logits.shape[0],
        "hidden_states and router_logits token counts must match",
    )
    _require(
        router_logits.shape[1] == GLM52_GATE_NUM_EXPERTS,
        "router_logits must contain 256 experts",
    )
    _require(
        router_logits.dtype == torch.float32,
        "router_logits must use torch.float32",
    )
    _require(
        router_logits.is_contiguous(),
        "router_logits must be contiguous",
    )
    _require(
        router_logits.device == hidden_states.device,
        "router_logits and hidden_states must share a device",
    )
    _require(
        indices_type in (None, torch.int32),
        "GLM-5.2 top-k indices must use torch.int32",
    )
    _require(
        hidden_states.device.type == "cuda",
        "router inputs must be on a ROCm GPU",
    )
    _require(
        torch.version.hip is not None,
        "a ROCm PyTorch build is required",
    )


def install_glm52_degenerate_grouped_topk(runner: object, model_config: object) -> None:
    """Install the fused one-group branch on an exact GLM-5.2 MoE runner."""

    model_mismatches = glm52_gate_config_mismatches(model_config)
    if model_mismatches:
        raise ValueError("degenerate grouped top-k requires the exact GLM-5.2 config: " + "; ".join(model_mismatches))

    from vllm import envs
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
        fused_topk_bias,
    )
    from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
        GroupedTopKRouter,
    )

    _require_exact_vllm_router_abi(GroupedTopKRouter, fused_topk_bias)
    _validate_batch_invariant_contract(envs.VLLM_BATCH_INVARIANT)
    _require(not rocm_aiter_ops.is_fused_moe_enabled(), "AITER fused MoE is outside the gfx90a routing contract")
    router = getattr(runner, "router", None)
    _require(type(router) is GroupedTopKRouter, f"expected GroupedTopKRouter, got {type(router).__name__}")
    contract_mismatches = degenerate_grouped_topk_contract_mismatches(router, runner)
    if contract_mismatches:
        raise ValueError("unsupported GLM-5.2 router contract: " + "; ".join(contract_mismatches))

    bias = router.e_score_correction_bias
    device_index = bias.device.index
    _require(bias.device.type == "cuda", "correction bias must be on a ROCm GPU")
    validate_gfx90a_device(torch.cuda.current_device() if device_index is None else device_index)

    original_compute = GroupedTopKRouter._compute_routing
    if not getattr(original_compute, _PATCH_MARKER, False):

        @functools.wraps(original_compute)
        def _patched_compute(
            self,
            hidden_states,
            router_logits,
            indices_type,
            *,
            input_ids=None,
        ):
            if not getattr(self, _ROUTER_MARKER, False):
                return original_compute(
                    self,
                    hidden_states,
                    router_logits,
                    indices_type,
                    input_ids=input_ids,
                )
            _validate_runtime_inputs(hidden_states, router_logits, indices_type)
            return fused_topk_bias(
                hidden_states=hidden_states,
                gating_output=router_logits,
                scoring_func=self.scoring_func,
                e_score_correction_bias=self.e_score_correction_bias.data,
                topk=self.top_k,
                renormalize=self.renormalize,
                indices_type=indices_type,
                # The stock grouped path ignores input IDs. No hash routing is
                # configured for GLM-5.2, so forwarding them could only add a
                # redundant dtype conversion in fused_topk_bias.
                input_tokens=None,
                hash_indices_table=None,
                routed_scaling_factor=self.routed_scaling_factor,
            )

        setattr(_patched_compute, _PATCH_MARKER, True)
        GroupedTopKRouter._compute_routing = _patched_compute

    setattr(router, _ROUTER_MARKER, True)


__all__ = [
    "GLM52_ROUTER_DP_SIZE",
    "GLM52_ROUTER_EP_SIZE",
    "GLM52_ROUTER_GLOBAL_MAX_TOKENS",
    "GLM52_ROUTER_LOCAL_MAX_TOKENS",
    "GLM52_ROUTER_MOE_TP_SIZE",
    "GLM52_ROUTER_PCP_SIZE",
    "GLM52_ROUTER_SP_SIZE",
    "degenerate_grouped_topk_contract_mismatches",
    "install_glm52_degenerate_grouped_topk",
    "is_degenerate_glm52_grouped_topk",
]
