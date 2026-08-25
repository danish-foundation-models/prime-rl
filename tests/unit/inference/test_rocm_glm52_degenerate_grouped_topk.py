from types import SimpleNamespace

import pytest
import torch

from prime_rl.inference.vllm.kernels import rocm_glm52_degenerate_grouped_topk
from prime_rl.inference.vllm.kernels.rocm_glm52_degenerate_grouped_topk import (
    GLM52_ROUTER_GLOBAL_MAX_TOKENS,
    GLM52_ROUTER_LOCAL_MAX_TOKENS,
    _require_exact_vllm_router_abi,
    _validate_batch_invariant_contract,
    _validate_router_token_count,
    _validate_runtime_inputs,
    degenerate_grouped_topk_contract_mismatches,
    is_degenerate_glm52_grouped_topk,
)


def make_router(**overrides):
    fields = {
        "top_k": 8,
        "global_num_experts": 256,
        "num_expert_group": 1,
        "topk_group": 1,
        "renormalize": True,
        "scoring_func": "sigmoid",
        "routed_scaling_factor": 1.0,
        "num_fused_shared_experts": 0,
        "eplb_state": None,
        "e_score_correction_bias": torch.linspace(-0.1, 0.1, 256),
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def make_moe_parallel_config(**overrides):
    fields = {
        "tp_size": 1,
        "dp_size": 2,
        "pcp_size": 1,
        "ep_size": 16,
        "sp_size": 8,
        "use_ep": True,
        "all2all_backend": "allgather_reducescatter",
        "enable_eplb": False,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def make_moe_config(*, parallel=None, **overrides):
    fields = {
        "max_num_tokens": GLM52_ROUTER_LOCAL_MAX_TOKENS,
        "moe_parallel_config": parallel or make_moe_parallel_config(),
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def make_runner(router, **overrides):
    fields = {
        "router": router,
        "routed_scaling_factor": 2.5,
        "moe_config": make_moe_config(),
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_degenerate_grouped_topk_contract_accepts_exact_glm52_router():
    router = make_router()
    runner = make_runner(router)

    assert is_degenerate_glm52_grouped_topk(router, runner)
    assert degenerate_grouped_topk_contract_mismatches(router, runner) == ()


def test_degenerate_grouped_topk_contract_derives_dp2_global_token_bound():
    assert GLM52_ROUTER_LOCAL_MAX_TOKENS == 2048
    assert GLM52_ROUTER_GLOBAL_MAX_TOKENS == 4096
    _validate_router_token_count(GLM52_ROUTER_GLOBAL_MAX_TOKENS)


def test_degenerate_grouped_topk_rejects_vllm_batch_invariant_mode():
    _validate_batch_invariant_contract(False)
    with pytest.raises(ValueError, match="VLLM_BATCH_INVARIANT=1"):
        _validate_batch_invariant_contract(True)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("tp_size", 8, "tp_size=8"),
        ("dp_size", 1, "dp_size=1"),
        ("dp_size", 4, "dp_size=4"),
        ("pcp_size", 2, "pcp_size=2"),
        ("ep_size", 8, "ep_size=8"),
        ("sp_size", 1, "sp_size=1"),
        ("use_ep", False, "use_ep=False"),
        ("all2all_backend", "deepep_high_throughput", "all2all_backend="),
        ("enable_eplb", True, "enable_eplb=True"),
    ),
)
def test_degenerate_grouped_topk_contract_rejects_unmeasured_parallel_topology(
    field,
    value,
    message,
):
    router = make_router()
    parallel = make_moe_parallel_config(**{field: value})
    runner = make_runner(router, moe_config=make_moe_config(parallel=parallel))

    mismatches = degenerate_grouped_topk_contract_mismatches(router, runner)

    assert any(message in mismatch for mismatch in mismatches)


def test_degenerate_grouped_topk_contract_rejects_unmeasured_local_token_bound():
    router = make_router()
    runner = make_runner(
        router,
        moe_config=make_moe_config(max_num_tokens=2049),
    )

    mismatches = degenerate_grouped_topk_contract_mismatches(router, runner)

    assert any("max_num_tokens=2049" in mismatch for mismatch in mismatches)


def test_degenerate_grouped_topk_contract_reports_semantic_changes():
    router = make_router(
        num_expert_group=8,
        topk_group=4,
        renormalize=False,
        scoring_func="softmax",
        routed_scaling_factor=2.5,
        eplb_state=object(),
        e_score_correction_bias=torch.zeros(128, dtype=torch.bfloat16),
    )
    runner = make_runner(router, routed_scaling_factor=1.0)

    mismatches = degenerate_grouped_topk_contract_mismatches(router, runner)

    assert len(mismatches) == 9
    assert any("router.num_expert_group=8" in mismatch for mismatch in mismatches)
    assert any("router.topk_group=4" in mismatch for mismatch in mismatches)
    assert any("router.renormalize=False" in mismatch for mismatch in mismatches)
    assert any("router.scoring_func='softmax'" in mismatch for mismatch in mismatches)
    assert any("router.routed_scaling_factor=2.5" in mismatch for mismatch in mismatches)
    assert any("router.eplb_state=" in mismatch for mismatch in mismatches)
    assert any("runner.routed_scaling_factor=1.0" in mismatch for mismatch in mismatches)
    assert any("shape (128,)" in mismatch for mismatch in mismatches)
    assert any("dtype torch.bfloat16" in mismatch for mismatch in mismatches)


def test_degenerate_grouped_topk_requires_contiguous_fp32_bias():
    bias = torch.zeros((2, 256), dtype=torch.float32).t()
    router = make_router(e_score_correction_bias=bias)
    runner = make_runner(router)

    mismatches = degenerate_grouped_topk_contract_mismatches(router, runner)

    assert any("shape (256, 2)" in mismatch for mismatch in mismatches)
    assert any("must be contiguous" in mismatch for mismatch in mismatches)


@pytest.mark.parametrize(
    ("hidden_states", "router_logits", "indices_type", "message"),
    (
        (
            torch.zeros((2, 6143), dtype=torch.bfloat16),
            torch.zeros((2, 256), dtype=torch.float32),
            None,
            "shape",
        ),
        (
            torch.zeros((2, 6144), dtype=torch.float32),
            torch.zeros((2, 256), dtype=torch.float32),
            None,
            "hidden_states must use torch.bfloat16",
        ),
        (
            torch.zeros((2, 6144), dtype=torch.bfloat16),
            torch.zeros((3, 256), dtype=torch.float32),
            None,
            "token counts",
        ),
        (
            torch.zeros((2, 6144), dtype=torch.bfloat16),
            torch.zeros((2, 255), dtype=torch.float32),
            None,
            "256 experts",
        ),
        (
            torch.zeros((2, 6144), dtype=torch.bfloat16),
            torch.zeros((2, 256), dtype=torch.bfloat16),
            None,
            "router_logits must use torch.float32",
        ),
        (
            torch.zeros((2, 6144), dtype=torch.bfloat16),
            torch.zeros((2, 256), dtype=torch.float32),
            torch.int64,
            "top-k indices",
        ),
    ),
)
def test_degenerate_grouped_topk_runtime_contract_fails_closed(
    hidden_states,
    router_logits,
    indices_type,
    message,
):
    with pytest.raises(ValueError, match=message):
        _validate_runtime_inputs(
            hidden_states,
            router_logits,
            indices_type,
        )


@pytest.mark.parametrize("num_tokens", (0, -1, 4097))
def test_degenerate_grouped_topk_runtime_token_bound_fails_closed(num_tokens):
    with pytest.raises(ValueError, match=r"M must be in \[1, 4096\]"):
        _validate_router_token_count(num_tokens)


def test_degenerate_grouped_topk_vllm_abi_is_exact(monkeypatch):
    monkeypatch.setattr(
        rocm_glm52_degenerate_grouped_topk,
        "version",
        lambda distribution: rocm_glm52_degenerate_grouped_topk._EXPECTED_VLLM_DISTRIBUTION,
    )

    class ExactRouter:
        def _compute_routing(
            self,
            hidden_states,
            router_logits,
            indices_type,
            *,
            input_ids=None,
        ):
            return hidden_states, router_logits, indices_type, input_ids

    def exact_fused(
        hidden_states,
        gating_output,
        scoring_func,
        e_score_correction_bias,
        topk,
        renormalize,
        indices_type=None,
        input_tokens=None,
        hash_indices_table=None,
        routed_scaling_factor=1.0,
    ):
        return hidden_states, gating_output

    _require_exact_vllm_router_abi(ExactRouter, exact_fused)

    def changed_fused(
        hidden_states,
        gating_output,
        scoring_func,
        e_score_correction_bias,
        topk,
        renormalize,
        indices_type=None,
        input_tokens=None,
        hash_indices_table=None,
        routed_scaling_factor=2.5,
    ):
        return hidden_states, gating_output

    with pytest.raises(RuntimeError, match="default changed"):
        _require_exact_vllm_router_abi(ExactRouter, changed_fused)


def test_degenerate_grouped_topk_vllm_abi_rejects_distribution_drift(
    monkeypatch,
):
    monkeypatch.setattr(
        rocm_glm52_degenerate_grouped_topk,
        "version",
        lambda distribution: "0.25.0",
    )

    class ExactRouter:
        def _compute_routing(
            self,
            hidden_states,
            router_logits,
            indices_type,
            *,
            input_ids=None,
        ):
            return hidden_states, router_logits, indices_type, input_ids

    with pytest.raises(RuntimeError, match="requires vLLM"):
        _require_exact_vllm_router_abi(ExactRouter, lambda: None)
