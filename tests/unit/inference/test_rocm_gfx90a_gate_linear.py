import functools
import inspect
from types import SimpleNamespace

import pytest
import torch

from prime_rl.inference.patches import (
    _align_deepseek_router_logits_dtype,
    _require_exact_vllm_fp32_router_patch_abi,
    monkey_patch_fp32_router_logits,
)
from prime_rl.inference.vllm.kernels import rocm_gfx90a_gate_linear
from prime_rl.inference.vllm.kernels.rocm_gfx90a_gate_linear import (
    GFX90A_GATE_LINEAR_CONFIGS,
    GFX90A_GATE_LINEAR_TOKEN_BUCKETS,
    Gfx90aGateLinearConfig,
    _custom_op_fake,
    _require_exact_vllm_gate_abi,
    _validate_gate_linear_inputs,
    glm52_gate_config_mismatches,
    is_glm52_gate_config,
    select_gfx90a_gate_linear_config,
    validate_gate_linear_config,
    validate_glm52_gate_scheduler_config,
)


def make_glm52_config(**overrides):
    fields = {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "model_type": "glm_moe_dsa",
        "hidden_size": 6144,
        "n_routed_experts": 256,
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
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_glm52_model_contract_is_exact_and_reports_all_mismatches():
    assert is_glm52_gate_config(make_glm52_config())
    mismatches = glm52_gate_config_mismatches(
        make_glm52_config(
            architectures=["DeepseekV3ForCausalLM"],
            hidden_size=7168,
            n_group=8,
        )
    )
    assert len(mismatches) == 3
    assert any("architectures=" in mismatch for mismatch in mismatches)
    assert any("hidden_size=7168" in mismatch for mismatch in mismatches)
    assert any("n_group=8" in mismatch for mismatch in mismatches)


def test_gate_config_dispatch_covers_only_bounded_token_buckets():
    assert tuple(GFX90A_GATE_LINEAR_CONFIGS) == GFX90A_GATE_LINEAR_TOKEN_BUCKETS
    assert select_gfx90a_gate_linear_config(1).split_k == 16
    assert select_gfx90a_gate_linear_config(64).split_k == 12
    assert select_gfx90a_gate_linear_config(256).split_k == 12
    assert select_gfx90a_gate_linear_config(2048).split_k == 2
    assert select_gfx90a_gate_linear_config(192) is GFX90A_GATE_LINEAR_CONFIGS[256]
    assert select_gfx90a_gate_linear_config(32).cache_modifier == ".cg"
    assert select_gfx90a_gate_linear_config(32).reduce_block_m == 16
    assert select_gfx90a_gate_linear_config(32).reduce_num_warps == 8
    assert select_gfx90a_gate_linear_config(64).waves_per_eu == 2
    assert select_gfx90a_gate_linear_config(128).block_k == 128
    assert select_gfx90a_gate_linear_config(128).block_n == 32
    assert select_gfx90a_gate_linear_config(128).reduce_num_warps == 4
    assert select_gfx90a_gate_linear_config(256).block_m == 32
    assert select_gfx90a_gate_linear_config(256).block_n == 64
    assert select_gfx90a_gate_linear_config(1024).block_n == 128
    assert select_gfx90a_gate_linear_config(1024).kpack == 2


@pytest.mark.parametrize(
    ("m", "message"),
    (
        (0, "positive integer"),
        (-1, "positive integer"),
        (2049, "gated through M=2048"),
    ),
)
def test_gate_config_dispatch_fails_closed(m, message):
    with pytest.raises(ValueError, match=message):
        select_gfx90a_gate_linear_config(m)


@pytest.mark.parametrize("max_num_batched_tokens", (1, 2048))
def test_gate_scheduler_contract_accepts_only_tuned_capacity(
    max_num_batched_tokens,
):
    assert (
        validate_glm52_gate_scheduler_config(
            SimpleNamespace(
                max_num_batched_tokens=max_num_batched_tokens,
            )
        )
        == max_num_batched_tokens
    )


@pytest.mark.parametrize(
    "max_num_batched_tokens",
    (None, True, 0, -1, 2049),
)
def test_gate_scheduler_contract_rejects_late_runtime_failure(
    max_num_batched_tokens,
):
    with pytest.raises(ValueError, match="max_num_batched_tokens"):
        validate_glm52_gate_scheduler_config(
            SimpleNamespace(
                max_num_batched_tokens=max_num_batched_tokens,
            )
        )


@pytest.mark.parametrize(
    ("config", "message"),
    (
        (Gfx90aGateLinearConfig(8, 64, 32, 8, 4), "block_m"),
        (
            Gfx90aGateLinearConfig(16, 48, 32, 8, 4),
            "block_n must be a power of two",
        ),
        (Gfx90aGateLinearConfig(16, 64, 32, 5, 4), "split_k must divide"),
        (Gfx90aGateLinearConfig(16, 64, 256, 16, 4), "K extent"),
        (Gfx90aGateLinearConfig(16, 64, 32, 8, 2), "num_warps"),
        (Gfx90aGateLinearConfig(16, 64, 32, 8, 4, waves_per_eu=5), "waves_per_eu"),
        (Gfx90aGateLinearConfig(16, 64, 32, 8, 4, group_m=3), "group_m"),
        (
            Gfx90aGateLinearConfig(
                16,
                64,
                32,
                8,
                4,
                cache_modifier=".ca",
            ),
            "cache_modifier",
        ),
        (
            Gfx90aGateLinearConfig(
                16,
                64,
                32,
                8,
                4,
                reduce_block_m=32,
            ),
            "reduction overrides",
        ),
        (
            Gfx90aGateLinearConfig(48, 64, 32, 8, 4),
            "block_m must be a power of two",
        ),
        (
            Gfx90aGateLinearConfig(16, 64, 48, 8, 4),
            "block_k must be a power of two",
        ),
        (
            Gfx90aGateLinearConfig(
                16,
                64,
                32,
                8,
                4,
                reduce_block_m=48,
                reduce_block_n=64,
                reduce_num_warps=4,
            ),
            "reduce_block_m must be a power of two",
        ),
        (
            Gfx90aGateLinearConfig(
                16,
                64,
                32,
                8,
                4,
                reduce_block_m=32,
                reduce_block_n=48,
                reduce_num_warps=4,
            ),
            "reduce_block_n must be a power of two",
        ),
        (
            Gfx90aGateLinearConfig(16, 64, 32, 8, 4, matrix_instr_nonkdim=32),
            "matrix_instr_nonkdim",
        ),
        (Gfx90aGateLinearConfig(16, 64, 32, 8, 4, kpack=4), "kpack"),
    ),
)
def test_gate_launch_config_rejects_invalid_mfma_shapes(config, message):
    with pytest.raises(ValueError, match=message):
        validate_gate_linear_config(config)


def test_gate_runtime_contract_rejects_cpu_before_runtime_import():
    hidden_states = torch.zeros((2, 6144), dtype=torch.bfloat16)
    weight = torch.zeros((256, 6144), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="ROCm GPU"):
        _validate_gate_linear_inputs(
            hidden_states,
            weight,
            select_gfx90a_gate_linear_config(2),
        )


@pytest.mark.parametrize(
    ("hidden_states", "weight", "message"),
    (
        (
            torch.zeros((2, 6143), dtype=torch.bfloat16),
            torch.zeros((256, 6144), dtype=torch.bfloat16),
            "K must be 6144",
        ),
        (
            torch.zeros((2, 6144), dtype=torch.float32),
            torch.zeros((256, 6144), dtype=torch.bfloat16),
            "hidden_states must use torch.bfloat16",
        ),
        (
            torch.zeros((2, 6144), dtype=torch.bfloat16),
            torch.zeros((255, 6144), dtype=torch.bfloat16),
            "weight must have shape",
        ),
        (
            torch.zeros((2, 6144), dtype=torch.bfloat16),
            torch.zeros((256, 6144), dtype=torch.float32),
            "weight must use torch.bfloat16",
        ),
    ),
)
def test_gate_runtime_contract_rejects_wrong_shape_or_dtype(hidden_states, weight, message):
    with pytest.raises(ValueError, match=message):
        _validate_gate_linear_inputs(
            hidden_states,
            weight,
            select_gfx90a_gate_linear_config(2),
        )


def test_gate_custom_op_fake_preserves_shape_and_emits_fp32():
    hidden_states = torch.empty((17, 6144), dtype=torch.bfloat16)
    weight = torch.empty((256, 6144), dtype=torch.bfloat16)

    output = _custom_op_fake(hidden_states, weight)

    assert output.shape == (17, 256)
    assert output.dtype == torch.float32


def test_gate_vllm_abi_guard_accepts_only_exact_forward(monkeypatch):
    monkeypatch.setattr(
        rocm_gfx90a_gate_linear,
        "version",
        lambda distribution: rocm_gfx90a_gate_linear._EXPECTED_VLLM_DISTRIBUTION,
    )

    class ExactGate:
        def forward(self, x):
            return x

    class ChangedGate:
        def forward(self, x, bias=None):
            return x, bias

    _require_exact_vllm_gate_abi(ExactGate)
    with pytest.raises(RuntimeError, match="GateLinear.forward"):
        _require_exact_vllm_gate_abi(ChangedGate)


def test_gate_vllm_abi_guard_rejects_distribution_drift(monkeypatch):
    monkeypatch.setattr(
        rocm_gfx90a_gate_linear,
        "version",
        lambda distribution: "0.25.0",
    )

    class ExactGate:
        def forward(self, x):
            return x

    with pytest.raises(RuntimeError, match="requires vLLM"):
        _require_exact_vllm_gate_abi(ExactGate)


def test_fp32_router_dtype_is_set_before_fused_moe_selection():
    class Gate:
        out_dtype = None

        def set_out_dtype(self, dtype):
            self.out_dtype = dtype

    gate = Gate()
    aligned = _align_deepseek_router_logits_dtype(gate, None)

    assert gate.out_dtype == torch.float32
    assert aligned == torch.float32


def test_fp32_router_dtype_alignment_preserves_aiter_and_rejects_disagreement():
    class Gate:
        out_dtype = torch.bfloat16

    assert (
        _align_deepseek_router_logits_dtype(Gate(), torch.bfloat16)
        == torch.bfloat16
    )

    Gate.out_dtype = torch.float32
    with pytest.raises(RuntimeError, match="disagrees"):
        _align_deepseek_router_logits_dtype(Gate(), torch.bfloat16)


def test_fp32_router_dtype_rejection_does_not_mutate_unset_gate():
    class Gate:
        out_dtype = None

        def set_out_dtype(self, dtype):
            self.out_dtype = dtype

    gate = Gate()
    with pytest.raises(RuntimeError, match="disagrees"):
        _align_deepseek_router_logits_dtype(gate, torch.bfloat16)
    assert gate.out_dtype is None


def test_fp32_router_backport_accepts_pinned_constructor_abi():
    from vllm.model_executor.models import deepseek_v2

    _require_exact_vllm_fp32_router_patch_abi(
        deepseek_v2.FusedMoE,
        deepseek_v2.DeepseekV2MoE.__init__,
    )


def test_fp32_router_backport_is_a_true_noop_without_config_context(
    monkeypatch,
):
    from vllm.model_executor.models import deepseek_v2

    monkeypatch.delenv("PRIME_ROCM_GFX90A_GATE_LINEAR", raising=False)
    monkeypatch.delenv("PRIME_ROCM_GFX90A_ONE_GROUP_TOPK", raising=False)
    calls = []

    def fake_fused_moe(*args, **kwargs):
        calls.append((args, kwargs))
        return "stock-result"

    original_init = deepseek_v2.DeepseekV2MoE.__init__
    monkeypatch.setattr(deepseek_v2, "FusedMoE", fake_fused_moe)
    # Record this class attribute with pytest so the direct production
    # monkey-patch is restored after the test.
    monkeypatch.setattr(deepseek_v2.DeepseekV2MoE, "__init__", original_init)

    monkey_patch_fp32_router_logits()

    gate = object()
    assert deepseek_v2.FusedMoE("unchanged", gate=gate) == "stock-result"
    assert calls == [(("unchanged",), {"gate": gate})]


def test_fp32_router_backport_rewrites_keyword_and_positional_gate_arguments(
    monkeypatch,
):
    import vllm.config as vllm_config
    from vllm.model_executor.models import deepseek_v2

    monkeypatch.delenv("PRIME_ROCM_GFX90A_GATE_LINEAR", raising=False)
    monkeypatch.delenv("PRIME_ROCM_GFX90A_ONE_GROUP_TOPK", raising=False)
    original_factory = deepseek_v2.FusedMoE
    original_init = deepseek_v2.DeepseekV2MoE.__init__
    signature = inspect.signature(original_factory)
    calls = []

    @functools.wraps(original_factory)
    def fake_fused_moe(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        calls.append(
            (
                bound.arguments["gate"].out_dtype,
                bound.arguments["router_logits_dtype"],
            )
        )
        return object()

    current_config = SimpleNamespace(
        additional_config={"fp32_router_logits": True},
    )
    monkeypatch.setattr(
        vllm_config,
        "get_current_vllm_config_or_none",
        lambda: current_config,
    )
    monkeypatch.setattr(deepseek_v2, "FusedMoE", fake_fused_moe)
    # Record this class attribute so pytest restores the direct production
    # monkey-patch after the test.
    monkeypatch.setattr(deepseek_v2.DeepseekV2MoE, "__init__", original_init)

    monkey_patch_fp32_router_logits()
    patched_factory = deepseek_v2.FusedMoE

    class Gate:
        out_dtype = None

        def set_out_dtype(self, dtype):
            self.out_dtype = dtype

    keyword_gate = Gate()
    patched_factory(
        num_experts=256,
        top_k=8,
        hidden_size=6144,
        intermediate_size=2048,
        router_logits_dtype=None,
        gate=keyword_gate,
    )

    positional_gate = Gate()
    positional_args = []
    required_values = {
        "num_experts": 256,
        "top_k": 8,
        "hidden_size": 6144,
        "intermediate_size": 2048,
    }
    for parameter in signature.parameters.values():
        if parameter.name == "gate":
            positional_args.append(positional_gate)
            break
        if parameter.default is inspect.Parameter.empty:
            positional_args.append(required_values[parameter.name])
        elif parameter.name == "router_logits_dtype":
            positional_args.append(None)
        else:
            positional_args.append(parameter.default)
    patched_factory(*positional_args)

    assert keyword_gate.out_dtype == torch.float32
    assert positional_gate.out_dtype == torch.float32
    assert calls == [
        (torch.float32, torch.float32),
        (torch.float32, torch.float32),
    ]


def test_gate_opt_in_rejects_oversized_scheduler_before_fused_moe_construction(
    monkeypatch,
):
    import vllm.config as vllm_config
    from vllm.model_executor.models import deepseek_v2

    monkeypatch.setenv("PRIME_ROCM_GFX90A_GATE_LINEAR", "1")
    monkeypatch.delenv("PRIME_ROCM_GFX90A_ONE_GROUP_TOPK", raising=False)
    original_factory = deepseek_v2.FusedMoE
    original_init = deepseek_v2.DeepseekV2MoE.__init__
    calls = []

    @functools.wraps(original_factory)
    def fake_fused_moe(*args, **kwargs):
        calls.append((args, kwargs))
        return object()

    monkeypatch.setattr(
        vllm_config,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(
            additional_config={"fp32_router_logits": True},
            scheduler_config=SimpleNamespace(
                max_num_batched_tokens=2049,
            ),
        ),
    )
    monkeypatch.setattr(deepseek_v2, "FusedMoE", fake_fused_moe)
    monkeypatch.setattr(
        deepseek_v2.DeepseekV2MoE,
        "__init__",
        original_init,
    )

    monkey_patch_fp32_router_logits()

    with pytest.raises(ValueError, match="max_num_batched_tokens"):
        deepseek_v2.FusedMoE(
            num_experts=256,
            top_k=8,
            hidden_size=6144,
            intermediate_size=2048,
            router_logits_dtype=None,
            gate=object(),
        )
    assert calls == []
