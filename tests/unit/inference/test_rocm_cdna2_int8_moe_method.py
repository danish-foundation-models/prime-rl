from dataclasses import dataclass

import pytest
import torch

from prime_rl.inference.vllm.kernels import rocm_cdna2_int8_moe_method as integration


@dataclass
class FakeParallelConfig:
    use_ep: bool = True
    ep_size: int = 16
    tp_size: int = integration.GLM52_MOE_TP_SIZE
    dp_size: int = integration.GLM52_DP_SIZE
    pcp_size: int = integration.GLM52_PCP_SIZE
    sp_size: int = integration.GLM52_SP_SIZE
    all2all_backend: str = "allgather_reducescatter"
    enable_eplb: bool = False


@dataclass
class FakeMoeConfig:
    num_experts: int = integration.GLM52_GLOBAL_EXPERTS
    num_local_experts: int = integration.GLM52_LOCAL_EXPERTS
    experts_per_token: int = integration.GLM52_TOP_K
    hidden_dim: int = integration.GLM52_HIDDEN_SIZE
    intermediate_size: int = integration.GLM52_INTERMEDIATE_SIZE
    activation: str = "silu"
    in_dtype: torch.dtype = torch.bfloat16
    has_bias: bool = False
    swiglu_limit: float | None = None
    swiglu_alpha: float | None = None
    swiglu_beta: float | None = None
    max_num_tokens: int = 2048
    moe_parallel_config: FakeParallelConfig | None = None

    def __post_init__(self):
        if self.moe_parallel_config is None:
            self.moe_parallel_config = FakeParallelConfig()


@dataclass
class FakeQuantConfig:
    is_checkpoint_fp8_serialized: bool = True
    activation_scheme: str = "dynamic"
    weight_block_size: list[int] | None = None

    def __post_init__(self):
        if self.weight_block_size is None:
            self.weight_block_size = [128, 128]


class FakeFp8MoEMethod:
    def __init__(self, moe=None):
        self.moe = moe or FakeMoeConfig()
        self.quant_config = FakeQuantConfig()
        self.block_quant = True
        self.weight_scale_name = "weight_scale_inv"
        self.create_calls = 0

    def create_weights(self, *args, **kwargs):
        self.create_calls += 1


@pytest.fixture(autouse=True)
def fake_fp8_method_type(monkeypatch):
    monkeypatch.setattr(
        integration,
        "_get_fp8_moe_method_type",
        lambda: FakeFp8MoEMethod,
    )


def test_moe_method_accepts_exact_glm_tp8_dp2_ep16_contract_before_weight_creation():
    source = FakeFp8MoEMethod()
    method = integration.Cdna2Int8Fp8MoEMethod(source, reload_mode="none")

    method.create_weights(
        torch.nn.Module(),
        integration.GLM52_LOCAL_EXPERTS,
        integration.GLM52_HIDDEN_SIZE,
        integration.GLM52_INTERMEDIATE_SIZE,
        torch.bfloat16,
    )
    assert source.create_calls == 1
    assert method.supports_internal_mk is False
    assert method.maybe_make_prepare_finalize() is None


def test_moe_method_accepts_exact_glm_tp8_dp4_ep32_contract():
    parallel = FakeParallelConfig(dp_size=4, ep_size=32)
    moe = FakeMoeConfig(
        num_local_experts=integration.GLM52_EP32_LOCAL_EXPERTS,
        max_num_tokens=1024,
        moe_parallel_config=parallel,
    )
    source = FakeFp8MoEMethod(moe)
    method = integration.Cdna2Int8Fp8MoEMethod(source, reload_mode="none")

    method.create_weights(
        torch.nn.Module(),
        integration.GLM52_EP32_LOCAL_EXPERTS,
        integration.GLM52_HIDDEN_SIZE,
        integration.GLM52_INTERMEDIATE_SIZE,
        torch.bfloat16,
    )

    assert source.create_calls == 1


def test_moe_method_accepts_exact_glm_tp8_dp8_ep64_contract():
    parallel = FakeParallelConfig(dp_size=8, ep_size=64)
    moe = FakeMoeConfig(
        num_local_experts=integration.GLM52_EP64_LOCAL_EXPERTS,
        max_num_tokens=1024,
        moe_parallel_config=parallel,
    )
    source = FakeFp8MoEMethod(moe)
    method = integration.Cdna2Int8Fp8MoEMethod(source, reload_mode="none")

    method.create_weights(
        torch.nn.Module(),
        integration.GLM52_EP64_LOCAL_EXPERTS,
        integration.GLM52_HIDDEN_SIZE,
        integration.GLM52_INTERMEDIATE_SIZE,
        torch.bfloat16,
    )

    assert source.create_calls == 1


@pytest.mark.parametrize("dp_size", [1, 3])
def test_moe_method_rejects_unmeasured_dp_topologies(dp_size):
    source = FakeFp8MoEMethod(FakeMoeConfig(moe_parallel_config=FakeParallelConfig(dp_size=dp_size)))
    method = integration.Cdna2Int8Fp8MoEMethod(source, reload_mode="none")

    with pytest.raises(ValueError, match="DP8/EP64/local4"):
        method.create_weights(
            torch.nn.Module(),
            integration.GLM52_LOCAL_EXPERTS,
            integration.GLM52_HIDDEN_SIZE,
            integration.GLM52_INTERMEDIATE_SIZE,
            torch.bfloat16,
        )
    assert source.create_calls == 0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("tp_size", 8, "MoE-local TP1"),
        ("pcp_size", 2, "PCP1"),
        ("sp_size", 1, "SP8"),
    ],
)
def test_moe_method_rejects_unmeasured_tp_pcp_and_sp_factors(
    field,
    value,
    message,
):
    parallel = FakeParallelConfig()
    setattr(parallel, field, value)
    source = FakeFp8MoEMethod(FakeMoeConfig(moe_parallel_config=parallel))
    method = integration.Cdna2Int8Fp8MoEMethod(source, reload_mode="none")

    with pytest.raises(ValueError, match=message):
        method.create_weights(
            torch.nn.Module(),
            integration.GLM52_LOCAL_EXPERTS,
            integration.GLM52_HIDDEN_SIZE,
            integration.GLM52_INTERMEDIATE_SIZE,
            torch.bfloat16,
        )
    assert source.create_calls == 0


def test_moe_method_rejects_unsupported_ep_topology():
    source = FakeFp8MoEMethod()
    method = integration.Cdna2Int8Fp8MoEMethod(source, reload_mode="none")

    source.moe.moe_parallel_config.ep_size = 8
    with pytest.raises(ValueError, match="DP8/EP64/local4"):
        method.create_weights(
            torch.nn.Module(),
            integration.GLM52_LOCAL_EXPERTS,
            integration.GLM52_HIDDEN_SIZE,
            integration.GLM52_INTERMEDIATE_SIZE,
            torch.bfloat16,
        )
    assert source.create_calls == 0


def test_moe_method_rejects_mismatched_dp4_ep32_local_expert_count():
    parallel = FakeParallelConfig(dp_size=4, ep_size=32)
    source = FakeFp8MoEMethod(
        FakeMoeConfig(
            num_local_experts=integration.GLM52_LOCAL_EXPERTS,
            max_num_tokens=1024,
            moe_parallel_config=parallel,
        )
    )
    method = integration.Cdna2Int8Fp8MoEMethod(source, reload_mode="none")

    with pytest.raises(ValueError, match="DP4/EP32/local16"):
        method.create_weights(
            torch.nn.Module(),
            integration.GLM52_LOCAL_EXPERTS,
            integration.GLM52_HIDDEN_SIZE,
            integration.GLM52_INTERMEDIATE_SIZE,
            torch.bfloat16,
        )

    assert source.create_calls == 0


def test_moe_process_replaces_fp8_parameters_without_retaining_fallback(monkeypatch):
    method = integration.Cdna2Int8Fp8MoEMethod(
        FakeFp8MoEMethod(),
        reload_mode="none",
        prefix="model.layers.4.mlp.experts",
        audit=True,
    )
    layer = torch.nn.Module()
    for name in ("w13_weight", "w2_weight"):
        layer.register_parameter(
            name,
            torch.nn.Parameter(
                torch.zeros((1,), dtype=torch.float8_e4m3fn),
                requires_grad=False,
            ),
        )
    for name in ("w13_weight_scale_inv", "w2_weight_scale_inv"):
        layer.register_parameter(
            name,
            torch.nn.Parameter(torch.ones((1,), dtype=torch.float32), requires_grad=False),
        )
    layer.weight_block_size = [128, 128]
    monkeypatch.setattr(method, "_validate_source_layer", lambda _layer: None)
    monkeypatch.setattr(method, "_validate_runtime_layer", lambda _layer: None)
    transformed = iter(
        (
            (torch.ones((1,), dtype=torch.int8), torch.ones((1,), dtype=torch.float16)),
            (torch.full((1,), 2, dtype=torch.int8), torch.ones((1,), dtype=torch.float16)),
        )
    )
    monkeypatch.setattr(
        integration,
        "transcode_ocp_fp8_expert_weight",
        lambda *_args, **_kwargs: next(transformed),
    )
    audit_rows = []

    def capture_audit(**kwargs):
        assert layer._prime_cdna2_int8_moe_ready is True
        audit_rows.append(kwargs)

    monkeypatch.setattr(integration, "log_transcode", capture_audit)

    method.process_weights_after_loading(layer)

    assert layer.w13_weight.dtype == torch.int8
    assert layer.w2_weight.dtype == torch.int8
    assert not any(parameter.dtype == torch.float8_e4m3fn for parameter in layer.parameters())
    assert layer.weight_block_size == [1, 32]
    assert layer._prime_cdna2_int8_moe_ready is True
    assert [row["kind"] for row in audit_rows] == ["moe_w13", "moe_w2"]
    assert all(row["prefix"] == "model.layers.4.mlp.experts" and row["enabled"] is True for row in audit_rows)


def test_moe_apply_uses_runtime_int8_weights_and_expert_map(monkeypatch):
    method = integration.Cdna2Int8Fp8MoEMethod(
        FakeFp8MoEMethod(),
        reload_mode="none",
    )
    monkeypatch.setattr(method, "_validate_runtime_layer", lambda _layer: None)
    seen = {}

    def fake_opaque_moe(*args):
        seen["args"] = args
        return torch.zeros((2, 3))

    monkeypatch.setattr(integration, "cdna2_int8_moe_opaque", fake_opaque_moe)
    layer = type(
        "Layer",
        (),
        {
            "w13_weight": object(),
            "w2_weight": object(),
            "w13_weight_scale_inv": object(),
            "w2_weight_scale_inv": object(),
            "expert_map": torch.arange(4, dtype=torch.int32),
            "global_num_experts": 4,
            "apply_router_weight_on_input": False,
        },
    )()
    x = torch.zeros((2, 3))
    topk_weights = torch.zeros((2, 1))
    topk_ids = torch.zeros((2, 1), dtype=torch.int32)

    output = method.apply(layer, x, topk_weights, topk_ids, None, None)

    assert output.shape == (2, 3)
    assert seen["args"][:3] == (x, layer.w13_weight, layer.w2_weight)
    assert seen["args"][-1] == 4


@pytest.mark.parametrize(
    ("shared_experts", "shared_experts_input", "message"),
    [
        (object(), None, "must be provided together"),
        (None, torch.zeros((2, 3)), "must be provided together"),
    ],
)
def test_moe_apply_rejects_mismatched_external_shared_expert_inputs(
    monkeypatch,
    shared_experts,
    shared_experts_input,
    message,
):
    method = integration.Cdna2Int8Fp8MoEMethod(
        FakeFp8MoEMethod(),
        reload_mode="none",
    )
    monkeypatch.setattr(method, "_validate_runtime_layer", lambda _layer: None)

    with pytest.raises(ValueError, match=message):
        method.apply(
            object(),
            torch.zeros((2, 3)),
            torch.zeros((2, 1)),
            torch.zeros((2, 1), dtype=torch.int32),
            shared_experts,
            shared_experts_input,
        )


def test_moe_apply_leaves_paired_shared_experts_to_vllm_runner(monkeypatch):
    method = integration.Cdna2Int8Fp8MoEMethod(
        FakeFp8MoEMethod(),
        reload_mode="none",
    )
    monkeypatch.setattr(method, "_validate_runtime_layer", lambda _layer: None)
    routed_output = torch.ones((2, 3))
    seen = {}

    def fake_opaque_moe(*args):
        seen["args"] = args
        return routed_output

    monkeypatch.setattr(integration, "cdna2_int8_moe_opaque", fake_opaque_moe)
    layer = type(
        "Layer",
        (),
        {
            "w13_weight": object(),
            "w2_weight": object(),
            "w13_weight_scale_inv": object(),
            "w2_weight_scale_inv": object(),
            "expert_map": torch.arange(4, dtype=torch.int32),
            "global_num_experts": 4,
            "apply_router_weight_on_input": False,
        },
    )()
    external_shared_experts = object()
    external_shared_experts_input = torch.zeros((2, 3))

    output = method.apply(
        layer,
        torch.zeros((2, 3)),
        torch.zeros((2, 1)),
        torch.zeros((2, 1), dtype=torch.int32),
        external_shared_experts,
        external_shared_experts_input,
    )

    assert output is routed_output
    assert all(arg is not external_shared_experts for arg in seen["args"])
    assert all(arg is not external_shared_experts_input for arg in seen["args"])
