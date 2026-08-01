from dataclasses import dataclass

import pytest
import torch

from prime_rl.inference.vllm.kernels import rocm_cdna2_int8_linear_method as integration
from prime_rl.inference.vllm.kernels.rocm_cdna2_int8_linear import (
    SOURCE_GROUP_K,
    SOURCE_GROUP_N,
    TARGET_GROUP_K,
    dequantize_block_weight,
)


@dataclass
class FakeFp8Config:
    is_checkpoint_fp8_serialized: bool = True
    activation_scheme: str = "dynamic"
    weight_block_size: list[int] | None = None

    def __post_init__(self):
        if self.weight_block_size is None:
            self.weight_block_size = [SOURCE_GROUP_N, SOURCE_GROUP_K]


class FakeFp8Method:
    def __init__(self, config: FakeFp8Config | None = None):
        self.quant_config = config or FakeFp8Config()
        self.block_quant = self.quant_config.weight_block_size is not None
        self.act_q_static = self.quant_config.activation_scheme == "static"
        self.is_scale_e8m0 = False
        self.create_weights_calls = 0

    def create_weights(self, *args, **kwargs):
        self.create_weights_calls += 1


class FakeLinearLayer(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, weight_scale: torch.Tensor):
        super().__init__()
        self.weight_block_size = [SOURCE_GROUP_N, SOURCE_GROUP_K]
        self.register_parameter(
            "weight",
            torch.nn.Parameter(weight, requires_grad=False),
        )
        self.register_parameter(
            "weight_scale_inv",
            torch.nn.Parameter(weight_scale, requires_grad=False),
        )


@pytest.fixture(autouse=True)
def fake_fp8_method_type(monkeypatch):
    monkeypatch.setattr(
        integration,
        "_get_fp8_linear_method_type",
        lambda: FakeFp8Method,
    )
    monkeypatch.setattr(
        integration,
        "select_gfx90a_int8_linear_config",
        lambda *_args: object(),
    )


def make_checkpoint_weight(
    out_features: int,
    in_features: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(53)
    padded_n = ((out_features + SOURCE_GROUP_N - 1) // SOURCE_GROUP_N) * SOURCE_GROUP_N
    padded_k = ((in_features + SOURCE_GROUP_K - 1) // SOURCE_GROUP_K) * SOURCE_GROUP_K
    raw = torch.zeros((padded_n, padded_k), dtype=torch.float32)
    raw[:out_features, :in_features] = torch.randn(
        (out_features, in_features),
        generator=generator,
    )
    blocks = raw.reshape(
        padded_n // SOURCE_GROUP_N,
        SOURCE_GROUP_N,
        padded_k // SOURCE_GROUP_K,
        SOURCE_GROUP_K,
    )
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = blocks.abs().amax(dim=(1, 3)).clamp_min(1.0e-12) / fp8_max
    weight = (
        (blocks / scale[:, None, :, None])
        .clamp(-fp8_max, fp8_max)
        .to(torch.float8_e4m3fn)
        .reshape(padded_n, padded_k)[:out_features, :in_features]
        .contiguous()
    )
    return weight, scale.contiguous()


def select(
    method: FakeFp8Method | None = None,
    *,
    reload_mode: integration.ReloadMode = "none",
):
    return integration.select_cdna2_int8_linear_method(
        method or FakeFp8Method(),
        environ={integration.ENABLE_ENV: "1"},
        gcn_arch_name="gfx90a:sramecc+:xnack-",
        reload_mode=reload_mode,
    )


def test_selector_is_a_strict_opt_in(monkeypatch):
    method = object()

    def unexpected_device_query():
        raise AssertionError("disabled selector queried the device")

    monkeypatch.setattr(integration, "_resolve_gfx90a_architecture", unexpected_device_query)

    assert (
        integration.select_cdna2_int8_linear_method(
            method,
            environ={},
        )
        is method
    )
    assert (
        integration.select_cdna2_int8_linear_method(
            method,
            environ={integration.ENABLE_ENV: "0"},
        )
        is method
    )


def test_architecture_validation_is_cached_per_device(monkeypatch):
    calls = []
    integration._resolve_gfx90a_architecture_for_device.cache_clear()
    monkeypatch.setattr(integration.torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(
        integration,
        "validate_gfx90a_device",
        lambda device: calls.append(device) or "gfx90a",
    )

    assert integration._resolve_gfx90a_architecture() == "gfx90a"
    assert integration._resolve_gfx90a_architecture() == "gfx90a"

    assert calls == [torch.device("cuda", 3)]
    integration._resolve_gfx90a_architecture_for_device.cache_clear()


@pytest.mark.parametrize("value", ("true", "yes", "", "2"))
def test_selector_rejects_ambiguous_environment_values(value):
    with pytest.raises(ValueError, match=integration.ENABLE_ENV):
        integration.select_cdna2_int8_linear_method(
            FakeFp8Method(),
            environ={integration.ENABLE_ENV: value},
            gcn_arch_name="gfx90a",
        )


def test_selector_wraps_only_the_measured_contract():
    wrapped = select()

    assert isinstance(wrapped, integration.Cdna2Int8Fp8LinearMethod)
    assert wrapped.gcn_arch_name == "gfx90a:sramecc+:xnack-"
    assert wrapped.weight_block_size == [1, TARGET_GROUP_K]
    assert wrapped.use_gfx90a_skinny_int8 is False


def test_selector_carries_skinny_opt_in_into_opaque_operator():
    wrapped = integration.select_cdna2_int8_linear_method(
        FakeFp8Method(),
        environ={
            integration.ENABLE_ENV: "1",
            "PRIME_ROCM_GFX90A_SKINNY_INT8": "1",
        },
        gcn_arch_name="gfx90a:sramecc+:xnack-",
    )

    assert isinstance(wrapped, integration.Cdna2Int8Fp8LinearMethod)
    assert wrapped.use_gfx90a_skinny_int8 is True


@pytest.mark.parametrize(
    ("method", "message"),
    (
        (
            FakeFp8Method(FakeFp8Config(is_checkpoint_fp8_serialized=False)),
            "serialized FP8 checkpoint",
        ),
        (
            FakeFp8Method(FakeFp8Config(activation_scheme="static")),
            "dynamic activation",
        ),
        (
            FakeFp8Method(FakeFp8Config(weight_block_size=[64, 128])),
            r"weight_block_size=\[128, 128\]",
        ),
    ),
)
def test_selector_rejects_incompatible_fp8_methods(method, message):
    with pytest.raises(ValueError, match=message):
        select(method)


def test_selector_rejects_wrong_architecture_and_any_live_reload():
    method = FakeFp8Method()
    with pytest.raises(ValueError, match="expected gfx90a"):
        integration.select_cdna2_int8_linear_method(
            method,
            environ={integration.ENABLE_ENV: "1"},
            gcn_arch_name="gfx942",
        )
    with pytest.raises(ValueError, match="live reload is not validated"):
        select(method, reload_mode="kernel")
    with pytest.raises(ValueError, match="live reload is not validated"):
        select(method, reload_mode="checkpoint")


def test_process_transcodes_and_preserves_weight_loader_metadata(monkeypatch):
    weight, source_scale = make_checkpoint_weight(131, 160)
    layer = FakeLinearLayer(weight, source_scale)

    def weight_loader(*args, **kwargs):
        return None

    def scale_loader(*args, **kwargs):
        return None

    layer.weight.weight_loader = weight_loader
    layer.weight_scale_inv.weight_loader = scale_loader
    expected = dequantize_block_weight(
        layer.weight,
        layer.weight_scale_inv,
        group_n=SOURCE_GROUP_N,
        group_k=SOURCE_GROUP_K,
    )
    method = select()
    method.audit = True
    method.prefix = "model.layers.4.mlp.down_proj"
    audit_rows = []

    def capture_audit(**kwargs):
        assert layer._prime_cdna2_int8_linear_ready is True
        audit_rows.append(kwargs)

    monkeypatch.setattr(integration, "log_transcode", capture_audit)

    method.process_weights_after_loading(layer)

    assert layer.weight.dtype == torch.int8
    assert layer.weight.shape == (131, 160)
    assert layer.weight_scale_inv.dtype == torch.float16
    assert layer.weight_scale_inv.shape == (131, 5)
    assert layer.weight_block_size == [1, 32]
    assert len(audit_rows) == 1
    assert audit_rows[0]["kind"] == "linear"
    assert audit_rows[0]["prefix"] == "model.layers.4.mlp.down_proj"
    assert audit_rows[0]["enabled"] is True
    assert layer.weight.weight_loader is weight_loader
    assert layer.weight_scale_inv.weight_loader is scale_loader
    assert layer.input_scale is None
    assert layer._prime_cdna2_int8_linear_ready is True
    actual = dequantize_block_weight(
        layer.weight,
        layer.weight_scale_inv,
        group_n=1,
        group_k=TARGET_GROUP_K,
    )
    relative_l2 = torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected)
    assert relative_l2.item() < 0.01

    with pytest.raises(RuntimeError, match="static-only"):
        method.process_weights_after_loading(layer)


@pytest.mark.parametrize(
    ("weight_dtype", "scale_shape", "message"),
    (
        (torch.bfloat16, (1, 1), "OCP torch.float8_e4m3fn"),
        (torch.float8_e4m3fn, (2, 1), "scales must have shape"),
    ),
)
def test_process_rejects_incompatible_layer_representation(
    weight_dtype,
    scale_shape,
    message,
):
    layer = FakeLinearLayer(
        torch.zeros((7, 64), dtype=weight_dtype),
        torch.ones(scale_shape, dtype=torch.float32),
    )

    with pytest.raises(ValueError, match=message):
        select().process_weights_after_loading(layer)


def test_process_rejects_unmeasured_shape_before_destructive_transcode(monkeypatch):
    weight, source_scale = make_checkpoint_weight(19, 64)
    layer = FakeLinearLayer(weight, source_scale)
    monkeypatch.setattr(
        integration,
        "select_gfx90a_int8_linear_config",
        lambda *_args: (_ for _ in ()).throw(ValueError("no measured config")),
    )

    def unexpected_transcode(*_args, **_kwargs):
        raise AssertionError("unmeasured shape reached transcode")

    monkeypatch.setattr(
        integration,
        "transcode_ocp_fp8_weight",
        unexpected_transcode,
    )

    with pytest.raises(ValueError, match="no measured config"):
        select().process_weights_after_loading(layer)


def test_apply_preserves_shape_bias_and_skips_device_query(monkeypatch):
    weight, source_scale = make_checkpoint_weight(19, 64)
    layer = FakeLinearLayer(weight, source_scale)
    method = select()
    method.process_weights_after_loading(layer)
    x = torch.randn((2, 3, 64), dtype=torch.bfloat16)
    bias = torch.randn((19,), dtype=torch.bfloat16)
    seen = {}

    def fake_opaque_linear(
        hidden_states,
        runtime_weight,
        runtime_scale,
        runtime_bias,
        use_gfx90a_skinny_int8,
    ):
        seen.update(
            hidden_states=hidden_states,
            runtime_weight=runtime_weight,
            runtime_scale=runtime_scale,
            runtime_bias=runtime_bias,
            use_gfx90a_skinny_int8=use_gfx90a_skinny_int8,
        )
        return torch.empty((*hidden_states.shape[:-1], runtime_weight.shape[0]))

    monkeypatch.setattr(
        integration,
        "cdna2_int8_linear_opaque",
        fake_opaque_linear,
    )

    output = method.apply(layer, x, bias)

    assert output.shape == (2, 3, 19)
    assert seen["hidden_states"] is x
    assert seen["runtime_weight"] is layer.weight
    assert seen["runtime_scale"] is layer.weight_scale_inv
    assert seen["runtime_bias"] is bias
    assert seen["use_gfx90a_skinny_int8"] is False


def test_apply_leaves_dynamic_token_gate_inside_opaque_operator(monkeypatch):
    weight, source_scale = make_checkpoint_weight(19, 64)
    layer = FakeLinearLayer(weight, source_scale)
    method = select()
    method.process_weights_after_loading(layer)
    seen = {}

    def fake_opaque_linear(hidden_states, *_args):
        seen["tokens"] = hidden_states.shape[0]
        return torch.empty(
            (hidden_states.shape[0], layer.weight.shape[0]),
            dtype=hidden_states.dtype,
        )

    monkeypatch.setattr(
        integration,
        "cdna2_int8_linear_opaque",
        fake_opaque_linear,
    )

    output = method.apply(
        layer,
        torch.zeros((2049, 64), dtype=torch.bfloat16),
    )

    assert seen["tokens"] == 2049
    assert output.shape == (2049, 19)
