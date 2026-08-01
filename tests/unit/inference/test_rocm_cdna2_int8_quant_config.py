import json
import logging
import pickle
from types import SimpleNamespace

import pytest

from prime_rl.inference.vllm.kernels import rocm_cdna2_int8_quant_config as integration

VALID_CONFIG = {
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
    "weight_block_size": [128, 128],
}


def test_custom_config_accepts_only_the_measured_checkpoint_contract():
    config = integration.Cdna2Int8Fp8Config.from_config(VALID_CONFIG)

    assert config.get_name() == integration.QUANTIZATION_NAME
    assert config.is_checkpoint_fp8_serialized is True
    assert config.activation_scheme == "dynamic"
    assert config.weight_block_size == [128, 128]

    with pytest.raises(ValueError, match=r"weight_block_size=\[128, 128\]"):
        integration.Cdna2Int8Fp8Config.from_config({**VALID_CONFIG, "weight_block_size": [64, 128]})
    with pytest.raises(ValueError, match="dynamic activation"):
        integration.Cdna2Int8Fp8Config.from_config({**VALID_CONFIG, "activation_scheme": "static"})


def test_mla_kv_b_keeps_stock_fp8_method_for_post_load_absorption(
    monkeypatch,
    caplog,
):
    from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod

    assert integration._requires_source_fp8_post_load("model.layers.7.self_attn.kv_b_proj")
    assert not integration._requires_source_fp8_post_load("model.layers.7.self_attn.o_proj")
    stock_method = object.__new__(Fp8LinearMethod)
    monkeypatch.setattr(
        integration.Fp8Config,
        "get_quant_method",
        lambda *_args, **_kwargs: stock_method,
    )
    monkeypatch.setattr(
        integration,
        "_resolve_gfx90a_architecture",
        lambda: (_ for _ in ()).throw(AssertionError("kv_b stock post-load path queried the device")),
    )
    monkeypatch.setenv(
        "PRIME_ROCM_CDNA2_INT8_AUDIT",
        "1",
    )
    monkeypatch.setenv("RANK", "7")
    monkeypatch.setenv("LOCAL_RANK", "3")
    config = integration.Cdna2Int8Fp8Config.from_config(VALID_CONFIG)

    with caplog.at_level(
        logging.INFO,
        logger=integration.log_exemption.__module__,
    ):
        selected = config.get_quant_method(
            object(),
            "model.layers.7.self_attn.kv_b_proj",
        )

    assert selected is stock_method
    payload = json.loads(caplog.records[-1].message)
    assert {key: payload[key] for key in ("event", "kind", "prefix", "reason")} == {
        "event": "prime_cdna2_int8_exemption",
        "kind": "linear",
        "prefix": "model.layers.7.self_attn.kv_b_proj",
        "reason": "stock_fp8_post_load_required",
    }
    assert isinstance(payload["pid"], int)
    assert {"rank", "local_rank", "hostname"} <= payload.keys()
    assert payload["rank"] == 7
    assert payload["local_rank"] == 3
    assert payload["identity_source"] == "environment"


def test_quant_config_passes_prefix_and_audit_to_linear_wrapper(monkeypatch):
    from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod

    stock_method = object.__new__(Fp8LinearMethod)
    seen = {}
    monkeypatch.setenv("PRIME_ROCM_CDNA2_INT8_AUDIT", "1")
    monkeypatch.setattr(
        integration.Fp8Config,
        "get_quant_method",
        lambda *_args, **_kwargs: stock_method,
    )
    monkeypatch.setattr(
        integration,
        "_resolve_gfx90a_architecture",
        lambda: "gfx90a",
    )
    monkeypatch.setattr(
        integration,
        "_validate_scheduler_contract",
        lambda: 2048,
    )

    def fake_wrapper(method, **kwargs):
        seen["method"] = method
        seen.update(kwargs)
        return "wrapped"

    monkeypatch.setattr(
        integration,
        "Cdna2Int8Fp8LinearMethod",
        fake_wrapper,
    )
    config = integration.Cdna2Int8Fp8Config.from_config(VALID_CONFIG)

    selected = config.get_quant_method(
        object(),
        "model.layers.7.self_attn.o_proj",
    )

    assert selected == "wrapped"
    assert seen == {
        "method": stock_method,
        "gcn_arch_name": "gfx90a",
        "reload_mode": "none",
        "prefix": "model.layers.7.self_attn.o_proj",
        "audit": True,
        "use_gfx90a_skinny_int8": False,
    }


@pytest.mark.parametrize("max_num_batched_tokens", (1, 2048))
def test_quant_config_scheduler_contract_accepts_tuned_capacity(
    monkeypatch,
    max_num_batched_tokens,
):
    import vllm.config as vllm_config

    monkeypatch.setattr(
        vllm_config,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(
            scheduler_config=SimpleNamespace(
                max_num_batched_tokens=max_num_batched_tokens,
            )
        ),
    )

    assert integration._validate_scheduler_contract() == max_num_batched_tokens


@pytest.mark.parametrize(
    "max_num_batched_tokens",
    (None, True, 0, -1, 2049),
)
def test_quant_config_scheduler_contract_rejects_before_transcode(
    monkeypatch,
    max_num_batched_tokens,
):
    import vllm.config as vllm_config

    monkeypatch.setattr(
        vllm_config,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(
            scheduler_config=SimpleNamespace(
                max_num_batched_tokens=max_num_batched_tokens,
            )
        ),
    )

    with pytest.raises(ValueError, match="max_num_batched_tokens"):
        integration._validate_scheduler_contract()


def test_quant_config_scheduler_contract_requires_construction_context(
    monkeypatch,
):
    import vllm.config as vllm_config

    monkeypatch.setattr(
        vllm_config,
        "get_current_vllm_config_or_none",
        lambda: None,
    )

    with pytest.raises(ValueError, match="current vLLM configuration"):
        integration._validate_scheduler_contract()


def test_quant_config_rejects_scheduler_before_linear_wrapper_or_device_query(
    monkeypatch,
):
    import vllm.config as vllm_config
    from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod

    stock_method = object.__new__(Fp8LinearMethod)
    monkeypatch.setattr(
        integration.Fp8Config,
        "get_quant_method",
        lambda *_args, **_kwargs: stock_method,
    )
    monkeypatch.setattr(
        vllm_config,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(
            scheduler_config=SimpleNamespace(
                max_num_batched_tokens=2049,
            )
        ),
    )
    monkeypatch.setattr(
        integration,
        "_resolve_gfx90a_architecture",
        lambda: (_ for _ in ()).throw(AssertionError("invalid scheduler queried the device")),
    )
    monkeypatch.setattr(
        integration,
        "Cdna2Int8Fp8LinearMethod",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("invalid scheduler constructed the wrapper")),
    )
    config = integration.Cdna2Int8Fp8Config.from_config(VALID_CONFIG)

    with pytest.raises(ValueError, match="max_num_batched_tokens"):
        config.get_quant_method(
            object(),
            "model.layers.7.self_attn.o_proj",
        )


def test_registration_is_disabled_without_the_explicit_opt_in(monkeypatch):
    def unexpected_architecture_query():
        raise AssertionError("disabled registration queried the device")

    monkeypatch.setattr(
        integration,
        "_resolve_gfx90a_architecture",
        unexpected_architecture_query,
    )

    assert integration.install_cdna2_int8_quantization(environ={}) is False


def test_registration_uses_vllm_public_custom_config_seam(monkeypatch):
    import vllm.model_executor.layers.quantization as quantization

    seen = {}

    def fake_register(name):
        seen["name"] = name

        def register(config_type):
            seen["config_type"] = config_type
            return config_type

        return register

    monkeypatch.setattr(quantization, "register_quantization_config", fake_register)

    def unexpected_architecture_query():
        raise AssertionError("frontend registration queried the device")

    monkeypatch.setattr(
        integration,
        "_resolve_gfx90a_architecture",
        unexpected_architecture_query,
    )
    monkeypatch.setattr(integration, "_REGISTERED", False)
    monkeypatch.delenv(integration.RELOAD_MODE_ENV, raising=False)
    abi_calls = []
    custom_op_calls = []
    monkeypatch.setattr(
        integration,
        "install_vllm_kernel_abi",
        lambda: abi_calls.append("validated"),
    )
    monkeypatch.setattr(
        integration,
        "register_cdna2_int8_custom_ops",
        lambda: custom_op_calls.append("registered"),
    )

    assert (
        integration.install_cdna2_int8_quantization(
            environ={"PRIME_ROCM_CDNA2_INT8": "1"},
        )
        is True
    )
    assert seen == {
        "name": integration.QUANTIZATION_NAME,
        "config_type": integration.Cdna2Int8Fp8Config,
    }
    assert abi_calls == ["validated"]
    assert custom_op_calls == ["registered"]
    assert integration.os.environ[integration.RELOAD_MODE_ENV] == "none"


def test_registration_rejects_ambiguous_audit_value_before_abi(monkeypatch):
    monkeypatch.setattr(
        integration,
        "install_vllm_kernel_abi",
        lambda: (_ for _ in ()).throw(AssertionError("invalid audit value reached ABI installation")),
    )

    with pytest.raises(ValueError, match="PRIME_ROCM_CDNA2_INT8_AUDIT"):
        integration.install_cdna2_int8_quantization(
            environ={
                "PRIME_ROCM_CDNA2_INT8": "1",
                "PRIME_ROCM_CDNA2_INT8_AUDIT": "yes",
            },
        )


def test_registration_rejects_ambiguous_skinny_value_before_abi(monkeypatch):
    monkeypatch.setattr(
        integration,
        "install_vllm_kernel_abi",
        lambda: (_ for _ in ()).throw(AssertionError("invalid skinny value reached ABI installation")),
    )

    with pytest.raises(ValueError, match="PRIME_ROCM_GFX90A_SKINNY_INT8"):
        integration.install_cdna2_int8_quantization(
            environ={
                "PRIME_ROCM_CDNA2_INT8": "1",
                "PRIME_ROCM_GFX90A_SKINNY_INT8": "yes",
            },
        )


def test_general_plugin_installs_enabled_quantization_before_model_config(monkeypatch):
    from prime_rl.inference import patches

    calls = []
    monkeypatch.setenv("PRIME_ROCM_CDNA2_INT8", "1")
    monkeypatch.setattr(
        integration,
        "install_cdna2_int8_quantization",
        lambda: calls.append("installed"),
    )

    patches._install_cdna2_int8_quantization()

    assert calls == ["installed"]


def test_reload_mode_is_serialized_on_config_for_fresh_workers(monkeypatch):
    monkeypatch.setenv(integration.RELOAD_MODE_ENV, "none")
    config = integration.Cdna2Int8Fp8Config.from_config(VALID_CONFIG)

    restored = pickle.loads(pickle.dumps(config))

    assert restored.cdna2_reload_mode == "none"
