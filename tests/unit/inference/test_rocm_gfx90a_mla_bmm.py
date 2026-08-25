import json
import logging
from types import SimpleNamespace

import pytest
import torch

from prime_rl.inference.vllm.kernels import rocm_gfx90a_mla_bmm_integration as integration
from prime_rl.inference.vllm.kernels.rocm_gfx90a_mla_bmm import (
    GFX90A_MLA_BMM_SEED_CONFIGS,
    GFX90A_MLA_BMM_TOKEN_BUCKETS,
    GFX90A_MLA_BMM_TUNED_CONFIGS,
    GFX90A_MLA_BMM_TUNED_TOKENS,
    Gfx90aMlaBmmConfig,
    _validate_q_absorb_inputs,
    _validate_v_up_inputs,
    get_gfx90a_mla_bmm_config,
    matches_gfx90a_mla_bmm_layout,
    select_gfx90a_mla_bmm_config,
    select_gfx90a_mla_bmm_tuning_seed,
    validate_mla_bmm_config,
)


def make_q_absorb_tensors(m: int):
    q_storage = torch.empty((m, 8, 256), dtype=torch.bfloat16)
    q_nope = q_storage[..., :192].transpose(0, 1)
    weight_storage = torch.empty((8, 448, 512), dtype=torch.bfloat16)
    weight = weight_storage[:, :192, :]
    output = torch.empty((8, m, 512), dtype=torch.bfloat16)
    return q_nope, weight, output


def make_v_up_tensors(m: int):
    attention_storage = torch.empty((m, 8, 512), dtype=torch.bfloat16)
    attention_output = attention_storage.transpose(0, 1)
    weight_storage = torch.empty((8, 448, 512), dtype=torch.bfloat16)
    weight = weight_storage[:, 192:, :].transpose(1, 2)
    output_storage = torch.empty((m, 8, 256), dtype=torch.bfloat16)
    output = output_storage.transpose(0, 1)
    return attention_output, weight, output


def test_mla_bmm_tables_separate_tuning_seeds_from_measured_dispatch():
    assert set(GFX90A_MLA_BMM_SEED_CONFIGS) == {"q_absorb", "v_up"}
    assert all(tuple(table) == GFX90A_MLA_BMM_TOKEN_BUCKETS for table in GFX90A_MLA_BMM_SEED_CONFIGS.values())
    assert GFX90A_MLA_BMM_TUNED_TOKENS == (16, 32)
    assert all(tuple(table) == GFX90A_MLA_BMM_TUNED_TOKENS for table in GFX90A_MLA_BMM_TUNED_CONFIGS.values())
    assert select_gfx90a_mla_bmm_config("q_absorb", 16) == Gfx90aMlaBmmConfig(
        16,
        64,
        64,
        4,
        num_stages=2,
        waves_per_eu=1,
    )
    assert select_gfx90a_mla_bmm_config("q_absorb", 32) == Gfx90aMlaBmmConfig(
        16,
        64,
        64,
        4,
        num_stages=1,
        waves_per_eu=2,
    )
    assert select_gfx90a_mla_bmm_config("v_up", 16) == Gfx90aMlaBmmConfig(
        16,
        64,
        64,
        4,
        num_stages=2,
        waves_per_eu=2,
    )
    assert select_gfx90a_mla_bmm_config("v_up", 32) == Gfx90aMlaBmmConfig(
        16,
        64,
        64,
        4,
        num_stages=2,
        waves_per_eu=1,
    )
    assert select_gfx90a_mla_bmm_tuning_seed("v_up", 192) is GFX90A_MLA_BMM_SEED_CONFIGS["v_up"][128]


@pytest.mark.parametrize(
    ("kind", "m", "message"),
    (
        ("unknown", 16, "kind must be one of"),
        ("q_absorb", 0, "measured production configs only"),
        ("q_absorb", 15, "measured production configs only"),
        ("v_up", 33, "measured production configs only"),
        ("v_up", 2049, "measured production configs only"),
    ),
)
def test_mla_bmm_dispatch_fails_closed(kind, m, message):
    with pytest.raises(ValueError, match=message):
        select_gfx90a_mla_bmm_config(kind, m)


@pytest.mark.parametrize("m", (1, 15, 17, 31, 33, 2048))
@pytest.mark.parametrize("kind", ("q_absorb", "v_up"))
def test_mla_bmm_unmeasured_token_counts_select_stock_fallback(kind, m):
    assert get_gfx90a_mla_bmm_config(kind, m) is None


@pytest.mark.parametrize(
    ("kind", "config", "message"),
    (
        ("q_absorb", Gfx90aMlaBmmConfig(8, 64, 32, 4), "block_m"),
        (
            "q_absorb",
            Gfx90aMlaBmmConfig(48, 64, 32, 4),
            "block_m must be a power of two",
        ),
        ("q_absorb", Gfx90aMlaBmmConfig(16, 96, 32, 4), "block_n"),
        ("q_absorb", Gfx90aMlaBmmConfig(16, 64, 256, 4), "block_k"),
        (
            "q_absorb",
            Gfx90aMlaBmmConfig(16, 64, 48, 4),
            "block_k must be a power of two",
        ),
        ("v_up", Gfx90aMlaBmmConfig(16, 64, 32, 2), "num_warps"),
    ),
)
def test_mla_bmm_config_rejects_invalid_mfma_tiles(kind, config, message):
    with pytest.raises(ValueError, match=message):
        validate_mla_bmm_config(kind, config)


def test_q_absorb_contract_matches_profiled_views_before_device_check():
    q_nope, weight, output = make_q_absorb_tensors(16)

    assert q_nope.shape == (8, 16, 192)
    assert q_nope.stride() == (256, 2048, 1)
    assert weight.shape == (8, 192, 512)
    assert weight.stride() == (229376, 512, 1)
    assert output.stride() == (8192, 512, 1)
    assert matches_gfx90a_mla_bmm_layout("q_absorb", q_nope, weight, output)
    with pytest.raises(ValueError, match="ROCm GPU"):
        _validate_q_absorb_inputs(
            q_nope,
            weight,
            output,
            select_gfx90a_mla_bmm_config("q_absorb", 16),
        )


def test_v_up_contract_matches_profiled_views_before_device_check():
    attention_output, weight, output = make_v_up_tensors(16)

    assert attention_output.shape == (8, 16, 512)
    assert attention_output.stride() == (512, 4096, 1)
    assert weight.shape == (8, 512, 256)
    assert weight.stride() == (229376, 1, 512)
    assert output.stride() == (256, 2048, 1)
    assert matches_gfx90a_mla_bmm_layout("v_up", attention_output, weight, output)
    with pytest.raises(ValueError, match="ROCm GPU"):
        _validate_v_up_inputs(
            attention_output,
            weight,
            output,
            select_gfx90a_mla_bmm_config("v_up", 16),
        )


def test_q_absorb_contract_rejects_materialized_contiguous_view():
    q_nope, weight, output = make_q_absorb_tensors(16)

    with pytest.raises(ValueError, match="q_nope must have stride"):
        _validate_q_absorb_inputs(
            q_nope.contiguous(),
            weight,
            output,
            select_gfx90a_mla_bmm_config("q_absorb", 16),
        )


def test_v_up_contract_rejects_wrong_weight_layout():
    attention_output, weight, output = make_v_up_tensors(16)

    with pytest.raises(ValueError, match="V-up weight must have stride"):
        _validate_v_up_inputs(
            attention_output,
            weight.contiguous(),
            output,
            select_gfx90a_mla_bmm_config("v_up", 16),
        )


@pytest.mark.parametrize("m", (15, 17, 31, 33))
@pytest.mark.parametrize(
    ("kind", "factory"),
    (
        ("q_absorb", make_q_absorb_tensors),
        ("v_up", make_v_up_tensors),
    ),
)
def test_mla_bmm_layout_selector_rejects_unmeasured_boundaries(kind, factory, m):
    left, weight, output = factory(m)

    assert not matches_gfx90a_mla_bmm_layout(kind, left, weight, output)


def test_mla_bmm_layout_selector_rejects_materialized_profiled_view():
    q_nope, weight, output = make_q_absorb_tensors(16)

    assert not matches_gfx90a_mla_bmm_layout("q_absorb", q_nope.contiguous(), weight, output)


@pytest.mark.parametrize(
    ("factory", "validator", "kind"),
    (
        (make_q_absorb_tensors, _validate_q_absorb_inputs, "q_absorb"),
        (make_v_up_tensors, _validate_v_up_inputs, "v_up"),
    ),
)
def test_mla_bmm_contract_rejects_non_bf16(factory, validator, kind):
    left, weight, output = factory(16)

    with pytest.raises(ValueError, match="weight must use torch.bfloat16"):
        validator(
            left,
            weight.float(),
            output,
            select_gfx90a_mla_bmm_config(kind, 16),
        )


def make_glm52_config(**overrides):
    fields = {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "model_type": "glm_moe_dsa",
        "hidden_size": 6144,
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "head_dim": 192,
        "q_lora_rank": 2048,
        "kv_lora_rank": 512,
        "qk_head_dim": 256,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "v_head_dim": 256,
        "num_hidden_layers": 78,
        "index_topk": 2048,
        "quantization_config": {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
        },
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def make_glm52_vllm_config(**overrides):
    fields = {
        "model_config": SimpleNamespace(
            hf_config=make_glm52_config(),
            dtype=torch.bfloat16,
        ),
        "parallel_config": SimpleNamespace(
            tensor_parallel_size=8,
            decode_context_parallel_size=1,
        ),
        "scheduler_config": SimpleNamespace(max_num_batched_tokens=512),
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def make_glm52_mla_layer(**overrides):
    weight_storage = torch.empty((8, 448, 512), dtype=torch.bfloat16)
    fields = {
        "num_heads": 8,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "v_head_dim": 256,
        "q_lora_rank": 2048,
        "kv_lora_rank": 512,
        "q_pad_num_heads": None,
        "use_sparse": True,
        "is_aiter_triton_fp8_bmm_enabled": False,
        "is_aiter_triton_fp4_bmm_enabled": False,
        "W_UK_T": weight_storage[:, :192, :],
        "W_UV": weight_storage[:, 192:, :].transpose(1, 2),
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


class ExactMlaAttentionAbi:
    def __init__(
        self,
        num_heads,
        scale,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        q_lora_rank,
        kv_lora_rank,
        kv_b_proj,
        cache_config=None,
        quant_config=None,
        prefix="",
        attn_backend=None,
        use_sparse=False,
        indexer=None,
        topk_indices_buffer=None,
        **extra_impl_args,
    ):
        pass

    def forward_impl(
        self,
        q,
        k_c_normed,
        k_pe,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
        quant_group_size=None,
        quant_scale_ue8m0=None,
        quant_col_major=None,
        quant_tma_aligned=None,
    ):
        pass

    def process_weights_after_loading(self, act_dtype):
        pass

    def _v_up_proj(self, x, out):
        return "stock-v-up"

    def _q_absorb(self, q_nope, out):
        return "stock-q-absorb"


class ExactMlaAttentionAbiWithoutQ:
    __init__ = ExactMlaAttentionAbi.__init__
    forward_impl = ExactMlaAttentionAbi.forward_impl
    process_weights_after_loading = ExactMlaAttentionAbi.process_weights_after_loading
    _v_up_proj = ExactMlaAttentionAbi._v_up_proj


def exact_unified_mla_attention_with_output(
    q,
    kv_c_normed,
    k_pe,
    output,
    layer_name,
    output_scale=None,
    output_block_scale=None,
    kv_cache_dummy_dep=None,
    quant_group_size=None,
    quant_scale_ue8m0=None,
    quant_col_major=None,
    quant_tma_aligned=None,
):
    pass


def configure_mla_audit_logger(caplog, monkeypatch):
    monkeypatch.setattr(integration.LOGGER, "handlers", [caplog.handler])
    monkeypatch.setattr(integration.LOGGER, "level", logging.INFO)
    monkeypatch.setattr(integration.LOGGER, "propagate", False)
    monkeypatch.setattr(integration.LOGGER, "disabled", False)


def make_process_patch_type(calls):
    class PatchableMlaAttention:
        def process_weights_after_loading(self, act_dtype):
            calls.append(act_dtype)
            return "processed"

    return PatchableMlaAttention


def test_mla_integration_accepts_exact_model_runtime_and_weight_views():
    assert integration.glm52_mla_config_mismatches(make_glm52_config()) == ()
    assert integration.glm52_mla_runtime_mismatches(make_glm52_vllm_config()) == ()
    layer = make_glm52_mla_layer()
    assert integration.glm52_mla_instance_mismatches(layer) == ()
    assert integration.glm52_mla_weight_layout_mismatches(layer) == ()


@pytest.mark.parametrize(
    ("field", "actual", "expected"),
    (
        ("qk_nope_head_dim", 128, 192),
        ("qk_rope_head_dim", 32, 64),
    ),
)
def test_mla_integration_uses_canonical_dimensions_with_transformers_head_dim_alias(field, actual, expected):
    config = make_glm52_config(head_dim=64)

    assert integration.glm52_mla_config_mismatches(config) == ()

    setattr(config, field, actual)
    assert f"{field}={actual!r} (expected {expected!r})" in integration.glm52_mla_config_mismatches(config)


def test_mla_integration_reports_model_topology_and_scheduler_drift():
    bad_model = make_glm52_config(qk_nope_head_dim=128)
    bad_model.quantization_config["weight_block_size"] = [64, 128]
    runtime = make_glm52_vllm_config(
        model_config=SimpleNamespace(hf_config=bad_model, dtype=torch.float16),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=4,
            decode_context_parallel_size=2,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4096),
    )

    mismatches = integration.glm52_mla_runtime_mismatches(runtime)

    assert any("qk_nope_head_dim=128" in mismatch for mismatch in mismatches)
    assert any("weight_block_size=(64, 128)" in mismatch for mismatch in mismatches)
    assert any("model_config.dtype=torch.float16" in mismatch for mismatch in mismatches)
    assert any("tensor_parallel_size=4" in mismatch for mismatch in mismatches)
    assert any("decode_context_parallel_size=2" in mismatch for mismatch in mismatches)
    assert any("max_num_batched_tokens=4096" in mismatch for mismatch in mismatches)


def test_mla_integration_rejects_aiter_and_materialized_weight_layouts():
    layer = make_glm52_mla_layer(
        is_aiter_triton_fp8_bmm_enabled=True,
    )
    assert integration.glm52_mla_instance_mismatches(layer) == (
        "is_aiter_triton_fp8_bmm_enabled=True (expected None or False)",
    )

    layer.W_UV = layer.W_UV.contiguous()
    mismatches = integration.glm52_mla_weight_layout_mismatches(layer)
    assert "W_UV.stride=(131072, 256, 1) (expected (229376, 1, 512))" in mismatches
    assert "W_UV.storage_offset=0 (expected 98304)" in mismatches
    assert "W_UK_T and W_UV do not share their post-load backing storage" in mismatches


@pytest.mark.parametrize("value", (None, False))
def test_mla_integration_accepts_inactive_aiter_flags(value):
    layer = make_glm52_mla_layer(
        is_aiter_triton_fp8_bmm_enabled=value,
        is_aiter_triton_fp4_bmm_enabled=value,
    )

    assert integration.glm52_mla_instance_mismatches(layer) == ()

    del layer.is_aiter_triton_fp8_bmm_enabled
    del layer.is_aiter_triton_fp4_bmm_enabled
    assert integration.glm52_mla_instance_mismatches(layer) == ()


@pytest.mark.parametrize("value", (True, 0, "", "false"))
def test_mla_integration_rejects_enabled_or_malformed_aiter_flags(value):
    layer = make_glm52_mla_layer(is_aiter_triton_fp4_bmm_enabled=value)

    assert integration.glm52_mla_instance_mismatches(layer) == (
        f"is_aiter_triton_fp4_bmm_enabled={value!r} (expected None or False)",
    )


def test_mla_audit_opt_in_accepts_only_zero_one_or_unset():
    assert integration.mla_audit_requested({}) is False
    assert integration.mla_audit_requested({integration.AUDIT_ENV: "0"}) is False
    assert integration.mla_audit_requested({integration.AUDIT_ENV: "1"}) is True


@pytest.mark.parametrize("value", ("", "true", "yes", "2"))
def test_mla_audit_opt_in_rejects_ambiguous_values(value):
    with pytest.raises(ValueError, match=integration.AUDIT_ENV):
        integration.mla_audit_requested({integration.AUDIT_ENV: value})


def test_mla_audit_invalid_opt_in_fails_before_import_or_mutation(monkeypatch):
    monkeypatch.setenv(integration.AUDIT_ENV, "yes")
    monkeypatch.setattr(
        integration,
        "_load_vllm_mla_symbols",
        lambda: pytest.fail("private vLLM symbols were imported before validating the audit opt-in"),
    )

    with pytest.raises(ValueError, match=integration.AUDIT_ENV):
        integration.install_glm52_gfx90a_mla_bmm()


def test_disabled_mla_audit_marks_validated_layer_without_identity_or_record(caplog, monkeypatch):
    calls = []
    mla_type = make_process_patch_type(calls)
    monkeypatch.setattr(integration, "_validate_layer_after_loading", lambda *_args: None)
    configure_mla_audit_logger(caplog, monkeypatch)
    capabilities = integration.MlaBmmIntegration(q_absorb=False, v_up=True)
    integration._install_process_weights_patch(
        mla_type,
        lambda: None,
        capabilities,
        audit_enabled=False,
    )
    layer = mla_type()
    layer.layer_name = "model.layers.3.self_attn.attn"

    assert layer.process_weights_after_loading(torch.bfloat16) == "processed"

    assert calls == [torch.bfloat16]
    assert getattr(layer, integration._MLA_MARKER) is True
    assert not hasattr(layer, integration._MLA_AUDIT_MARKER)
    assert caplog.records == []


def test_enabled_mla_audit_emits_exactly_once_after_repeated_processing(caplog, monkeypatch):
    calls = []
    validations = []
    mla_type = make_process_patch_type(calls)
    monkeypatch.setattr(
        integration,
        "_validate_layer_after_loading",
        lambda *args: validations.append(args),
    )
    monkeypatch.setattr(integration.dist, "is_available", lambda: True)
    monkeypatch.setattr(integration.dist, "is_initialized", lambda: False)
    monkeypatch.setenv("RANK", "7")
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setattr(integration.os, "getpid", lambda: 1234)
    monkeypatch.setattr(integration.socket, "gethostname", lambda: "worker.example")
    configure_mla_audit_logger(caplog, monkeypatch)
    capabilities = integration.MlaBmmIntegration(q_absorb=False, v_up=True)
    integration._install_process_weights_patch(
        mla_type,
        lambda: None,
        capabilities,
        audit_enabled=True,
    )
    layer = mla_type()
    layer.layer_name = "model.layers.3.self_attn.attn"

    assert layer.process_weights_after_loading(torch.bfloat16) == "processed"
    assert layer.process_weights_after_loading(torch.bfloat16) == "processed"

    assert calls == [torch.bfloat16, torch.bfloat16]
    assert len(validations) == 2
    assert getattr(layer, integration._MLA_MARKER) is True
    assert getattr(layer, integration._MLA_AUDIT_MARKER) is True
    assert len(caplog.records) == 1
    payload = json.loads(caplog.records[0].message)
    assert payload == {
        "event": "prime_glm52_gfx90a_mla_bmm_layer",
        "hostname": "worker.example",
        "identity_source": "environment",
        "layer_name": "model.layers.3.self_attn.attn",
        "local_rank": 3,
        "pid": 1234,
        "prefix": "model.layers.3.self_attn.attn",
        "q_absorb": False,
        "rank": 7,
        "schema_version": 1,
        "v_up": True,
    }
    assert caplog.records[0].message == json.dumps(payload, sort_keys=True, separators=(",", ":"))


def test_mla_audit_prefers_initialized_distributed_identity(monkeypatch):
    monkeypatch.setenv("RANK", "99")
    monkeypatch.setenv("LOCAL_RANK", "98")
    monkeypatch.setattr(integration.dist, "is_available", lambda: True)
    monkeypatch.setattr(integration.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(integration.dist, "get_rank", lambda: 11)
    monkeypatch.setattr(integration.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(integration.torch.cuda, "current_device", lambda: 3)

    payload = integration._layer_audit_payload(
        SimpleNamespace(layer_name="model.layers.4.self_attn.attn"),
        integration.MlaBmmIntegration(q_absorb=True, v_up=True),
    )

    assert payload["identity_source"] == "torch.distributed"
    assert payload["rank"] == 11
    assert payload["local_rank"] == 3


def test_enabled_mla_audit_fails_closed_without_process_identity(caplog, monkeypatch):
    mla_type = make_process_patch_type([])
    monkeypatch.setattr(integration, "_validate_layer_after_loading", lambda *_args: None)
    monkeypatch.setattr(integration.dist, "is_available", lambda: True)
    monkeypatch.setattr(integration.dist, "is_initialized", lambda: False)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    configure_mla_audit_logger(caplog, monkeypatch)
    integration._install_process_weights_patch(
        mla_type,
        lambda: None,
        integration.MlaBmmIntegration(q_absorb=False, v_up=True),
        audit_enabled=True,
    )
    layer = mla_type()
    layer.layer_name = "model.layers.3.self_attn.attn"

    with pytest.raises(RuntimeError, match="integer RANK"):
        layer.process_weights_after_loading(torch.bfloat16)

    assert not hasattr(layer, integration._MLA_MARKER)
    assert not hasattr(layer, integration._MLA_AUDIT_MARKER)
    assert caplog.records == []


def test_mla_audit_emits_nothing_when_post_load_validation_fails(caplog, monkeypatch):
    mla_type = make_process_patch_type([])

    def fail_validation(*_args):
        raise RuntimeError("post-load validation failed")

    monkeypatch.setattr(integration, "_validate_layer_after_loading", fail_validation)
    monkeypatch.setenv("RANK", "7")
    monkeypatch.setenv("LOCAL_RANK", "3")
    configure_mla_audit_logger(caplog, monkeypatch)
    integration._install_process_weights_patch(
        mla_type,
        lambda: None,
        integration.MlaBmmIntegration(q_absorb=False, v_up=True),
        audit_enabled=True,
    )
    layer = mla_type()
    layer.layer_name = "model.layers.3.self_attn.attn"

    with pytest.raises(RuntimeError, match="post-load validation failed"):
        layer.process_weights_after_loading(torch.bfloat16)

    assert not hasattr(layer, integration._MLA_MARKER)
    assert not hasattr(layer, integration._MLA_AUDIT_MARKER)
    assert caplog.records == []


@pytest.mark.parametrize("m", (15, 17, 31, 33))
def test_mla_integration_uses_stock_paths_at_unmeasured_token_boundaries(m):
    calls = []

    class PatchableMlaAttention:
        def _v_up_proj(self, x, out):
            calls.append(("stock-v-up", x, out))
            return "stock-v-up"

        def _q_absorb(self, q_nope, out):
            calls.append(("stock-q-absorb", q_nope, out))
            return "stock-q-absorb"

    integration._install_v_up_patch(PatchableMlaAttention)
    integration._install_q_absorb_patch(PatchableMlaAttention)
    layer = PatchableMlaAttention()
    setattr(layer, integration._MLA_MARKER, True)
    layer.num_heads = 8
    layer.kv_lora_rank = 512
    layer.v_head_dim = 256
    layer.W_UK_T = make_q_absorb_tensors(m)[1]
    layer.W_UV = make_v_up_tensors(m)[1]
    q_nope, _, q_out = make_q_absorb_tensors(m)
    v_in = torch.empty((m, 8, 512), dtype=torch.bfloat16)
    v_out = torch.empty((m, 8, 256), dtype=torch.bfloat16)

    assert layer._q_absorb(q_nope, q_out) == "stock-q-absorb"
    assert layer._v_up_proj(v_in, v_out) == "stock-v-up"
    assert [call[0] for call in calls] == ["stock-q-absorb", "stock-v-up"]


@pytest.mark.parametrize("m", (16, 32))
def test_mla_integration_falls_back_when_measured_layout_is_not_on_rocm(m):
    calls = []

    class PatchableMlaAttention:
        def _v_up_proj(self, x, out):
            calls.append("stock-v-up")
            return "stock-v-up"

        def _q_absorb(self, q_nope, out):
            calls.append("stock-q-absorb")
            return "stock-q-absorb"

    integration._install_v_up_patch(PatchableMlaAttention)
    integration._install_q_absorb_patch(PatchableMlaAttention)
    layer = PatchableMlaAttention()
    setattr(layer, integration._MLA_MARKER, True)
    layer.num_heads = 8
    layer.kv_lora_rank = 512
    layer.v_head_dim = 256
    layer.W_UK_T = make_q_absorb_tensors(m)[1]
    layer.W_UV = make_v_up_tensors(m)[1]
    q_nope, _, q_out = make_q_absorb_tensors(m)
    v_in = torch.empty((m, 8, 512), dtype=torch.bfloat16)
    v_out = torch.empty((m, 8, 256), dtype=torch.bfloat16)

    assert layer._q_absorb(q_nope, q_out) == "stock-q-absorb"
    assert layer._v_up_proj(v_in, v_out) == "stock-v-up"
    assert calls == ["stock-q-absorb", "stock-v-up"]


@pytest.mark.parametrize("m", (16, 32))
def test_mla_integration_dispatches_exact_measured_cells(monkeypatch, m):
    custom_calls = []

    class PatchableMlaAttention:
        def _v_up_proj(self, x, out):
            return "stock-v-up"

        def _q_absorb(self, q_nope, out):
            return "stock-q-absorb"

    def fake_q_absorb(left, weight, output, **kwargs):
        custom_calls.append(("q_absorb", left, weight, output, kwargs))
        return output

    def fake_v_up(left, weight, output, **kwargs):
        custom_calls.append(("v_up", left, weight, output, kwargs))
        return output

    monkeypatch.setattr(integration, "can_dispatch_gfx90a_mla_bmm", lambda *_args: True)
    monkeypatch.setattr(integration, "gfx90a_glm52_mla_q_absorb", fake_q_absorb)
    monkeypatch.setattr(integration, "gfx90a_glm52_mla_v_up", fake_v_up)
    integration._install_v_up_patch(PatchableMlaAttention)
    integration._install_q_absorb_patch(PatchableMlaAttention)
    layer = PatchableMlaAttention()
    setattr(layer, integration._MLA_MARKER, True)
    layer.num_heads = 8
    layer.kv_lora_rank = 512
    layer.v_head_dim = 256
    layer.W_UK_T = make_q_absorb_tensors(m)[1]
    layer.W_UV = make_v_up_tensors(m)[1]
    q_nope, _, q_out = make_q_absorb_tensors(m)
    v_in = torch.empty((m, 8, 512), dtype=torch.bfloat16)
    v_out = torch.empty((m, 8, 256), dtype=torch.bfloat16)

    assert layer._q_absorb(q_nope, q_out) is None
    assert layer._v_up_proj(v_in, v_out) is None
    assert [call[0] for call in custom_calls] == ["q_absorb", "v_up"]
    for kind, left, _, _, kwargs in custom_calls:
        assert left.shape[1] == m
        assert kwargs == {
            "config": get_gfx90a_mla_bmm_config(kind, m),
            "device_is_validated": True,
        }


def test_mla_integration_accepts_exact_vllm_abi(monkeypatch):
    monkeypatch.setattr(
        integration,
        "distribution_version",
        lambda _distribution: integration.EXPECTED_VLLM_DISTRIBUTION,
    )

    assert (
        integration._require_exact_vllm_mla_abi(
            ExactMlaAttentionAbi,
            exact_unified_mla_attention_with_output,
            require_q_absorb=True,
        )
        is True
    )


def test_mla_integration_requires_exact_vllm_distribution(monkeypatch):
    monkeypatch.setattr(integration, "distribution_version", lambda _distribution: "0.25.0")
    monkeypatch.setattr(
        integration,
        "_load_vllm_mla_symbols",
        lambda: pytest.fail("private vLLM symbols were imported before the distribution check"),
    )

    with pytest.raises(RuntimeError, match="requires vLLM"):
        integration.install_glm52_gfx90a_mla_bmm()


def test_mla_integration_requires_exact_method_abi(monkeypatch):
    class DriftedMlaAttentionAbi(ExactMlaAttentionAbi):
        def _v_up_proj(self, x, out, stream=None):
            pass

    monkeypatch.setattr(
        integration,
        "distribution_version",
        lambda _distribution: integration.EXPECTED_VLLM_DISTRIBUTION,
    )

    with pytest.raises(RuntimeError, match=r"MLAAttention\._v_up_proj"):
        integration._require_exact_vllm_mla_abi(
            DriftedMlaAttentionAbi,
            exact_unified_mla_attention_with_output,
            require_q_absorb=True,
        )


def test_mla_integration_requires_upstream_q_absorb_hook_before_mutation(monkeypatch):
    monkeypatch.setattr(
        integration,
        "distribution_version",
        lambda _distribution: integration.EXPECTED_VLLM_DISTRIBUTION,
    )
    monkeypatch.setattr(
        integration,
        "_load_vllm_mla_symbols",
        lambda: (
            ExactMlaAttentionAbiWithoutQ,
            exact_unified_mla_attention_with_output,
            lambda: make_glm52_vllm_config(),
        ),
    )
    original_process = ExactMlaAttentionAbiWithoutQ.process_weights_after_loading
    original_v_up = ExactMlaAttentionAbiWithoutQ._v_up_proj

    with pytest.raises(RuntimeError, match=r"MLAAttention\._q_absorb"):
        integration.install_glm52_gfx90a_mla_bmm()

    assert ExactMlaAttentionAbiWithoutQ.process_weights_after_loading is original_process
    assert ExactMlaAttentionAbiWithoutQ._v_up_proj is original_v_up


def test_mla_integration_can_install_v_up_only_without_intercepting_stock_instances(monkeypatch):
    monkeypatch.delenv(integration.AUDIT_ENV, raising=False)
    monkeypatch.setattr(
        integration,
        "distribution_version",
        lambda _distribution: integration.EXPECTED_VLLM_DISTRIBUTION,
    )
    monkeypatch.setattr(
        integration,
        "_load_vllm_mla_symbols",
        lambda: (
            ExactMlaAttentionAbiWithoutQ,
            exact_unified_mla_attention_with_output,
            lambda: make_glm52_vllm_config(),
        ),
    )

    installed = integration.install_glm52_gfx90a_mla_bmm(require_q_absorb=False)
    stock_instance = object.__new__(ExactMlaAttentionAbiWithoutQ)

    assert installed == integration.MlaBmmIntegration(q_absorb=False, v_up=True)
    assert (
        getattr(
            ExactMlaAttentionAbiWithoutQ.process_weights_after_loading,
            integration._PROCESS_CAPABILITIES_MARKER,
        )
        is installed
    )
    assert stock_instance._v_up_proj(None, None) == "stock-v-up"
