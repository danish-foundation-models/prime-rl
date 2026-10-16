import pytest
import torch
from transformers.models.apertus.configuration_apertus import ApertusConfig
from transformers.models.apertus.modeling_apertus import ApertusForCausalLM

from prime_rl.utils.cp import assert_hf_ulysses_supports_model


def test_apertus_hf_model_supports_ulysses_cp4():
    config = ApertusConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=131_072,
    )
    with torch.device("meta"):
        model = ApertusForCausalLM(config)

    assert_hf_ulysses_supports_model(model, cp_world_size=4)


def test_hf_ulysses_rejects_non_softmax_layers_and_incompatible_heads():
    class Config:
        model_type = "apertus"
        num_attention_heads = 32
        num_key_value_heads = 8

    class HybridLayer(torch.nn.Module):
        layer_type = "linear_attention"

    class HybridInner:
        layers = torch.nn.Sequential(HybridLayer())

    class HybridModel:
        config = Config()
        model = HybridInner()

    with pytest.raises(ValueError, match="softmax-attention"):
        assert_hf_ulysses_supports_model(HybridModel(), cp_world_size=4)

    class MambaConfig(Config):
        layers_block_type = ["mamba", "attention"]

    class MambaModel:
        config = MambaConfig()

    with pytest.raises(ValueError, match="Mamba/SSM"):
        assert_hf_ulysses_supports_model(MambaModel(), cp_world_size=4)

    class IncompatibleConfig:
        model_type = "apertus"
        num_attention_heads = 30
        num_key_value_heads = 8

    class IncompatibleModel:
        config = IncompatibleConfig()

    with pytest.raises(ValueError, match="num_attention_heads"):
        assert_hf_ulysses_supports_model(IncompatibleModel(), cp_world_size=4)


def test_hf_ulysses_rejects_unvalidated_model_type():
    class Config:
        model_type = "qwen3"
        num_attention_heads = 32
        num_key_value_heads = 8

    class Model:
        config = Config()

    with pytest.raises(ValueError, match="only model_type='apertus'"):
        assert_hf_ulysses_supports_model(Model(), cp_world_size=4)


def test_apertus_meta_load_restores_checkpoint_buffers_and_logits():
    from copy import deepcopy

    from prime_rl.trainer.apertus_buffers import can_reinit_apertus_buffers, normalize_apertus_parameter_dtype
    from prime_rl.trainer.model import can_reinit_empty_buffers, fix_model_post_empty

    config = ApertusConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=64, dtype="bfloat16", hidden_act="xielu",
    )
    config._attn_implementation = "eager"
    fresh = ApertusForCausalLM._from_config(deepcopy(config), dtype=torch.float32).eval()
    normalize_apertus_parameter_dtype(fresh, torch.float32)
    state = fresh.state_dict()
    assert "model.layers.0.mlp.act_fn.beta" in state
    assert "model.layers.0.mlp.act_fn.eps" in state
    with torch.device("meta"):
        restored = ApertusForCausalLM._from_config(deepcopy(config), dtype=torch.float32).eval()
        normalize_apertus_parameter_dtype(restored, torch.float32)
    assert all(p.dtype == torch.float32 for p in restored.parameters())
    assert can_reinit_apertus_buffers(restored)
    assert can_reinit_empty_buffers(restored)
    restored.to_empty(device="cpu")
    fix_model_post_empty(restored)
    restored.load_state_dict(state)
    expected_buffers = dict(fresh.named_buffers())
    for name, value in restored.named_buffers():
        torch.testing.assert_close(value, expected_buffers[name], rtol=0, atol=0)
    ids = torch.arange(32)[None]
    with torch.no_grad():
        torch.testing.assert_close(
            fresh(ids, use_cache=False).logits,
            restored(ids, use_cache=False).logits,
            rtol=0, atol=0,
        )
    restored.register_buffer("unknown_runtime_cache", torch.zeros(1), persistent=False)
    assert not can_reinit_apertus_buffers(restored)
    assert not can_reinit_empty_buffers(restored)
