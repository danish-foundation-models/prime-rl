from types import SimpleNamespace

import pytest

from prime_rl.inference.vllm.kernels.rocm_nemotron_router_linear import (
    NEMOTRON_ROUTER_MAX_TOKENS,
    nemotron_router_config_mismatches,
    select_nemotron_router_config,
    validate_nemotron_router_scheduler,
)


def _config(**overrides):
    values = {
        "architectures": ["NemotronHForCausalLM"],
        "model_type": "nemotron_h",
        "hidden_size": 4096,
        "n_routed_experts": 512,
        "num_experts_per_tok": 22,
        "moe_intermediate_size": 2688,
        "moe_latent_size": 1024,
        "mlp_hidden_act": "relu2",
        "num_hidden_layers": 88,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_nemotron_router_model_family_validation():
    assert nemotron_router_config_mismatches(_config()) == ()
    assert nemotron_router_config_mismatches(_config(n_routed_experts=256)) == ("n_routed_experts=256 (expected 512)",)


def test_nemotron_router_selects_ranges_not_one_exact_m():
    small = select_nemotron_router_config(1)
    assert select_nemotron_router_config(512) is small

    large = select_nemotron_router_config(513)
    assert select_nemotron_router_config(16384) is large
    assert select_nemotron_router_config(NEMOTRON_ROUTER_MAX_TOKENS) is large
    assert large is not small


@pytest.mark.parametrize("tokens", [0, -1, 32769])
def test_nemotron_router_rejects_unsupported_token_counts(tokens):
    with pytest.raises(ValueError):
        select_nemotron_router_config(tokens)


def test_nemotron_router_scheduler_bound():
    assert validate_nemotron_router_scheduler(SimpleNamespace(max_num_batched_tokens=32768)) == 32768
    with pytest.raises(ValueError, match="max_num_batched_tokens"):
        validate_nemotron_router_scheduler(SimpleNamespace(max_num_batched_tokens=65536))
