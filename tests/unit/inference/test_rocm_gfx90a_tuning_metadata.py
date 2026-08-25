import json
from dataclasses import asdict
from pathlib import Path

from prime_rl.inference.vllm.kernels import rocm_cdna2_int8_linear as linear
from prime_rl.inference.vllm.kernels import rocm_cdna2_int8_moe as moe
from prime_rl.inference.vllm.kernels import rocm_cdna2_int8_vllm_abi as vllm_abi


def _load_tuning_metadata() -> dict:
    path = Path(linear.__file__).with_name("gfx90a_int8_tuning.json")
    return json.loads(path.read_text())


def test_gfx90a_tuning_metadata_matches_live_dispatch_and_runtime_contract():
    metadata = _load_tuning_metadata()
    launch_configs = metadata["launch_configs"]

    expected_dense = {
        tuple(int(value) for value in shape.split("x")): {
            int(tokens): launch_configs[config_name] for tokens, config_name in rows.items()
        }
        for shape, rows in metadata["selected_dense"].items()
    }
    actual_dense = {
        shape: {tokens: asdict(config) for tokens, config in rows.items()}
        for shape, rows in linear.GFX90A_INT8_LINEAR_CONFIGS.items()
    }
    assert expected_dense == actual_dense
    assert tuple(metadata["dispatch"]["dense_token_buckets"]) == (linear.GFX90A_INT8_TOKEN_BUCKETS)

    moe_overrides = metadata["moe_launch_overrides"]
    expected_moe = {
        int(tokens): {
            **launch_configs[config_name],
            **moe_overrides,
        }
        for tokens, config_name in metadata["selected_moe"].items()
    }
    actual_moe = {tokens: asdict(config) for tokens, config in moe.GFX90A_INT8_MOE_CONFIGS.items()}
    assert expected_moe == actual_moe
    assert tuple(metadata["dispatch"]["moe_token_buckets"]) == (moe.GFX90A_INT8_MOE_TOKEN_BUCKETS)

    assert metadata["runtime"]["vllm_distribution"] == (vllm_abi.EXPECTED_VLLM_DISTRIBUTION)
    assert metadata["runtime_options"]["required_environment"] == {"PRIME_ROCM_CDNA2_INT8": "1"}
    assert metadata["topology"]["local_scheduler_max_tokens"] == (linear.PREFILL_MAX_TOKENS)
    assert metadata["topology"]["global_ep_max_tokens"] == (moe.GLOBAL_EP_MAX_TOKENS)
