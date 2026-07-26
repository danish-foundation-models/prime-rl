import inspect

from prime_rl.inference import patches
from prime_rl.inference.vllm.kernels.rocm_nemotron_moe_config import (
    can_dispatch_nemotron_moe_config,
    wrap_nemotron_moe_config,
)


def stock(
    w1_shape,
    w2_shape,
    top_k,
    dtype,
    M,
    block_shape=None,
):
    return "stock"


def exact_inputs(*, tokens=4096):
    return (
        (64, 2688, 1024),
        (64, 1024, 2688),
        22,
        None,
        tokens,
        None,
    )


def test_nemotron_moe_config_accepts_measured_prefill_range():
    assert can_dispatch_nemotron_moe_config(*exact_inputs())
    assert can_dispatch_nemotron_moe_config(*exact_inputs(tokens=32768))


def test_nemotron_moe_config_rejects_decode_and_other_shapes():
    assert not can_dispatch_nemotron_moe_config(*exact_inputs(tokens=8))
    values = list(exact_inputs())
    values[0] = (512, 2688, 1024)
    assert not can_dispatch_nemotron_moe_config(*values)


def test_nemotron_moe_config_wrapper_dispatches_and_delegates():
    wrapped = wrap_nemotron_moe_config(stock)
    selected = wrapped(*exact_inputs(tokens=16384))

    assert selected["BLOCK_SIZE_N"] == 256
    assert selected["GROUP_SIZE_M"] == 1
    assert wrapped(*exact_inputs(tokens=8)) == "stock"
    assert inspect.signature(wrapped) == inspect.signature(stock)
    assert wrap_nemotron_moe_config(wrapped) is wrapped


def test_nemotron_moe_config_installer_wraps_both_references(monkeypatch):
    from vllm.model_executor.layers.fused_moe import fused_moe
    from vllm.model_executor.layers.fused_moe.experts import triton_moe

    original = fused_moe.try_get_optimal_moe_config
    monkeypatch.setattr(fused_moe, "try_get_optimal_moe_config", original)
    monkeypatch.setattr(triton_moe, "try_get_optimal_moe_config", original)
    monkeypatch.setenv("PRIME_ROCM_NEMOTRON_MOE_CONFIG", "1")
    monkeypatch.setattr(patches, "_require_rocm_gfx90a", lambda _feature: "gfx90a")
    monkeypatch.setattr(
        patches,
        "_installed_vllm_distribution_version",
        lambda: "0.24.0+lumi_aif_gfx90a_ee0da84",
    )

    patches.monkey_patch_rocm_nemotron_moe_config()

    assert fused_moe.try_get_optimal_moe_config is not original
    assert triton_moe.try_get_optimal_moe_config is fused_moe.try_get_optimal_moe_config
    assert getattr(
        fused_moe.try_get_optimal_moe_config,
        "_prime_rl_gfx90a_nemotron_moe_config",
    )
