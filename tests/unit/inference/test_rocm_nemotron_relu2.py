import inspect
from types import SimpleNamespace

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from prime_rl.inference import patches
from prime_rl.inference.vllm.kernels import rocm_nemotron_relu2
from prime_rl.inference.vllm.kernels.rocm_nemotron_relu2 import (
    can_dispatch_nemotron_ep_relu2,
    can_dispatch_nemotron_relu2,
    wrap_nemotron_experts_activation,
)

RELU2 = SimpleNamespace(value="relu2_no_mul")
SILU = SimpleNamespace(value="silu_no_mul")


def make_fake_inputs(*, rows=5632, width=2688):
    with FakeTensorMode():
        input_tensor = torch.empty(
            (rows, width),
            device="cuda",
            dtype=torch.bfloat16,
        )
        output = torch.empty_like(input_tensor)
    return output, input_tensor


def make_fake_assignment(*, rows=5632, block_size_m=64):
    with FakeTensorMode():
        sorted_token_ids = torch.empty((rows,), device="cuda", dtype=torch.int32)
        expert_ids = torch.empty(
            ((rows + block_size_m - 1) // block_size_m,),
            device="cuda",
            dtype=torch.int32,
        )
        num_tokens_post_padded = torch.empty((1,), device="cuda", dtype=torch.int32)
    return {
        "sorted_token_ids": sorted_token_ids,
        "expert_ids": expert_ids,
        "num_tokens_post_padded": num_tokens_post_padded,
        "block_size_m": block_size_m,
    }


def stock(self, activation, output, input, **kwargs):
    return "stock"


def test_nemotron_relu2_accepts_exact_prefill_cell():
    assert can_dispatch_nemotron_relu2(RELU2, *make_fake_inputs())
    assert can_dispatch_nemotron_relu2(RELU2, *make_fake_inputs(rows=32768 * 22))


def test_nemotron_relu2_rejects_decode_and_other_activation():
    assert not can_dispatch_nemotron_relu2(RELU2, *make_fake_inputs(rows=8 * 22))
    assert not can_dispatch_nemotron_relu2(SILU, *make_fake_inputs())
    assert not can_dispatch_nemotron_relu2(RELU2, *make_fake_inputs(width=1024))


def test_nemotron_ep_relu2_accepts_local_assignment_metadata():
    assert can_dispatch_nemotron_ep_relu2(
        RELU2,
        *make_fake_inputs(),
        **make_fake_assignment(),
    )


def test_nemotron_relu2_wrapper_dispatches_and_delegates(monkeypatch):
    calls = []
    ep_calls = []
    wrapped = wrap_nemotron_experts_activation(stock)
    supported = make_fake_inputs()
    monkeypatch.setattr(
        rocm_nemotron_relu2,
        "nemotron_relu2_gfx90a",
        lambda *args: calls.append(args),
    )
    monkeypatch.setattr(
        rocm_nemotron_relu2,
        "nemotron_ep_relu2_gfx90a",
        lambda *args, **kwargs: ep_calls.append((args, kwargs)),
    )

    assert wrapped("self", RELU2, *supported) is None
    assert calls == [supported]
    assignment = make_fake_assignment()
    assert wrapped("self", RELU2, *supported, **assignment) is None
    assert ep_calls == [(supported, assignment)]
    assert wrapped("self", RELU2, *make_fake_inputs(rows=176)) == "stock"
    assert inspect.signature(wrapped) == inspect.signature(stock)
    assert wrap_nemotron_experts_activation(wrapped) is wrapped


def test_nemotron_relu2_installer_wraps_triton_experts(monkeypatch):
    from vllm.model_executor.layers.fused_moe.experts import triton_moe
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts

    original = TritonExperts.activation
    monkeypatch.setattr(TritonExperts, "activation", original)
    monkeypatch.setattr(triton_moe, "_PRIME_NEMOTRON_EP_RELU2_VERSION", 1, raising=False)
    monkeypatch.setattr(triton_moe, "_PRIME_NEMOTRON_EP_RELU2_ENABLED", False, raising=False)
    monkeypatch.setenv("PRIME_ROCM_NEMOTRON_RELU2", "1")
    monkeypatch.setattr(patches, "_require_rocm_gfx90a", lambda _feature: "gfx90a")
    monkeypatch.setattr(
        patches,
        "_installed_vllm_distribution_version",
        lambda: "0.24.0+lumi_aif_gfx90a_ee0da84",
    )

    patches.monkey_patch_rocm_nemotron_relu2()

    assert TritonExperts.activation is not original
    assert getattr(
        TritonExperts.activation,
        "_prime_rl_gfx90a_nemotron_relu2",
    )
    assert triton_moe._PRIME_NEMOTRON_EP_RELU2_ENABLED
