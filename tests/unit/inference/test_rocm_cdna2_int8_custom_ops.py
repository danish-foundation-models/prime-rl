from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from prime_rl.inference.vllm.kernels import (
    rocm_cdna2_int8_custom_ops as custom_ops,
)


def test_linear_implementation_owns_dynamic_config_selection(monkeypatch):
    selected_config = object()
    seen = {}

    def fake_select(m, n, k):
        seen["shape"] = (m, n, k)
        return selected_config

    def fake_linear(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return torch.empty((2, 3, 5), dtype=torch.bfloat16)

    monkeypatch.setattr(
        custom_ops,
        "select_gfx90a_int8_linear_config",
        fake_select,
    )
    monkeypatch.setattr(custom_ops, "cdna2_int8_linear", fake_linear)
    hidden_states = torch.empty((2, 3, 7), dtype=torch.bfloat16)
    weight = torch.empty((5, 7), dtype=torch.int8)
    weight_scale = torch.empty((5, 1), dtype=torch.float16)
    bias = torch.empty((5,), dtype=torch.bfloat16)

    output = custom_ops._linear_impl(
        hidden_states,
        weight,
        weight_scale,
        bias,
        True,
    )

    assert output.shape == (2, 3, 5)
    assert seen["shape"] == (6, 5, 7)
    assert seen["args"] == (
        hidden_states,
        weight,
        weight_scale,
        bias,
    )
    assert seen["kwargs"] == {
        "config": selected_config,
        "device_is_validated": True,
        "use_gfx90a_skinny_int8": True,
    }


def test_linear_implementation_keeps_high_m_gate_inside_operator(monkeypatch):
    monkeypatch.setattr(
        custom_ops,
        "select_gfx90a_int8_linear_config",
        lambda *_args: (_ for _ in ()).throw(ValueError("CDNA2 INT8 linear is gated through M=2048")),
    )
    monkeypatch.setattr(
        custom_ops,
        "cdna2_int8_linear",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("high-M input reached the kernel")),
    )

    with pytest.raises(ValueError, match="gated through M=2048"):
        custom_ops._linear_impl(
            torch.empty((2049, 32), dtype=torch.bfloat16),
            torch.empty((64, 32), dtype=torch.int8),
            torch.empty((64, 1), dtype=torch.float16),
            None,
            True,
        )


def test_moe_implementation_keeps_dynamic_selector_inside_operator(monkeypatch):
    seen = {}

    def fake_fused_experts(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return torch.empty_like(args[0])

    monkeypatch.setattr(
        custom_ops,
        "cdna2_int8_fused_experts",
        fake_fused_experts,
    )
    tensors = (
        torch.empty((6, 8), dtype=torch.bfloat16),
        torch.empty((2, 16, 8), dtype=torch.int8),
        torch.empty((2, 8, 8), dtype=torch.int8),
        torch.empty((2, 16, 1), dtype=torch.float16),
        torch.empty((2, 8, 1), dtype=torch.float16),
        torch.empty((6, 2), dtype=torch.float32),
        torch.empty((6, 2), dtype=torch.int32),
        torch.empty((4,), dtype=torch.int32),
    )

    output = custom_ops._moe_impl(*tensors, 4)

    assert output.shape == (6, 8)
    assert seen["args"] == tensors
    assert seen["kwargs"] == {
        "global_num_experts": 4,
        "device_is_validated": True,
    }


def test_fake_implementations_preserve_output_contract():
    dense = custom_ops._linear_fake(
        torch.empty((2, 3, 7), dtype=torch.bfloat16),
        torch.empty((5, 7), dtype=torch.int8),
        torch.empty((5, 1), dtype=torch.float16),
        None,
        True,
    )
    moe_hidden = torch.empty((6, 8), dtype=torch.bfloat16)
    moe = custom_ops._moe_fake(
        moe_hidden,
        torch.empty((2, 16, 8), dtype=torch.int8),
        torch.empty((2, 8, 8), dtype=torch.int8),
        torch.empty((2, 16, 1), dtype=torch.float16),
        torch.empty((2, 8, 1), dtype=torch.float16),
        torch.empty((6, 2), dtype=torch.float32),
        torch.empty((6, 2), dtype=torch.int32),
        torch.empty((4,), dtype=torch.int32),
        4,
    )

    assert dense.shape == (2, 3, 5)
    assert dense.dtype == torch.bfloat16
    assert moe.shape == moe_hidden.shape
    assert moe.dtype == moe_hidden.dtype


def test_registration_is_rocm_only_and_idempotent(monkeypatch):
    import vllm.platforms
    import vllm.utils.torch_utils

    calls = []
    monkeypatch.setattr(custom_ops, "_REGISTERED", False)
    monkeypatch.setattr(
        custom_ops,
        "LINEAR_OP_NAME",
        "test_prime_rocm_gfx90a_cdna2_int8_linear",
    )
    monkeypatch.setattr(
        custom_ops,
        "MOE_OP_NAME",
        "test_prime_rocm_gfx90a_cdna2_int8_moe",
    )
    monkeypatch.setattr(
        vllm.platforms,
        "current_platform",
        SimpleNamespace(is_rocm=lambda: True, dispatch_key="CUDA"),
    )
    monkeypatch.setattr(
        vllm.utils.torch_utils,
        "direct_register_custom_op",
        lambda **kwargs: calls.append(kwargs),
    )

    custom_ops.register_cdna2_int8_custom_ops()
    custom_ops.register_cdna2_int8_custom_ops()

    assert [call["op_name"] for call in calls] == [
        custom_ops.LINEAR_OP_NAME,
        custom_ops.MOE_OP_NAME,
    ]
    assert all(call["dispatch_key"] == "CUDA" for call in calls)
    assert calls[0]["fake_impl"] is custom_ops._linear_fake
    assert calls[1]["fake_impl"] is custom_ops._moe_fake


def test_registration_rejects_non_rocm_platform(monkeypatch):
    import vllm.platforms

    monkeypatch.setattr(custom_ops, "_REGISTERED", False)
    monkeypatch.setattr(
        vllm.platforms,
        "current_platform",
        SimpleNamespace(is_rocm=lambda: False, dispatch_key="CUDA"),
    )

    with pytest.raises(RuntimeError, match="require a ROCm vLLM platform"):
        custom_ops.register_cdna2_int8_custom_ops()


def test_symbolic_dense_and_moe_calls_export_as_opaque_ops(monkeypatch):
    import vllm.platforms

    monkeypatch.setattr(custom_ops, "_REGISTERED", False)
    monkeypatch.setattr(
        vllm.platforms,
        "current_platform",
        SimpleNamespace(is_rocm=lambda: True, dispatch_key="CPU"),
    )
    custom_ops.register_cdna2_int8_custom_ops()

    class Graph(torch.nn.Module):
        def forward(
            self,
            dense_hidden,
            dense_weight,
            dense_scale,
            moe_hidden,
            w13_weight,
            w2_weight,
            w13_scale,
            w2_scale,
            topk_weights,
            topk_ids,
            expert_map,
        ):
            dense = custom_ops.cdna2_int8_linear_opaque(
                dense_hidden,
                dense_weight,
                dense_scale,
                None,
                True,
            )
            moe = custom_ops.cdna2_int8_moe_opaque(
                moe_hidden,
                w13_weight,
                w2_weight,
                w13_scale,
                w2_scale,
                topk_weights,
                topk_ids,
                expert_map,
                4,
            )
            return dense, moe

    inputs = (
        torch.empty((2, 7), dtype=torch.bfloat16),
        torch.empty((5, 7), dtype=torch.int8),
        torch.empty((5, 1), dtype=torch.float16),
        torch.empty((2, 8), dtype=torch.bfloat16),
        torch.empty((2, 16, 8), dtype=torch.int8),
        torch.empty((2, 8, 8), dtype=torch.int8),
        torch.empty((2, 16, 1), dtype=torch.float16),
        torch.empty((2, 8, 1), dtype=torch.float16),
        torch.empty((2, 2), dtype=torch.float32),
        torch.empty((2, 2), dtype=torch.int32),
        torch.empty((4,), dtype=torch.int32),
    )
    dense_tokens = torch.export.Dim("dense_tokens", min=1, max=2048)
    moe_tokens = torch.export.Dim("moe_tokens", min=1, max=4096)
    exported = torch.export.export(
        Graph(),
        inputs,
        dynamic_shapes=(
            {0: dense_tokens},
            None,
            None,
            {0: moe_tokens},
            None,
            None,
            None,
            None,
            {0: moe_tokens},
            {0: moe_tokens},
            None,
        ),
        strict=True,
    )
    targets = {node.target for node in exported.graph.nodes if node.op == "call_function"}

    assert torch.ops.vllm.prime_rocm_gfx90a_cdna2_int8_linear.default in targets
    assert torch.ops.vllm.prime_rocm_gfx90a_cdna2_int8_moe.default in targets
