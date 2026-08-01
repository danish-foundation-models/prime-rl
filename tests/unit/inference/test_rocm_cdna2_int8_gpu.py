"""Small production-path correctness cells for the gfx90a INT8 kernels."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F

from prime_rl.inference.vllm.kernels.rocm_cdna2_int8_linear import (
    TARGET_GROUP_K,
    cdna2_int8_linear,
    select_gfx90a_int8_linear_config,
)
from prime_rl.inference.vllm.kernels.rocm_cdna2_int8_moe import (
    cdna2_int8_fused_experts,
    select_gfx90a_int8_moe_config,
)
from prime_rl.inference.vllm.kernels.rocm_cdna2_int8_vllm_abi import (
    install_vllm_kernel_abi,
)


def _is_gfx90a() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return getattr(properties, "gcnArchName", "").split(":")[0] == "gfx90a"


pytestmark = pytest.mark.skipif(
    not _is_gfx90a(),
    reason="requires a gfx90a ROCm GPU",
)


def _dequantize_rows(
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return weight.float() * scale.float().repeat_interleave(
        TARGET_GROUP_K,
        dim=-1,
    )


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    delta = actual.float() - expected.float()
    denominator = torch.linalg.vector_norm(expected.float()).clamp_min(1.0e-12)
    return float((torch.linalg.vector_norm(delta) / denominator).item())


def _assert_graph_replay_exact(fn: Callable[[], torch.Tensor]) -> None:
    eager = fn().clone()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = fn()
    graph.replay()
    torch.cuda.synchronize()
    first = captured.clone()
    torch.testing.assert_close(first, eager, rtol=0, atol=0)
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, first, rtol=0, atol=0)


def test_dense_production_api_is_numerical_and_graph_safe():
    install_vllm_kernel_abi()
    generator = torch.Generator(device="cuda").manual_seed(520241)
    m, n, k = 64, 2048, 2048
    hidden = (
        torch.randn(
            (m, k),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.2
    ).contiguous()
    weight = torch.randint(
        -16,
        17,
        (n, k),
        device="cuda",
        dtype=torch.int8,
        generator=generator,
    )
    scale = (
        torch.rand(
            (n, k // TARGET_GROUP_K),
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        * 0.01
        + 0.005
    ).half()
    config = select_gfx90a_int8_linear_config(m, n, k)

    def run() -> torch.Tensor:
        return cdna2_int8_linear(
            hidden,
            weight,
            scale,
            config=config,
            device_is_validated=True,
        )

    actual = run()
    expected = hidden.float() @ _dequantize_rows(weight, scale).T
    assert _relative_l2(actual, expected) <= 0.025
    assert (
        float(
            F.cosine_similarity(
                actual.float().flatten(),
                expected.flatten(),
                dim=0,
            ).item()
        )
        >= 0.999
    )
    _assert_graph_replay_exact(run)


def test_moe_m4096_production_api_is_numerical_and_graph_safe():
    install_vllm_kernel_abi()
    generator = torch.Generator(device="cuda").manual_seed(520242)
    tokens = 4096
    hidden_size = 128
    intermediate_size = 64
    experts = 4
    top_k = 2
    hidden = (
        torch.randn(
            (tokens, hidden_size),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.2
    ).contiguous()
    w1 = torch.randint(
        -12,
        13,
        (experts, 2 * intermediate_size, hidden_size),
        device="cuda",
        dtype=torch.int8,
        generator=generator,
    )
    w2 = torch.randint(
        -12,
        13,
        (experts, hidden_size, intermediate_size),
        device="cuda",
        dtype=torch.int8,
        generator=generator,
    )
    w1_scale = (
        torch.rand(
            (
                experts,
                2 * intermediate_size,
                hidden_size // TARGET_GROUP_K,
            ),
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        * 0.01
        + 0.005
    ).half()
    w2_scale = (
        torch.rand(
            (
                experts,
                hidden_size,
                intermediate_size // TARGET_GROUP_K,
            ),
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        * 0.01
        + 0.005
    ).half()
    token = torch.arange(tokens, device="cuda", dtype=torch.int32)[:, None]
    route = torch.arange(top_k, device="cuda", dtype=torch.int32)[None, :]
    topk_ids = ((token * top_k + route) % experts).contiguous()
    topk_weights = torch.full(
        (tokens, top_k),
        1.0 / top_k,
        device="cuda",
        dtype=torch.float32,
    )
    expert_map = torch.arange(
        experts,
        device="cuda",
        dtype=torch.int32,
    )
    config = select_gfx90a_int8_moe_config(tokens)

    def run() -> torch.Tensor:
        return cdna2_int8_fused_experts(
            hidden,
            w1,
            w2,
            w1_scale,
            w2_scale,
            topk_weights,
            topk_ids,
            expert_map,
            global_num_experts=experts,
            config=config,
            device_is_validated=True,
        )

    dequantized_w1 = _dequantize_rows(w1, w1_scale)
    dequantized_w2 = _dequantize_rows(w2, w2_scale)
    expected = torch.zeros_like(hidden, dtype=torch.float32)
    for expert in range(experts):
        selected = torch.nonzero(topk_ids == expert, as_tuple=False)
        token_ids = selected[:, 0]
        route_ids = selected[:, 1]
        expert_hidden = hidden[token_ids].float()
        gate_up = expert_hidden @ dequantized_w1[expert].T
        activation = F.silu(gate_up[:, :intermediate_size])
        activation *= gate_up[:, intermediate_size:]
        expert_output = activation @ dequantized_w2[expert].T
        expert_output *= topk_weights[token_ids, route_ids, None]
        expected.index_add_(0, token_ids.long(), expert_output)

    actual = run()
    assert _relative_l2(actual, expected) <= 0.06
    assert (
        float(
            F.cosine_similarity(
                actual.float().flatten(),
                expected.flatten(),
                dim=0,
            ).item()
        )
        >= 0.999
    )
    _assert_graph_replay_exact(run)
