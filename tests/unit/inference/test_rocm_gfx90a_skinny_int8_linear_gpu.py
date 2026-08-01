"""Production-path correctness and graph replay for the gfx90a candidate."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from prime_rl.inference.vllm.kernels.rocm_cdna2_int8_linear import (
    TARGET_GROUP_K,
    cdna2_int8_linear,
    select_gfx90a_int8_linear_config,
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


@pytest.mark.parametrize(
    ("m", "n", "k"),
    (
        (8, 4096, 6144),
        (64, 2624, 6144),
    ),
)
def test_skinny_int8_matches_stock_and_is_graph_safe(m, n, k):
    install_vllm_kernel_abi()
    generator = torch.Generator(device="cuda").manual_seed(520252 + m)
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
    stock_config = select_gfx90a_int8_linear_config(m, n, k)

    def run_candidate() -> torch.Tensor:
        return cdna2_int8_linear(
            hidden,
            weight,
            scale,
            config=stock_config,
            device_is_validated=True,
            use_gfx90a_skinny_int8=True,
        )

    stock = cdna2_int8_linear(
        hidden,
        weight,
        scale,
        config=stock_config,
        device_is_validated=True,
    )
    candidate = run_candidate()
    assert _relative_l2(candidate, stock) <= 0.002
    torch.testing.assert_close(candidate, stock, rtol=0.01, atol=0.5)
    _assert_graph_replay_exact(run_candidate)
