from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from prime_rl.inference.vllm.kernels import (
    rocm_cdna2_int8_linear as linear,
)
from prime_rl.inference.vllm.kernels import (
    rocm_gfx90a_skinny_int8_linear as skinny,
)


@pytest.mark.parametrize(
    ("environ", "expected"),
    (
        ({}, False),
        ({skinny.ENABLE_ENV: "0"}, False),
        ({skinny.ENABLE_ENV: "1"}, True),
    ),
)
def test_skinny_int8_opt_in_is_strict(environ, expected):
    assert skinny.gfx90a_skinny_int8_requested(environ) is expected


def test_skinny_int8_opt_in_rejects_ambiguous_values():
    with pytest.raises(ValueError, match=skinny.ENABLE_ENV):
        skinny.gfx90a_skinny_int8_requested({skinny.ENABLE_ENV: "true"})


def test_selector_covers_only_profiled_c128_shapes():
    m8 = skinny.select_gfx90a_skinny_int8_config(8, 4096, 6144)
    assert m8 is not None
    assert (
        m8.block_m,
        m8.block_n,
        m8.block_k,
        m8.num_warps,
        m8.num_stages,
    ) == (16, 32, 32, 4, 2)

    m64 = skinny.select_gfx90a_skinny_int8_config(64, 2624, 6144)
    assert m64 is not None
    assert (
        m64.block_m,
        m64.block_n,
        m64.block_k,
        m64.num_warps,
        m64.num_stages,
    ) == (64, 32, 32, 8, 1)

    assert skinny.select_gfx90a_skinny_int8_config(8, 6144, 2048) is None
    assert skinny.select_gfx90a_skinny_int8_config(32, 2624, 6144) is None
    assert skinny.select_gfx90a_skinny_int8_config(64, 7168, 6144) is None


def test_every_gated_shape_obeys_the_kernel_tile_contract():
    for m, shapes in skinny.GFX90A_SKINNY_INT8_SHAPES.items():
        for n, k in shapes:
            config = skinny.select_gfx90a_skinny_int8_config(m, n, k)
            assert config is not None
            assert m <= config.block_m
            assert n % config.block_n == 0
            assert k % config.block_k == 0


@pytest.mark.parametrize(
    ("m", "enabled", "uses_skinny"),
    (
        (64, True, True),
        (64, False, False),
        (32, True, False),
    ),
)
def test_production_dispatch_is_opt_in_and_falls_back(
    monkeypatch,
    m,
    enabled,
    uses_skinny,
):
    n = k = 2048
    hidden = torch.empty((m, k), dtype=torch.bfloat16)
    weight = torch.empty((n, k), dtype=torch.int8)
    weight_scale = torch.empty((n, k // 32), dtype=torch.float16)
    candidate_calls = []
    stock = _KernelRecorder()

    monkeypatch.setattr(
        linear,
        "_validate_linear_inputs",
        lambda *_args: (m, n, k, (m, n)),
    )
    monkeypatch.setattr(
        linear,
        "_VLLM_ACTIVATION_QUANTIZER",
        lambda tensor, _group_k: (
            torch.empty_like(tensor, dtype=torch.int8),
            torch.empty((m, k // 32), dtype=torch.float32),
        ),
    )
    monkeypatch.setattr(linear, "_VLLM_DENSE_INT8_KERNEL", stock)
    monkeypatch.setattr(
        linear,
        "invoke_gfx90a_skinny_int8_linear",
        lambda *args, **kwargs: candidate_calls.append((args, kwargs)),
    )

    output = linear.cdna2_int8_linear(
        hidden,
        weight,
        weight_scale,
        config=linear.select_gfx90a_int8_linear_config(m, n, k),
        device_is_validated=True,
        use_gfx90a_skinny_int8=enabled,
    )

    assert output.shape == (m, n)
    assert bool(candidate_calls) is uses_skinny
    assert (stock.grid is not None) is not uses_skinny


@dataclass
class _FakeTensor:
    shape: tuple[int, int]
    strides: tuple[int, int]
    ndim: int = 2

    def stride(self, dimension: int) -> int:
        return self.strides[dimension]


class _KernelRecorder:
    def __init__(self) -> None:
        self.grid = None
        self.args = None
        self.kwargs = None

    def __getitem__(self, grid):
        self.grid = grid
        return self

    def __call__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs


def test_launcher_uses_one_complete_m_tile_per_n32_program(monkeypatch):
    recorder = _KernelRecorder()
    monkeypatch.setattr(
        skinny,
        "_gfx90a_k32_skinny_int8_kernel",
        recorder,
    )
    m, n, k = 64, 2624, 6144
    activation = _FakeTensor((m, k), (k, 1))
    weight = _FakeTensor((n, k), (k, 1))
    output = _FakeTensor((m, n), (n, 1))
    activation_scale = _FakeTensor((m, k // 32), (k // 32, 1))
    weight_scale = _FakeTensor((n, k // 32), (k // 32, 1))

    skinny.invoke_gfx90a_skinny_int8_linear(
        activation,
        weight,
        output,
        activation_scale,
        weight_scale,
    )

    assert recorder.grid == (82,)
    assert recorder.args[5:8] == (m, n, k)
    assert recorder.args[8:18] == (
        k,
        1,
        1,
        k,
        n,
        1,
        k // 32,
        1,
        1,
        k // 32,
    )
    assert recorder.kwargs == {
        "BLOCK_M": 64,
        "BLOCK_N": 32,
        "BLOCK_K": 32,
        "num_warps": 8,
        "num_stages": 1,
    }


def test_launcher_rejects_an_unsupported_shape_before_launch(monkeypatch):
    recorder = _KernelRecorder()
    monkeypatch.setattr(
        skinny,
        "_gfx90a_k32_skinny_int8_kernel",
        recorder,
    )

    with pytest.raises(ValueError, match="unsupported skinny INT8 shape"):
        skinny.invoke_gfx90a_skinny_int8_linear(
            _FakeTensor((8, 2048), (2048, 1)),
            _FakeTensor((6144, 2048), (2048, 1)),
            _FakeTensor((8, 6144), (6144, 1)),
            _FakeTensor((8, 64), (64, 1)),
            _FakeTensor((6144, 64), (64, 1)),
        )

    assert recorder.grid is None
