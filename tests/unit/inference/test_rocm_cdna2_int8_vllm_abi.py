from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from prime_rl.inference.vllm.kernels import rocm_cdna2_int8_vllm_abi as abi


@pytest.fixture(autouse=True)
def clear_abi_cache():
    abi.validate_vllm_kernel_abi.cache_clear()
    yield
    abi.validate_vllm_kernel_abi.cache_clear()


def test_vllm_abi_pin_matches_exact_gfx90a_stack():
    assert abi.DEFAULT_VLLM_DISTRIBUTION == "0.25.0.dev0+g27d0e3ef3.gfx90a"
    assert abi.EXPECTED_VLLM_DISTRIBUTION == os.environ.get(
        "PRIME_CDNA2_EXPECTED_VLLM_DISTRIBUTION",
        abi.DEFAULT_VLLM_DISTRIBUTION,
    )
    assert abi.EXPECTED_FUSED_MOE_KERNEL_ARGS[-2:] == (
        "HAS_BIAS",
        "SWAP_AB",
    )
    assert abi.EXPECTED_MOE_SUM_ARGS == (
        "input",
        "output",
        "topk_ids",
        "expert_map",
    )


def test_vllm_abi_rejects_wrong_distribution_before_symbol_import(
    monkeypatch,
):
    monkeypatch.setattr(abi, "distribution_version", lambda _name: "0.24.1")
    monkeypatch.setattr(
        abi,
        "_load_vllm_symbols",
        lambda: (_ for _ in ()).throw(AssertionError("wrong version imported private symbols")),
    )

    with pytest.raises(RuntimeError, match="requires vLLM distribution"):
        abi.validate_vllm_kernel_abi()


@pytest.mark.parametrize(
    ("symbol_index", "message"),
    (
        (0, "dense INT8 kernel ABI changed"),
        (1, "INT8 activation quantizer ABI changed"),
        (2, "fused MoE kernel ABI changed"),
        (3, "moe_align_block_size ABI changed"),
        (4, "moe_sum ABI changed"),
    ),
)
def test_vllm_abi_rejects_changed_symbol_signatures(
    monkeypatch,
    symbol_index,
    message,
):
    symbols = [
        SimpleNamespace(arg_names=abi.EXPECTED_DENSE_KERNEL_ARGS),
        SimpleNamespace(arg_names=abi.EXPECTED_ACTIVATION_QUANTIZER_ARGS),
        SimpleNamespace(arg_names=abi.EXPECTED_FUSED_MOE_KERNEL_ARGS),
        SimpleNamespace(arg_names=abi.EXPECTED_MOE_ALIGN_ARGS),
        SimpleNamespace(arg_names=abi.EXPECTED_MOE_SUM_ARGS),
    ]
    symbols[symbol_index] = SimpleNamespace(arg_names=("changed",))
    monkeypatch.setattr(
        abi,
        "distribution_version",
        lambda _name: abi.EXPECTED_VLLM_DISTRIBUTION,
    )
    monkeypatch.setattr(abi, "_load_vllm_symbols", lambda: tuple(symbols))

    with pytest.raises(RuntimeError, match=message):
        abi.validate_vllm_kernel_abi()


def test_vllm_abi_accepts_and_caches_exact_symbols(monkeypatch):
    symbols = (
        SimpleNamespace(arg_names=abi.EXPECTED_DENSE_KERNEL_ARGS),
        SimpleNamespace(arg_names=abi.EXPECTED_ACTIVATION_QUANTIZER_ARGS),
        SimpleNamespace(arg_names=abi.EXPECTED_FUSED_MOE_KERNEL_ARGS),
        SimpleNamespace(arg_names=abi.EXPECTED_MOE_ALIGN_ARGS),
        SimpleNamespace(arg_names=abi.EXPECTED_MOE_SUM_ARGS),
    )
    calls = 0

    def load_symbols():
        nonlocal calls
        calls += 1
        return symbols

    monkeypatch.setattr(
        abi,
        "distribution_version",
        lambda _name: abi.EXPECTED_VLLM_DISTRIBUTION,
    )
    monkeypatch.setattr(abi, "_load_vllm_symbols", load_symbols)

    first = abi.validate_vllm_kernel_abi()
    second = abi.validate_vllm_kernel_abi()

    assert first is second
    assert first.dense_kernel is symbols[0]
    assert first.activation_quantizer is symbols[1]
    assert first.fused_moe_kernel is symbols[2]
    assert first.moe_align_block_size is symbols[3]
    assert first.moe_sum is symbols[4]
    assert calls == 1


def test_install_vllm_abi_binds_every_validated_runtime_symbol(monkeypatch):
    from prime_rl.inference.vllm.kernels import (
        rocm_cdna2_int8_linear,
        rocm_cdna2_int8_moe,
    )

    symbols = (
        object(),
        object(),
        object(),
        object(),
        object(),
    )
    validated = abi.VllmKernelAbi(
        dense_kernel=symbols[0],
        activation_quantizer=symbols[1],
        fused_moe_kernel=symbols[2],
        moe_align_block_size=symbols[3],
        moe_sum=symbols[4],
    )
    dense_bindings = []
    moe_bindings = []
    monkeypatch.setattr(abi, "validate_vllm_kernel_abi", lambda: validated)
    monkeypatch.setattr(
        rocm_cdna2_int8_linear,
        "bind_vllm_dense_int8_symbols",
        lambda *runtime_symbols: dense_bindings.append(runtime_symbols),
    )
    monkeypatch.setattr(
        rocm_cdna2_int8_moe,
        "bind_vllm_moe_symbols",
        lambda *runtime_symbols: moe_bindings.append(runtime_symbols),
    )

    installed = abi.install_vllm_kernel_abi()

    assert installed is validated
    assert dense_bindings == [(symbols[0], symbols[1])]
    assert moe_bindings == [(symbols[2], symbols[1], symbols[3], symbols[4])]
