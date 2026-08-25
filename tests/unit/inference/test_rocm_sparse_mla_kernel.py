from types import SimpleNamespace

import pytest
import torch

from prime_rl.inference.vllm.kernels import rocm_sparse_mla_triton
from prime_rl.inference.vllm.kernels.rocm_sparse_mla import (
    _flatten_kv_cache,
    _unpad_repeated_heads,
    rocm_sparse_mla_bf16,
)


def test_sparse_mla_direct_launch_rejects_non_gfx90a(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.1")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(gcnArchName="gfx942:sramecc+:xnack-"),
    )
    rocm_sparse_mla_triton._require_gfx90a.cache_clear()
    q = SimpleNamespace(device=torch.device("cuda", 0))

    with pytest.raises(RuntimeError, match="validated only on gfx90a"):
        rocm_sparse_mla_triton.launch_sparse_mla_bf16(
            q,
            None,
            None,
            None,
            0.125,
        )


def test_flatten_kv_cache_preserves_flat_and_paged_storage():
    flat = torch.empty((8, 576), dtype=torch.bfloat16)
    paged = torch.empty((2, 4, 576), dtype=torch.bfloat16)

    flat_view = _flatten_kv_cache(flat)
    paged_view = _flatten_kv_cache(paged)

    assert flat_view.shape == (8, 576)
    assert paged_view.shape == (8, 576)
    assert flat_view.data_ptr() == flat.data_ptr()
    assert paged_view.data_ptr() == paged.data_ptr()


def test_unpad_repeated_heads_matches_repeat_interleave_inverse():
    source = torch.arange(2 * 8 * 576, dtype=torch.float32).view(2, 8, 576)
    padded = source.repeat_interleave(2, dim=1)

    output = _unpad_repeated_heads(padded, num_heads=8)

    torch.testing.assert_close(output, source[..., :512])
    assert output.is_contiguous()


@pytest.mark.parametrize(
    ("q", "kv", "indices", "indptr", "error", "match"),
    [
        (
            torch.empty((2, 8, 575), dtype=torch.bfloat16),
            torch.empty((8, 576), dtype=torch.bfloat16),
            torch.empty(0, dtype=torch.int32),
            torch.zeros(3, dtype=torch.int32),
            ValueError,
            r"q must have shape \[tokens, heads, 576\]",
        ),
        (
            torch.empty((2, 8, 576), dtype=torch.float32),
            torch.empty((8, 576), dtype=torch.bfloat16),
            torch.empty(0, dtype=torch.int32),
            torch.zeros(3, dtype=torch.int32),
            TypeError,
            "q must have dtype torch.bfloat16",
        ),
        (
            torch.empty((2, 8, 576), dtype=torch.bfloat16),
            torch.empty((8, 576), dtype=torch.bfloat16),
            torch.empty(0, dtype=torch.int64),
            torch.zeros(3, dtype=torch.int32),
            TypeError,
            "selected_indices must have dtype torch.int32",
        ),
        (
            torch.empty((2, 8, 576), dtype=torch.bfloat16),
            torch.empty((8, 576), dtype=torch.bfloat16),
            torch.empty(0, dtype=torch.int32),
            torch.zeros(2, dtype=torch.int32),
            ValueError,
            "selected_indptr must contain 3 entries",
        ),
    ],
)
def test_sparse_mla_rejects_invalid_contract_before_runtime_import(q, kv, indices, indptr, error, match):
    with pytest.raises(error, match=match):
        rocm_sparse_mla_bf16(q, kv, indices, indptr, 0.125)


def test_sparse_mla_rejects_cpu_tensors_deterministically():
    q = torch.empty((2, 8, 576), dtype=torch.bfloat16)
    kv = torch.empty((8, 576), dtype=torch.bfloat16)
    indices = torch.tensor([0, -1], dtype=torch.int32)
    indptr = torch.tensor([0, 1, 2], dtype=torch.int32)

    with pytest.raises(ValueError, match="must be on a ROCm device"):
        rocm_sparse_mla_bf16(q, kv, indices, indptr, 0.125)


def _is_gfx90a() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return getattr(properties, "gcnArchName", "").split(":")[0] == "gfx90a"


@pytest.mark.skipif(not _is_gfx90a(), reason="requires a gfx90a ROCm GPU")
@pytest.mark.parametrize(
    "indptr",
    (
        [-1, 1],
        [1, 0],
        [0, 3],
    ),
)
def test_sparse_mla_malformed_indptr_is_memory_safe_and_empty(indptr):
    q = torch.randn((1, 16, 576), device="cuda", dtype=torch.bfloat16)
    kv = torch.randn((2, 576), device="cuda", dtype=torch.bfloat16)
    indices = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    row_ptr = torch.tensor(indptr, device="cuda", dtype=torch.int32)

    output = rocm_sparse_mla_bf16(
        q,
        kv,
        indices,
        row_ptr,
        0.125,
        num_heads=16,
    )

    torch.cuda.synchronize()
    assert torch.count_nonzero(output).item() == 0


@pytest.mark.skipif(not _is_gfx90a(), reason="requires a gfx90a ROCm GPU")
def test_sparse_mla_overlong_row_is_empty_instead_of_truncated():
    q = torch.randn((1, 16, 576), device="cuda", dtype=torch.bfloat16)
    kv = torch.randn((2, 576), device="cuda", dtype=torch.bfloat16)
    indices = torch.arange(2049, device="cuda", dtype=torch.int32) % kv.shape[0]
    indptr = torch.tensor([0, 2049], device="cuda", dtype=torch.int32)

    output = rocm_sparse_mla_bf16(
        q,
        kv,
        indices,
        indptr,
        0.125,
        num_heads=16,
    )

    torch.cuda.synchronize()
    assert torch.count_nonzero(output).item() == 0
