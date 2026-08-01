# SPDX-License-Identifier: Apache-2.0

"""Bounded-memory BF16 sparse MLA for ROCm.

The Triton implementation adapts vLLM's Apache-2.0 sparse-MLA online softmax
at commit ``ee0da84ab9e04ac7610e28580af62c365e898389``. It separates the
512-wide value accumulator from GLM's 512 NoPE + 64 RoPE score dimension and
tiles that accumulator so it remains tractable on gfx90a.
"""

import math

import torch

_GLM_NOPE_DIM = 512
_GLM_ROPE_DIM = 64
_GLM_HEAD_DIM = _GLM_NOPE_DIM + _GLM_ROPE_DIM
_GLM_VALUE_DIM = _GLM_NOPE_DIM


def _require_tensor(name: str, value: object) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    return value


def _flatten_kv_cache(kv_cache: torch.Tensor) -> torch.Tensor:
    """Return a zero-copy ``[tokens, head_dim]`` view of flat or paged KV."""
    if kv_cache.ndim == 2:
        if kv_cache.shape[1] != _GLM_HEAD_DIM:
            raise ValueError(f"kv_cache must have shape [tokens, {_GLM_HEAD_DIM}], got {tuple(kv_cache.shape)}")
        return kv_cache
    if kv_cache.ndim == 3:
        if kv_cache.shape[2] != _GLM_HEAD_DIM:
            raise ValueError(
                "kv_cache must have shape "
                f"[tokens, 1, {_GLM_HEAD_DIM}] or [blocks, block_size, {_GLM_HEAD_DIM}], "
                f"got {tuple(kv_cache.shape)}"
            )
        return kv_cache.view(-1, _GLM_HEAD_DIM)
    raise ValueError(
        "kv_cache must have shape "
        f"[tokens, {_GLM_HEAD_DIM}], [tokens, 1, {_GLM_HEAD_DIM}], "
        f"or [blocks, block_size, {_GLM_HEAD_DIM}], got {tuple(kv_cache.shape)}"
    )


def _unpad_repeated_heads(output: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Undo vLLM MLA's ``repeat_interleave`` query-head padding."""
    padded_heads = output.shape[1]
    if not 1 <= num_heads <= padded_heads:
        raise ValueError(f"num_heads must be in [1, {padded_heads}], got {num_heads}")
    if padded_heads % num_heads != 0:
        raise ValueError(f"padded query heads ({padded_heads}) must be divisible by num_heads ({num_heads})")
    repeat = padded_heads // num_heads
    return output[:, ::repeat, :_GLM_VALUE_DIM].contiguous()


def _validate_inputs(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    selected_indices: torch.Tensor,
    selected_indptr: torch.Tensor,
    sm_scale: float,
    num_heads: int,
) -> torch.Tensor:
    if q.ndim != 3 or q.shape[2] != _GLM_HEAD_DIM:
        raise ValueError(f"q must have shape [tokens, heads, {_GLM_HEAD_DIM}], got {tuple(q.shape)}")
    if q.shape[1] < 1:
        raise ValueError("q must contain at least one query head")
    if q.dtype != torch.bfloat16:
        raise TypeError(f"q must have dtype torch.bfloat16, got {q.dtype}")
    if kv_cache.dtype != torch.bfloat16:
        raise TypeError(f"kv_cache must have dtype torch.bfloat16, got {kv_cache.dtype}")
    if selected_indices.dtype != torch.int32:
        raise TypeError(f"selected_indices must have dtype torch.int32, got {selected_indices.dtype}")
    if selected_indptr.dtype != torch.int32:
        raise TypeError(f"selected_indptr must have dtype torch.int32, got {selected_indptr.dtype}")
    if not q.is_contiguous():
        raise ValueError("q must be contiguous")
    if not kv_cache.is_contiguous():
        raise ValueError("kv_cache must be contiguous")
    if selected_indices.ndim != 1 or not selected_indices.is_contiguous():
        raise ValueError("selected_indices must be a contiguous one-dimensional ragged values tensor")
    if selected_indptr.ndim != 1 or not selected_indptr.is_contiguous():
        raise ValueError("selected_indptr must be a contiguous one-dimensional row-pointer tensor")
    if selected_indptr.numel() != q.shape[0] + 1:
        raise ValueError(f"selected_indptr must contain {q.shape[0] + 1} entries, got {selected_indptr.numel()}")
    tensors = {
        "kv_cache": kv_cache,
        "selected_indices": selected_indices,
        "selected_indptr": selected_indptr,
    }
    for name, tensor in tensors.items():
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on {q.device}, got {tensor.device}")
    if q.device.type != "cuda":
        raise ValueError(f"q, kv_cache, and sparse indices must be on a ROCm device, got {q.device}")
    if not math.isfinite(sm_scale) or sm_scale <= 0.0:
        raise ValueError(f"sm_scale must be finite and positive, got {sm_scale}")
    if not 1 <= num_heads <= q.shape[1]:
        raise ValueError(f"num_heads must be in [1, {q.shape[1]}], got {num_heads}")
    if q.shape[1] % num_heads != 0:
        raise ValueError(f"padded query heads ({q.shape[1]}) must be divisible by num_heads ({num_heads})")
    return _flatten_kv_cache(kv_cache)


def rocm_sparse_mla_bf16(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    selected_indices: torch.Tensor,
    selected_indptr: torch.Tensor,
    sm_scale: float,
    *,
    num_heads: int | None = None,
) -> torch.Tensor:
    """Compute GLM sparse MLA from ragged selected-token indices.

    Args:
        q: Contiguous BF16 query tensor ``[M, padded_heads, 576]``.
        kv_cache: Contiguous BF16 flat ``[tokens, 576]`` / ``[tokens, 1, 576]``
            or paged ``[blocks, block_size, 576]`` cache.
        selected_indices: Contiguous int32 ragged values ``[nnz]``. ``-1`` and
            out-of-range entries are ignored by the kernel.
        selected_indptr: Contiguous int32 row pointers ``[M + 1]``. Each row
            contains at most GLM's configured 2048 selected-token entries.
            Malformed negative, decreasing, or out-of-bounds rows are treated
            as empty without dereferencing outside ``selected_indices``.
        sm_scale: Positive finite softmax scale.
        num_heads: Unpadded query-head count. If q was formed with
            ``repeat_interleave``, the corresponding repeated output heads are
            removed. Defaults to ``q.shape[1]``.

    Returns:
        A contiguous BF16 tensor ``[M, num_heads, 512]``.

    The kernel streams selected KV rows. Its temporary storage is one int32
    ``[M, 2048]`` index tensor; it does not construct an
    ``M * topk * 576`` KV gather, synchronize with the host, or mutate inputs.
    """
    q = _require_tensor("q", q)
    kv_cache = _require_tensor("kv_cache", kv_cache)
    selected_indices = _require_tensor("selected_indices", selected_indices)
    selected_indptr = _require_tensor("selected_indptr", selected_indptr)
    actual_heads = q.shape[1] if num_heads is None and q.ndim == 3 else num_heads
    if not isinstance(actual_heads, int):
        raise TypeError("num_heads must be an integer")
    flat_kv = _validate_inputs(
        q,
        kv_cache,
        selected_indices,
        selected_indptr,
        sm_scale,
        actual_heads,
    )

    if q.shape[0] == 0:
        return torch.empty((0, actual_heads, _GLM_VALUE_DIM), dtype=q.dtype, device=q.device)

    try:
        from prime_rl.inference.vllm.kernels.rocm_sparse_mla_triton import (
            launch_sparse_mla_bf16,
        )
    except ImportError as exc:
        raise RuntimeError("rocm_sparse_mla_bf16 requires vLLM's Triton runtime") from exc

    padded_output = launch_sparse_mla_bf16(
        q,
        flat_kv,
        selected_indices,
        selected_indptr,
        sm_scale,
    )
    return _unpad_repeated_heads(padded_output, actual_heads)
