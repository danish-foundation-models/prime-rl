# SPDX-License-Identifier: Apache-2.0

import inspect
import os
import sys
from pathlib import Path

import torch

from prime_rl.inference.vllm.kernels.rocm_hybrid_paged_attention import (
    _require_gfx90a,
)

_INSTALL_MARKER = "_prime_gfx90a_aiter_attention_v1"
_EXPECTED_EXTEND_PARAMETERS = (
    "self",
    "attn_metadata",
    "query",
    "key",
    "value",
    "key_cache",
    "value_cache",
    "output",
    "cu_seqlens_q",
    "max_seqlen_q",
    "min_seqlen_q",
    "max_seqlen_k",
    "block_table",
    "slot_mapping",
    "k_scale",
    "v_scale",
)
def _strict_one(name: str) -> None:
    if os.environ.get(name) != "1":
        raise RuntimeError(f"PRIME_ROCM_GFX90A_AITER_ATTN=1 requires {name}=1")


def _prepare_aiter_source() -> Path:
    source_value = os.environ.get("PRIME_AITER_SOURCE_DIR")
    if not source_value:
        raise RuntimeError("PRIME_ROCM_GFX90A_AITER_ATTN=1 requires PRIME_AITER_SOURCE_DIR")
    source_dir = Path(source_value)
    if not source_dir.is_absolute():
        raise RuntimeError("PRIME_AITER_SOURCE_DIR must be an absolute path")
    expected = (
        source_dir
        / "3rdparty"
        / "composable_kernel"
        / "example"
        / "ck_tile"
        / "01_fmha"
        / "codegen"
        / "ops"
        / "fmha_batch_prefill.py"
    )
    if not expected.is_file():
        raise RuntimeError("PRIME_AITER_SOURCE_DIR lacks the CK batch-prefill generator")
    cache_dir = os.environ.get("AITER_ROOT_DIR")
    if not cache_dir or not Path(cache_dir).is_absolute():
        raise RuntimeError("PRIME_ROCM_GFX90A_AITER_ATTN=1 requires an absolute AITER_ROOT_DIR")
    if os.environ.get("GPU_ARCHS") != "gfx90a":
        raise RuntimeError("PRIME_ROCM_GFX90A_AITER_ATTN=1 requires GPU_ARCHS=gfx90a")
    source_string = str(source_dir)
    if source_string not in sys.path:
        sys.path.insert(0, source_string)
    return source_dir


def _can_use_batch_prefill(
    self,
    attn_metadata,
    query,
    key_cache,
    value_cache,
    output,
    cu_seqlens_q,
    max_seqlen_q,
    block_table,
) -> bool:
    return (
        self.num_heads == 4
        and self.num_kv_heads == 1
        and self.head_size == 128
        and self.kv_cache_dtype == "auto"
        and self.alibi_slopes is None
        and self.sliding_window == (-1, -1)
        and self.logits_soft_cap == 0.0
        and self.sinks is None
        and attn_metadata.causal
        and query.dtype == torch.bfloat16
        and output.dtype == torch.bfloat16
        and key_cache.dtype == torch.bfloat16
        and value_cache.dtype == torch.bfloat16
        and query.ndim == 3
        and tuple(query.shape[1:]) == (4, 128)
        and output.shape == query.shape
        and key_cache.ndim == 4
        and value_cache.shape == key_cache.shape
        and tuple(key_cache.shape[1:]) == (2048, 1, 128)
        and cu_seqlens_q.dtype == torch.int32
        and cu_seqlens_q.ndim == 1
        and block_table.dtype == torch.int32
        and block_table.ndim == 2
        and block_table.stride(-1) == 1
        and 2 <= max_seqlen_q <= 32768
    )


def install_gfx90a_aiter_attention() -> None:
    _prepare_aiter_source()
    _strict_one("VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT")
    _require_gfx90a(torch.cuda.current_device())

    import aiter
    from vllm.v1.attention.backends import rocm_aiter_fa

    backend = rocm_aiter_fa.AiterFlashAttentionBackend
    impl = rocm_aiter_fa.AiterFlashAttentionImpl
    if getattr(backend, _INSTALL_MARKER, False):
        return

    actual = tuple(inspect.signature(impl.extend_forward).parameters)
    if actual != _EXPECTED_EXTEND_PARAMETERS:
        raise RuntimeError(
            "PRIME_ROCM_GFX90A_AITER_ATTN requires "
            f"AiterFlashAttentionImpl.extend_forward{_EXPECTED_EXTEND_PARAMETERS}; "
            f"found {actual}"
        )
    if not hasattr(aiter, "mha_batch_prefill_func"):
        raise RuntimeError("AITER lacks mha_batch_prefill_func")

    original_extend = impl.extend_forward

    @classmethod
    def _shuffle_cache_enabled(cls) -> bool:
        del cls
        return True

    @staticmethod
    def _supported_block_sizes() -> list[int]:
        return [2048]

    @classmethod
    def _supports_compute_capability(cls, capability) -> bool:
        del cls, capability
        return True

    def _extend_forward(
        self,
        attn_metadata,
        query,
        key,
        value,
        key_cache,
        value_cache,
        output,
        cu_seqlens_q,
        max_seqlen_q,
        min_seqlen_q,
        max_seqlen_k,
        block_table,
        slot_mapping,
        k_scale,
        v_scale,
    ):
        if not _can_use_batch_prefill(
            self,
            attn_metadata,
            query,
            key_cache,
            value_cache,
            output,
            cu_seqlens_q,
            max_seqlen_q,
            block_table,
        ):
            return original_extend(
                self,
                attn_metadata,
                query,
                key,
                value,
                key_cache,
                value_cache,
                output,
                cu_seqlens_q,
                max_seqlen_q,
                min_seqlen_q,
                max_seqlen_k,
                block_table,
                slot_mapping,
                k_scale,
                v_scale,
            )

        del key, value, min_seqlen_q, slot_mapping, k_scale, v_scale
        vector_size = 16 // key_cache.element_size()
        num_blocks, block_size, num_kv_heads, head_size = key_cache.shape
        vector_key_cache = key_cache.reshape(
            num_blocks,
            num_kv_heads,
            head_size // vector_size,
            block_size,
            vector_size,
        )
        vector_value_cache = value_cache.reshape(
            num_blocks,
            num_kv_heads,
            block_size // vector_size,
            head_size,
            vector_size,
        )
        start = attn_metadata.num_decodes
        stop = start + attn_metadata.num_extends
        seq_lens = attn_metadata.seq_lens[start:stop]
        aiter.mha_batch_prefill_func(
            query,
            vector_key_cache,
            vector_value_cache,
            cu_seqlens_q,
            cu_seqlens_q,
            block_table,
            max_seqlen_q,
            max_seqlen_k,
            causal=True,
            out=output,
            block_table=block_table,
            seqlen_k=seq_lens,
        )

    rocm_aiter_fa.rocm_aiter_ops.is_shuffle_kv_cache_enabled = _shuffle_cache_enabled
    backend.get_supported_kernel_block_sizes = _supported_block_sizes
    backend.supports_compute_capability = _supports_compute_capability
    impl.extend_forward = _extend_forward
    setattr(backend, _INSTALL_MARKER, True)
