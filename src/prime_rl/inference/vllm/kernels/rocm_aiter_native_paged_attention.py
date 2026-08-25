# SPDX-License-Identifier: Apache-2.0

import functools
import importlib
import inspect
import math
import os
import sys
from pathlib import Path

import torch

_PARTITION_SIZE = 256
_CUSTOM_OP_NAME = "prime_gfx90a_aiter_paged_attention"
_CUSTOM_OP_REGISTERED = False
_EXPECTED_PARAMETERS = (
    "out",
    "exp_sums",
    "max_logits",
    "tmp_out",
    "query",
    "key_cache",
    "value_cache",
    "num_kv_heads",
    "scale",
    "block_tables",
    "context_lens",
    "block_size",
    "max_context_len",
    "alibi_slopes",
    "kv_cache_dtype",
    "key_scale",
    "value_scale",
    "fp8_out_scale",
    "partition_size",
    "mtp",
    "query_scale",
)


@functools.lru_cache(maxsize=1)
def _load_paged_attention_rocm():
    source_dir_value = os.environ.get("PRIME_AITER_SOURCE_DIR")
    if not source_dir_value:
        raise RuntimeError(
            "PRIME_ROCM_GFX90A_AITER_PAGED_ATTN=1 requires PRIME_AITER_SOURCE_DIR to name an AITER source checkout"
        )
    source_dir = Path(source_dir_value)
    if not source_dir.is_absolute():
        raise RuntimeError("PRIME_AITER_SOURCE_DIR must be an absolute path")
    expected_source = source_dir / "csrc" / "cpp_itfs" / "pa" / "pa.py"
    if not expected_source.is_file():
        raise RuntimeError(f"PRIME_AITER_SOURCE_DIR does not contain {expected_source.relative_to(source_dir)}")
    cache_dir = os.environ.get("AITER_ROOT_DIR")
    if not cache_dir or not Path(cache_dir).is_absolute():
        raise RuntimeError(
            "PRIME_ROCM_GFX90A_AITER_PAGED_ATTN=1 requires an absolute "
            "AITER_ROOT_DIR so JIT artifacts stay in the bench cache"
        )
    if os.environ.get("GPU_ARCHS") != "gfx90a":
        raise RuntimeError("PRIME_ROCM_GFX90A_AITER_PAGED_ATTN=1 requires GPU_ARCHS=gfx90a")

    source_dir_string = str(source_dir)
    if source_dir_string not in sys.path:
        sys.path.insert(0, source_dir_string)
    module = importlib.import_module("csrc.cpp_itfs.pa.pa")
    function = module.paged_attention_rocm
    actual_parameters = tuple(inspect.signature(function).parameters)
    if actual_parameters != _EXPECTED_PARAMETERS:
        raise RuntimeError(
            "AITER paged_attention_rocm has an unsupported signature: "
            f"expected {_EXPECTED_PARAMETERS}, found {actual_parameters}"
        )
    return function


def _launch_aiter_native_paged_attention(
    query,
    output,
    key_cache,
    value_cache,
    block_table,
    seq_lens,
    max_seq_len,
    sm_scale,
) -> None:
    paged_attention_rocm = _load_paged_attention_rocm()
    max_partitions = math.ceil(max_seq_len / _PARTITION_SIZE)
    scalar_shape = (query.shape[0], query.shape[1], max_partitions)
    exp_sums = torch.empty(scalar_shape, device=query.device, dtype=torch.float32)
    max_logits = torch.empty_like(exp_sums)
    tmp_out = torch.empty(
        (*scalar_shape, query.shape[2]),
        device=query.device,
        dtype=query.dtype,
    )
    paged_attention_rocm(
        out=output,
        exp_sums=exp_sums,
        max_logits=max_logits,
        tmp_out=tmp_out,
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        num_kv_heads=key_cache.shape[1],
        scale=sm_scale if sm_scale is not None else query.shape[2] ** -0.5,
        block_tables=block_table,
        context_lens=seq_lens,
        block_size=key_cache.shape[3],
        max_context_len=max_seq_len,
        alibi_slopes=None,
        kv_cache_dtype="auto",
        partition_size=_PARTITION_SIZE,
    )


def _custom_op_impl(
    query: torch.Tensor,
    output: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    sm_scale: float,
) -> None:
    _launch_aiter_native_paged_attention(
        query=query,
        output=output,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        sm_scale=sm_scale,
    )


def _custom_op_fake(
    query: torch.Tensor,
    output: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    sm_scale: float,
) -> None:
    del (
        query,
        output,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        max_seq_len,
        sm_scale,
    )


def register_aiter_native_paged_attention_op() -> None:
    global _CUSTOM_OP_REGISTERED

    if _CUSTOM_OP_REGISTERED:
        return
    from vllm.platforms import current_platform
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name=_CUSTOM_OP_NAME,
        op_func=_custom_op_impl,
        mutates_args=["output"],
        fake_impl=_custom_op_fake,
        dispatch_key=current_platform.dispatch_key,
    )
    _CUSTOM_OP_REGISTERED = True


def aiter_native_paged_attention_gfx90a(
    query,
    output,
    key_cache,
    value_cache,
    block_table,
    seq_lens,
    max_seq_len,
    sm_scale,
) -> None:
    register_aiter_native_paged_attention_op()
    scale = sm_scale if sm_scale is not None else query.shape[2] ** -0.5
    torch.ops.vllm.prime_gfx90a_aiter_paged_attention(
        query,
        output,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        max_seq_len,
        scale,
    )
