# SPDX-License-Identifier: Apache-2.0

"""Fail-closed vLLM ABI boundary for the gfx90a INT8 kernels."""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import version as distribution_version
from typing import Any

DEFAULT_VLLM_DISTRIBUTION = "0.25.0.dev0+g27d0e3ef3.gfx90a"
EXPECTED_VLLM_DISTRIBUTION = os.environ.get(
    "PRIME_CDNA2_EXPECTED_VLLM_DISTRIBUTION",
    DEFAULT_VLLM_DISTRIBUTION,
)
EXPECTED_DENSE_KERNEL_ARGS = (
    "A",
    "B",
    "C",
    "As",
    "Bs",
    "M",
    "N",
    "K",
    "group_n",
    "group_k",
    "stride_am",
    "stride_ak",
    "stride_bk",
    "stride_bn",
    "stride_cm",
    "stride_cn",
    "stride_As_m",
    "stride_As_k",
    "stride_Bs_k",
    "stride_Bs_n",
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_N",
    "BLOCK_SIZE_K",
    "GROUP_SIZE_M",
)
EXPECTED_ACTIVATION_QUANTIZER_ARGS = (
    "x",
    "group_size",
    "eps",
    "dtype",
)
EXPECTED_FUSED_MOE_KERNEL_ARGS = (
    "a_ptr",
    "b_ptr",
    "c_ptr",
    "b_bias_ptr",
    "a_scale_ptr",
    "b_scale_ptr",
    "topk_weights_ptr",
    "sorted_token_ids_ptr",
    "expert_ids_ptr",
    "num_tokens_post_padded_ptr",
    "N",
    "K",
    "EM",
    "num_valid_tokens",
    "stride_am",
    "stride_ak",
    "stride_be",
    "stride_bk",
    "stride_bn",
    "stride_cm",
    "stride_cn",
    "stride_asm",
    "stride_ask",
    "stride_bse",
    "stride_bsk",
    "stride_bsn",
    "stride_bbe",
    "stride_bbn",
    "group_n",
    "group_k",
    "naive_block_assignment",
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_N",
    "BLOCK_SIZE_K",
    "GROUP_SIZE_M",
    "SPLIT_K",
    "MUL_ROUTED_WEIGHT",
    "top_k",
    "compute_type",
    "use_fp8_w8a8",
    "use_int8_w8a8",
    "use_int8_w8a16",
    "per_channel_quant",
    "HAS_BIAS",
    "SWAP_AB",
)
EXPECTED_MOE_ALIGN_ARGS = (
    "topk_ids",
    "block_size",
    "num_experts",
    "expert_map",
    "pad_sorted_ids",
    "ignore_invalid_experts",
)
EXPECTED_MOE_SUM_ARGS = (
    "input",
    "output",
    "topk_ids",
    "expert_map",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _argument_names(function: Any) -> tuple[str, ...]:
    names = getattr(function, "arg_names", None)
    if names is not None:
        return tuple(names)
    return tuple(inspect.signature(getattr(function, "fn", function)).parameters)


def _load_vllm_symbols() -> tuple[Any, Any, Any, Any, Any]:
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        fused_moe_kernel,
    )
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )
    from vllm.model_executor.layers.quantization.utils.int8_utils import (
        _w8a8_block_int8_matmul,
        per_token_group_quant_int8,
    )

    return (
        _w8a8_block_int8_matmul,
        per_token_group_quant_int8,
        fused_moe_kernel,
        moe_align_block_size,
        ops.moe_sum,
    )


@dataclass(frozen=True)
class VllmKernelAbi:
    dense_kernel: Any
    activation_quantizer: Any
    fused_moe_kernel: Any
    moe_align_block_size: Any
    moe_sum: Any


@lru_cache(maxsize=1)
def validate_vllm_kernel_abi() -> VllmKernelAbi:
    """Load private symbols once after proving their exact distribution ABI."""

    actual_version = distribution_version("vllm")
    _require(
        actual_version == EXPECTED_VLLM_DISTRIBUTION,
        f"CDNA2 INT8 requires vLLM distribution {EXPECTED_VLLM_DISTRIBUTION!r}; got {actual_version!r}",
    )
    dense, quantizer, fused_moe, moe_align, moe_sum = _load_vllm_symbols()
    dense_args = _argument_names(dense)
    quantizer_args = _argument_names(quantizer)
    fused_moe_args = _argument_names(fused_moe)
    moe_align_args = _argument_names(moe_align)
    moe_sum_args = _argument_names(moe_sum)
    _require(
        dense_args == EXPECTED_DENSE_KERNEL_ARGS,
        f"vLLM dense INT8 kernel ABI changed: expected {EXPECTED_DENSE_KERNEL_ARGS}, got {dense_args}",
    )
    _require(
        quantizer_args == EXPECTED_ACTIVATION_QUANTIZER_ARGS,
        "vLLM INT8 activation quantizer ABI changed: "
        f"expected {EXPECTED_ACTIVATION_QUANTIZER_ARGS}, got {quantizer_args}",
    )
    _require(
        fused_moe_args == EXPECTED_FUSED_MOE_KERNEL_ARGS,
        f"vLLM fused MoE kernel ABI changed: expected {EXPECTED_FUSED_MOE_KERNEL_ARGS}, got {fused_moe_args}",
    )
    _require(
        moe_align_args == EXPECTED_MOE_ALIGN_ARGS,
        f"vLLM moe_align_block_size ABI changed: expected {EXPECTED_MOE_ALIGN_ARGS}, got {moe_align_args}",
    )
    _require(
        moe_sum_args == EXPECTED_MOE_SUM_ARGS,
        f"vLLM moe_sum ABI changed: expected {EXPECTED_MOE_SUM_ARGS}, got {moe_sum_args}",
    )
    return VllmKernelAbi(
        dense_kernel=dense,
        activation_quantizer=quantizer,
        fused_moe_kernel=fused_moe,
        moe_align_block_size=moe_align,
        moe_sum=moe_sum,
    )


def install_vllm_kernel_abi() -> VllmKernelAbi:
    """Validate once and bind symbols outside the inference hot paths."""

    abi = validate_vllm_kernel_abi()
    from prime_rl.inference.vllm.kernels import (
        rocm_cdna2_int8_linear,
        rocm_cdna2_int8_moe,
    )

    rocm_cdna2_int8_linear.bind_vllm_dense_int8_symbols(
        abi.dense_kernel,
        abi.activation_quantizer,
    )
    rocm_cdna2_int8_moe.bind_vllm_moe_symbols(
        abi.fused_moe_kernel,
        abi.activation_quantizer,
        abi.moe_align_block_size,
        abi.moe_sum,
    )
    return abi
