# SPDX-License-Identifier: Apache-2.0

import functools
import inspect

import torch
from vllm.triton_utils import tl, triton

_DISPATCH_MARKER = "_prime_rl_gfx90a_nemotron_relu2"
_INTERMEDIATE = 2688
_MIN_ROWS = 5632
_MAX_ROWS = 32768 * 22
_BLOCK = 1024
_EP_PROGRAMS = 8192


@functools.lru_cache(maxsize=None)
def _require_gfx90a(device_index: int) -> None:
    if torch.version.hip is None:
        raise RuntimeError("Nemotron ReLU-squared requires a ROCm PyTorch build")
    properties = torch.cuda.get_device_properties(device_index)
    architecture = getattr(properties, "gcnArchName", "").split(":")[0]
    if architecture != "gfx90a":
        raise RuntimeError(
            f"Nemotron ReLU-squared is validated only on gfx90a; device {device_index} reports {properties.gcnArchName}"
        )


@triton.jit
def _nemotron_relu2_inplace_kernel(
    input_ptr,
    output_ptr,
    elements,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < elements
    values = tl.load(input_ptr + offsets, mask=mask)
    activated = tl.maximum(values, 0.0)
    tl.store(input_ptr + offsets, activated, mask=mask)
    tl.store(output_ptr + offsets, activated * activated, mask=mask)


@triton.jit
def _nemotron_ep_relu2_inplace_kernel(
    input_ptr,
    output_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    total_rows,
    width: tl.constexpr,
    block_size_m: tl.constexpr,
    block_n: tl.constexpr,
    num_column_blocks: tl.constexpr,
):
    task = tl.program_id(0)
    task_count = tl.load(num_tokens_post_padded_ptr) * num_column_blocks
    stride = tl.num_programs(0)

    while task < task_count:
        sorted_position = task // num_column_blocks
        column_block = task % num_column_blocks
        expert = tl.load(expert_ids_ptr + sorted_position // block_size_m)

        if expert >= 0:
            row = tl.load(sorted_token_ids_ptr + sorted_position)
            columns = column_block * block_n + tl.arange(0, block_n)
            mask = (row < total_rows) & (columns < width)
            offsets = row * width + columns
            values = tl.load(input_ptr + offsets, mask=mask)
            activated = tl.maximum(values, 0.0)
            tl.store(input_ptr + offsets, activated, mask=mask)
            tl.store(output_ptr + offsets, activated * activated, mask=mask)

        task += stride


def can_dispatch_nemotron_relu2(activation, output, input_tensor) -> bool:
    if getattr(activation, "value", None) != "relu2_no_mul":
        return False
    if not isinstance(output, torch.Tensor) or not isinstance(input_tensor, torch.Tensor):
        return False
    if (
        input_tensor.device.type != "cuda"
        or output.device != input_tensor.device
        or input_tensor.dtype != torch.bfloat16
        or output.dtype != torch.bfloat16
        or input_tensor.ndim != 2
        or output.shape != input_tensor.shape
        or not input_tensor.is_contiguous()
        or not output.is_contiguous()
        or input_tensor.shape[1] != _INTERMEDIATE
        or not _MIN_ROWS <= input_tensor.shape[0] <= _MAX_ROWS
    ):
        return False
    return True


def can_dispatch_nemotron_ep_relu2(
    activation,
    output,
    input_tensor,
    *,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    block_size_m,
) -> bool:
    if not can_dispatch_nemotron_relu2(activation, output, input_tensor):
        return False
    if not isinstance(block_size_m, int) or block_size_m <= 0:
        return False
    for tensor in (sorted_token_ids, expert_ids, num_tokens_post_padded):
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device != input_tensor.device
            or tensor.dtype != torch.int32
            or not tensor.is_contiguous()
        ):
            return False
    return sorted_token_ids.ndim == 1 and expert_ids.ndim == 1 and num_tokens_post_padded.shape == (1,)


def nemotron_relu2_gfx90a(output: torch.Tensor, input_tensor: torch.Tensor) -> None:
    _require_gfx90a(input_tensor.device.index)
    elements = input_tensor.numel()
    _nemotron_relu2_inplace_kernel[(triton.cdiv(elements, _BLOCK),)](
        input_tensor,
        output,
        elements,
        BLOCK=_BLOCK,
        num_warps=4,
    )


def nemotron_ep_relu2_gfx90a(
    output: torch.Tensor,
    input_tensor: torch.Tensor,
    *,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    block_size_m: int,
) -> None:
    _require_gfx90a(input_tensor.device.index)
    column_blocks = triton.cdiv(_INTERMEDIATE, _BLOCK)
    max_tasks = sorted_token_ids.numel() * column_blocks
    programs = min(_EP_PROGRAMS, max_tasks)
    _nemotron_ep_relu2_inplace_kernel[(programs,)](
        input_tensor,
        output,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        input_tensor.shape[0],
        width=_INTERMEDIATE,
        block_size_m=block_size_m,
        block_n=_BLOCK,
        num_column_blocks=column_blocks,
        num_warps=4,
    )


def wrap_nemotron_experts_activation(original):
    if getattr(original, _DISPATCH_MARKER, False):
        return original

    signature = inspect.signature(original)

    @functools.wraps(original)
    def wrapped(self, activation, output, input, **kwargs):
        assignment = {
            name: kwargs.get(name)
            for name in (
                "sorted_token_ids",
                "expert_ids",
                "num_tokens_post_padded",
                "block_size_m",
            )
        }
        if can_dispatch_nemotron_ep_relu2(
            activation,
            output,
            input,
            **assignment,
        ):
            nemotron_ep_relu2_gfx90a(output, input, **assignment)
            return None
        if can_dispatch_nemotron_relu2(activation, output, input):
            nemotron_relu2_gfx90a(output, input)
            return None
        return original(self, activation, output, input, **kwargs)

    if inspect.signature(wrapped) != signature:
        raise RuntimeError("Nemotron ReLU-squared wrapper changed the vLLM activation signature")
    setattr(wrapped, _DISPATCH_MARKER, True)
    return wrapped
