# SPDX-License-Identifier: Apache-2.0

"""CDNA2 fused gated-SiLU and fine-group INT8 activation quantization.

vLLM 0.24's ROCm fused ``silu_and_mul_per_block_quant`` kernel accepts the
coarser FP8 group sizes used by its existing MoE path, but rejects the
32-element K groups selected for native gfx90a INT8 MFMA. This kernel performs
the same gate/up activation and emits per-row, per-32-element INT8 scales in a
single pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from vllm.triton_utils import tl, triton

from prime_rl.inference.vllm.kernels.rocm_cdna2_int8_linear import (
    GFX90A_INT8_TOKEN_BUCKETS,
    PREFILL_MAX_TOKENS,
    RUNTIME_WEIGHT_SCALE_DTYPE,
    TARGET_GROUP_K,
    TARGET_GROUP_N,
    transcode_ocp_fp8_weight_batch,
    validate_gfx90a_device,
)

_VLLM_FUSED_MOE_KERNEL = None
_VLLM_ACTIVATION_QUANTIZER = None
_VLLM_MOE_ALIGN_BLOCK_SIZE = None
_VLLM_MOE_SUM = None


@dataclass(frozen=True)
class Gfx90aInt8MoeConfig:
    block_m: int
    block_n: int
    block_k: int = TARGET_GROUP_K
    group_size_m: int = 1
    split_k: int = 1
    num_warps: int = 4
    num_stages: int = 1

    def as_vllm_config(self) -> dict[str, int]:
        """Return the launch dictionary accepted by vLLM fused MoE."""

        return {
            "BLOCK_SIZE_M": self.block_m,
            "BLOCK_SIZE_N": self.block_n,
            "BLOCK_SIZE_K": self.block_k,
            "GROUP_SIZE_M": self.group_size_m,
            "SPLIT_K": self.split_k,
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
        }


DEFAULT_GFX90A_INT8_MOE_CONFIG = Gfx90aInt8MoeConfig(
    block_m=16,
    block_n=64,
)
GLOBAL_EP_MAX_TOKENS = 4 * PREFILL_MAX_TOKENS
GFX90A_INT8_MOE_TOKEN_BUCKETS = (
    *GFX90A_INT8_TOKEN_BUCKETS,
    GLOBAL_EP_MAX_TOKENS,
)
# Measured winners over the exact GLM-5.2 EP16, EP32, and EP64 cells.
GFX90A_INT8_MOE_CONFIGS = {
    1: Gfx90aInt8MoeConfig(
        block_m=16,
        block_n=64,
        num_stages=2,
    ),
    8: DEFAULT_GFX90A_INT8_MOE_CONFIG,
    32: Gfx90aInt8MoeConfig(
        block_m=16,
        block_n=64,
        group_size_m=8,
    ),
    64: Gfx90aInt8MoeConfig(
        block_m=16,
        block_n=64,
        group_size_m=8,
    ),
    128: Gfx90aInt8MoeConfig(
        block_m=16,
        block_n=64,
        num_stages=2,
    ),
    256: Gfx90aInt8MoeConfig(
        block_m=64,
        block_n=64,
        num_stages=2,
    ),
    512: Gfx90aInt8MoeConfig(
        block_m=64,
        block_n=128,
        num_warps=8,
        num_stages=2,
    ),
    1024: Gfx90aInt8MoeConfig(
        block_m=128,
        block_n=128,
        num_warps=8,
        num_stages=2,
    ),
    2048: Gfx90aInt8MoeConfig(
        block_m=64,
        block_n=128,
        group_size_m=8,
        num_warps=8,
        num_stages=2,
    ),
    # DP2 tokens are all-gathered across global EP16 before expert dispatch,
    # doubling the local 2,048-token scheduler limit.
    4096: Gfx90aInt8MoeConfig(
        block_m=64,
        block_n=128,
        group_size_m=8,
        num_warps=8,
        num_stages=2,
    ),
    # DP8 tokens are all-gathered across global EP64 before expert dispatch.
    8192: Gfx90aInt8MoeConfig(
        block_m=128,
        block_n=256,
        group_size_m=32,
        num_warps=8,
        num_stages=2,
    ),
}


def select_gfx90a_int8_moe_config(num_tokens: int) -> Gfx90aInt8MoeConfig:
    """Select only launch configs measured for the GLM-5.2 EP path."""

    _require(num_tokens > 0, "num_tokens must be positive")
    _require(
        num_tokens <= GLOBAL_EP_MAX_TOKENS,
        f"CDNA2 INT8 MoE is gated through M={GLOBAL_EP_MAX_TOKENS}; got {num_tokens}",
    )
    token_bucket = min(
        GFX90A_INT8_MOE_TOKEN_BUCKETS,
        key=lambda value: abs(value - num_tokens),
    )
    return GFX90A_INT8_MOE_CONFIGS[token_bucket]


def bind_vllm_moe_symbols(
    fused_moe_kernel,
    activation_quantizer,
    moe_align_block_size,
    moe_sum,
) -> None:
    """Bind the ABI-validated vLLM MoE helpers once."""

    global _VLLM_ACTIVATION_QUANTIZER
    global _VLLM_FUSED_MOE_KERNEL
    global _VLLM_MOE_ALIGN_BLOCK_SIZE
    global _VLLM_MOE_SUM

    _require(callable(fused_moe_kernel), "fused_moe_kernel must be callable")
    _require(
        callable(activation_quantizer),
        "activation_quantizer must be callable",
    )
    _require(
        callable(moe_align_block_size),
        "moe_align_block_size must be callable",
    )
    _require(callable(moe_sum), "moe_sum must be callable")
    _require(
        _VLLM_FUSED_MOE_KERNEL is None or _VLLM_FUSED_MOE_KERNEL is fused_moe_kernel,
        "a different vLLM fused MoE kernel is already bound",
    )
    _require(
        _VLLM_ACTIVATION_QUANTIZER is None or _VLLM_ACTIVATION_QUANTIZER is activation_quantizer,
        "a different vLLM activation quantizer is already bound",
    )
    _require(
        _VLLM_MOE_ALIGN_BLOCK_SIZE is None or _VLLM_MOE_ALIGN_BLOCK_SIZE is moe_align_block_size,
        "a different vLLM MoE alignment helper is already bound",
    )
    _require(
        _VLLM_MOE_SUM is None or _VLLM_MOE_SUM is moe_sum,
        "a different vLLM MoE reduction helper is already bound",
    )
    _VLLM_FUSED_MOE_KERNEL = fused_moe_kernel
    _VLLM_ACTIVATION_QUANTIZER = activation_quantizer
    _VLLM_MOE_ALIGN_BLOCK_SIZE = moe_align_block_size
    _VLLM_MOE_SUM = moe_sum


@torch.no_grad()
def transcode_ocp_fp8_expert_weight(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    device_is_validated: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transcode contiguous ``[experts, N, K]`` OCP-FP8 weights in one GPU launch."""

    _require(weight.ndim == 3, "expert weight must have shape [experts, N, K]")
    _require(weight.shape[0] > 0, "expert weight must contain at least one expert")
    _require(weight.is_contiguous(), "expert weight must be contiguous")
    _require(
        weight_scale.ndim == 3,
        "expert weight_scale must have shape [experts, N / 128, K / 128]",
    )
    _require(
        weight_scale.shape[0] == weight.shape[0],
        "expert weight and scale counts must match",
    )
    _require(weight_scale.is_contiguous(), "expert weight_scale must be contiguous")

    return transcode_ocp_fp8_weight_batch(
        weight,
        weight_scale,
        device_is_validated=device_is_validated,
    )


def _validate_fused_moe_inputs(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor,
) -> tuple[int, int, int, int]:
    _require(
        hidden_states.ndim == 2 and hidden_states.is_contiguous(),
        "hidden_states must be contiguous [tokens, hidden]",
    )
    _require(
        hidden_states.dtype == torch.bfloat16,
        "the measured CDNA2 INT8 MoE path requires BF16 activations",
    )
    _require(hidden_states.device.type == "cuda", "hidden_states must be on a ROCm GPU")
    _require(torch.version.hip is not None, "a ROCm PyTorch build is required")
    _require(
        w1.ndim == 3 and w2.ndim == 3 and w1.is_contiguous() and w2.is_contiguous(),
        "MoE weights must be contiguous [experts, N, K]",
    )
    _require(w1.dtype == torch.int8 and w2.dtype == torch.int8, "MoE weights must use INT8")
    _require(
        w1.device == hidden_states.device and w2.device == hidden_states.device,
        "MoE weights must be on the activation device",
    )
    tokens, hidden_size = hidden_states.shape
    local_experts, w1_out, w1_k = w1.shape
    _require(w1_k == hidden_size, "w1 K must equal hidden_size")
    _require(w1_out % 2 == 0, "w1 output must contain equal gate and up halves")
    intermediate_size = w1_out // 2
    _require(
        tuple(w2.shape) == (local_experts, hidden_size, intermediate_size),
        "w2 must have shape [local_experts, hidden_size, intermediate_size]",
    )
    _require(
        w1_scale.dtype == RUNTIME_WEIGHT_SCALE_DTYPE
        and w2_scale.dtype == RUNTIME_WEIGHT_SCALE_DTYPE
        and w1_scale.is_contiguous()
        and w2_scale.is_contiguous(),
        "MoE weight scales must be contiguous FP16",
    )
    _require(
        tuple(w1_scale.shape) == (local_experts, w1_out, hidden_size // TARGET_GROUP_K),
        "w1_scale must have shape [experts, w1_out, hidden / 32]",
    )
    _require(
        tuple(w2_scale.shape) == (local_experts, hidden_size, intermediate_size // TARGET_GROUP_K),
        "w2_scale must have shape [experts, hidden, intermediate / 32]",
    )
    _require(
        topk_weights.ndim == 2
        and topk_ids.ndim == 2
        and tuple(topk_weights.shape) == tuple(topk_ids.shape)
        and topk_weights.shape[0] == tokens,
        "topk weights and ids must have matching [tokens, top_k] shapes",
    )
    _require(
        topk_weights.device == hidden_states.device and topk_ids.device == hidden_states.device,
        "routing tensors must be on the activation device",
    )
    _require(
        topk_weights.dtype == torch.float32 and topk_weights.is_contiguous(),
        "topk_weights must be contiguous FP32",
    )
    _require(
        topk_ids.dtype == torch.int32 and topk_ids.is_contiguous(),
        "topk_ids must be contiguous INT32",
    )
    _require(
        expert_map.ndim == 1
        and expert_map.dtype == torch.int32
        and expert_map.is_contiguous()
        and expert_map.device == hidden_states.device,
        "expert_map must be contiguous device INT32",
    )
    return tokens, hidden_size, intermediate_size, topk_ids.shape[1]


def invoke_gfx90a_block_int8_moe(
    activation: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    *,
    mul_routed_weight: bool,
    top_k: int,
    config: Gfx90aInt8MoeConfig,
    assignment_capacity: int | None = None,
) -> None:
    """Invoke vLLM's fused kernel with the correct K cap for ``[1, 32]`` scales."""

    _require(
        _VLLM_FUSED_MOE_KERNEL is not None,
        "vLLM kernel ABI is not installed",
    )
    launch = config.as_vllm_config()
    m = activation.size(0)
    num_tokens = m * top_k
    em = sorted_token_ids.size(0)
    if assignment_capacity is not None:
        _require(
            0 < assignment_capacity <= em,
            "assignment_capacity must be within sorted_token_ids",
        )
        em = assignment_capacity
    if m < launch["BLOCK_SIZE_M"]:
        em = min(
            em,
            m * top_k * launch["BLOCK_SIZE_M"],
        )
    grid = lambda meta: (triton.cdiv(em, meta["BLOCK_SIZE_M"]) * triton.cdiv(weight.size(1), meta["BLOCK_SIZE_N"]),)
    launch["SPLIT_K"] = 1
    # vLLM 0.24 incorrectly also caps this by block_shape[0]. For per-row
    # weights block_shape[0] is one, but the reduction tile is governed only
    # by the K grouping and must remain 32 to emit native INT8 MFMA.
    block_size_k = min(launch.pop("BLOCK_SIZE_K"), TARGET_GROUP_K)
    _VLLM_FUSED_MOE_KERNEL[grid](
        activation,
        weight,
        output,
        None,
        activation_scale,
        weight_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        weight.size(1),
        weight.size(2),
        em,
        num_tokens,
        activation.stride(0),
        activation.stride(1),
        weight.stride(0),
        weight.stride(2),
        weight.stride(1),
        output.stride(1),
        output.stride(2),
        activation_scale.stride(0),
        activation_scale.stride(1),
        weight_scale.stride(0),
        weight_scale.stride(2),
        weight_scale.stride(1),
        0,
        0,
        TARGET_GROUP_N,
        TARGET_GROUP_K,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=tl.bfloat16,
        use_fp8_w8a8=False,
        use_int8_w8a8=True,
        use_int8_w8a16=False,
        per_channel_quant=False,
        naive_block_assignment=False,
        HAS_BIAS=False,
        BLOCK_SIZE_K=block_size_k,
        SWAP_AB=False,
        **launch,
    )


def _select_local_expert_assignment_capacity(
    sorted_token_ids: torch.Tensor,
    *,
    num_route_slots: int,
    num_local_experts: int,
    block_size_m: int,
    expert_map: torch.Tensor | None,
    ignore_invalid_experts: bool,
) -> int | None:
    """Bound the launch grid when vLLM packs only local expert assignments."""

    if (
        not ignore_invalid_experts
        or expert_map is None
        or expert_map.ndim != 1
        or expert_map.dtype != torch.int32
        or not expert_map.is_contiguous()
        or sorted_token_ids.ndim != 1
        or sorted_token_ids.dtype != torch.int32
        or not sorted_token_ids.is_contiguous()
        or num_route_slots <= 0
        or block_size_m <= 0
        or num_local_experts <= 0
        or num_local_experts >= expert_map.numel()
        or expert_map.numel() % num_local_experts != 0
    ):
        return None

    capacity = num_route_slots + num_local_experts * (block_size_m - 1)
    if capacity > sorted_token_ids.numel():
        return None
    return capacity


def _allocate_moe_workspaces(
    hidden_states: torch.Tensor,
    *,
    tokens: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Alias w13/w2 caches because w13 is dead before the w2 launch."""

    workspace_width = max(2 * intermediate_size, hidden_size)
    common = torch.empty(
        tokens * top_k * workspace_width,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    cache1_elements = tokens * top_k * 2 * intermediate_size
    cache3_elements = tokens * top_k * hidden_size
    cache1 = common[:cache1_elements].view(tokens, top_k, 2 * intermediate_size)
    cache3 = common[:cache3_elements].view(tokens, top_k, hidden_size)
    output = torch.empty_like(hidden_states)
    return cache1, cache3, output


def cdna2_int8_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor,
    *,
    global_num_experts: int,
    config: Gfx90aInt8MoeConfig | None = None,
    device_is_validated: bool = False,
) -> torch.Tensor:
    """Run the measured EP block-INT8 routed experts without FP8 fallback weights."""

    _require(
        _VLLM_FUSED_MOE_KERNEL is not None
        and _VLLM_ACTIVATION_QUANTIZER is not None
        and _VLLM_MOE_ALIGN_BLOCK_SIZE is not None
        and _VLLM_MOE_SUM is not None,
        "vLLM kernel ABI is not installed",
    )
    tokens, hidden_size, intermediate_size, top_k = _validate_fused_moe_inputs(
        hidden_states,
        w1,
        w2,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids,
        expert_map,
    )
    _require(
        global_num_experts == expert_map.numel(),
        "global_num_experts must equal expert_map length",
    )
    if not device_is_validated:
        validate_gfx90a_device(hidden_states.device)
    if config is None:
        config = select_gfx90a_int8_moe_config(tokens)
    launch = config.as_vllm_config()
    ignore_invalid_experts = True
    assignment = _VLLM_MOE_ALIGN_BLOCK_SIZE(
        topk_ids=topk_ids,
        block_size=launch["BLOCK_SIZE_M"],
        num_experts=global_num_experts,
        expert_map=expert_map,
        ignore_invalid_experts=ignore_invalid_experts,
    )
    sorted_token_ids, expert_ids, num_tokens_post_padded = assignment
    _require(
        sorted_token_ids is not None,
        "CDNA2 INT8 MoE requires vLLM's sorted expert assignment",
    )
    assignment_capacity = _select_local_expert_assignment_capacity(
        sorted_token_ids,
        num_route_slots=topk_ids.numel(),
        num_local_experts=w1.size(0),
        block_size_m=launch["BLOCK_SIZE_M"],
        expert_map=expert_map,
        ignore_invalid_experts=ignore_invalid_experts,
    )

    cache1, cache3, output = _allocate_moe_workspaces(
        hidden_states,
        tokens=tokens,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    quantized_hidden, a1_scale = _VLLM_ACTIVATION_QUANTIZER(
        hidden_states,
        TARGET_GROUP_K,
    )
    invoke_gfx90a_block_int8_moe(
        quantized_hidden,
        w1,
        cache1,
        a1_scale,
        w1_scale,
        None,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight=False,
        top_k=top_k,
        config=config,
        assignment_capacity=assignment_capacity,
    )
    quantized_activation, a2_scale = cdna2_silu_and_mul_group_quant_int8(
        cache1.view(-1, 2 * intermediate_size),
        device_is_validated=True,
    )
    # Expert-parallel assignments leave nonlocal route slots unwritten. Since
    # moe_sum reduces every top-k slot, match stock fused_experts and clear the
    # aliased w2 view after w13 has been consumed by activation quantization.
    cache3.zero_()
    invoke_gfx90a_block_int8_moe(
        quantized_activation,
        w2,
        cache3,
        a2_scale,
        w2_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight=True,
        top_k=1,
        config=config,
        assignment_capacity=assignment_capacity,
    )
    _VLLM_MOE_SUM(cache3, output)
    return output


@dataclass(frozen=True)
class Gfx90aSiluInt8Config:
    block_groups: int = 8
    num_warps: int = 4
    num_stages: int = 1


DEFAULT_GFX90A_SILU_INT8_CONFIG = Gfx90aSiluInt8Config()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@triton.jit
def _silu_and_mul_group_quant_int8_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    input_stride_row: tl.int64,
    output_stride_row: tl.int64,
    scale_stride_row: tl.int64,
    hidden_size,
    num_groups,
    eps,
    GROUP_SIZE: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
):
    row_id = tl.program_id(0)
    group_block_id = tl.program_id(1)
    group_offsets = group_block_id * BLOCK_GROUPS + tl.arange(0, BLOCK_GROUPS)
    element_offsets = tl.arange(0, GROUP_SIZE)
    columns = group_offsets[:, None] * GROUP_SIZE + element_offsets[None, :]
    mask = (group_offsets[:, None] < num_groups) & (columns < hidden_size)

    gate = tl.load(
        input_ptr + row_id * input_stride_row + columns,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    up = tl.load(
        input_ptr + row_id * input_stride_row + hidden_size + columns,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    activated = gate * tl.sigmoid(gate) * up
    absmax = tl.maximum(tl.max(tl.abs(activated), axis=1), eps)
    scale = absmax / 127.0
    scaled = tl.clamp(
        activated / scale[:, None],
        -128.0,
        127.0,
    )
    # Match vLLM's ROCm INT8 quantizers: round to nearest before conversion,
    # rather than relying on backend-dependent float-to-int truncation.
    quantized = tl.extra.hip.libdevice.round(scaled).to(tl.int8)

    tl.store(
        output_ptr + row_id * output_stride_row + columns,
        quantized,
        mask=mask,
    )
    tl.store(
        scale_ptr + row_id * scale_stride_row + group_offsets,
        scale,
        mask=group_offsets < num_groups,
    )


def _validate_silu_inputs(
    input_tensor: torch.Tensor,
    eps: float,
    config: Gfx90aSiluInt8Config,
) -> tuple[int, int, int]:
    _require(input_tensor.ndim == 2, "input must have shape [tokens, 2 * hidden]")
    _require(input_tensor.shape[0] > 0, "input must contain at least one token")
    _require(
        input_tensor.shape[1] > 0 and input_tensor.shape[1] % 2 == 0,
        "input width must be positive and even for gate || up layout",
    )
    _require(
        input_tensor.dtype == torch.bfloat16,
        "the measured CDNA2 INT8 SiLU path requires BF16",
    )
    _require(input_tensor.is_contiguous(), "input must be contiguous")
    _require(eps > 0.0, "eps must be positive")
    _require(
        config.block_groups > 0 and config.block_groups & (config.block_groups - 1) == 0,
        "block_groups must be a positive power of two",
    )
    _require(config.num_warps in (1, 2, 4, 8), "num_warps must be one of 1, 2, 4, 8")
    _require(config.num_stages > 0, "num_stages must be positive")

    tokens = input_tensor.shape[0]
    hidden_size = input_tensor.shape[1] // 2
    _require(
        hidden_size % TARGET_GROUP_K == 0,
        f"activated hidden size must be divisible by {TARGET_GROUP_K}",
    )
    _require(input_tensor.device.type == "cuda", "input must be on a ROCm GPU")
    _require(torch.version.hip is not None, "a ROCm PyTorch build is required")
    return tokens, hidden_size, hidden_size // TARGET_GROUP_K


def cdna2_silu_and_mul_group_quant_int8(
    input_tensor: torch.Tensor,
    *,
    eps: float = 1.0e-10,
    config: Gfx90aSiluInt8Config = DEFAULT_GFX90A_SILU_INT8_CONFIG,
    device_is_validated: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse gated SiLU with per-token, per-32-element INT8 quantization.

    Args:
        input_tensor: Contiguous BF16 ``[tokens, 2 * hidden]`` tensor in
            ``gate || up`` layout.
        eps: Positive lower bound for the group absolute maximum.
        config: Static gfx90a launch configuration.
        device_is_validated: Skip the architecture query when the caller
            already selected this path on gfx90a outside the hot path.

    Returns:
        Contiguous INT8 activations ``[tokens, hidden]`` and FP32 scales
        ``[tokens, hidden / 32]``.
    """

    tokens, hidden_size, num_groups = _validate_silu_inputs(input_tensor, eps, config)
    if not device_is_validated:
        validate_gfx90a_device(input_tensor.device)

    output = torch.empty(
        (tokens, hidden_size),
        device=input_tensor.device,
        dtype=torch.int8,
    )
    scales = torch.empty(
        (tokens, num_groups),
        device=input_tensor.device,
        dtype=torch.float32,
    )
    grid = (tokens, triton.cdiv(num_groups, config.block_groups))
    _silu_and_mul_group_quant_int8_kernel[grid](
        input_tensor,
        output,
        scales,
        input_tensor.stride(0),
        output.stride(0),
        scales.stride(0),
        hidden_size,
        num_groups,
        eps,
        GROUP_SIZE=TARGET_GROUP_K,
        BLOCK_GROUPS=config.block_groups,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    return output, scales
