"""EP-compact Triton MoE for non-gated (relu²) experts under DP+EP with allgather dispatch.

With allgather/reduce-scatter EP every rank sees all gathered tokens (8 x the local batch) with
their global top-22 ids, but holds only 64 of 512 experts. vLLM 0.28's TritonExperts handles
this by aligning *all* 512 experts and marking the 448 remote ones -1, so per layer it
  - launches GEMM blocks for every remote expert just to write zeros (GEMM1 and GEMM2),
  - runs relu² over the whole (tokens x 22 x 2688) intermediate, ~8x the useful rows,
  - runs moe_sum over all 22 slots per token, 19 of which are zeros on average.
This patch keeps vLLM's fused_moe GEMM kernel (so tuned configs apply and the GEMM numerics are
unchanged) and only changes what surrounds it:
  - moe_align_block_size(ignore_invalid_experts=True): blocks for local experts only,
  - relu² over the rows listed in sorted_token_ids only,
  - a combine kernel that sums only the slots whose expert is local (same fp32 sum as moe_sum
    over zeros, so outputs match stock up to fp32 summation order).
Everything is fixed-shape with device-side counts, so it is CUDA-graph safe.

Enable with INFER_OPT=ep_compact_moe. Falls back to stock for LoRA, quantized, gated or non-EP layers.
"""
import torch

from infer_opt import _log, when_imported

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None


if triton is not None:

    @triton.jit
    def _relu2_rows_kernel(src, dst, sorted_ids, post_pad_ptr, num_pairs, N, stride_src, stride_dst,
                           BLOCK_P: tl.constexpr, BLOCK_N: tl.constexpr):
        pid_p = tl.program_id(0)
        pid_n = tl.program_id(1)
        post_pad = tl.load(post_pad_ptr)
        if pid_p * BLOCK_P >= post_pad:
            return
        pos = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
        pair = tl.load(sorted_ids + pos, mask=pos < post_pad, other=num_pairs)
        valid = pair < num_pairs
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m = valid[:, None] & (cols[None, :] < N)
        x = tl.load(src + pair[:, None].to(tl.int64) * stride_src + cols[None, :], mask=m, other=0.0)
        x = tl.maximum(x.to(tl.float32), 0.0)
        tl.store(dst + pair[:, None].to(tl.int64) * stride_dst + cols[None, :], (x * x).to(dst.dtype.element_ty), mask=m)

    @triton.jit
    def _combine_local_kernel(src, out, topk_ids, expert_map, K, stride_src, stride_out,
                              TOPK: tl.constexpr, BLOCK_K: tl.constexpr):
        t = tl.program_id(0)
        pid_k = tl.program_id(1)
        cols = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        cm = cols < K
        acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for j in tl.static_range(TOPK):
            e = tl.load(topk_ids + t * TOPK + j)
            loc = tl.load(expert_map + e)
            row = (t * TOPK + j).to(tl.int64)
            v = tl.load(src + row * stride_src + cols, mask=cm & (loc >= 0), other=0.0)
            acc += v.to(tl.float32)
        tl.store(out + t.to(tl.int64) * stride_out + cols, acc.to(out.dtype.element_ty), mask=cm)


def relu2_rows(src: torch.Tensor, dst: torch.Tensor, sorted_ids: torch.Tensor, post_pad: torch.Tensor,
               num_pairs: int) -> None:
    N = src.shape[1]
    BLOCK_P, BLOCK_N = 16, 256
    grid = (triton.cdiv(sorted_ids.numel(), BLOCK_P), triton.cdiv(N, BLOCK_N))
    _relu2_rows_kernel[grid](src, dst, sorted_ids, post_pad, num_pairs, N, src.stride(0), dst.stride(0),
                             BLOCK_P=BLOCK_P, BLOCK_N=BLOCK_N, num_warps=4)


def combine_local(src: torch.Tensor, out: torch.Tensor, topk_ids: torch.Tensor, expert_map: torch.Tensor) -> None:
    M, topk = topk_ids.shape
    K = out.shape[1]
    if M == 0:
        return
    BLOCK_K = 1024 if K >= 1024 else triton.next_power_of_2(K)
    grid = (M, triton.cdiv(K, BLOCK_K))
    _combine_local_kernel[grid](src, out, topk_ids.contiguous(), expert_map, K, src.stride(0), out.stride(0),
                                TOPK=topk, BLOCK_K=BLOCK_K, num_warps=4)


def compact_apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation, global_num_experts,
                  expert_map, workspace13, workspace2, apply_router_weight_on_input):
    """Body of TritonExperts.apply for the bf16, non-gated, EP (expert_map) case."""
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        _prepare_expert_assignment, invoke_fused_moe_triton_kernel, try_get_optimal_moe_config)
    from vllm.model_executor.layers.fused_moe.utils import _resize_cache
    from vllm.triton_utils import tl as vtl

    E, num_tokens, N, K, top_k = self.moe_problem_size(hidden_states, w1, w2, topk_ids)
    if global_num_experts == -1:
        global_num_experts = E
    config = try_get_optimal_moe_config(w1.size(), w2.size(), top_k, self.quant_config.config_name(hidden_states.dtype),
                                        num_tokens, block_shape=self.block_shape)
    # Same buffers as stock: cache1 and cache3 share workspace2 (cache1 is dead once relu² has run).
    cache1 = _resize_cache(workspace2, (num_tokens, top_k, N))
    cache2 = _resize_cache(workspace13, (num_tokens * top_k, N))
    cache3 = _resize_cache(workspace2, (num_tokens, top_k, K))
    sorted_ids, expert_ids, post_pad = _prepare_expert_assignment(
        topk_ids, config, num_tokens, top_k, global_num_experts, expert_map, ignore_invalid_experts=True)
    common = dict(compute_type=vtl.bfloat16, use_fp8_w8a8=False, use_int8_w8a8=False, use_int8_w8a16=False,
                  use_int4_w4a16=False, per_channel_quant=False, block_shape=None)
    invoke_fused_moe_triton_kernel(hidden_states, w1, cache1, None, None, None, sorted_ids, expert_ids, post_pad,
                                   False, top_k, config, B_bias=self.w1_bias, **common)
    relu2_rows(cache1.view(-1, N), cache2, sorted_ids, post_pad, num_tokens * top_k)
    invoke_fused_moe_triton_kernel(cache2, w2, cache3, None, None, topk_weights, sorted_ids, expert_ids, post_pad,
                                   not apply_router_weight_on_input, 1, config, B_bias=self.w2_bias, **common)
    combine_local(cache3.view(-1, K), output, topk_ids, expert_map)


def install() -> None:
    def patch(mod):
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation

        T = mod.TritonExperts
        orig = T.apply
        relu2 = MoEActivation.RELU2_NO_MUL
        state = {"logged": False}

        def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation, global_num_experts,
                  expert_map, a1q_scale, a2_scale, workspace13, workspace2, expert_tokens_meta,
                  apply_router_weight_on_input):
            if (expert_map is None or activation != relu2 or hidden_states.dtype != torch.bfloat16
                    or self._lora_context is not None or self.quant_config.use_fp8_w8a8
                    or self.quant_config.use_int8_w8a8 or self.quant_config.use_int8_w8a16
                    or self.quant_config.use_int4_w4a16 or type(self) is not T):
                return orig(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation,
                            global_num_experts, expert_map, a1q_scale, a2_scale, workspace13, workspace2,
                            expert_tokens_meta, apply_router_weight_on_input)
            if not state["logged"]:
                state["logged"] = True
                _log(f"ep_compact_moe active (E_local={w1.shape[0]}, N={w1.shape[1]}, K={w1.shape[2]})")
            compact_apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation,
                          global_num_experts, expert_map, workspace13, workspace2, apply_router_weight_on_input)

        T.apply = apply
        _log("ep_compact_moe: TritonExperts.apply patched")

    when_imported("vllm.model_executor.layers.fused_moe.experts.triton_moe", patch)
