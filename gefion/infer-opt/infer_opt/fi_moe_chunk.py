"""Token-chunked FlashInfer CUTLASS MoE, to cap its internal workspace under DP+EP allgather.

With allgather EP a rank's MoE sees all gathered tokens (8 x 4096 at a prefill step), and
FlashInfer's cutlass_fused_moe allocates its permuted/intermediate buffers for tokens x top-22
rows at once. On DP8 online FP8 (fp8_per_block) that showed up as 14.0 GiB of profiled peak
activation per GPU, against 2.3 GiB for the bf16 Triton path, eating most of the 19 GiB FP8
frees for KV/Mamba state. Calling it in token chunks caps the workspace. Every token's result is
independent (per-token activation quantization, per-token routing), so outputs are unchanged.
Decode batches are below the chunk size and take the stock path, so CUDA graphs are unaffected.

Enable with INFER_OPT=fi_moe_chunk; INFER_OPT_FI_MOE_CHUNK sets the chunk (default 4096 tokens).
"""
import os

from infer_opt import _log, when_imported


def install() -> None:
    chunk = int(os.environ.get("INFER_OPT_FI_MOE_CHUNK", "4096"))

    def patch(mod):
        F = mod.FlashInferExperts
        orig = F.apply

        def rows(t, m):
            return t is not None and t.dim() > 0 and t.shape[0] == m

        def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation, global_num_experts,
                  expert_map, a1q_scale, a2_scale, workspace13, workspace2, expert_tokens_meta,
                  apply_router_weight_on_input):
            m = hidden_states.shape[0]
            # nvfp4 / mxfp8 activation scales are swizzled, not per-row: leave those paths alone
            chunkable = (m > chunk and expert_tokens_meta is None and self.quant_dtype not in ("nvfp4", "mxfp8")
                         and (a1q_scale is None or rows(a1q_scale, m)))
            if not chunkable:
                return orig(self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation,
                            global_num_experts, expert_map, a1q_scale, a2_scale, workspace13, workspace2,
                            expert_tokens_meta, apply_router_weight_on_input)
            for s in range(0, m, chunk):
                e = min(m, s + chunk)
                sl = slice(s, e)
                orig(self, output[sl], hidden_states[sl], w1, w2, topk_weights[sl], topk_ids[sl], activation,
                     global_num_experts, expert_map, a1q_scale[sl] if a1q_scale is not None else None, a2_scale,
                     workspace13[sl] if rows(workspace13, m) else workspace13,
                     workspace2[sl] if rows(workspace2, m) else workspace2,
                     expert_tokens_meta, apply_router_weight_on_input)

        F.apply = apply
        _log(f"fi_moe_chunk: FlashInferExperts.apply chunked at {chunk} tokens")

    when_imported("vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe", patch)
