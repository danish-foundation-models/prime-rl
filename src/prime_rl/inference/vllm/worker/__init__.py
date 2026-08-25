import logging
import os

from prime_rl.inference.patches import (
    monkey_patch_fp32_lm_head,
    monkey_patch_fp32_router_logits,
    monkey_patch_glm52_gfx90a_mla_bmm,
    monkey_patch_minimax_m2_for_lora,
    monkey_patch_no_moe_lora,
    monkey_patch_rocm_gfx90a_aiter_attention,
    monkey_patch_rocm_gfx90a_aiter_legacy_allreduce,
    monkey_patch_rocm_hybrid_paged_attention,
    monkey_patch_rocm_nemotron_grouped_topk,
    monkey_patch_rocm_nemotron_int8_paged_attention,
    monkey_patch_rocm_nemotron_moe_config,
    monkey_patch_rocm_nemotron_relu2,
    monkey_patch_rocm_nemotron_router_linear,
    monkey_patch_rocm_sparse_indexer_fallback,
    monkey_patch_rocm_sparse_mla_fallback,
)

logger = logging.getLogger(__name__)

# Monkeypatch MiniMaxM2 MoE gate dtype and adapter key mapping for LoRA compatibility
monkey_patch_minimax_m2_for_lora()
# Disable LoRA on MoE layers so vLLM picks better kernels (e.g. TRTLLMFlashInfer on Blackwell)
if os.environ.get("PRIME_NO_MOE_LORA") == "1":
    logger.info("PRIME_NO_MOE_LORA=1: disabling LoRA on MoE layers")
    monkey_patch_no_moe_lora()
else:
    logger.info("PRIME_NO_MOE_LORA=0: no patch applied")

# Install fp32 lm_head patch; self-gates on additional_config["fp32_lm_head"] at call time
monkey_patch_fp32_lm_head()

# Install fp32 router logits patch; self-gates on additional_config["fp32_router_logits"]
monkey_patch_fp32_router_logits()

# Install exact-shape GLM-5.2 MLA BMMs only when explicitly selected.
monkey_patch_glm52_gfx90a_mla_bmm()

# Install the guarded AITER MHA backend before attention methods are captured.
monkey_patch_rocm_gfx90a_aiter_attention()

# Install the exact Nemotron TP8 long-context kernel only when explicitly selected.
monkey_patch_rocm_hybrid_paged_attention()

# Install exact Nemotron TP8 dynamic-INT8 decode only when explicitly selected.
monkey_patch_rocm_nemotron_int8_paged_attention()

# Install exact Nemotron-H one-group decode routing only when selected.
monkey_patch_rocm_nemotron_grouped_topk()

# GLM-5.2's sparse attention requires AITER by default on ROCm. AITER does not
# support LUMI's gfx90a GPUs, so permit explicitly selected non-AITER indexer
# and attention backends only when the launcher opts in.
monkey_patch_rocm_sparse_indexer_fallback()
monkey_patch_rocm_sparse_mla_fallback()

# Install the validated gfx90a TP8 BF16 all-reduce path when explicitly selected.
monkey_patch_rocm_gfx90a_aiter_legacy_allreduce()

# Skip non-local expert rows in large ReLU-squared activations when selected.
monkey_patch_rocm_nemotron_relu2()

# Select the measured large-prefill expert-GEMM configuration when selected.
monkey_patch_rocm_nemotron_moe_config()

# Preserve FP32 router logits while using BF16 MFMA for large prefills.
monkey_patch_rocm_nemotron_router_linear()
