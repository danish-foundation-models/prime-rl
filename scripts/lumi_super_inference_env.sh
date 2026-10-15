#!/usr/bin/env bash
# Inference-node environment for LUMI RL, sourced by the multi-node RL template
# inside the container before the per-config [inference.env_vars] exports.

# RCCL caches channel parameters at first use. Match the trainer defaults
# before importing GPU libraries; evaluation-only tuning must not leak into RL.
unset NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS

if [[ -n "${PRIME_RL_SUPER_PROFILE:-}" || -n "${PRIME_RL_SUPER_RELOAD_OVERLAY:-}" ]]; then
    echo "PRIME_RL_SUPER_PROFILE and PRIME_RL_SUPER_RELOAD_OVERLAY target the retired r71 image;" \
        "use PRIME_RL_INFERENCE_PROFILE=nemotron-super-gfx90a with PRIME_RL_VLLM_OVERLAY" >&2
    return 1
fi

case "${PRIME_RL_INFERENCE_PROFILE:-}" in
    "") ;;
    nemotron-super-gfx90a)
        # Validated Nemotron-3-Super TP8/EP8 settings on MI250X. The kernels come
        # from the gfx90a vLLM overlay (PRIME_RL_VLLM_OVERLAY), bound at launch.
        [[ -n "${PRIME_RL_VLLM_OVERLAY:-}" ]] || {
            echo "nemotron-super-gfx90a requires PRIME_RL_VLLM_OVERLAY" >&2
            return 1
        }
        export VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_CUSTOM_AR=1
        export VLLM_ROCM_USE_AITER_MHA=1 VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1
        export VLLM_ROCM_USE_AITER_LINEAR=0 VLLM_ROCM_USE_AITER_MOE=0
        export VLLM_ROCM_USE_AITER_RMSNORM=0 VLLM_ROCM_USE_AITER_MLA=0
        # Keep compilation, graph capture and kernel caches; skip whole-model AOT reload.
        export VLLM_USE_AOT_COMPILE=0 VLLM_USE_V2_MODEL_RUNNER=0
        cache_root="$LUMI_ROOT/.cache/vllm-super/$(basename "$SIF" .sif)"
        export VLLM_CACHE_ROOT="$cache_root/vllm"
        export TRITON_CACHE_DIR="$cache_root/triton"
        export TORCHINDUCTOR_CACHE_DIR="$cache_root/torchinductor"
        mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
        ;;
    *)
        echo "Unknown PRIME_RL_INFERENCE_PROFILE: $PRIME_RL_INFERENCE_PROFILE" >&2
        return 1
        ;;
esac
