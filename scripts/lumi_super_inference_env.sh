#!/usr/bin/env bash
# Use the frozen, validated ROCm kernels from the Super evaluation profile.
source "$LUMI_ROOT/artifacts/training/nemotron-super-r5-launch-20260914/optimized-eval/environment.sh"
source "$LUMI_ROOT/artifacts/training/nemotron-super-r5-launch-20260914/optimized-eval-mtp4-c32/environment.sh"
# RCCL caches channel parameters at first use. Match the trainer defaults
# before importing GPU libraries; evaluation-only tuning must not leak into RL.
unset NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS
export NEMOTRON_MTP_TOKENS=0
export VLLM_USE_V2_MODEL_RUNNER=0
export TRITON_CACHE_DIR=$DFM_EVALS_TRITON_CACHE_DIR
export TORCHINDUCTOR_CACHE_DIR=$DFM_EVALS_TORCHINDUCTOR_CACHE_DIR
export VLLM_CACHE_ROOT=$TARGET_SERVER_CACHE_ROOT
export PYTHONPATH="$MTP_PROFILE_ROOT:$LUMI_ROOT/artifacts/training/nemotron-super-r5-rl-main-20260916/runtime-helpers:$MTP_PROFILE_ROOT/source:$NEMOTRON_EVAL_LAYER_SITE:$PYTHONPATH"

if [[ -n "${PRIME_RL_SUPER_RELOAD_OVERLAY:-}" ]]; then
    [[ -r "$PRIME_RL_SUPER_RELOAD_OVERLAY/sitecustomize.py" && -r "$PRIME_RL_SUPER_RELOAD_OVERLAY/manifest.json" ]] || {
        echo "Missing Super reload overlay: $PRIME_RL_SUPER_RELOAD_OVERLAY" >&2
        return 1
    }
    export PRIME_RL_BASE_SITECUSTOMIZE="$MTP_PROFILE_ROOT/sitecustomize.py"
    export PYTHONPATH="$PRIME_RL_SUPER_RELOAD_OVERLAY:$PYTHONPATH"
fi

# The consolidated profile contains the same validated modules, including reload.
if [[ -n "${PRIME_RL_SUPER_PROFILE:-}" ]]; then
    [[ -r "$PRIME_RL_SUPER_PROFILE/manifest.json" && -r "$PRIME_RL_SUPER_PROFILE/sitecustomize.py" ]] || {
        echo "Missing unified Super profile: $PRIME_RL_SUPER_PROFILE" >&2
        return 1
    }
    export VLLM_SRC_OVERRIDE="$PRIME_RL_SUPER_PROFILE/source"
    export DFM_EVALS_VLLM_SOURCE="$PRIME_RL_SUPER_PROFILE:$PRIME_RL_SUPER_PROFILE/source:$NEMOTRON_EVAL_LAYER_SITE"
    export PYTHONPATH="$DFM_EVALS_VLLM_SOURCE:$PYTHONPATH"
fi
