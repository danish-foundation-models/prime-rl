#!/usr/bin/env bash
set -euo pipefail
: "${PROJECT_DIR:?}" "${LUMI_ROOT:?}" "${SIF:?}" "${PRIME_RL_METADATA_DIR:?}"
[[ -f "$PRIME_RL_METADATA_DIR/.ready" ]]
source "$PROJECT_DIR/scripts/lumi_runtime_env.sh"
prime_rl_lumi_export_pythonpath "$PRIME_RL_METADATA_DIR" "$PROJECT_DIR"
export HF_HOME=/scratch/project_465002751/.cache/huggingface
export UV_CACHE_DIR=$LUMI_ROOT/.cache/uv XDG_CACHE_HOME=$LUMI_ROOT/.cache
export WANDB_CACHE_DIR=$LUMI_ROOT/.cache/wandb WANDB_CONFIG_DIR=$LUMI_ROOT/.cache/wandb-config
export MPLCONFIGDIR=$LUMI_ROOT/.cache/matplotlib
aiter_cache=$(prime_rl_lumi_aiter_cache_dir "$LUMI_ROOT" "$SIF")
export AITER_JIT_DIR=${AITER_JIT_DIR:-$aiter_cache} AITER_ROOT_DIR=${AITER_ROOT_DIR:-$aiter_cache}
mkdir -p "$AITER_JIT_DIR" "$AITER_ROOT_DIR"
export VIRTUAL_ENV=/opt/venv UV_PROJECT_ENVIRONMENT=/opt/venv PYTHONNOUSERSITE=1
binds=(-B /scratch/project_465002751:/scratch/project_465002751)
# Optional gfx90a vLLM source overlay (scripts/lumi_prepare_vllm_overlay.sh).
if [[ -n ${PRIME_RL_VLLM_OVERLAY:-} ]]; then
    [[ -f "$PRIME_RL_VLLM_OVERLAY/.ready" ]] || {
        echo "Missing vLLM overlay: $PRIME_RL_VLLM_OVERLAY" >&2; exit 1;
    }
    while IFS= read -r file; do
        binds+=(-B "$PRIME_RL_VLLM_OVERLAY/$file:/opt/venv/lib/python3.12/site-packages/$file:ro")
    done < "$PRIME_RL_VLLM_OVERLAY/FILES"
fi
exec singularity exec \
    "${binds[@]}" \
    --home "$LUMI_ROOT/.home" --pwd "$PROJECT_DIR" \
    --env "PATH=$PRIME_RL_METADATA_DIR/bin:/opt/venv/bin:/usr/local/bin:/usr/bin:/bin" \
    --env "PYTHONPATH=$PYTHONPATH" "$SIF" \
    "$LUMI_ROOT/bin/uv" run --no-project "$@"
