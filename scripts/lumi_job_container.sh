#!/usr/bin/env bash
set -euo pipefail
: "${PROJECT_DIR:?}" "${LUMI_ROOT:?}" "${SIF:?}" "${PRIME_RL_METADATA_DIR:?}"
[[ -f "$PRIME_RL_METADATA_DIR/.ready" ]]
source "$PROJECT_DIR/scripts/lumi_runtime_env.sh"
prime_rl_lumi_export_pythonpath "$PRIME_RL_METADATA_DIR" "$PROJECT_DIR"
mask=$(prime_rl_lumi_connect_python_mask "$LUMI_ROOT")
export HF_HOME=/scratch/project_465002751/.cache/huggingface
export UV_CACHE_DIR=$LUMI_ROOT/.cache/uv XDG_CACHE_HOME=$LUMI_ROOT/.cache
export WANDB_CACHE_DIR=$LUMI_ROOT/.cache/wandb WANDB_CONFIG_DIR=$LUMI_ROOT/.cache/wandb-config
export MPLCONFIGDIR=$LUMI_ROOT/.cache/matplotlib
export VIRTUAL_ENV=/opt/venv UV_PROJECT_ENVIRONMENT=/opt/venv PYTHONNOUSERSITE=1
exec singularity exec \
    -B /scratch/project_465002751:/scratch/project_465002751 \
    -B "$mask:/opt/venv/lib/python3.12/site-packages/connect_python-0.9.0.dist-info" \
    --home "$LUMI_ROOT/.home" --pwd "$PROJECT_DIR" \
    --env "PATH=$PRIME_RL_METADATA_DIR/bin:/opt/venv/bin:/usr/local/bin:/usr/bin:/bin" \
    --env "PYTHONPATH=$PYTHONPATH" "$SIF" \
    "$LUMI_ROOT/bin/uv" run --no-project "$@"
