#!/usr/bin/env bash
set -euo pipefail

[[ $# -ge 1 ]] || {
  echo "usage: $0 COMMAND [ARGS ...]" >&2
  exit 2
}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_DIR=${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd -P)}
LUMI_ROOT=${LUMI_ROOT:-/scratch/project_465002751/rasmus}
SIF=${SIF:-/scratch/project_465002751/shared/sif/lumi-posttrain-u24r71-t211-vllm024.sif}
LUMI_UV=${LUMI_UV:-$LUMI_ROOT/bin/uv}
HF_HOME=${PRIME_RL_HF_HOME:-/scratch/project_465002751/.cache/huggingface}
UV_CACHE_DIR=${PRIME_RL_UV_CACHE_DIR:-$LUMI_ROOT/.cache/uv}
XDG_CACHE_HOME=${PRIME_RL_XDG_CACHE_HOME:-$LUMI_ROOT/.cache}
WANDB_CACHE_DIR=${PRIME_RL_WANDB_CACHE_DIR:-$LUMI_ROOT/.cache/wandb}
WANDB_CONFIG_DIR=${PRIME_RL_WANDB_CONFIG_DIR:-$LUMI_ROOT/.cache/wandb-config}
MPLCONFIGDIR=${PRIME_RL_MPLCONFIGDIR:-$LUMI_ROOT/.cache/matplotlib}
TMPDIR=${PRIME_RL_TMPDIR:-$LUMI_ROOT/tmp/submit}
CONTAINER_HOME=${CONTAINER_HOME:-$LUMI_ROOT/.home}

[[ -x "$LUMI_UV" ]] || {
  echo "FATAL: uv >=0.11.1 is not installed at $LUMI_UV" >&2
  echo "Install it outside the repository and set LUMI_UV to that executable." >&2
  exit 1
}

mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$XDG_CACHE_HOME" "$WANDB_CACHE_DIR" \
  "$WANDB_CONFIG_DIR" "$MPLCONFIGDIR" "$TMPDIR" "$CONTAINER_HOME"

export PROJECT_DIR SIF LUMI_ROOT
PRIME_RL_METADATA_DIR=$($PROJECT_DIR/scripts/lumi_prepare_runtime.sh)

PYTHONPATH=/workdir/src:/workdir/packages/prime-rl-configs/src:/workdir/deps/renderers:/workdir/deps/verifiers:/workdir/deps/pydantic-config/src:/workdir/deps/research-environments/environments/mini_swe_agent_v2:$PRIME_RL_METADATA_DIR
PATH_VALUE=$PRIME_RL_METADATA_DIR/bin:/opt/venv/bin:/usr/local/bin:/usr/bin:/bin
PASSTHROUGH_ENV=()
for name in \
  HF_TOKEN HUGGING_FACE_HUB_TOKEN WANDB_API_KEY \
  NEVER_CLEAN_OUTPUT_DIR \
  PRIME_RL_SFT_DEBUG_GRADS PRIME_RL_SFT_DEBUG_GRADS_STEPS PRIME_RL_SFT_DEBUG_GRADS_TOPK
do
  if [[ -n ${!name+x} ]]; then
    PASSTHROUGH_ENV+=(--env "$name=${!name}")
  fi
done

exec singularity exec \
  --cleanenv \
  -B "$PROJECT_DIR:/workdir" \
  -B /scratch/project_465002751:/scratch/project_465002751 \
  --home "$CONTAINER_HOME" \
  --pwd /workdir \
  --env "PATH=$PATH_VALUE" \
  --env "PYTHONPATH=$PYTHONPATH" \
  --env PYTHONNOUSERSITE=1 \
  --env VIRTUAL_ENV=/opt/venv \
  --env UV_PROJECT_ENVIRONMENT=/opt/venv \
  --env "HF_HOME=$HF_HOME" \
  --env "UV_CACHE_DIR=$UV_CACHE_DIR" \
  --env "UV_PYTHON_INSTALL_DIR=$LUMI_ROOT/.uv/python" \
  --env "UV_TOOL_BIN_DIR=$LUMI_ROOT/.local/bin" \
  --env "UV_TOOL_DIR=$LUMI_ROOT/.local/share/uv/tools" \
  --env "XDG_CACHE_HOME=$XDG_CACHE_HOME" \
  --env "WANDB_CACHE_DIR=$WANDB_CACHE_DIR" \
  --env "WANDB_CONFIG_DIR=$WANDB_CONFIG_DIR" \
  --env "MPLCONFIGDIR=$MPLCONFIGDIR" \
  --env "TMPDIR=$TMPDIR" \
  "${PASSTHROUGH_ENV[@]}" \
  "$SIF" \
  "$LUMI_UV" run --no-project "$@"
