#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

set -a
for ENV_FILE in "$REPO_DIR/../post/.env" "$REPO_DIR/.env"; do
  if [[ -f "$ENV_FILE" ]]; then
    source "$ENV_FILE"
  fi
done
set +a

LUMI_ROOT=${LUMI_ROOT:-/scratch/project_465002751/rasmus}
SIF=${SIF:-/scratch/project_465002751/shared/sif/lumi-posttrain-u24r71-t211-vllm024.sif}
HF_HOME=${PRIME_RL_HF_HOME:-/scratch/project_465002751/.cache/huggingface}
UV_CACHE_DIR=${PRIME_RL_UV_CACHE_DIR:-$LUMI_ROOT/.cache/uv}
XDG_CACHE_HOME=${PRIME_RL_XDG_CACHE_HOME:-$LUMI_ROOT/.cache}
WANDB_CACHE_DIR=${PRIME_RL_WANDB_CACHE_DIR:-$LUMI_ROOT/.cache/wandb}
WANDB_CONFIG_DIR=${PRIME_RL_WANDB_CONFIG_DIR:-$LUMI_ROOT/.cache/wandb-config}
MPLCONFIGDIR=${PRIME_RL_MPLCONFIGDIR:-$LUMI_ROOT/.cache/matplotlib}
TMPDIR=${PRIME_RL_TMPDIR:-$LUMI_ROOT/tmp/submit}
CONFIG=${CONFIG:-$REPO_DIR/configs/lumi/sft_single_node.toml}

mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$XDG_CACHE_HOME" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$MPLCONFIGDIR" "$TMPDIR"

export LUMI_ROOT SIF HF_HOME UV_CACHE_DIR XDG_CACHE_HOME WANDB_CACHE_DIR WANDB_CONFIG_DIR MPLCONFIGDIR TMPDIR

CONFIG=$(readlink -f "$CONFIG")
[[ -f "$CONFIG" ]] || {
  echo "FATAL: config not found: $CONFIG" >&2
  exit 1
}
CONTAINER_CONFIG=$CONFIG
if [[ "$CONFIG" == "$REPO_DIR/"* ]]; then
  CONTAINER_CONFIG=/workdir/${CONFIG#"$REPO_DIR/"}
fi

USER_DRY_RUN=false
for arg in "$@"; do
  if [[ "$arg" == "--dry-run" || "$arg" == "--dry-run=true" ]]; then
    USER_DRY_RUN=true
  fi
done

RENDER_ARGS=("$@")
if [[ "$USER_DRY_RUN" == false ]]; then
  RENDER_ARGS+=(--dry-run)
fi

render_log=$(mktemp "$TMPDIR/prime-rl-render.XXXXXX")
trap 'rm -f "$render_log"' EXIT
"$SCRIPT_DIR/lumi_run_in_container.sh" \
  sft @ "$CONTAINER_CONFIG" \
  --slurm.project-dir "$REPO_DIR" \
  "${RENDER_ARGS[@]}" 2>&1 | tee "$render_log"

if [[ "$USER_DRY_RUN" == true ]]; then
  exit 0
fi

SBATCH_PATH=$(sed -E 's/\x1B\[[0-9;]*[mK]//g' "$render_log" \
  | sed -nE 's/^[[:space:]]*sbatch ([^[:space:]]+).*$/\1/p' \
  | tail -n 1)
[[ -n "$SBATCH_PATH" && -f "$SBATCH_PATH" ]] || {
  echo "FATAL: could not find rendered SFT sbatch path in launcher output" >&2
  exit 1
}
SBATCH_BIN=${SBATCH_BIN:-sbatch}
command -v "$SBATCH_BIN" >/dev/null || {
  echo "FATAL: host sbatch command not found: $SBATCH_BIN" >&2
  exit 1
}
"$SBATCH_BIN" "$SBATCH_PATH"
