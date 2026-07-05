#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

set -a
for ENV_FILE in "$REPO_DIR/../../.env" "$REPO_DIR/.env"; do
  if [[ -f "$ENV_FILE" ]]; then
    source "$ENV_FILE"
  fi
done
set +a

FLASH_PROJECT_ROOT=${FLASH_PROJECT_ROOT:-/flash/project_465002183}
HF_HOME=${HF_HOME:-$FLASH_PROJECT_ROOT/.cache/huggingface}
UV_CACHE_DIR=${UV_CACHE_DIR:-$FLASH_PROJECT_ROOT/.cache/uv}
UV_NO_SYNC=${UV_NO_SYNC:-1}
CONFIG=${CONFIG:-$REPO_DIR/configs/lumi/rl_single_node.toml}
ORCH_USE_TOKEN_CLIENT=${ORCH_USE_TOKEN_CLIENT:-}

mkdir -p "$HF_HOME" "$UV_CACHE_DIR"

export FLASH_PROJECT_ROOT
export HF_HOME
export UV_CACHE_DIR
export UV_NO_SYNC
export ORCH_USE_TOKEN_CLIENT

cd "$REPO_DIR"
ORCH_TOKEN_CLIENT_FLAG=()
if [[ "$ORCH_USE_TOKEN_CLIENT" == "true" ]]; then
  ORCH_TOKEN_CLIENT_FLAG=(--orchestrator.use-token-client)
fi

uv run rl @ "$CONFIG" "${ORCH_TOKEN_CLIENT_FLAG[@]}" "$@"
