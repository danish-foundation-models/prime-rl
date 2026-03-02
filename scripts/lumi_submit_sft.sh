#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

FLASH_PROJECT_ROOT=${FLASH_PROJECT_ROOT:-/flash/project_465002183}
HF_HOME=${HF_HOME:-$FLASH_PROJECT_ROOT/.cache/huggingface}
UV_CACHE_DIR=${UV_CACHE_DIR:-$FLASH_PROJECT_ROOT/.cache/uv}
CONFIG=${CONFIG:-$REPO_DIR/configs/lumi/sft_single_node.toml}

mkdir -p "$HF_HOME" "$UV_CACHE_DIR"

export FLASH_PROJECT_ROOT
export HF_HOME
export UV_CACHE_DIR

cd "$REPO_DIR"
uv run sft @ "$CONFIG" "$@"
