#!/usr/bin/env bash
set -euo pipefail
REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ROUTER_BIN="$REPO_DIR/third_party/llmd/bin"
for binary in epp envoy; do
  [[ -x "$ROUTER_BIN/$binary" ]] || {
    echo "FATAL: missing executable $ROUTER_BIN/$binary; provision third_party/llmd in this checkout before submitting" >&2
    exit 1
  }
done
timeout 20 "$ROUTER_BIN/epp" --help >/dev/null 2>&1
timeout 20 "$ROUTER_BIN/envoy" --version
if [[ ${1:-} == --pd ]]; then
  [[ -x "$ROUTER_BIN/pd-sidecar" ]] || { echo "FATAL: missing pd-sidecar" >&2; exit 1; }
  timeout 20 "$ROUTER_BIN/pd-sidecar" --help >/dev/null 2>&1
fi
echo "llm-d executable preflight passed inside the operational container"
