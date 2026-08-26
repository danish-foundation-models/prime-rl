#!/usr/bin/env bash
set -euo pipefail

[[ $# -ge 1 ]] || {
  echo "usage: $0 TASKSET_ID [VALIDATE_ARGS ...]" >&2
  exit 2
}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd -P)
TASKSET_ID=$1
shift

set -a
for ENV_FILE in "$PROJECT_DIR/../post/.env" "$PROJECT_DIR/.env"; do
  if [[ -f "$ENV_FILE" ]]; then
    source "$ENV_FILE"
  fi
done
set +a

exec "$SCRIPT_DIR/lumi_run_in_container.sh" \
  python -m verifiers.v1.cli.validate "$TASKSET_ID" \
  --only-setup \
  --num-tasks 1 \
  --max-concurrent 1 \
  --no-rich \
  --runtime.type ucloud \
  --runtime.user root \
  --runtime.request-timeout-seconds 600 \
  --runtime.create-timeout-seconds 600 \
  "$@"
