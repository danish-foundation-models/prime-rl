#!/usr/bin/env bash
set -euo pipefail
: "${PROJECT_DIR:?}" "${LUMI_ROOT:?}" "${CALIBRATION_ROOT:?}"
source "$PROJECT_DIR/scripts/lumi_super_inference_env.sh"
export TMPDIR="$LUMI_ROOT/tmp/rl-calibration-${SLURM_JOB_ID:?}"
mkdir -p "$TMPDIR"
"$LUMI_ROOT/bin/uv" run --no-project python "$CALIBRATION_ROOT/kernel_preflight.py"
"$LUMI_ROOT/bin/uv" run --no-project inference @ "$CALIBRATION_ROOT/calibration-inference.json" >"$CALIBRATION_ROOT/calibration-server-${SLURM_JOB_ID}.log" 2>&1 &
server_pid=$!
trap 'kill -TERM "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' EXIT
ready=0
for ((attempt=0; attempt<240; attempt++)); do
    kill -0 "$server_pid" 2>/dev/null || { echo "Inference exited before readiness"; exit 1; }
    if curl --silent --fail --max-time 3 http://127.0.0.1:8125/health >/dev/null; then ready=1; break; fi
    sleep 5
done
[[ "$ready" == 1 ]] || { echo "Inference did not become ready within 20 minutes"; exit 1; }
"$LUMI_ROOT/bin/uv" run --no-project python "$CALIBRATION_ROOT/calibration_probe.py"
"$LUMI_ROOT/bin/uv" run --no-project eval @ "$CALIBRATION_ROOT/calibration-configs/smoke.json"
echo "SMOKE_COMPLETE: inspect traces before expanding calibration"
