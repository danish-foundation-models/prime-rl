#!/usr/bin/env bash
set -euo pipefail
: "${CAPACITY_ROOT:?}" "${PROJECT_DIR:?}" "${LUMI_ROOT:?}"
source "$PROJECT_DIR/scripts/lumi_super_inference_env.sh"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
if [[ "${CAPACITY_HOST_CACHE:-0}" == 1 ]]; then export PRIME_RL_CACHE_DIAGNOSTICS=1; fi
export FI_PROVIDER=cxi FI_PROVIDER_NAME=cxi FI_CXI_DISABLE_HMEM=1 FI_CXI_DISABLE_HMEM_FLAG=1
export GLOO_SOCKET_IFNAME=hsn0 NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_PLUGIN=/usr/lib/x86_64-linux-gnu/librccl-net.so NCCL_NET=Libfabric NCCL_NET_GDR_LEVEL=0
export NCCL_DMABUF_ENABLE=0 TORCH_NCCL_HIGH_PRIORITY=1 GPU_MAX_HW_QUEUES=2
export PRIME_RL_WEIGHT_NCCL_ALGO=Ring PRIME_RL_WEIGHT_NCCL_PROTO=LL PRIME_RL_WEIGHT_HIGH_PRIORITY=1
export PRIME_RL_WEIGHT_MEMORY_PROBE=1
export PRIME_RL_WEIGHT_BUCKET_BYTES=268435456
# Diagnostic workaround for the HSA copy-path crash during communicator setup.
export HSA_ENABLE_SDMA=${HSA_ENABLE_SDMA:-0}
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET PYTHONFAULTHANDLER=1
export TMPDIR="$LUMI_ROOT/tmp/capacity-$SLURM_JOB_ID-$SLURM_PROCID"
mkdir -p "$TMPDIR"
if [[ "${CAPACITY_WIRE_PROBE:-0}" == 1 ]]; then
    wire_args=()
    [[ "$SLURM_PROCID" == 1 ]] && wire_args+=(--sender)
    PRIME_RL_PROBE_NATIVE_WRAPPER=1 PRIME_RL_WEIGHT_BUCKET_BYTES=33554432 "$LUMI_ROOT/bin/uv" run --no-project python "$PROJECT_DIR/scripts/lumi_super_nccl_probe.py" \
        --host "$SENDER_HOST" --port 29611 --output "$CAPACITY_ROOT/job-$SLURM_JOB_ID/wire" "${wire_args[@]}"
fi
budget=$1
stage="$CAPACITY_ROOT/job-$SLURM_JOB_ID/cache-$budget"
mkdir -p "$stage"
if [[ "$SLURM_PROCID" == 0 ]]; then
    "$LUMI_ROOT/bin/uv" run --no-project inference @ "$CAPACITY_ROOT/${CAPACITY_CONFIG_PREFIX:-inference}-$budget.json" > "$stage/server.log" 2>&1 &
    server_pid=$!
    trap 'kill -TERM "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' EXIT
    for ((i=0; i<5400; i++)); do
        [[ -f "$stage/done" ]] && exit 0
        kill -0 "$server_pid" 2>/dev/null || { echo 'Inference service died'; exit 1; }
        sleep 2
    done
    echo 'Capacity test timed out'; exit 1
else
    "$LUMI_ROOT/bin/uv" run --no-project python "$PROJECT_DIR/scripts/lumi_super_capacity_probe.py" \
        ${CAPACITY_HOST_CACHE:+--host-cache} ${CAPACITY_PREFIX_PROBE:+--prefix-probe} ${CAPACITY_PREFIX_STRICT:+--prefix-strict} --server "http://$INFERENCE_HOST:8125" --sender "$SENDER_HOST" --cache-gib "$budget" \
        --model "$LUMI_ROOT/artifacts/training/nemotron-super-base-r5-16n-cp8-ep8-bs32-lr1e5-r1/eval-exports/step_1250" \
        --output "$stage" > "$stage/probe.log" 2>&1
fi
