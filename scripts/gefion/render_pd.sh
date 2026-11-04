#!/bin/bash
# Render a PD (inference.deployment.type = "disaggregated") RL config: render.sh (dry run + Gefion
# patches), then switch the node ordering to order_nodes_pd.sh (iommu=pt nodes to the prefill/decode
# pair) and check that the per-role vLLM overrides reach the rendered script.
# Usage: render_pd.sh <config.toml>
set -euo pipefail
out=$(bash /dcai/projects/iu_0145/rasmus/phase0/full/render.sh "$1") || { echo "$out"; exit 1; }
echo "$out"
L=$(echo "$out" | sed -n 's/^rendered: //p')
[ -f "$L" ] || { echo "render_pd: no rendered sbatch"; exit 1; }
grep -q 'full/order_nodes.sh)' "$L" || { echo "render_pd: order_nodes.sh hook not found"; exit 1; }
sed -i 's|full/order_nodes.sh)|full/order_nodes_pd.sh)|' "$L"
# LOCAL_IP: `hostname -I` lists bmc_redfish0 (169.254.0.18, the SAME link-local address on every
# Gefion node) first. It becomes VLLM_NIXL_SIDE_CHANNEL_HOST, so D handshook with its own engines
# ("Remote NIXL agent engine ID mismatch", every transfer failed; job 533137). Skip link-local.
grep -q 'LOCAL_IP=$(hostname -I | awk "{print \\$1}")' "$L" || { echo "render_pd: LOCAL_IP line not found"; exit 1; }
sed -i 's/LOCAL_IP=$(hostname -I | awk "{print \\$1}")/LOCAL_IP=$(hostname -I | tr " " "\\n" | grep -v "^169\\.254\\." | head -1)/' "$L"
grep -q 'grep -v "^169' "$L" || { echo "render_pd: LOCAL_IP patch failed"; exit 1; }
grep -q 'ROLE="prefill"' "$L" || { echo "render_pd: not a PD script"; exit 1; }
# TP8 PD (one engine per role group, DP=1): vLLM rejects the external-LB DP args when
# data_parallel_size == 1 ("data_parallel_external_lb can only be set when data_parallel_size > 1",
# job 533478), so pass them only for a real DP group. No change for DP8.
old_extra='        RANK_EXTRA="{\"data_parallel_address\": \"$DP_ADDR\", \"data_parallel_rank\": $GLOBAL_RANK, $ROLE_EXTRA}"'
grep -qxF "$old_extra" "$L" || { echo "render_pd: RANK_EXTRA line not found"; exit 1; }
python3 - "$L" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
old_extra = '        RANK_EXTRA="{\\"data_parallel_address\\": \\"$DP_ADDR\\", \\"data_parallel_rank\\": $GLOBAL_RANK, $ROLE_EXTRA}"\n'
new_extra = ('        if [ "$DP" -gt 1 ]; then\n'
             '            RANK_EXTRA="{\\"data_parallel_address\\": \\"$DP_ADDR\\", \\"data_parallel_rank\\": $GLOBAL_RANK, $ROLE_EXTRA}"; RANK_RPC=$INFERENCE_DATA_PARALLEL_RPC_PORT\n'
             '        else\n'
             '            RANK_EXTRA="{$ROLE_EXTRA}"; RANK_RPC=""\n'
             '        fi\n')
old_launch = 'launch_inference_rank "$((PORT + k))" "$RANK_GPUS" "$DP" "$INFERENCE_DATA_PARALLEL_RPC_PORT" "$RANK_EXTRA"'
new_launch = 'launch_inference_rank "$((PORT + k))" "$RANK_GPUS" "$DP" "$RANK_RPC" "$RANK_EXTRA"'
assert s.count(old_extra) == 1 and s.count(old_launch) == 1, "render_pd: DP=1 patch does not apply"
open(p, "w").write(s.replace(old_extra, new_extra).replace(old_launch, new_launch))
PY
grep -c 'allgather_reducescatter' "$L" | grep -qx 2 || { echo "render_pd: all2all overrides missing"; exit 1; }
# Prefill routing: consistent_hash pins each session to one P with no load awareness. In job 534149
# one P thrashed its prefix cache (~900 queued, 17% hits) while the other idled. sticky_load (our
# build of PrimeIntellect-ai/router fix/cache-aware-load-accounting-v0.2.0, phase0/router-src) keeps a
# session on its P until that P's in-flight load exceeds max(1.5 x mean, mean + 32), then moves it to
# the least-loaded P; new sessions go to the least-loaded P. Decode uses it too since step 15: with 4 D
# nodes, consistent_hash left one D at 85% KV with 98 queued and preempting (job 534552) while the
# others idled; a session's KV comes from P every turn, so moving it between D nodes is cheap.
old_router='launch_router pd "$ROUTER_ARGS" "$ROUTER_PORT" "consistent_hash" "$LOG_DIR/inference/router.log"'
new_router='launch_router pd "$ROUTER_ARGS --prefill-policy sticky_load --decode-policy sticky_load --balance-rel-threshold 1.5 --balance-abs-threshold 32" "$ROUTER_PORT" "consistent_hash" "$LOG_DIR/inference/router.log"'
grep -qF "$old_router" "$L" || { echo "render_pd: router launch line not found"; exit 1; }
python3 - "$L" "$old_router" "$new_router" <<'PY'
import sys
p, old, new = sys.argv[1:]
s = open(p).read()
assert s.count(old) == 1
open(p, "w").write(s.replace(old, new))
PY
grep -qF -- '--prefill-policy sticky_load' "$L" || { echo "render_pd: router policy patch failed"; exit 1; }
# Broadcast markers left by a previous job: on resume the orchestrator can find the old v{resume}
# .sender_ready before the trainer resets that dir, ack it, and deadlock the startup weight sync
# (535952). Nothing is running yet when the batch script reaches this line, so clear them all.
logs_line='ln -sfn inference/node_0.log $LOG_DIR/inference.log'
grep -qxF "$logs_line" "$L" || { echo "render_pd: log symlink line not found"; exit 1; }
sed -i "\|^$logs_line\$|a rm -rf \$OUTPUT_DIR/broadcasts/step_*" "$L"
grep -qxF 'rm -rf $OUTPUT_DIR/broadcasts/step_*' "$L" || { echo "render_pd: broadcast cleanup patch failed"; exit 1; }
bash -n "$L"
echo "render_pd: $L"
grep -nE "order_nodes_pd|ROLE_EXTRA=|UCX_NET_DEVICES=|VLLM_SSM_CONV_STATE_LAYOUT|INFER_OPT=" "$L"
