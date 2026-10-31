#!/bin/bash
#SBATCH --account=iu_0145
#SBATCH --partition=defq
#SBATCH --job-name=io-profile1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --propagate=NONE
#SBATCH --mem=0
#SBATCH --exclude=dgx035,dgx157
#SBATCH --time=01:20:00
#SBATCH --output=/dcai/projects/iu_0145/rasmus/infer-opt/results/profile1-%j.out
# Baseline DP8+EP8 bf16 under nsys: synthetic decode, RL-shaped load with/without top-33 logprobs,
# long-context decode; then two zero-code variants (FlashInfer CUTLASS bf16 MoE, ReplaySSM).
OUT=/dcai/projects/iu_0145/rasmus/infer-opt/results/profile1-$SLURM_JOB_ID
source /dcai/projects/iu_0145/rasmus/infer-opt/jobs/lib.sh
cat /proc/sys/kernel/yama/ptrace_scope 2>/dev/null | sed 's/^/ptrace_scope=/'

api_pid() { ss -ltnpH 'sport = :8000' 2>/dev/null | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2; }
drain() {  # wait until the server has no running/waiting requests (client disconnects should abort them)
  local t0=$(date +%s) r
  while :; do
    r=$(curl -s localhost:8000/metrics | grep -E "^vllm:num_requests_(running|waiting)\{" | awk '{s+=$2} END {print s+0}')
    [ "${r%.*}" = 0 ] && break
    [ $(( $(date +%s) - t0 )) -gt 400 ] && { echo "drain timeout ($r left)"; break; }
    sleep 5
  done
  echo "drained in $(( $(date +%s) - t0 )) s"
}
spy() {  # spy <tag> <pid> <seconds>
  [ -n "$2" ] && timeout $(( $3 + 30 )) $PYSPY record --pid $2 --duration $3 --rate 50 --format raw --nonblocking \
    -o $OUT/pyspy-$1.txt >/dev/null 2>&1 && echo "py-spy $1 ok"
}

export INFER_OPT=nvtx
if ! start_server base '{}' nsys; then
  echo "retrying baseline without nsys"; NSYS_OFF=1; start_server base '{}' || exit 1
fi
API=$(api_pid); ENG=$(pgrep -P "$API" | head -1); echo "api_pid=$API engine_pid=$ENG"; ps -o pid,args -p $(pgrep -P "$API" | tr '\n' ',' | sed 's/,$//') | cut -c1-120

parity_run ref
cp $OUT/par-ref.json $OUT/par-ref-keep.json
$PY -I $IO/bench/parity.py gen --model "$MODEL" --out $OUT/par-ref2.json | tail -1  # run-to-run noise floor
$PY -I $IO/bench/parity.py compare --a $OUT/par-ref.json --b $OUT/par-ref2.json --out $OUT/parity-noise.json | tail -1

# A: synthetic (section-11 load), nsys capture in steady decode
( sleep 85; [ -z "${NSYS_OFF:-}" ] && nsys_capture nsys-A-synthetic 4; spy A-api "$API" 15; cpu_snapshot ) &
synthetic base; wait
drain

# B: RL shape, top-33 logprobs (what PRIME-RL requests for score centering)
( sleep 300; [ -z "${NSYS_OFF:-}" ] && nsys_capture nsys-B-rl 4; spy B-api "$API" 20; spy B-engine "$ENG" 20; cpu_snapshot ) &
rl B-rl-lp33 --concurrency 768 --warmup 240 --duration 180 --prompt 1024:8192 --output 8192:32768 --logprobs 33; wait
drain
# C: same without logprobs, to price the logprob path
rl C-rl-lp0 --concurrency 768 --warmup 180 --duration 120 --prompt 1024:8192 --output 8192:32768 --logprobs 0
drain
# D: long contexts (late-rollout shape), distinct prompts
( sleep 200; [ -z "${NSYS_OFF:-}" ] && nsys_capture nsys-D-long 4 ) &
rl D-long-lp33 --concurrency 768 --warmup 150 --duration 120 --prompt 16384:24576 --output 2048:4096 --group 1 --logprobs 33; wait
grep -E "KV cache|Maximum concurrency" $OUT/base.log | sort -u | head -3
stop_server

for v in fi_cutlass replayssm; do
  case $v in
    fi_cutlass) ov='{"vllm.moe_backend": "flashinfer_cutlass"}' ;;
    replayssm) ov='{"vllm.use_replayssm": true}' ;;
  esac
  if start_server $v "$ov"; then
    parity_run $v
    synthetic $v
    drain
    rl $v-rl-lp33 --concurrency 768 --warmup 240 --duration 120 --prompt 1024:8192 --output 8192:32768 --logprobs 33
  fi
  stop_server
done

# nsys reports -> sqlite -> per-step breakdown (device 0 = DP rank 0)
for f in $OUT/nsys-*.nsys-rep; do
  [ -f "$f" ] || continue
  $NSYS export --type sqlite --force-overwrite true -o ${f%.nsys-rep}.sqlite $f >/dev/null 2>&1
  $PY -I $IO/analysis/nsys_breakdown.py ${f%.nsys-rep}.sqlite --json ${f%.nsys-rep}.json > /dev/null && echo "breakdown ${f##*/} ok"
done
echo "end=$(date -Is)"
