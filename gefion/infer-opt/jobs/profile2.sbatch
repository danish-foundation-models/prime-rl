#!/bin/bash
#SBATCH --account=iu_0145
#SBATCH --partition=defq
#SBATCH --job-name=io-profile2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --propagate=NONE
#SBATCH --mem=0
#SBATCH --exclude=dgx035,dgx157
#SBATCH --time=00:55:00
#SBATCH --output=/dcai/projects/iu_0145/rasmus/infer-opt/results/profile2-%j.out
# Stock DP8+EP8 bf16 under nsys: RL-shaped load with and without top-33 logprobs and long-context
# decode. nsys captures run after each measurement window (the flush stalls the workers), and
# py-spy samples the API server and one engine core inside the window.
OUT=/dcai/projects/iu_0145/rasmus/infer-opt/results/profile2-$SLURM_JOB_ID
source /dcai/projects/iu_0145/rasmus/infer-opt/jobs/lib.sh

api_pid() { ss -ltnpH 'sport = :8000' 2>/dev/null | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2; }
drain() {
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
cap() {  # post-hook command string: 3 s nsys capture
  echo "$NSYS start --session=io$SLURM_JOB_ID --sample=none --cpuctxsw=none -f true -o $OUT/nsys-$1 >/dev/null 2>&1 && sleep 3 && $NSYS stop --session=io$SLURM_JOB_ID >/dev/null 2>&1 && echo 'nsys capture $1 done'"
}

export INFER_OPT=nvtx
start_server base '{}' nsys || exit 1
API=$(api_pid); ENG=$(pgrep -f "VLLM::EngineCore_DP0" | head -1); echo "api_pid=$API engine_pid=$ENG"

# B: RL shape with top-33 logprobs, as PRIME-RL requests them for score centering
( sleep 300; spy B-api "$API" 20; spy B-engine "$ENG" 20; cpu_snapshot ) &
rl B-rl-lp33 --concurrency 768 --warmup 240 --duration 150 --prompt 1024:8192 --output 8192:32768 --logprobs 33 \
  --post-hook "$(cap B-rl)"; wait
drain
# C: same without logprobs
( sleep 200; spy C-api "$API" 20; cpu_snapshot ) &
rl C-rl-lp0 --concurrency 768 --warmup 180 --duration 120 --prompt 1024:8192 --output 8192:32768 --logprobs 0 \
  --post-hook "$(cap C-rl)"; wait
drain
# D: long contexts (late-rollout shape), distinct prompts
rl D-long-lp33 --concurrency 768 --warmup 150 --duration 120 --prompt 16384:24576 --output 2048:4096 --group 1 --logprobs 33 \
  --post-hook "$(cap D-long)"
stop_server

for f in $OUT/nsys-*.nsys-rep; do
  [ -f "$f" ] || continue
  $NSYS export --type sqlite --force-overwrite true -o ${f%.nsys-rep}.sqlite $f >/dev/null 2>&1
  $PY -I $IO/analysis/nsys_breakdown.py ${f%.nsys-rep}.sqlite --json ${f%.nsys-rep}.json > /dev/null && echo "breakdown ${f##*/} ok"
done
echo "end=$(date -Is)"
