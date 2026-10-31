#!/bin/bash
#SBATCH --account=iu_0145
#SBATCH --partition=defq
#SBATCH --job-name=io-final1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --propagate=NONE
#SBATCH --mem=0
#SBATCH --exclude=dgx035,dgx157
#SBATCH --time=01:15:00
#SBATCH --output=/dcai/projects/iu_0145/rasmus/infer-opt/results/final1-%j.out
# RL-shaped load, nsys and py-spy for stock vs the recommended stack, plus 8K-token
# sampler-vs-teacher-forced parity (does decode error grow with length?) for stock, the
# recommended fp32-state stack and the fp16-state variant.
OUT=/dcai/projects/iu_0145/rasmus/infer-opt/results/final1-$SLURM_JOB_ID
source /dcai/projects/iu_0145/rasmus/infer-opt/jobs/lib.sh
TUNED=$IO/moe_configs

api_pid() { ss -ltnpH 'sport = :8000' 2>/dev/null | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2; }
drain() {
  local t0=$(date +%s) r
  while :; do
    r=$(curl -s -m 120 localhost:8000/metrics | grep -E "^vllm:num_requests_(running|waiting)\{" | awk '{s+=$2} END {print s+0}')
    [ "${r%.*}" = 0 ] && break
    [ $(( $(date +%s) - t0 )) -gt 400 ] && { echo "drain timeout ($r left)"; break; }
    sleep 5
  done
  echo "drained in $(( $(date +%s) - t0 )) s"
}
spy() {
  [ -n "$2" ] && timeout $(( $3 + 30 )) $PYSPY record --pid $2 --duration $3 --rate 50 --format raw --nonblocking \
    -o $OUT/pyspy-$1.txt >/dev/null 2>&1 && echo "py-spy $1 ok"
}
cap() {
  echo "$NSYS start --session=io$SLURM_JOB_ID --sample=none --cpuctxsw=none -f true -o $OUT/nsys-$1 >/dev/null 2>&1 && sleep 3 && $NSYS stop --session=io$SLURM_JOB_ID >/dev/null 2>&1 && echo 'nsys capture $1 done'"
}
long_parity() {  # 32 prompts x 8192 tokens: own decode logprobs vs own teacher-forced logprobs
  $PY -I $IO/bench/parity.py gen --model "$MODEL" --reps 1 --max-tokens 8192 --out $OUT/long-$1.json 2>&1 | tail -1
  $PY -I $IO/bench/parity.py score --model "$MODEL" --seqs $OUT/long-$1.json --out $OUT/long-$1-tf.json 2>&1 | tail -1
  $PY -I $IO/bench/parity.py compare --b $OUT/long-$1.json --b-self-tf $OUT/long-$1-tf.json --out $OUT/long-parity-$1.json 2>&1 | tail -1
}
rl_profiled() {  # rl_profiled <tag>
  local API ENG
  API=$(api_pid); ENG=$(pgrep -f "VLLM::EngineCore_DP0" | head -1)
  ( sleep 270; spy $1-api "$API" 20; spy $1-engine "$ENG" 20; cpu_snapshot ) &
  rl $1-rl-lp33 --concurrency 768 --warmup 240 --duration 150 --prompt 1024:8192 --output 8192:32768 --logprobs 33 \
    --post-hook "$(cap $1-rl)"
  wait
  grep "Engine 000:" $OUT/$1.log | tail -4 | cut -c40-250
  drain
}

REC='{"vllm.use_replayssm": true, "vllm.gpu_memory_utilization": 0.93}'

# 1. stock
export INFER_OPT=nvtx; unset VLLM_TUNED_CONFIG_FOLDER
if start_server stock '{}' nsys; then
  long_parity stock
  rl_profiled stock
fi
stop_server

# 2. recommended: EP-compact MoE + tuned configs + fast generate path + ReplaySSM (fp32 state)
export INFER_OPT=nvtx,ep_compact_moe,fast_generate VLLM_TUNED_CONFIG_FOLDER=$TUNED
if start_server rec "$REC" nsys; then
  long_parity rec
  rl_profiled rec
  synthetic rec
fi
stop_server

# 3. recommended with an fp16 SSM state cache (halves Mamba state memory and traffic)
export INFER_OPT=ep_compact_moe,fast_generate
if start_server rec_fp16 '{"vllm.use_replayssm": true, "vllm.gpu_memory_utilization": 0.93, "vllm.mamba_ssm_cache_dtype": "float16"}'; then
  long_parity rec_fp16
  rl rec_fp16-rl-lp33 --concurrency 768 --warmup 240 --duration 150 --prompt 1024:8192 --output 8192:32768 --logprobs 33
  grep "Engine 000:" $OUT/rec_fp16.log | tail -4 | cut -c40-250
fi
stop_server

for f in $OUT/nsys-*.nsys-rep; do
  [ -f "$f" ] || continue
  $NSYS export --type sqlite --force-overwrite true -o ${f%.nsys-rep}.sqlite $f >/dev/null 2>&1
  $PY -I $IO/analysis/nsys_breakdown.py ${f%.nsys-rep}.sqlite --json ${f%.nsys-rep}.json > /dev/null && echo "breakdown ${f##*/} ok"
done
echo "end=$(date -Is)"
