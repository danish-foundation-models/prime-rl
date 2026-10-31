#!/bin/bash
#SBATCH --account=iu_0145
#SBATCH --partition=defq
#SBATCH --job-name=io-fp8mtp-l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --propagate=NONE
#SBATCH --mem=0
#SBATCH --exclude=dgx035,dgx157,dgx150
#SBATCH --time=01:50:00
#SBATCH --output=/dcai/projects/iu_0145/rasmus/infer-opt/results/fp8mtp-l-%j.out
# FP8 + MTP with prefix caching on and host-RAM KV offload, on a multi-turn agent-style load.
# Multi-turn rerun behind PRIME-RL's own router: vllm-router, consistent_hash on x-session-id across the 8
# local DP ranks (the inference entrypoint's default), with rl_load sending one x-session-id per rollout as
# the verifiers clients do. No dp_affinity patch. Engine (and /metrics) on backend port 8100.
# max_num_batched_tokens 8704 (job 531287), prefix caching (align) + 96 GiB/engine host offload.
# Job 531399 (same, MTP at 176 seqs/rank) was cancelled before the load started.
# Last step: a bf16 server without MTP (mnbt 4176, no router) scores job 531249's FP8+MTP sequences, for
# FP8 sampler logprobs vs bf16 teacher forcing (the scoring OOMed at mnbt 8704 in job 531287).
OUT=/dcai/projects/iu_0145/rasmus/infer-opt/results/fp8mtp-l-$SLURM_JOB_ID
ulimit -n $(ulimit -Hn)
ulimit -l unlimited 2>/dev/null || true  # pinned host buffers for the offload tier
source /dcai/projects/iu_0145/rasmus/infer-opt/jobs/lib.sh
export VLLM_TUNED_CONFIG_FOLDER=$IO/moe_configs INFER_OPT_DG_CACHE_BASE=/tmp/$SLURM_JOB_ID/dg-cache
H=$IO/results/fp8mtp-h-531249
PROMPTS=$IO/bench/data/i3math_prompts.json
COMMON='"vllm.gpu_memory_utilization": 0.91, "vllm.enable_prefix_caching": true, "vllm.mamba_cache_mode": "align", "vllm.max_num_batched_tokens": 8704, "kv_cache_offload": {"type": "native", "cpu": {"num_bytes": 103079215104}}, "router": {"type": "vllm-router", "policy": "consistent_hash"}'
FP8='"vllm.quantization": "fp8_per_block", "vllm.max_num_seqs": 176'
mtp() { echo "\"vllm.speculative_config\": {\"method\": \"mtp\", \"num_speculative_tokens\": $1}"; }

run() {  # run <name> <INFER_OPT> <overrides> <concurrency> [noparity]
  local name=$1 conc=$4 noparity=${5:-}
  export INFER_OPT=$2
  rm -f /dev/shm/vllm_offload_* 2>/dev/null
  if start_server $name "{$3}"; then
    grep -m4 -oE "Available KV cache memory: [0-9.]+ GiB|Setting attention block size to [0-9]+|Using [A-Z_]+ Fp8 MoE backend|OffloadingConnector[^,]*" $OUT/$name.log | sort -u
    # 8-turn agent-style rollouts: each turn re-sends the history + a synthetic tool result
    rl $name-multiturn --concurrency $conc --warmup 360 --duration 240 --prompts-file $PROMPTS --output 1:4096 \
      --turns 8 --tool-tokens 256:2048 --max-context 65536 --logprobs 33 \
      --session-header --metrics-url http://localhost:8100
    grep "Engine 000:" $OUT/$name.log | tail -2 | cut -c40-260
    grep "SpecDecoding metrics" $OUT/$name.log | tail -1 | grep -oE "Mean acceptance length: [0-9.]+"
    grep -m3 -iE "offload.*(hit|load|store)[^ ]* [0-9]" $OUT/$name.log | tail -2 | cut -c40-260
    [ -n "$noparity" ] || { $PY -I $IO/bench/parity.py gen --model "$MODEL" --prompts-file $PROMPTS --n-prompts 32 --reps 1 \
      --max-tokens 16384 --out $OUT/math-$name.json 2>&1 | tail -1
    $PY -I $IO/bench/parity.py score --model "$MODEL" --seqs $OUT/math-$name.json --out $OUT/math-$name-tf.json 2>&1 | tail -1
    $PY -I $IO/bench/parity.py compare --b $OUT/math-$name.json --b-self-tf $OUT/math-$name-tf.json \
      --out $OUT/math-parity-$name.json 2>&1 | tail -1 | cut -c1-220
    $PY -I $IO/analysis/mismatch_bursts.py $OUT/math-$name.json $OUT/math-$name-tf.json | tail -1; }
  fi
  stop_server
}

P=ep_compact_moe,fast_generate
# MTP keeps ~1.6x the cache blocks per request: at 176 seqs/rank job 531287 filled the GPU cache at ~105
# running and fell into a preempt/re-prefill storm (hits ~0). Cap MTP at 96 seqs/rank.
run fp8_mtp1 $P,fi_moe_chunk,dg_jit_per_proc "$COMMON, $FP8, $(mtp 1), \"vllm.max_num_seqs\": 96" 1408 noparity
run fp8_nomtp $P,fi_moe_chunk,dg_jit_per_proc "$COMMON, $FP8" 1408 noparity

export INFER_OPT=$P
if start_server bf16_score '{"vllm.gpu_memory_utilization": 0.90, "vllm.max_num_batched_tokens": 4176}'; then
  f=fp8_mtp1_h; cp $H/math-fp8_mtp1.json $OUT/math-$f.json
  $PY -I $IO/bench/parity.py score --model "$MODEL" --seqs $OUT/math-$f.json --out $OUT/math-$f-tf-bf16.json 2>&1 | tail -1
  $PY -I $IO/bench/parity.py compare --b $OUT/math-$f.json --b-self-tf $OUT/math-$f-tf-bf16.json \
    --out $OUT/math-parity-$f-vs-bf16.json 2>&1 | tail -1 | sed "s/^/$f vs bf16: /" | cut -c1-240
fi
stop_server
echo "end=$(date -Is)"
