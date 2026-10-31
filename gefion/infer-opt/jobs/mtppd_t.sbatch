#!/bin/bash
#SBATCH --account=iu_0145
#SBATCH --partition=defq
#SBATCH --job-name=io-mtppd-t
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --propagate=NONE
#SBATCH --mem=0
#SBATCH --exclude=dgx035,dgx150,dgx157
#SBATCH --time=00:50:00
#SBATCH --output=/dcai/projects/iu_0145/rasmus/infer-opt/results/mtppd-t-%j.out
# Follow-up to mtppd_s (job 533117): the equal-memory point. MTP doubles the Mamba state blocks per request
# (1 + num_speculative_tokens), so at 96 seqs/rank it fills the cache (kv 0.98) where no-MTP sits at 0.49.
# Here: no-MTP at 176 seqs/rank (its production cap) vs MTP at 80/rank (just under its memory limit).
# What MTP k=1 would buy on a PD decode node: decode-only step time with and without MTP, FP8 DP8+EP8
# (allgather_reducescatter), at 16/32/64/96 sequences per rank, plus prefill-only throughput (does MTP
# make prefill chunks costlier?). Prefix caching off (a decode node gets its KV over NIXL; also avoids
# the MTP + prefix caching < 4176 bug). Synthetic prompts, ignore_eos, 32768-token outputs: nothing
# finishes inside the window, so the window has no prefill at all (pure decode steps).
# Step time per rank = tokens_per_step * concurrency / gen_tok_s (independent of acceptance).
OUT=/dcai/projects/iu_0145/rasmus/infer-opt/results/mtppd-t-$SLURM_JOB_ID
ulimit -n $(ulimit -Hn)
source /dcai/projects/iu_0145/rasmus/infer-opt/jobs/lib.sh
export VLLM_TUNED_CONFIG_FOLDER=$IO/moe_configs INFER_OPT_DG_CACHE_BASE=/tmp/$SLURM_JOB_ID/dg-cache
export INFER_OPT=ep_compact_moe,fast_generate,fi_moe_chunk,dg_jit_per_proc
COMMON='"vllm.gpu_memory_utilization": 0.91, "vllm.enable_prefix_caching": false, "vllm.mamba_cache_mode": "none", "vllm.max_num_batched_tokens": 8704, "vllm.quantization": "fp8_per_block", "vllm.max_num_seqs": 176'
MTP='"vllm.speculative_config": {"method": "mtp", "num_speculative_tokens": 1}'

drain() {  # wait for aborted requests of the previous load to leave the engine
  for i in $(seq 1 36); do
    n=$(curl -s -m 10 localhost:8000/metrics | awk '/^vllm:num_requests_(running|waiting)\{/ {s+=$2} END {print s+0}')
    [ "${n:-1}" = 0 ] && return 0; sleep 5
  done
  echo "drain: $n requests still in the engine"
}

run() {  # run <name> <overrides>
  local name=$1
  if start_server $name "{$2}"; then
    for c in $CONCS; do
      rl $name-dec$c --concurrency $c --warmup 75 --duration 90 --prompt 2048:4096 --output 32768:32768 \
        --logprobs 33 --group 8
      drain
    done
    grep "SpecDecoding metrics" $OUT/$name.log | tail -1 | cut -c40-400
  fi
  stop_server
}

CONCS="1408" run fp8_nomtp "$COMMON"
CONCS="640" run fp8_mtp1 "$COMMON, $MTP"
echo "end=$(date -Is)"
