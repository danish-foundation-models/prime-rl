#!/bin/bash
#SBATCH --account=iu_0145
#SBATCH --partition=defq
#SBATCH --job-name=io-offloadfix-m
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --propagate=NONE
#SBATCH --mem=0
#SBATCH --exclude=dgx035,dgx157,dgx150
#SBATCH --time=01:15:00
#SBATCH --output=/dcai/projects/iu_0145/rasmus/infer-opt/results/offloadfix-m-%j.out
# infer_opt.offload_fix on FP8 + MTP k=1 (align, mnbt 8704, 96 GiB/engine host tier):
#  1. bench/cache_probe.py: decoded history and full prompt chunks should now come back from host RAM
#     (job 531395 without the patch: 0 and 8352 tokens).
#  2. Correctness: 32 natural-length math decodes on the patched MTP server, scored by an FP8 server
#     without MTP or offload (mnbt 4176: prompt_logprobs at 8704 OOMs), plus the burst check.
OUT=/dcai/projects/iu_0145/rasmus/infer-opt/results/offloadfix-m-$SLURM_JOB_ID
ulimit -n $(ulimit -Hn)
ulimit -l unlimited 2>/dev/null || true
source /dcai/projects/iu_0145/rasmus/infer-opt/jobs/lib.sh
export VLLM_TUNED_CONFIG_FOLDER=$IO/moe_configs INFER_OPT_DG_CACHE_BASE=/tmp/$SLURM_JOB_ID/dg-cache
export PRIME_RL_CACHE_DIAGNOSTICS=1
PROMPTS=$IO/bench/data/i3math_prompts.json
FP8='"vllm.quantization": "fp8_per_block", "vllm.max_num_seqs": 96'
MTP='"vllm.speculative_config": {"method": "mtp", "num_speculative_tokens": 1}'
CACHE='"vllm.enable_prefix_caching": true, "vllm.mamba_cache_mode": "align", "vllm.max_num_batched_tokens": 8704, "kv_cache_offload": {"type": "native", "cpu": {"num_bytes": 103079215104}}'

export INFER_OPT=ep_compact_moe,fast_generate,dp_affinity,fi_moe_chunk,dg_jit_per_proc,offload_fix
rm -f /dev/shm/vllm_offload_* 2>/dev/null
if start_server fp8_mtp1_fix "{\"vllm.gpu_memory_utilization\": 0.91, $CACHE, $FP8, $MTP}"; then
  grep -a -m3 "offload_fix:\|EAGLE/MTP draft attention groups" $OUT/fp8_mtp1_fix.log | sort -u | cut -c1-260
  timeout 1500 $PY -I $IO/bench/cache_probe.py --model "$MODEL" --tag fp8_mtp1_fix 2>&1 | tail -20
  $PY -I $IO/bench/parity.py gen --model "$MODEL" --prompts-file $PROMPTS --n-prompts 32 --reps 1 \
    --max-tokens 16384 --out $OUT/math-fp8_mtp1_fix.json 2>&1 | tail -1
fi
stop_server

export INFER_OPT=ep_compact_moe,fast_generate,fi_moe_chunk,dg_jit_per_proc
if start_server fp8_score "{\"vllm.gpu_memory_utilization\": 0.90, \"vllm.max_num_batched_tokens\": 4176, $FP8}"; then
  f=fp8_mtp1_fix
  $PY -I $IO/bench/parity.py score --model "$MODEL" --seqs $OUT/math-$f.json --out $OUT/math-$f-tf.json 2>&1 | tail -1
  $PY -I $IO/bench/parity.py compare --b $OUT/math-$f.json --b-self-tf $OUT/math-$f-tf.json \
    --out $OUT/math-parity-$f.json 2>&1 | tail -1 | cut -c1-220
  $PY -I $IO/analysis/mismatch_bursts.py $OUT/math-$f.json $OUT/math-$f-tf.json | tail -1
fi
stop_server
echo "end=$(date -Is)"
