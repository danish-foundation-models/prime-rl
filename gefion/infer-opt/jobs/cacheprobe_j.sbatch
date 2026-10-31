#!/bin/bash
#SBATCH --account=iu_0145
#SBATCH --partition=defq
#SBATCH --job-name=io-cacheprobe-j
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --propagate=NONE
#SBATCH --mem=0
#SBATCH --exclude=dgx035,dgx157,dgx150
#SBATCH --time=01:00:00
#SBATCH --output=/dcai/projects/iu_0145/rasmus/infer-opt/results/cacheprobe-j-%j.out
# Controlled prefix-cache and host-offload probe (bench/cache_probe.py), FP8 with and without MTP.
# Why: on the multi-turn load (job 531287) FP8 got 42% prefix hits without MTP and ~0% with MTP k=1,
# and the host offload tier never served a hit.
OUT=/dcai/projects/iu_0145/rasmus/infer-opt/results/cacheprobe-j-$SLURM_JOB_ID
ulimit -n $(ulimit -Hn)
ulimit -l unlimited 2>/dev/null || true
source /dcai/projects/iu_0145/rasmus/infer-opt/jobs/lib.sh
export VLLM_TUNED_CONFIG_FOLDER=$IO/moe_configs INFER_OPT_DG_CACHE_BASE=/tmp/$SLURM_JOB_ID/dg-cache
export PRIME_RL_CACHE_DIAGNOSTICS=1 INFER_OPT=ep_compact_moe,fast_generate,dp_affinity,fi_moe_chunk,dg_jit_per_proc
COMMON='"vllm.gpu_memory_utilization": 0.91, "vllm.enable_prefix_caching": true, "vllm.mamba_cache_mode": "align", "vllm.max_num_batched_tokens": 8704, "kv_cache_offload": {"type": "native", "cpu": {"num_bytes": 103079215104}}, "vllm.quantization": "fp8_per_block", "vllm.max_num_seqs": 176'
for v in mtp1 nomtp; do
  ov="$COMMON"
  [ $v = mtp1 ] && ov="$COMMON, \"vllm.speculative_config\": {\"method\": \"mtp\", \"num_speculative_tokens\": 1}"
  rm -f /dev/shm/vllm_offload_* 2>/dev/null
  if start_server fp8_$v "{$ov}"; then
    timeout 1500 $PY -I $IO/bench/cache_probe.py --model "$MODEL" --tag fp8_$v 2>&1 | tail -20
  fi
  stop_server
done
echo "end=$(date -Is)"
