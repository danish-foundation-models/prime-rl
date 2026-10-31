#!/bin/bash
#SBATCH --account=iu_0145
#SBATCH --partition=defq
#SBATCH --job-name=io-offloadfix-o
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --propagate=NONE
#SBATCH --mem=0
#SBATCH --exclude=dgx035,dgx157,dgx150
#SBATCH --time=00:45:00
#SBATCH --output=/dcai/projects/iu_0145/rasmus/infer-opt/results/offloadfix-o-%j.out
# infer_opt.offload_fix with the vllm#52771/#54288/#52807 backports, plus prefix_match_unit 16, on FP8 + MTP k=1
# (align, mnbt 8704, 96 GiB/engine host tier): bench/cache_probe.py --consistency. Job 531411 (no backports,
# default prefix_match_unit) got 0 host-tier tokens for decoded history.
OUT=/dcai/projects/iu_0145/rasmus/infer-opt/results/offloadfix-o-$SLURM_JOB_ID
ulimit -n $(ulimit -Hn)
ulimit -l unlimited 2>/dev/null || true
source /dcai/projects/iu_0145/rasmus/infer-opt/jobs/lib.sh
export VLLM_TUNED_CONFIG_FOLDER=$IO/moe_configs INFER_OPT_DG_CACHE_BASE=/tmp/$SLURM_JOB_ID/dg-cache
export PRIME_RL_CACHE_DIAGNOSTICS=1
PROMPTS=$IO/bench/data/i3math_prompts.json
FP8='"vllm.quantization": "fp8_per_block", "vllm.max_num_seqs": 96'
MTP='"vllm.speculative_config": {"method": "mtp", "num_speculative_tokens": 1}'
CACHE='"vllm.enable_prefix_caching": true, "vllm.mamba_cache_mode": "align", "vllm.max_num_batched_tokens": 8704, "vllm.prefix_match_unit": 16, "kv_cache_offload": {"type": "native", "cpu": {"num_bytes": 103079215104}}'

export INFER_OPT=ep_compact_moe,fast_generate,dp_affinity,fi_moe_chunk,dg_jit_per_proc,offload_fix
rm -f /dev/shm/vllm_offload_* 2>/dev/null
if start_server fp8_mtp1_bp "{\"vllm.gpu_memory_utilization\": 0.91, $CACHE, $FP8, $MTP}"; then
  grep -a -m4 "offload_fix:\|EAGLE/MTP draft attention groups\|prefix_match_unit" $OUT/fp8_mtp1_bp.log | sort -u | cut -c1-260
  timeout 1500 $PY -I $IO/bench/cache_probe.py --model "$MODEL" --tag fp8_mtp1_bp --consistency 2>&1 | tail -24
fi
stop_server

echo "end=$(date -Is)"
