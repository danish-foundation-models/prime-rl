#!/bin/bash
#SBATCH --account=iu_0145
#SBATCH --partition=defq
#SBATCH --job-name=io-nopc-e
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --propagate=NONE
#SBATCH --mem=0
#SBATCH --exclude=dgx035,dgx157
#SBATCH --time=01:50:00
#SBATCH --output=/dcai/projects/iu_0145/rasmus/infer-opt/results/nopc-e-%j.out
# Prefix caching off (mamba_cache_mode "none"): one Mamba state block per request (+1 per draft token) instead of one per 4176 tokens. bf16.
# All variants: ep_compact_moe + tuned tiles + fast_generate, gmu 0.91. Load: real i3-math prompts,
# natural EOS (max 16384), GRPO groups of 8, top-33 logprobs, 768 in flight. Parity: the same
# prompts, natural-length seeded decodes, sampler vs own teacher-forced logprobs.
OUT=/dcai/projects/iu_0145/rasmus/infer-opt/results/nopc-e-$SLURM_JOB_ID
VARIANTS=(
  'nopc|{"vllm.enable_prefix_caching": false, "vllm.mamba_cache_mode": "none"}'
  'nopc_replay|{"vllm.enable_prefix_caching": false, "vllm.mamba_cache_mode": "none", "vllm.use_replayssm": true}'
  'nopc_mtp1|{"vllm.enable_prefix_caching": false, "vllm.mamba_cache_mode": "none", "vllm.speculative_config": {"method": "mtp", "num_speculative_tokens": 1}}'
  'nopc_mtp2|{"vllm.enable_prefix_caching": false, "vllm.mamba_cache_mode": "none", "vllm.speculative_config": {"method": "mtp", "num_speculative_tokens": 2}}'
)
ulimit -n $(ulimit -Hn)
source /dcai/projects/iu_0145/rasmus/infer-opt/jobs/mtp_lib.sh
