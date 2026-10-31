#!/bin/bash
#SBATCH --account=iu_0145
#SBATCH --partition=defq
#SBATCH --job-name=io-mtp-a
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --propagate=NONE
#SBATCH --mem=0
#SBATCH --exclude=dgx035,dgx157
#SBATCH --time=01:30:00
#SBATCH --output=/dcai/projects/iu_0145/rasmus/infer-opt/results/mtp-a-%j.out
# MTP speculative decoding on DP8+EP8 (ReplaySSM off: vLLM rejects it with spec decoding).
# All variants: ep_compact_moe + tuned tiles + fast_generate, gmu 0.91. Load: real i3-math prompts,
# natural EOS (max 16384), GRPO groups of 8, top-33 logprobs, 768 in flight. Parity: the same
# prompts, natural-length seeded decodes, sampler vs own teacher-forced logprobs.
OUT=/dcai/projects/iu_0145/rasmus/infer-opt/results/mtp-a-$SLURM_JOB_ID
VARIANTS=(
  'nospec|{}'
  'mtp1|{"vllm.speculative_config": {"method": "mtp", "num_speculative_tokens": 1}}'
  'mtp2|{"vllm.speculative_config": {"method": "mtp", "num_speculative_tokens": 2}}'
)
source /dcai/projects/iu_0145/rasmus/infer-opt/jobs/mtp_lib.sh
