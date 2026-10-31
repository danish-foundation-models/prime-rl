#!/bin/bash
#SBATCH --account=iu_0145
#SBATCH --partition=defq
#SBATCH --job-name=io-mtp-c
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --propagate=NONE
#SBATCH --mem=0
#SBATCH --exclude=dgx035,dgx157
#SBATCH --time=01:30:00
#SBATCH --output=/dcai/projects/iu_0145/rasmus/infer-opt/results/mtp-c-%j.out
# Online FP8 (fp8_per_block, FlashInfer CUTLASS block-FP8 relu² MoE) on DP8+EP8, alone and with MTP.
# All variants: ep_compact_moe + tuned tiles + fast_generate, gmu 0.91. Load: real i3-math prompts,
# natural EOS (max 16384), GRPO groups of 8, top-33 logprobs, 768 in flight. Parity: the same
# prompts, natural-length seeded decodes, sampler vs own teacher-forced logprobs.
OUT=/dcai/projects/iu_0145/rasmus/infer-opt/results/mtp-c-$SLURM_JOB_ID
VARIANTS=(
  'fp8|{"vllm.quantization": "fp8_per_block", "vllm.max_num_seqs": 128}|1024'
  'fp8_mtp2|{"vllm.quantization": "fp8_per_block", "vllm.max_num_seqs": 128, "vllm.speculative_config": {"method": "mtp", "num_speculative_tokens": 2}}|1024'
  'fp8_mtp2_fp16|{"vllm.quantization": "fp8_per_block", "vllm.max_num_seqs": 128, "vllm.speculative_config": {"method": "mtp", "num_speculative_tokens": 2}, "vllm.mamba_ssm_cache_dtype": "float16"}|1024'
)
source /dcai/projects/iu_0145/rasmus/infer-opt/jobs/mtp_lib.sh
