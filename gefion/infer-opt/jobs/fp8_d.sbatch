#!/bin/bash
#SBATCH --account=iu_0145
#SBATCH --partition=defq
#SBATCH --job-name=io-fp8-d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --propagate=NONE
#SBATCH --mem=0
#SBATCH --exclude=dgx035,dgx157
#SBATCH --time=01:30:00
#SBATCH --output=/dcai/projects/iu_0145/rasmus/infer-opt/results/fp8-d-%j.out
# Online FP8 with token-chunked FlashInfer CUTLASS MoE (infer_opt.fi_moe_chunk), alone and with MTP.
# All variants: ep_compact_moe + tuned tiles + fast_generate, gmu 0.91. Load: real i3-math prompts,
# natural EOS (max 16384), GRPO groups of 8, top-33 logprobs, 768 in flight. Parity: the same
# prompts, natural-length seeded decodes, sampler vs own teacher-forced logprobs.
OUT=/dcai/projects/iu_0145/rasmus/infer-opt/results/fp8-d-$SLURM_JOB_ID
VARIANTS=(
  'fp8_chunk|{"vllm.quantization": "fp8_per_block", "vllm.max_num_seqs": 176}|1408'
  'fp8_chunk_mtp2|{"vllm.quantization": "fp8_per_block", "vllm.max_num_seqs": 176, "vllm.speculative_config": {"method": "mtp", "num_speculative_tokens": 2}}|1408'
  'fp8_chunk_mtp2_fp16|{"vllm.quantization": "fp8_per_block", "vllm.max_num_seqs": 176, "vllm.speculative_config": {"method": "mtp", "num_speculative_tokens": 2}, "vllm.mamba_ssm_cache_dtype": "float16"}|1408'
)
ulimit -n $(ulimit -Hn)  # >1024 connections: server and client fds (ReadErrors at 1024 in job 531185)
EXTRA_OPT=fi_moe_chunk
source /dcai/projects/iu_0145/rasmus/infer-opt/jobs/mtp_lib.sh
