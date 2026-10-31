#!/bin/bash
#SBATCH --account=iu_0145
#SBATCH --partition=defq
#SBATCH --job-name=io-moe-tune
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --propagate=NONE
#SBATCH --mem=200G
#SBATCH --exclude=dgx035,dgx157
#SBATCH --time=00:50:00
#SBATCH --output=/dcai/projects/iu_0145/rasmus/infer-opt/results/moe-tune-%j.out
# Kernel microbenchmark (stock vs EP-compact MoE) and Triton config tuning for E=64,N=2688,K=1024 relu².
OUT=/dcai/projects/iu_0145/rasmus/infer-opt/results/moe-tune-$SLURM_JOB_ID
source /dcai/projects/iu_0145/rasmus/infer-opt/jobs/lib.sh
cd $IO
echo "--- bench, vLLM default configs"
CUDA_VISIBLE_DEVICES=0 $PY tools/moe_tune.py bench --json $OUT/bench-default.json
echo "--- tune (compact path)"
$PY tools/moe_tune.py tune --out $OUT/configs
echo "--- bench, tuned configs"
CUDA_VISIBLE_DEVICES=0 $PY tools/moe_tune.py bench --configs $OUT/configs --json $OUT/bench-tuned.json
echo "end=$(date -Is)"
