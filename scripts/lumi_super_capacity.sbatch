#!/usr/bin/env bash
#SBATCH --job-name=super-128k-cache
#SBATCH --account=project_465002751
#SBATCH --partition=standard-g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/project_465002751/rasmus/artifacts/training/nemotron-super-r5-rl-main-20260916/capacity/job-%j.log
set -euo pipefail
export PROJECT_DIR=/scratch/project_465002751/rasmus/tmp/nemotron-super-rl-20260916
export LUMI_ROOT=/scratch/project_465002751/rasmus
export SIF=/scratch/project_465002751/shared/sif/lumi-posttrain-u24r71-t211-vllm024.sif
export CAPACITY_ROOT="$LUMI_ROOT/artifacts/training/nemotron-super-r5-rl-main-20260916/capacity"
export PRIME_RL_METADATA_DIR=${PRIME_RL_METADATA_DIR:?Submit with prepared metadata}
mapfile -t hosts < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
export INFERENCE_HOST=${hosts[0]} SENDER_HOST=${hosts[1]}
read -r -a budgets <<< "${CAPACITY_BUDGETS:-8 10}"
for budget in "${budgets[@]}"; do
    srun --kill-on-bad-exit=1 "$PROJECT_DIR/scripts/lumi_job_container.sh" bash "$PROJECT_DIR/scripts/lumi_super_capacity_role.sh" "$budget"
done
