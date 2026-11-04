#!/bin/bash
# Make an nsys-profiled copy of a rendered RL sbatch: every inference rank runs under
# `nsys launch` (one session per rank, named prof<jobid>-<port>) with the infer_opt nvtx patch, so
# a window of the real RL load can be captured later with profile_capture.sh. Nothing is
# recorded until a capture starts.
# Usage: make_profile_sbatch.sh <rl.sbatch> <out.sbatch>
set -euo pipefail
src=$1 out=$2
NSYS=/dcai/software/tenant/apps/CUDA/13.0.0/bin/nsys
cp "$src" "$out"
grep -q 'local -a cmd=(uv run inference @ "$CONFIG_PATH"$' "$out"
sed -i "s|local -a cmd=(uv run inference @ \"\$CONFIG_PATH\"\$|local -a cmd=($NSYS launch --session-new=prof\$SLURM_JOB_ID-\$port --trace=cuda,nvtx --cuda-graph-trace=node --wait=all uv run inference @ \"\$CONFIG_PATH\"|" "$out"
grep -q '^    export INFER_OPT="' "$out"
sed -i 's|^    export INFER_OPT="\([^"]*\)"$|    export INFER_OPT="\1,nvtx"\n    # nsys flushes stall every worker at once; do not let vLLM declare the engine dead meanwhile.\n    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600|' "$out"
sed -i 's|^#SBATCH --job-name=.*|#SBATCH --job-name=gefion-super-shakedown|' "$out"
bash -n "$out"
grep -nE "nsys launch|INFER_OPT=|EXECUTE_MODEL_TIMEOUT" "$out"
