#!/bin/bash
# Render an RL config's sbatch with the PRIME-RL launcher (dry run) and apply the Gefion patches:
# - --propagate=NONE: jobs otherwise inherit the login node's 512 MB memlock;
# - --mem=0: DefMemPerGPU otherwise caps an exclusive 8-GPU node at ~248 GB of its 2 TiB;
# - CUDA 13.0 nvcc on PATH: vLLM otherwise treats FlashInfer as unavailable and falls back to Triton MoE;
# - user-space uv on PATH, and no `uv sync` in the job (UV_NO_SYNC; not UV_FROZEN, which also breaks
#   the harnesses' own `uv sync --script`): the venv is prepared ahead with the UCloud
#   pins (verifiers fork, verifiers-ucloud, SDK) installed on top, which a sync would remove.
# Usage: render.sh <config.toml>
set -euo pipefail
CFG=$(readlink -f "$1")
cd /dcai/projects/iu_0145/rasmus/phase0/prime-rl
# The template sources $PROJECT_DIR/.env on every node; keep the credentials in one place.
[ -e .env ] || ln -s /dcai/projects/iu_0145/rasmus/.env .env
out=$(.venv/bin/rl @ "$CFG" --dry-run 2>&1) || { echo "$out" | tail -3; echo "render: dry run failed"; exit 1; }
echo "$out" | grep -E "SUCCESS|rror" | head -3
L=$(echo "$out" | grep -oE "sbatch \S+/launcher/rl.sbatch" | awk '{print $2}' | head -1)
[ -f "$L" ] || { echo "no rendered sbatch found"; exit 1; }
sed -i 's|^#SBATCH --exclusive$|#SBATCH --exclusive\n#SBATCH --propagate=NONE\n#SBATCH --mem=0|' "$L"
sed -i 's|^# Configs$|# Gefion: user-space uv, never re-sync the prepared venv inside the job.\nexport PATH=$PATH:/dcai/projects/iu_0145/rasmus/phase0/uvenv/bin UV_NO_SYNC=1\n# nvcc on PATH: without it vLLM deems FlashInfer unavailable and silently runs the FP8 MoE on Triton (~5x slower decode).\nexport CUDA_HOME=/dcai/software/tenant/apps/CUDA/13.0.0\nexport PATH=$PATH:$CUDA_HOME/bin\nexport UV_CACHE_DIR=/dcai/projects/iu_0145/rasmus/phase0/uvcache\n\n# Configs|' "$L"
sed -i 's|^uv sync --all-extras --all-packages$|# Gefion: the venv is prepared ahead (UCloud pins on top); a sync here would remove them.|' "$L"
# Nodes without iommu=pt go to inference: order the hostnames with them first, and give each launch
# task the role of its hostname's position (the template otherwise maps roles by SLURM_PROCID, i.e.
# Slurm's sorted node order).
sed -i 's|^HOSTNAMES=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )$|ORDERED=$(bash /dcai/projects/iu_0145/rasmus/phase0/full/order_nodes.sh) \|\| exit 1\nHOSTNAMES=( $ORDERED )|' "$L"
sed -i "/<<'LAUNCH_SH'\$/a\\    # Gefion: role by position of this host in the IOMMU-ordered HOSTNAMES (see order_nodes.sh).\n    _h=\$(hostname -s); _i=0; for _x in \$HOSTNAMES_STR; do [ \"\$_x\" = \"\$_h\" ] \&\& break; _i=\$((_i + 1)); done\n    export SLURM_PROCID=\$_i" "$L"
# The template runs `source .env`, and our .env has no `export`s: auto-export it so the UCloud
# credentials reach the env servers and orchestrator.
sed -i 's|\[ -f \.env \] \&\& source \.env$|[ -f .env ] \&\& { set -a; source .env; set +a; }|' "$L"
# vllm-router at debug logs every request body (token arrays): 8.5 GB/h in the PD shakedown (533326).
sed -i 's|--worker-startup-timeout-secs 4200 --log-level debug)|--worker-startup-timeout-secs 4200 --log-level info)|' "$L"
echo "rendered: $L"
grep -nE "^#SBATCH --(nodes|propagate|mem|exclude|time)|UV_NO_SYNC|CUDA_HOME|^uv sync|venv is prepared|preflight|ORDERED|^HOSTNAMES|export SLURM_PROCID|source .env" "$L"
