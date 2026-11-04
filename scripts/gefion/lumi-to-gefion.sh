#!/bin/bash
# Run ON LUMI. Copies the LUMI-only inputs of the Gefion full-run config (gefion_full_r1.toml) to
# Gefion's mirror, keeping paths relative to /scratch/project_465002751/. Resumable: rerun to finish.
# Usage: bash lumi-to-gefion.sh <gefion-ssh-target>   e.g.  larras@<gefion-login>
# First run with DRY_RUN=1 to see sizes:  DRY_RUN=1 bash lumi-to-gefion.sh larras@<gefion-login>
set -euo pipefail
DEST=${1:?gefion ssh target}
cd /scratch/project_465002751
LIST=$(mktemp)
cat > "$LIST" <<'PATHS'
oliver/data/bird
rasmus/.cache/browsecomp-plus
rasmus/.cache/nltk_data
rasmus/artifacts/manifests/ucloud-task-ids-20261010b/scaleswe-train.task-ids.json
rasmus/artifacts/manifests/ucloud-task-ids-20261010b/terminal-lego.task-ids.json
rasmus/artifacts/manifests/ucloud-task-ids-20261010b/tmax.task-ids.json
rasmus/artifacts/training/nemo-local-20260923/data-v2/calendar-train.jsonl
rasmus/artifacts/training/nemo-local-20260923/data-v2/calendar-validation.jsonl
rasmus/artifacts/training/nemo-local-20260923/data-v2/instruction-train.jsonl
rasmus/artifacts/training/nemo-local-20260923/data-v2/instruction-validation.jsonl
rasmus/artifacts/training/nemo-local-20260923/data-v2/pivot-train.jsonl
rasmus/artifacts/training/nemo-local-20260923/data-v2/pivot-validation.jsonl
rasmus/artifacts/training/nemo-local-20260923/data-v2/workplace-train.jsonl
rasmus/artifacts/training/nemo-local-20260923/data-v2/workplace-validation.jsonl
rasmus/artifacts/training/nemotron-super-r5-rl-scale-20260917/all-tasks/task-metadata-20261010/terminal-lego.jsonl
rasmus/artifacts/training/nemotron-super-r5-rl-scale-20260917/all-tasks/task-metadata-20261010/tmax.jsonl
rasmus/artifacts/training/nemotron-super-r5-rl-scale-20260917/expanded-sources/prime-tasks/datasets/tmax
rasmus/artifacts/training/nemotron-super-r5-rl-scale-20260917/offline-compaction-20260922/python-0.tar.gz
rasmus/artifacts/training/nemotron-super-r5-rl-scale-20260917/offline-compaction-20260922/python-1.tar.gz
.cache/huggingface/terminal-lego-git/PrimeIntellect--Terminal-Lego-15k
PATHS
if [ "${DRY_RUN:-0}" = 1 ]; then
  du -sh --total $(cat "$LIST") 2>&1 | sort -h | tail -25
  exit 0
fi
rsync -a --partial --info=progress2 --files-from="$LIST" -r ./ "$DEST:/dcai/projects/iu_0145/lumi-mirror/"
