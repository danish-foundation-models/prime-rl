#!/bin/bash
# Capture <seconds> of nsys on every inference rank of a job started from make_profile_sbatch.sh,
# then export each report to sqlite and run infer-opt's per-step breakdown on it.
# Usage: profile_capture.sh <jobid> <infer-host> <out-dir> [seconds]
set -uo pipefail
J=$1 H=$2 OUT=$3 SECS=${4:-5}
mkdir -p "$OUT"
srun --jobid="$J" --overlap -w "$H" -N1 -n1 bash -c "
NSYS=/dcai/software/tenant/apps/CUDA/13.0.0/bin/nsys
sessions=\$(\$NSYS sessions list 2>/dev/null | grep -oE 'prof$J-[0-9]+' | sort -u)
echo \"sessions: \$sessions\"
for s in \$sessions; do \$NSYS start --session=\$s --sample=none --cpuctxsw=none -f true -o $OUT/nsys-\$s & done; wait
sleep $SECS
for s in \$sessions; do \$NSYS stop --session=\$s & done; wait
echo capture done"
PY=/dcai/projects/iu_0145/rasmus/phase0/prime-rl/.venv/bin/python
NSYS=/dcai/software/tenant/apps/CUDA/13.0.0/bin/nsys
for f in "$OUT"/nsys-*.nsys-rep; do
  $NSYS export --type sqlite --force-overwrite true -o "${f%.nsys-rep}.sqlite" "$f" >/dev/null 2>&1
  $PY -I /dcai/projects/iu_0145/rasmus/infer-opt/analysis/nsys_breakdown.py "${f%.nsys-rep}.sqlite" --json "${f%.nsys-rep}.json" > "${f%.nsys-rep}.txt" && echo "breakdown ${f##*/} ok"
done
