#!/bin/bash
# Gefion preflight for the RL job (runs once on the batch node via slurm.pre_run_command):
# every LUMI-mirrored input the config references must exist. (Node IOMMU placement: order_nodes.sh.)
set -euo pipefail
missing=0
while read -r p; do
  [ -z "$p" ] && continue
  g=/dcai/projects/iu_0145/lumi-mirror/${p#/scratch/project_465002751/}
  [ -e "$g" ] || { echo "PREFLIGHT: missing $g" >&2; missing=1; }
done < /dcai/projects/iu_0145/rasmus/phase0/full/lumi-paths.txt
[ $missing = 0 ] || { echo "PREFLIGHT FAILED: missing LUMI inputs" >&2; exit 1; }
# The UCloud sandbox relay must be up: env servers register with it at startup and exit on a 503
# ("Job is unavailable" while the relay's UCloud job is down; job 533619). Unauthenticated probe.
# Since 2026-10-30 the relay runs on Hetzner (https://<ip>/relay, answers /healthz).
relay=$(grep -E '^UCLOUD_RELAY_URL=' /dcai/projects/iu_0145/rasmus/.env | head -1 | cut -d= -f2- | tr -d "\"'")
code=$(curl -s -o /dev/null -w '%{http_code}' -m 20 "${relay%/}/healthz" || true)
[ "$code" = 200 ] || { echo "PREFLIGHT FAILED: sandbox relay ${relay%/}/healthz answers $code" >&2; exit 1; }
echo "preflight ok: sandbox relay ${relay%/} healthy"
echo "preflight ok: all mirrored inputs present"
