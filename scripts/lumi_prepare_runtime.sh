#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_DIR=${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd -P)}
LUMI_ROOT=${LUMI_ROOT:-/scratch/project_465002751/rasmus}
SIF=${SIF:-/scratch/project_465002751/shared/sif/lumi-posttrain-u24r71-t211-vllm024.sif}
PRIME_RL_CACHE_DIR=${PRIME_RL_CACHE_DIR:-$LUMI_ROOT/.cache/prime-rl}

[[ -f "$PROJECT_DIR/pyproject.toml" ]] || {
  echo "FATAL: not a PRIME-RL checkout: $PROJECT_DIR" >&2
  exit 1
}
[[ -f "$SIF" ]] || {
  echo "FATAL: SIF not found: $SIF" >&2
  exit 1
}

commit=$(git -C "$PROJECT_DIR" rev-parse HEAD)
project_hash=$(sha256sum "$PROJECT_DIR/pyproject.toml" | cut -c1-12)
metadata_root=$PRIME_RL_CACHE_DIR/metadata
metadata_dir=$metadata_root/$commit-$project_hash
lock_file=$metadata_root/.lock

mkdir -p "$metadata_root" "$LUMI_ROOT/.cache/pip" "$LUMI_ROOT/tmp"
exec 9>"$lock_file"
flock 9

if [[ ! -f "$metadata_dir/.ready" ]]; then
  build_dir=$(mktemp -d "$metadata_root/.build.XXXXXX")
  trap 'rm -rf "$build_dir"' EXIT

  echo "+ Creating PRIME-RL distribution metadata in $metadata_dir" >&2
  singularity exec \
    -B "$PROJECT_DIR:$PROJECT_DIR" \
    -B "$LUMI_ROOT:$LUMI_ROOT" \
    --env "PIP_CACHE_DIR=$LUMI_ROOT/.cache/pip,TMPDIR=$LUMI_ROOT/tmp" \
    "$SIF" \
    /opt/venv/bin/python -m pip install \
      --no-deps \
      --no-build-isolation \
      --target="$build_dir" \
      "$PROJECT_DIR" >&2

  printf '%s\n' "$commit" > "$build_dir/.ready"
  if ! mv "$build_dir" "$metadata_dir" 2>/dev/null; then
    [[ -f "$metadata_dir/.ready" ]] || {
      echo "FATAL: could not publish PRIME-RL metadata: $metadata_dir" >&2
      exit 1
    }
  fi
  trap - EXIT
fi

printf '%s\n' "$metadata_dir"
