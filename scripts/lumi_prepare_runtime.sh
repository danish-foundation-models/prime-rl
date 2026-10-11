#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_DIR=${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd -P)}
LUMI_ROOT=${LUMI_ROOT:-/scratch/project_465002751/rasmus}
SIF=${SIF:-/scratch/project_465002751/shared/sif/lumi-posttrain-u24r71-t211-vllm024.sif}
PRIME_RL_CACHE_DIR=${PRIME_RL_CACHE_DIR:-$LUMI_ROOT/.cache/prime-rl}
# The UCloud client is released packages, not sibling checkouts: verifiers-ucloud
# pins the verifiers fork and the sandbox SDK it was tested with; installing with
# --no-deps lists all three. Git pins are the release tags' commits:
# verifiers v0.3.1.post1 and verifiers-ucloud v0.3.1.
LUMI_RUNTIME_PACKAGES=(
  "$PROJECT_DIR/packages/prime-rl-configs"
  "$PROJECT_DIR/deps/renderers"
  "verifiers @ git+https://github.com/rlrs/verifiers@29ea1a32f22cc5eda385579573a1d890f9c3e49c"
  "verifiers-ucloud @ git+https://github.com/rlrs/verifiers-ucloud@4c99bab86df4422467409de2aa4aa4ee377c5c1d"
  "mcp==2.0.0"
  "httpcore2==2.12.0"
  "httpx2==2.12.0"
  "mcp-types==2.0.0"
  "truststore==0.10.4"
  "prime-runs==0.1.1"
  "prime-traces==0.0.3"
  "pyzstd==0.19.1"
  "backports-zstd==1.7.0"
  "https://github.com/rlrs/ucloud-sandboxes-sdk/releases/download/v0.4.38/ucloud_sandboxes_sdk-0.4.38-py3-none-any.whl"
)

[[ -f "$PROJECT_DIR/pyproject.toml" ]] || {
  echo "FATAL: not a PRIME-RL checkout: $PROJECT_DIR" >&2
  exit 1
}
[[ -f "$SIF" ]] || {
  echo "FATAL: SIF not found: $SIF" >&2
  exit 1
}

commit=$(git -C "$PROJECT_DIR" rev-parse HEAD)
runtime_hash=$(
  sha256sum \
    "$PROJECT_DIR/pyproject.toml" \
    "$PROJECT_DIR/scripts/lumi_prepare_runtime.sh" \
    | sha256sum \
    | cut -c1-12
)
metadata_root=$PRIME_RL_CACHE_DIR/metadata
metadata_dir=$metadata_root/$commit-$runtime_hash
lock_file=$metadata_root/.lock

mkdir -p "$metadata_root" "$LUMI_ROOT/.cache/pip" "$LUMI_ROOT/tmp"
exec 9>"$lock_file"
flock 9

if [[ ! -f "$metadata_dir/.ready" ]]; then
  build_dir=$(mktemp -d "$metadata_root/.build.XXXXXX")
  # Retain failed private builds for diagnosis; never remove shared caches.

  echo "+ Creating PRIME-RL distribution metadata in $metadata_dir" >&2
  singularity exec \
    -B "$PROJECT_DIR:$PROJECT_DIR" \
    -B "$LUMI_ROOT:$LUMI_ROOT" \
    --env "PIP_CACHE_DIR=$LUMI_ROOT/.cache/pip,TMPDIR=$LUMI_ROOT/tmp,HIP_ARCHITECTURES=gfx90a,MAX_JOBS=2" \
    "$SIF" \
    /opt/venv/bin/python -m pip install \
      --no-deps \
      --no-build-isolation \
      --target="$build_dir" \
      "$PROJECT_DIR" \
      "${LUMI_RUNTIME_PACKAGES[@]}" >&2

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
