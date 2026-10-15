#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_DIR=${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd -P)}
LUMI_ROOT=${LUMI_ROOT:-/scratch/project_465002751/rasmus}
SIF=${SIF:-/scratch/project_465002751/shared/sif/lumi-posttrain-u24r72-t213-vllm028.sif}
PRIME_RL_CACHE_DIR=${PRIME_RL_CACHE_DIR:-$LUMI_ROOT/.cache/prime-rl}
# The UCloud client is released packages, not sibling checkouts: verifiers-ucloud
# pins the verifiers fork and the sandbox SDK it was tested with; installing with
# --no-deps lists all three. Git pins are the release tags' commits:
# verifiers v0.3.1.post1 and verifiers-ucloud v0.3.7.
LUMI_RUNTIME_PACKAGES=(
  "$PROJECT_DIR/packages/prime-rl-configs"
  "$PROJECT_DIR/deps/renderers"
  "verifiers @ git+https://github.com/rlrs/verifiers@29ea1a32f22cc5eda385579573a1d890f9c3e49c"
  "verifiers-ucloud @ git+https://github.com/rlrs/verifiers-ucloud@966927c95e9df8108b1af32cd90fb376e1c751f2"
  "mcp==2.0.0"
  "httpcore2==2.12.0"
  "httpx2==2.12.0"
  "mcp-types==2.0.0"
  "truststore==0.10.4"
  "prime-runs==0.1.1"
  "prime-traces==0.0.3"
  "pyzstd==0.19.1"
  "backports-zstd==1.7.0"
  "https://github.com/rlrs/ucloud-sandboxes-sdk/releases/download/v0.4.39/ucloud_sandboxes_sdk-0.4.39-py3-none-any.whl"
)

# Third-party packages the SIF lacks. They change rarely, so they live in one
# image-keyed base layer shared by every commit instead of each commit layer.
# torchao (CUDA-only mxfp8) is not installed; mamba-ssm comes from the SIF.
LUMI_BASE_PACKAGES=(
  "bashlex==0.18"
  "dataclasses-json==0.6.7"
  "harbor==0.14.0"
  "marshmallow==3.26.2"
  "modelexpress==0.3.0"
  "multi-swe-bench==1.1.2"
  "mypy-extensions==1.1.0"
  "prime-sandboxes==0.2.39"
  "protobuf-py==0.1.1"
  "protobuf-py-ext==0.1.1"
  "PyGithub==2.9.1"
  "PyNaCl==1.6.2"
  "swe-rex==1.4.0"
  "typing-inspect==0.9.0"
  "unidiff==0.7.5"
  "vllm-router @ https://github.com/PrimeIntellect-ai/router/releases/download/v0.2.0/vllm_router-0.2.0-cp38-abi3-manylinux_2_28_x86_64.whl"
)

[[ -f "$PROJECT_DIR/pyproject.toml" ]] || {
  echo "FATAL: not a PRIME-RL checkout: $PROJECT_DIR" >&2
  exit 1
}
[[ -f "$SIF" ]] || {
  echo "FATAL: SIF not found: $SIF" >&2
  exit 1
}

sif_identity=$(stat -L -c '%n %s %Y' "$(readlink -f "$SIF")")
commit=$(git -C "$PROJECT_DIR" rev-parse HEAD)
# Key the layer on the image too: the same commit needs separate metadata per SIF.
runtime_hash=$(
  {
    sha256sum \
      "$PROJECT_DIR/pyproject.toml" \
      "$PROJECT_DIR/scripts/lumi_prepare_runtime.sh"
    printf '%s\n' "$sif_identity"
  } | sha256sum | cut -c1-12
)
base_hash=$(
  printf '%s\n' "$sif_identity" "${LUMI_BASE_PACKAGES[@]}" | sha256sum | cut -c1-12
)
metadata_root=$PRIME_RL_CACHE_DIR/metadata
metadata_dir=$metadata_root/$commit-$runtime_hash
base_dir=$PRIME_RL_CACHE_DIR/base/$base_hash
lock_file=$metadata_root/.lock

mkdir -p "$metadata_root" "$PRIME_RL_CACHE_DIR/base" "$LUMI_ROOT/.cache/pip" "$LUMI_ROOT/tmp"
exec 9>"$lock_file"
flock 9

pip_install_target() {
  local target=${1:?}
  shift
  singularity exec \
    -B "$PROJECT_DIR:$PROJECT_DIR" \
    -B "$LUMI_ROOT:$LUMI_ROOT" \
    --env "PIP_CACHE_DIR=$LUMI_ROOT/.cache/pip,TMPDIR=$LUMI_ROOT/tmp,HIP_ARCHITECTURES=gfx90a,MAX_JOBS=2" \
    "$SIF" \
    /opt/venv/bin/python -m pip install \
      --no-deps \
      --no-build-isolation \
      --target="$target" \
      "$@" >&2
}

if [[ ! -f "$base_dir/.ready" ]]; then
  build_dir=$(mktemp -d "$PRIME_RL_CACHE_DIR/base/.build.XXXXXX")
  echo "+ Creating SIF base package layer in $base_dir" >&2
  pip_install_target "$build_dir" "${LUMI_BASE_PACKAGES[@]}"
  printf '%s\n' "$sif_identity" > "$build_dir/.ready"
  mv "$build_dir" "$base_dir" 2>/dev/null || [[ -f "$base_dir/.ready" ]] || {
    echo "FATAL: could not publish base package layer: $base_dir" >&2
    exit 1
  }
fi

if [[ ! -f "$metadata_dir/.ready" ]]; then
  build_dir=$(mktemp -d "$metadata_root/.build.XXXXXX")
  # Retain failed private builds for diagnosis; never remove shared caches.

  echo "+ Creating PRIME-RL distribution metadata in $metadata_dir" >&2
  pip_install_target "$build_dir" "$PROJECT_DIR" "${LUMI_RUNTIME_PACKAGES[@]}"

  # Jobs read the base layer path from here (scripts/lumi_runtime_env.sh).
  printf '%s\n' "$base_dir" > "$build_dir/.base"
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
