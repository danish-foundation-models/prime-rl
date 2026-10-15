#!/usr/bin/env bash

# Build the live-source Python path used by LUMI runtime jobs. Keep package
# code outside the immutable SIF while exposing the research environments as
# importable top-level modules. verifiers and verifiers-ucloud come from the
# prepared metadata layer (lumi_prepare_runtime.sh), never from sibling checkouts.
prime_rl_lumi_export_pythonpath() {
  local metadata_dir=${1:?metadata directory is required}
  local project_dir=${2:-/workdir}
  local scan_dir=${3:-$project_dir}
  local path
  local env_root
  local env_dir
  local relative_env_dir

  path="$project_dir/src"
  path="$path:$project_dir/packages/prime-rl-configs/src"
  path="$path:$project_dir/deps/renderers"
  path="$path:$project_dir/deps/prime-kernels"
  path="$path:$project_dir/deps/pydantic-config/src"

  for env_root in \
    "$scan_dir/deps/prime-envs/environments" \
    "$scan_dir/deps/research-environments/environments"
  do
    [[ -d "$env_root" ]] || continue
    while IFS= read -r -d '' pyproject; do
      env_dir=${pyproject%/pyproject.toml}
      relative_env_dir=${env_dir#"$scan_dir"/}
      path="$path:$project_dir/$relative_env_dir"
    done < <(
      find "$env_root" -mindepth 2 -maxdepth 4 -type f \
        -name pyproject.toml -print0 | sort -z
    )
  done

  # lumi_prepare_runtime.sh records the image-keyed base package layer.
  local base_metadata
  base_metadata=$(cat "$metadata_dir/.base" 2>/dev/null) || {
    echo "Runtime metadata predates the base layer; rerun lumi_prepare_runtime.sh" >&2; return 1;
  }
  [[ -f "$base_metadata/.ready" ]] || { echo "Missing SIF base package layer: $base_metadata" >&2; return 1; }
  if [[ -n ${PRIME_RL_ENVIRONMENT_SITE_PACKAGES:-} ]]; then
    [[ -f "$PRIME_RL_ENVIRONMENT_SITE_PACKAGES/.ready" ]] || {
      echo "Missing validated agentic environment dependency layer" >&2; return 1;
    }
    path="$path:$PRIME_RL_ENVIRONMENT_SITE_PACKAGES"
  fi
  if [[ -d "$scan_dir/deps/text2sql-env/environments/text2sql/text2sql" ]]; then
    path="$path:$project_dir/deps/text2sql-env/environments/text2sql"
  fi
  export PYTHONPATH="$path:$metadata_dir:$base_metadata"
}

# AITER compiles JIT modules into AITER_JIT_DIR and template ops into
# AITER_ROOT_DIR (default ~/.aiter). Keep both durable, off $HOME, and per image.
prime_rl_lumi_aiter_cache_dir() {
  local lumi_root=${1:?LUMI root is required}
  local sif=${2:?SIF is required}
  local name
  name=$(basename "$sif" .sif)
  printf '%s\n' "$lumi_root/.cache/aiter/$name"
}
