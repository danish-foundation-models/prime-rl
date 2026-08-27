#!/usr/bin/env bash

# Build the live-source Python path used by LUMI runtime jobs. Keep package
# code outside the immutable SIF while exposing all repository-bundled
# verifier/research environments as importable top-level modules.
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
  path="$path:$project_dir/deps/verifiers"
  path="$path:$project_dir/deps/ucloud-sandboxes-sdk/src"
  path="$path:$project_dir/deps/pydantic-config/src"

  for env_root in \
    "$scan_dir/deps/prime-envs/environments" \
    "$scan_dir/deps/verifiers/environments" \
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

  export PYTHONPATH="$path:$metadata_dir"
}

# The current supported SIF still contains connect-python 0.9.0. New
# prime-sandboxes releases intentionally reject that distribution even when
# the replacement connectrpc module is first on PYTHONPATH. Bind this empty
# directory over the legacy dist-info metadata for container executions.
prime_rl_lumi_connect_python_mask() {
  local lumi_root=${1:?LUMI root is required}
  local mask_dir="$lumi_root/.runtime/masks/connect_python-0.9.0.dist-info"

  mkdir -p "$mask_dir"
  printf '%s\n' "$mask_dir"
}
