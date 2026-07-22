#!/usr/bin/env bash

# Build the live-source Python path used by LUMI runtime jobs. Keep package
# code outside the immutable SIF while exposing all repository-bundled
# verifier/research environments as importable top-level modules.
prime_rl_lumi_export_pythonpath() {
  local metadata_dir=${1:?metadata directory is required}
  local project_dir=${2:-/workdir}
  local path
  local env_root
  local env_dir

  path="$project_dir/src"
  path="$path:$project_dir/packages/prime-rl-configs/src"
  path="$path:$project_dir/deps/renderers"
  path="$path:$project_dir/deps/verifiers"
  path="$path:$project_dir/deps/pydantic-config/src"

  for env_root in \
    "$project_dir/deps/verifiers/environments" \
    "$project_dir/deps/research-environments/environments"
  do
    [[ -d "$env_root" ]] || continue
    for env_dir in "$env_root"/*; do
      [[ -f "$env_dir/pyproject.toml" ]] || continue
      path="$path:$env_dir"
    done
  done

  export PYTHONPATH="$path:$metadata_dir"
}
