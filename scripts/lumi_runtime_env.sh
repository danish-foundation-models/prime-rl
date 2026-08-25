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
  path="$path:$project_dir/deps/ucloud-sandboxes-sdk/src"
  path="$path:$project_dir/deps/pydantic-config/src"

  for env_root in \
    "$project_dir/deps/verifiers/environments" \
    "$project_dir/deps/research-environments/environments"
  do
    [[ -d "$env_root" ]] || continue
    while IFS= read -r -d '' pyproject; do
      env_dir=${pyproject%/pyproject.toml}
      path="$path:$env_dir"
    done < <(
      find "$env_root" -mindepth 2 -maxdepth 4 -type f \
        -name pyproject.toml -print0 | sort -z
    )
  done

  export PYTHONPATH="$path:$metadata_dir"
}
