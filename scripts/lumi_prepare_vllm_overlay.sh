#!/usr/bin/env bash
# Materialize a gfx90a vLLM source overlay (rlrs/vllm) for the LUMI SIF.
#
# usage: scripts/lumi_prepare_vllm_overlay.sh REV
#
# Exports every vllm/ file that REV changes relative to the vLLM commit the
# SIF was built from into $LUMI_ROOT/.runtime/vllm-overlays/<full REV>.
# lumi_job_container.sh bind-mounts those files read-only over the SIF's vLLM
# when PRIME_RL_VLLM_OVERLAY names that directory. Compiled extensions stay
# from the SIF, so REV may only change Python and tuning files.
set -euo pipefail

rev=${1:?usage: $0 REV}
LUMI_ROOT=${LUMI_ROOT:-/scratch/project_465002751/rasmus}
SIF=${SIF:-/scratch/project_465002751/shared/sif/lumi-posttrain-u24r72-t213-vllm028.sif}
repo=${PRIME_RL_VLLM_OVERLAY_REPO:-$LUMI_ROOT/.runtime/src/vllm.git}
overlay_root=$LUMI_ROOT/.runtime/vllm-overlays

[[ -d "$repo" ]] || {
  echo "+ Cloning rlrs/vllm into $repo" >&2
  git clone --bare --filter=blob:none https://github.com/rlrs/vllm "$repo" >&2
}
git -C "$repo" cat-file -e "$rev^{commit}" 2>/dev/null \
  || git -C "$repo" fetch -q origin "$rev" >&2 \
  || git -C "$repo" fetch -q origin '+refs/heads/*:refs/heads/*' >&2
rev=$(git -C "$repo" rev-parse "$rev^{commit}")
dir=$overlay_root/$rev
if [[ -f "$dir/.ready" ]]; then
  printf '%s\n' "$dir"
  exit 0
fi

# The SIF's vLLM version ends in the short commit it was built from.
version=$(singularity exec "$SIF" /opt/venv/bin/python -c \
  'import importlib.metadata as m; print(m.version("vllm"))')
base=${version##*.}
git -C "$repo" cat-file -e "$base^{commit}" 2>/dev/null || {
  echo "FATAL: cannot resolve the SIF vLLM commit '$base' (vllm $version) in $repo" >&2
  exit 1
}
git -C "$repo" merge-base --is-ancestor "$base" "$rev" || {
  echo "FATAL: $rev does not descend from the SIF vLLM commit $base" >&2
  exit 1
}
if git -C "$repo" diff --name-only --diff-filter=DR "$base" "$rev" -- vllm | grep -q .; then
  echo "FATAL: $rev deletes or renames vllm files; a bind overlay cannot express that" >&2
  exit 1
fi
mapfile -t files < <(git -C "$repo" diff --name-only "$base" "$rev" -- vllm)
for file in "${files[@]}"; do
  case $file in
    *.py|*.json) ;;
    *) echo "FATAL: overlay changes a non-Python file: $file" >&2; exit 1 ;;
  esac
done
((${#files[@]})) || { echo "FATAL: $rev does not change vllm/" >&2; exit 1; }

# Singularity splits bind specs on commas, and tuning files are named
# "E=64,N=2688,...json". Bind such a file's whole directory, taken from REV.
declare -A seen=()
binds=()
for file in "${files[@]}"; do
  [[ $file == *,* ]] && file=$(dirname "$file")
  [[ $file == *,* ]] && { echo "FATAL: comma in overlay directory: $file" >&2; exit 1; }
  [[ -n ${seen[$file]:-} ]] || binds+=("$file")
  seen[$file]=1
done

mkdir -p "$overlay_root"
build_dir=$(mktemp -d "$overlay_root/.build.XXXXXX")
git -C "$repo" archive "$rev" -- "${binds[@]}" | tar -x -C "$build_dir"
printf '%s\n' "${binds[@]}" > "$build_dir/FILES"
printf 'rev=%s\nbase=%s\nsif_vllm=%s\n' "$rev" "$base" "$version" > "$build_dir/SOURCE"
printf '%s\n' "$rev" > "$build_dir/.ready"
mv "$build_dir" "$dir" 2>/dev/null || [[ -f "$dir/.ready" ]]
echo "+ vLLM overlay $rev over $base: ${#files[@]} files in ${#binds[@]} binds" >&2
printf '%s\n' "$dir"
