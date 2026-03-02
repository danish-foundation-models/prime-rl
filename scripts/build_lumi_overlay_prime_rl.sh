#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

BASE_DIR=/pfs/lustref1/appl/local/laifs
LAIFS_APPL_DIR=/appl/local/laifs

: "${SIF:=$BASE_DIR/containers/lumi-multitorch-u24r64f21m43t29-20260216_093549/lumi-multitorch-full-u24r64f21m43t29-20260216_093549.sif}"
: "${OVERLAY_DIR:=$REPO_DIR/overlay_prime_rl_lumi}"
: "${OVERLAY_ENV_NAME:=prime-rl-lumi}"
: "${EXTRA_PIP_PACKAGES:=tomli tomli_w beartype jaxtyping torchdata ring-flash-attn aiolimiter tenacity}"
: "${PRIMEINTELLECT_INDEX_URL:=https://hub.primeintellect.ai/primeintellect/simple/}"
: "${DION_REF:=d891eeb}"
: "${TORCHTITAN_REF:=a1fdd7e}"
: "${VERIFIERS_REF:=b35d0c7}"

if [[ ! -f "$SIF" ]]; then
  echo "FATAL: SIF not found: $SIF" >&2
  exit 1
fi

mkdir -p "$OVERLAY_DIR"/venv "$OVERLAY_DIR"/cache
chmod 700 "$OVERLAY_DIR"

echo "+ SIF: $SIF"
echo "+ Overlay: $OVERLAY_DIR"
echo "+ Overlay env: $OVERLAY_ENV_NAME"
echo "+ Repo: $REPO_DIR"
echo "+ Extra pip packages: ${EXTRA_PIP_PACKAGES:-<none>}"

export OVERLAY_ENV_NAME EXTRA_PIP_PACKAGES
export PRIMEINTELLECT_INDEX_URL DION_REF TORCHTITAN_REF VERIFIERS_REF

singularity exec --rocm \
  -B "$BASE_DIR:$LAIFS_APPL_DIR" \
  -B "$OVERLAY_DIR:/overlay" \
  -B "$REPO_DIR:/workdir" \
  "$SIF" bash -eu -s <<'INSIDE'
source /opt/venv/bin/activate

OVERLAY_VENV_DIR="/overlay/venv/${OVERLAY_ENV_NAME}"
if [[ ! -d "$OVERLAY_VENV_DIR" ]]; then
  python3 -m venv --system-site-packages "$OVERLAY_VENV_DIR"
fi

source "${OVERLAY_VENV_DIR}/bin/activate"
OVERLAY_SITE="${OVERLAY_VENV_DIR}/lib/python3.12/site-packages"
export PYTHONPATH="${OVERLAY_SITE}${PYTHONPATH:+:$PYTHONPATH}"

export PIP_USER=0
unset PYTHONUSERBASE
export XDG_CACHE_HOME=/overlay/cache
export PIP_CACHE_DIR=/overlay/cache/pip
mkdir -p "$XDG_CACHE_HOME" "$PIP_CACHE_DIR"

python -m pip install --no-user -U pip 'setuptools>=79,<80' wheel
python -m pip install --no-user --no-deps -e /workdir

if [[ -n "${EXTRA_PIP_PACKAGES}" ]]; then
  python -m pip install --no-user ${EXTRA_PIP_PACKAGES}
fi

python -m pip install --no-user --extra-index-url "${PRIMEINTELLECT_INDEX_URL}" \
  "prime>=0.5.37" \
  "math-env" \
  "reverse-text"

python -m pip install --no-user \
  "liger-kernel>=0.5.10" \
  "tilelang>=0.1.8"

python -m pip install --no-user \
  "git+https://github.com/samsja/dion.git@${DION_REF}"

python -m pip install --no-user \
  "git+https://github.com/pytorch/torchtitan@${TORCHTITAN_REF}"

python -m pip install --no-user \
  "git+https://github.com/PrimeIntellect-ai/verifiers.git@${VERIFIERS_REF}"

python -m pip install --no-user "compressed-tensors==0.13.0"

TORCH_TRITON_VERSION="$(python - <<'PY'
import importlib.metadata as md

versions = {}
for dist in md.distributions():
    name = (dist.metadata.get("Name") or "").lower()
    if name:
        versions[name] = dist.version

print(versions.get("pytorch-triton-rocm", ""))
PY
)"

if [[ -n "$TORCH_TRITON_VERSION" ]]; then
  python -m pip install --no-user --upgrade "triton==${TORCH_TRITON_VERSION}"
  echo "+ Aligned triton to pytorch-triton-rocm==${TORCH_TRITON_VERSION}"
else
  echo "+ WARNING: pytorch-triton-rocm not found; skipping triton alignment"
fi

python - <<'PY'
import prime_rl
import torch

print("prime_rl:", prime_rl.__file__)
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
PY

python -m pip freeze > /overlay/overlay-pip-freeze.txt
echo "+ Wrote /overlay/overlay-pip-freeze.txt"
INSIDE

echo "+ Build complete."
echo "+ Activate in container with: source /overlay/venv/${OVERLAY_ENV_NAME}/bin/activate"
