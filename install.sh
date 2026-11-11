#!/usr/bin/env bash
# Alpharidge environment setup (CPU only, no model downloads).
#
#   ./install.sh validator
#   ./install.sh miner
set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-miner}"
PY="${PYTHON:-python3}"

case "$MODE" in
  validator) ENV_EXAMPLE=vali_env.example ENV_FILE=.vali_env ;;
  miner) ENV_EXAMPLE=miner_env.example ENV_FILE=.miner_env ;;
  *)
    echo "usage: ./install.sh [miner|validator]" >&2
    exit 1
    ;;
esac

[ -d .venv ] || "$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
if [ "$MODE" = validator ]; then
  echo ">> Installing the hash-pinned requirements.lock and the package"
  pip install --require-hashes -r requirements.lock
  pip install --no-deps --no-build-isolation -e .
else
  echo ">> Installing package with the verify extra"
  pip install -e ".[verify]"
fi

echo
echo "Environment ready (./.venv)."
echo "Next: cp $ENV_EXAMPLE $ENV_FILE, then start the neuron (see docs/$MODE.md)."
