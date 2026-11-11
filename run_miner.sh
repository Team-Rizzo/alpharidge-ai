#!/usr/bin/env bash
# Reference miner launcher. Settings come from .miner_env; wallet and network from here.
set -euo pipefail
cd "$(dirname "$0")"
exec ./.venv/bin/python -m neurons.miner \
  --netuid "${NETUID:-45}" \
  --subtensor.network "${SUBTENSOR_NETWORK:-finney}" \
  --wallet.name "${WALLET_NAME:?set WALLET_NAME}" \
  --wallet.hotkey "${WALLET_HOTKEY:?set WALLET_HOTKEY}" \
  --logging.info "$@"
