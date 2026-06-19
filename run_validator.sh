#!/usr/bin/env bash
# Talisman validator — local subtensor netuid 2, deepseek-v4-flash via OpenRouter
set -euo pipefail
cd /home/rizzo/talisman/talisman-ai
# bittensor 10.4 defaults to NOT parsing CLI args; this re-enables it
export BT_NO_PARSE_CLI_ARGS=0
# Deterministic cuBLAS workspace (must be set before CUDA init) — shrinks GPU
# jitter so miner/validator per-asset sentiment matches across the consensus gate.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec /home/rizzo/miniconda3/envs/talisman_ai/bin/python -m neurons.validator \
  --netuid 2 \
  --subtensor.network ws://127.0.0.1:9946 \
  --subtensor.chain_endpoint ws://127.0.0.1:9946 \
  --wallet.name sn45_vali --wallet.hotkey sn45_vali \
  --axon.port 18091 --axon.external_ip 192.168.69.157 --axon.external_port 18091 \
  --logging.info
