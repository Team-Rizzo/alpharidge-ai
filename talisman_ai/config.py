"""
Centralized configuration loader for talisman_ai_subnet.
Loads environment variables from .miner_env and .vali_env files.

Remote config: the API serves recommended values at /config/subnet.
Validators fetch these once per hour and apply them unless the operator
has set OVERRIDE_<KEY> in their local environment.

Priority chain:
  1. OVERRIDE_<KEY> in local env  (operator explicitly opts out)
  2. Remote /config/subnet value  (subnet-wide recommendation)
  3. Hardcoded default below       (fallback if API is unreachable)

This module should be imported at the top of any module that needs configuration:
    from talisman_ai import config
    
Then access config values as:
    config.MODEL
    config.BLOCKS_PER_WINDOW
    etc.
"""

from pathlib import Path
import os
import time
import threading
import logging

_config_logger = logging.getLogger("talisman_ai.config")

# Find the talisman_ai_subnet root directory
# This file is at talisman_ai_subnet/talisman_ai/config.py
_SUBNET_ROOT = Path(__file__).resolve().parent.parent

# Paths to environment files
_MINER_ENV_PATH = _SUBNET_ROOT / ".miner_env"
_VALI_ENV_PATH = _SUBNET_ROOT / ".vali_env"

# Load environment files
try:
    from dotenv import load_dotenv
    
    # Load miner env file (if it exists)
    if _MINER_ENV_PATH.exists():
        load_dotenv(str(_MINER_ENV_PATH), override=True)
        print(f"[CONFIG] Loaded {_MINER_ENV_PATH}")
    else:
        print(f"[CONFIG] Warning: {_MINER_ENV_PATH} not found")
    
    # Load validator env file (if it exists)
    # Note: validator vars will override miner vars if both exist
    if _VALI_ENV_PATH.exists():
        load_dotenv(str(_VALI_ENV_PATH), override=True)
        print(f"[CONFIG] Loaded {_VALI_ENV_PATH}")
    else:
        print(f"[CONFIG] Warning: {_VALI_ENV_PATH} not found")
        
except ImportError:
    print("[CONFIG] Warning: python-dotenv not installed, using system environment variables only")


# ============================================================================
# Shared Configuration (available to both miners and validators)
# ============================================================================

# LLM Analysis
MODEL = os.getenv("MODEL", "null")
API_KEY = os.getenv("API_KEY", "null")
LLM_BASE = os.getenv("LLM_BASE", "null")

# X/Twitter API Configuration
X_BEARER_TOKEN = os.getenv("X_BEARER_TOKEN", "null")
X_API_BASE = os.getenv("X_API_BASE", "null")

# SN13/Macro API Configuration
SN13_API_KEY = os.getenv("SN13_API_KEY", "null")
SN13_API_URL = os.getenv("SN13_API_URL", "https://constellation.api.cloud.macrocosmos.ai/sn13.v1.Sn13Service/OnDemandData")

# API Source Selection (for validators)
# Set to "x_api" or "sn13_api" to choose which API to use for validation
X_API_SOURCE = os.getenv("X_API_SOURCE", "x_api")


# ============================================================================
# Miner-Specific Configuration
# ============================================================================

# V3 miners process TweetBatch requests from validators - no scraping/submission config needed


# ============================================================================
# Validator-Specific Configuration
# ============================================================================

# Miner API configuration
MINER_API_URL = os.getenv("MINER_API_URL", "null")
BATCH_HTTP_TIMEOUT = float(os.getenv("BATCH_HTTP_TIMEOUT", "30.0"))
VOTE_ENDPOINT = os.getenv("VOTE_ENDPOINT", "null")
# Backward compatibility: support both old and new names
VALIDATION_POLL_SECONDS = int(os.getenv("VALIDATION_POLL_SECONDS", os.getenv("BATCH_POLL_SECONDS", "10")))
SCORES_BLOCK_INTERVAL = int(os.getenv("SCORES_BLOCK_INTERVAL", "100"))

MINER_BATCH_SIZE = int(os.getenv("MINER_BATCH_SIZE", "3"))
# How many tweets/messages to fetch from the API per poll cycle.
# Fetched items are split into MINER_BATCH_SIZE chunks and dispatched to different miners.
VALIDATION_FETCH_LIMIT = int(os.getenv("VALIDATION_FETCH_LIMIT", "24"))
BLOCK_LENGTH = int(os.getenv("BLOCK_LENGTH", "100"))
START_BLOCK = int(os.getenv("START_BLOCK", "0"))

# Validator -> miner dispatch behavior (push-based mining).
# The validator should only "dispatch" work; miners will push results back asynchronously.
MINER_SEND_TIMEOUT = float(os.getenv("MINER_SEND_TIMEOUT", "6.0"))
VALIDATOR_MINER_QUERY_CONCURRENCY = int(os.getenv("VALIDATOR_MINER_QUERY_CONCURRENCY", "8"))
VALIDATOR_MAX_PENDING_MINER_TASKS = int(os.getenv("VALIDATOR_MAX_PENDING_MINER_TASKS", "256"))

# Validation thread pool: controls how many concurrent LLM-based validations run.
# Lower values reduce LLM API pressure at the cost of slower validation throughput.
VALIDATION_MAX_WORKERS = int(os.getenv("VALIDATION_MAX_WORKERS", "8"))

# LLM result cache: avoids redundant API calls for identical post text.
LLM_CACHE_TTL = float(os.getenv("LLM_CACHE_TTL", "300"))
LLM_CACHE_MAX_SIZE = int(os.getenv("LLM_CACHE_MAX_SIZE", "1024"))

# Tweet store configuration
TWEET_STORE_LOCATION = os.getenv("TWEET_STORE_LOCATION", str(_SUBNET_ROOT / ".tweet_store.json"))
TWEET_MAX_PROCESS_TIME = float(os.getenv("TWEET_MAX_PROCESS_TIME", "300.0"))  # 5 minutes default

# Telegram store configuration
TELEGRAM_STORE_LOCATION = os.getenv("TELEGRAM_STORE_LOCATION", str(_SUBNET_ROOT / ".telegram_store.json"))

# Message max process time (shared for tweets and telegram, with fallback chain for backward compatibility)
MESSAGE_MAX_PROCESS_TIME = float(os.getenv("MESSAGE_MAX_PROCESS_TIME", os.getenv("TWEET_MAX_PROCESS_TIME", "300.0")))

# Penalty and reward store configuration
PENALTY_STORE_LOCATION = os.getenv("PENALTY_STORE_LOCATION", str(_SUBNET_ROOT / ".penalty_store.json"))
REWARD_STORE_LOCATION = os.getenv("REWARD_STORE_LOCATION", str(_SUBNET_ROOT / ".reward_store.json"))

USD_PRICE_PER_POINT = float(os.getenv("USD_PRICE_PER_POINT", "0.040"))
MIN_PERCENT_PER_POINT = float(os.getenv("MIN_PERCENT_PER_POINT", "0.003"))
FINNEY_RPC = os.getenv("FINNEY_RPC", "wss://entrypoint-finney.opentensor.ai:443")

EPOCH_LENGTH = int(os.getenv("EPOCH_LENGTH", "100"))

BURN_UID = int(os.getenv("BURN_UID", "189"))

# Validator↔validator broadcast state (rewards and penalties)
BROADCAST_STATE_LOCATION = os.getenv("BROADCAST_STATE_LOCATION", str(_SUBNET_ROOT / ".broadcast_state.json"))
PENALTY_BROADCAST_STATE_LOCATION = os.getenv("PENALTY_BROADCAST_STATE_LOCATION", str(_SUBNET_ROOT / ".penalty_broadcast_state.json"))
VALIDATOR_BROADCAST_MAX_TARGETS = int(os.getenv("VALIDATOR_BROADCAST_MAX_TARGETS", "32"))

# Validator allowlist selection
VALIDATOR_STAKE_THRESHOLD = float(os.getenv("VALIDATOR_STAKE_THRESHOLD", "0"))
VALIDATOR_CACHE_SECONDS = float(os.getenv("VALIDATOR_CACHE_SECONDS", "120"))
ALLOW_MANUAL_VALIDATOR_HOTKEYS = os.getenv("ALLOW_MANUAL_VALIDATOR_HOTKEYS", "false").lower() == "true"
MANUAL_VALIDATOR_HOTKEYS = [hk.strip() for hk in os.getenv("MANUAL_VALIDATOR_HOTKEYS", "").split(",") if hk.strip()]


# ============================================================================
# Remote Config (fetched from coordination API)
# ============================================================================

# Keys managed by the remote /config/subnet endpoint.
# Maps config key -> (type_cast, module-level attribute name)
_REMOTE_CONFIG_KEYS = {
    "USD_PRICE_PER_POINT": (float, "USD_PRICE_PER_POINT"),
    "MINER_BATCH_SIZE":    (int,   "MINER_BATCH_SIZE"),
    "VALIDATION_FETCH_LIMIT": (int, "VALIDATION_FETCH_LIMIT"),
    "MIN_PERCENT_PER_POINT":  (float, "MIN_PERCENT_PER_POINT"),
}

REMOTE_CONFIG_REFRESH_SECONDS = int(os.getenv("REMOTE_CONFIG_REFRESH_SECONDS", "3600"))
_remote_config_last_fetch: float = 0.0
_remote_config_lock = threading.Lock()


_wallet_ref = None  # set by validation_client before first call


def set_wallet(wallet) -> None:
    """Store a reference to the validator wallet for authenticated API calls."""
    global _wallet_ref
    _wallet_ref = wallet


def _build_auth_headers() -> dict:
    """Build auth headers using the stored wallet, matching TalismanAPIClient format."""
    if _wallet_ref is None:
        return {}
    try:
        timestamp = time.time()
        message = f"talisman-ai-auth:{int(timestamp)}"
        signature = _wallet_ref.hotkey.sign(message).hex()
        return {
            "X-Auth-SS58Address": _wallet_ref.hotkey.ss58_address,
            "X-Auth-Signature": signature,
            "X-Auth-Message": message,
            "X-Auth-Timestamp": str(timestamp),
        }
    except Exception:
        return {}


def refresh_remote_config(force: bool = False) -> dict:
    """
    Fetch recommended config from the coordination API and apply values.

    For each key in _REMOTE_CONFIG_KEYS:
      - If OVERRIDE_<key> is set in the environment, use that (operator opt-out).
      - Else if the API returned a value, apply it.
      - Else keep the current (hardcoded) default.

    Returns a dict of {key: (value, source)} for logging.
    """
    global _remote_config_last_fetch
    import requests as _requests  # local import to avoid top-level dep in miners

    module = __import__(__name__)

    now = time.time()
    if not force and (now - _remote_config_last_fetch) < REMOTE_CONFIG_REFRESH_SECONDS:
        return {}

    with _remote_config_lock:
        # Double-check after acquiring lock
        if not force and (time.time() - _remote_config_last_fetch) < REMOTE_CONFIG_REFRESH_SECONDS:
            return {}

        api_url = MINER_API_URL
        if not api_url or api_url == "null":
            print("[REMOTE_CONFIG] MINER_API_URL not set, skipping remote config fetch")
            _remote_config_last_fetch = time.time()
            return {}

        remote_values = {}
        try:
            headers = _build_auth_headers()
            resp = _requests.get(
                f"{api_url.rstrip('/')}/config/subnet",
                headers=headers,
                timeout=10.0,
            )
            resp.raise_for_status()
            data = resp.json()
            remote_values = data.get("config", {})
            print(f"[REMOTE_CONFIG] Fetched remote config (version={data.get('version')}): {remote_values}")
        except Exception as e:
            print(f"[REMOTE_CONFIG] Failed to fetch remote config: {e}")
            _remote_config_last_fetch = time.time()
            return {}

        applied = {}
        for key, (cast_fn, attr_name) in _REMOTE_CONFIG_KEYS.items():
            override_env = os.getenv(f"OVERRIDE_{key}")
            if override_env is not None:
                try:
                    value = cast_fn(override_env)
                    setattr(module, attr_name, value)
                    applied[key] = (value, "local_override")
                    print(f"[REMOTE_CONFIG] {key}={value} (OVERRIDE_{key} set locally)")
                except (ValueError, TypeError) as e:
                    print(f"[REMOTE_CONFIG] Bad OVERRIDE_{key}={override_env!r}: {e}")
            elif key in remote_values:
                try:
                    value = cast_fn(remote_values[key])
                    setattr(module, attr_name, value)
                    applied[key] = (value, "remote_api")
                    print(f"[REMOTE_CONFIG] {key}={value} (from API)")
                except (ValueError, TypeError) as e:
                    print(f"[REMOTE_CONFIG] Bad remote value for {key}={remote_values[key]!r}: {e}")
            else:
                current = getattr(module, attr_name, None)
                applied[key] = (current, "default")

        _remote_config_last_fetch = time.time()
        return applied