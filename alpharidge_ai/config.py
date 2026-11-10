"""Configuration: env flags from .miner_env / .vali_env, served remote config,
and the neuron command-line arguments.

    from alpharidge_ai import config
    config.BLOCK_LENGTH
"""

from pathlib import Path
import os
import threading
import time
from typing import Optional

from alpharidge_ai.ledger.attestation import BLOCK_LENGTH  # noqa: F401
from alpharidge_ai.schema.types import reverify_rate

# Repo root (this file is alpharidge_ai/config.py)
_SUBNET_ROOT = Path(__file__).resolve().parent.parent

# Paths to environment files
_MINER_ENV_PATH = _SUBNET_ROOT / ".miner_env"
_VALI_ENV_PATH = _SUBNET_ROOT / ".vali_env"

# Load environment files (validator vars override miner vars if both exist)
try:
    from dotenv import load_dotenv

    found = [p for p in (_MINER_ENV_PATH, _VALI_ENV_PATH) if p.exists()]
    for path in found:
        load_dotenv(str(path), override=True)
        print(f"[CONFIG] Loaded {path}")
    if not found:
        print(f"[CONFIG] Warning: neither {_MINER_ENV_PATH} nor {_VALI_ENV_PATH} found")

except ImportError:
    print("[CONFIG] Warning: python-dotenv not installed, using system environment variables only")


# ============================================================================
# Neuron settings
# ============================================================================

# The coordination API, for miners and validators alike.
MINER_API_URL = os.getenv("MINER_API_URL", "null")
BATCH_HTTP_TIMEOUT = float(os.getenv("BATCH_HTTP_TIMEOUT", "30.0"))
VALIDATION_POLL_SECONDS = int(os.getenv("VALIDATION_POLL_SECONDS", "10"))

# Reference miner.
MINER_FEEDS = os.getenv("MINER_FEEDS", "")
MINER_CATEGORIES = tuple(c.strip() for c in os.getenv(
    "MINER_CATEGORIES", "finance").split(",") if c.strip())
MINER_POLL_SECONDS = float(os.getenv("MINER_POLL_SECONDS", "300"))
MINER_MAX_AGE_H = float(os.getenv("MINER_MAX_AGE_H", "48"))
MINER_ITEMS_PER_TICK = int(os.getenv("MINER_ITEMS_PER_TICK", "40"))
MINER_STATUS_PER_TICK = int(os.getenv("MINER_STATUS_PER_TICK", "50"))
MINER_LEAD_SENTENCES = int(os.getenv("MINER_LEAD_SENTENCES", "6"))
# Contact URL or email added to the crawler user agent.
USER_AGENT_CONTACT = os.getenv("USER_AGENT_CONTACT", "")
MINER_USER_AGENT = os.getenv("MINER_USER_AGENT", "")
MINER_FETCH_TIMEOUT_S = float(os.getenv("MINER_FETCH_TIMEOUT_S", "10"))
MINER_DOMAIN_RPS = float(os.getenv("MINER_DOMAIN_RPS", "0.5"))
MINER_FETCH_PUBLIC_ONLY = os.getenv("MINER_FETCH_PUBLIC_ONLY", "true").lower() == "true"
MINER_STATE_PATH = os.getenv("MINER_STATE_PATH", "")

# Shared by every validator, so not settings.
BURN_UID = 189
# The API attestation pubkey (sr25519 ss58). Public by design.
API_ATTESTATION_PUBKEY = "5DqYRNaJ9FJ2cuJTFxbU5HDLeeotpgj6Zgkrkw4RgS6SA4nf"

# --mock only: blocks follow the wall clock (0 keeps the chain still), and these hotkeys
# are registered in order from uid 1 instead of placeholder miners.
MOCK_BLOCK_SECONDS = float(os.getenv("MOCK_BLOCK_SECONDS", "0"))
MOCK_GENESIS_TS = float(os.getenv("MOCK_GENESIS_TS", "0"))
MOCK_HOTKEYS = tuple(h.strip() for h in os.getenv("MOCK_HOTKEYS", "").split(",") if h.strip())
# "hotkey:coldkey" pairs; unlisted hotkeys share one mock coldkey.
MOCK_COLDKEYS = dict(p.strip().split(":", 1) for p in os.getenv("MOCK_COLDKEYS", "").split(",")
                     if ":" in p)
# A file that gets one JSON line per set_weights call.
MOCK_WEIGHTS_LOG = os.getenv("MOCK_WEIGHTS_LOG", "")

# Share of paid claims each validator re-verifies (shared with the API). A signed profile's
# ledger.reverify_rate overrides it.
LEDGER_REVERIFY_RATE = reverify_rate()
if LEDGER_REVERIFY_RATE != 1.0:
    print(f"[CONFIG] WARNING: LEDGER_REVERIFY_RATE={LEDGER_REVERIFY_RATE} from the environment, "
          "not the default 1.0; remove it unless the operator asked for it")

# Re-verifying the keyed sample. Check values are constants in verify/check.py.
REVERIFY_CONCURRENCY = int(os.getenv("REVERIFY_CONCURRENCY", "4"))
REVERIFY_USER_AGENT = os.getenv("REVERIFY_USER_AGENT", "")
REVERIFY_FETCH_TIMEOUT_S = float(os.getenv("REVERIFY_FETCH_TIMEOUT_S", "15"))
REVERIFY_FETCH_RETRY = int(os.getenv("REVERIFY_FETCH_RETRY", "1"))
REVERIFY_DOMAIN_RPS = float(os.getenv("REVERIFY_DOMAIN_RPS", "1"))
REVERIFY_FETCH_PUBLIC_ONLY = os.getenv("REVERIFY_FETCH_PUBLIC_ONLY", "true").lower() == "true"


def check_fetch_guard(mock: bool, network: Optional[str]) -> None:
    """Fetching private addresses is for mock and local chains only."""
    if REVERIFY_FETCH_PUBLIC_ONLY:
        return
    if not mock and network != "local":
        raise SystemExit("[CONFIG] REVERIFY_FETCH_PUBLIC_ONLY=false is for mock and local "
                         "chains only")
    print("[CONFIG] WARNING: REVERIFY_FETCH_PUBLIC_ONLY=false: page fetches may reach "
          "private addresses")

# Reading the trusted list's feeds while scoops are open (listed validators only); fetches use
# the re-verify settings above.
SCAN_INTERVAL_S = float(os.getenv("SCAN_INTERVAL_S", "600"))
SCAN_CONCURRENCY = int(os.getenv("SCAN_CONCURRENCY", "4"))

# Judging queued claims' pages (listed validators only). The endpoint and model are served to
# listed validators by the API; the key is the validator's own (API_KEY). Fetches use the
# re-verify settings above.
JUDGE_URL = os.getenv("EL_JUDGE_URL", "")
JUDGE_MODEL = os.getenv("EL_JUDGE_MODEL", "")
JUDGE_API_KEY = os.getenv("API_KEY") or os.getenv("EL_JUDGE_API_KEY", "")
JUDGE_POLL_S = float(os.getenv("JUDGE_POLL_S", "10"))
JUDGE_DAILY_CAP_USD = float(os.getenv("EL_JUDGE_DAILY_CAP_USD", "10"))
# Hotkeys whose posted verdicts re-verify accepts, when the signed profile names none.
JUDGE_VALIDATORS = tuple(h.strip() for h in os.getenv("JUDGE_VALIDATORS", "").split(",")
                         if h.strip())


# ============================================================================
# Served remote config and the other requests outside the ledger client
# ============================================================================

# Served by the API on /config/subnet and applied unless OVERRIDE_<KEY> is set locally.
_REMOTE_CONFIG_KEYS = {
    "EL_JUDGE_URL":             (str,   "JUDGE_URL"),
    "EL_JUDGE_MODEL":           (str,   "JUDGE_MODEL"),
    "USER_AGENT_CONTACT":       (str,   "USER_AGENT_CONTACT"),
    "REVERIFY_FETCH_TIMEOUT_S": (float, "REVERIFY_FETCH_TIMEOUT_S"),
    "REVERIFY_DOMAIN_RPS":      (float, "REVERIFY_DOMAIN_RPS"),
    "SCAN_INTERVAL_S":          (float, "SCAN_INTERVAL_S"),
    "JUDGE_POLL_S":             (float, "JUDGE_POLL_S"),
}

REMOTE_CONFIG_REFRESH_SECONDS = int(os.getenv("REMOTE_CONFIG_REFRESH_SECONDS", "3600"))
_remote_config_last_fetch: float = 0.0
_remote_config_lock = threading.Lock()
_wallet_ref = None


def set_wallet(wallet) -> None:
    global _wallet_ref
    _wallet_ref = wallet


def _build_auth_headers(method: str = "GET", url: str = "/") -> dict:
    if _wallet_ref is None:
        return {}
    try:
        from alpharidge_ai import __version__
        from alpharidge_ai.ledger.request_auth import signed_headers
        headers = signed_headers(_wallet_ref.hotkey, method, url)
        headers["X-Validator-Version"] = __version__
        return headers
    except Exception:
        return {}


def _version_tuple(v: str) -> tuple:
    return tuple(int(x) for x in v.split("."))


def refresh_remote_config(force: bool = False) -> dict:
    """Fetch the served config from the API and apply it, unless a local OVERRIDE_<KEY> is set.
    Warns when this validator is below the minimum version. Returns the response ({} on
    failure or when not yet due)."""
    global _remote_config_last_fetch
    import bittensor as bt
    import requests

    if not force and time.time() - _remote_config_last_fetch < REMOTE_CONFIG_REFRESH_SECONDS:
        return {}
    with _remote_config_lock:
        if not force and time.time() - _remote_config_last_fetch < REMOTE_CONFIG_REFRESH_SECONDS:
            return {}
        api_url = (MINER_API_URL or "").rstrip("/")
        if api_url in ("", "null"):
            return {}
        try:
            url = f"{api_url}/config/subnet"
            resp = requests.get(url, headers=_build_auth_headers("GET", url), timeout=15)
            resp.raise_for_status()
            data = resp.json() or {}
        except Exception as e:
            bt.logging.warning(f"[REMOTE_CONFIG] Failed to fetch config: {e}")
            return {}
        _remote_config_last_fetch = time.time()

        served = data.get("config") or {}
        for key, (cast, attr) in _REMOTE_CONFIG_KEYS.items():
            override = os.getenv(f"OVERRIDE_{key}")
            raw, source = (override, "local OVERRIDE") if override is not None else (
                served.get(key), "from API")
            if raw is None:
                continue
            try:
                value = cast(raw)
            except (TypeError, ValueError):
                continue
            if globals()[attr] != value:
                globals()[attr] = value
                shown = value if key not in ("EL_JUDGE_URL", "EL_JUDGE_MODEL") else "set"
                bt.logging.info(f"[REMOTE_CONFIG] {key} = {shown} ({source})")

        try:
            from alpharidge_ai import __version__
            minimum = data.get("min_validator_version", "0.0.0")
            if _version_tuple(__version__) < _version_tuple(minimum):
                bt.logging.warning(f"[REMOTE_CONFIG] Validator version {__version__} is below "
                                   f"minimum {minimum}. Run 'git pull && pm2 restart' to fix.")
        except ValueError:
            pass
        return data


# ============================================================================
# Neuron command-line arguments
# ============================================================================

_EVENTS_LEVEL = 38


def _setup_events_logger(full_path, retention_size):
    import logging
    from logging.handlers import RotatingFileHandler

    logging.addLevelName(_EVENTS_LEVEL, "EVENT")
    logger = logging.getLogger("event")
    logger.setLevel(_EVENTS_LEVEL)

    def event(self, message, *args, **kws):
        if self.isEnabledFor(_EVENTS_LEVEL):
            self._log(_EVENTS_LEVEL, message, args, **kws)

    logging.Logger.event = event
    handler = RotatingFileHandler(os.path.join(full_path, "events.log"),
                                  maxBytes=retention_size, backupCount=10)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                                           datefmt="%Y-%m-%d %H:%M:%S"))
    handler.setLevel(_EVENTS_LEVEL)
    logger.addHandler(handler)
    return logger


def check_config(cls, cfg):
    import bittensor as bt
    bt.logging.check_config(cfg)
    full_path = os.path.expanduser("{}/{}/{}/netuid{}/{}".format(
        cfg.logging.logging_dir, cfg.wallet.name, cfg.wallet.hotkey, cfg.netuid, cfg.neuron.name))
    cfg.neuron.full_path = full_path
    os.makedirs(full_path, exist_ok=True)
    if not cfg.neuron.dont_save_events:
        events_logger = _setup_events_logger(full_path, cfg.neuron.events_retention_size)
        bt.logging.register_primary_logger(events_logger.name)


def add_args(cls, parser):
    parser.add_argument("--netuid", type=int, help="Subnet netuid", default=1)
    parser.add_argument("--neuron.epoch_length", type=int, default=100,
                        help="Blocks between metagraph syncs and weight setting.")
    parser.add_argument("--mock", action="store_true", default=False,
                        help="Mock neuron and all network components.")
    parser.add_argument("--neuron.events_retention_size", type=int, default=2 * 1024 * 1024 * 1024,
                        help="Events log retention size in bytes.")
    parser.add_argument("--neuron.dont_save_events", action="store_true", default=False,
                        help="Do not write the events log.")


def add_miner_args(cls, parser):
    parser.add_argument("--neuron.name", type=str, default="miner",
                        help="State directory name under the wallet path.")


def add_validator_args(cls, parser):
    parser.add_argument("--neuron.name", type=str, default="validator",
                        help="State directory name under the wallet path.")
    parser.add_argument("--neuron.disable_set_weights", action="store_true", default=False,
                        help="Disables setting weights.")


def neuron_config(cls):
    import argparse
    import bittensor as bt
    parser = argparse.ArgumentParser()
    bt.Wallet.add_args(parser)
    bt.Subtensor.add_args(parser)
    bt.logging.add_args(parser)
    cls.add_args(parser)
    return bt.Config(parser)
