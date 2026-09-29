"""Local, append-only record of the validator's triage draws. One JSON line per
draw, one file per UTC day, kept for DRAW_LOG_KEEP_DAYS."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import bittensor as bt

from alpharidge_ai import config

_lock = threading.Lock()
_state = {"day": None, "fh": None}


def _dir() -> Path:
    return Path(getattr(config, "DRAW_LOG_LOCATION", ".triage_draws"))


def _prune(folder: Path, today: str) -> None:
    keep = int(getattr(config, "DRAW_LOG_KEEP_DAYS", 14))
    cutoff = time.strftime("%Y%m%d", time.gmtime(time.time() - keep * 86400))
    for f in folder.glob("draws-*.jsonl"):
        if f.stem[6:] < cutoff:
            f.unlink(missing_ok=True)


def write(kind: str, **fields) -> None:
    now = time.time()
    day = time.strftime("%Y%m%d", time.gmtime(now))
    line = json.dumps({"t": round(now, 3), "kind": kind, **fields}, default=str)
    try:
        with _lock:
            if _state["day"] != day:
                if _state["fh"]:
                    _state["fh"].close()
                folder = _dir()
                folder.mkdir(parents=True, exist_ok=True)
                _prune(folder, day)
                _state["fh"] = open(folder / f"draws-{day}.jsonl", "a", encoding="utf-8")
                _state["day"] = day
            _state["fh"].write(line + "\n")
            _state["fh"].flush()
    except Exception as e:
        bt.logging.warning(f"[TRIAGE] draw log unavailable ({e}): {line}")
