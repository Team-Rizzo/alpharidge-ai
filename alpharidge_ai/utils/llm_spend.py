"""What the validator's own LLM calls cost and how long they take.

Costs are OpenRouter's per-call accounting, requested only when the client points at
OpenRouter. A summary per purpose and model is logged once an hour.
"""
from __future__ import annotations

import statistics
import threading
import time
from collections import defaultdict, deque

import bittensor as bt

REPORT_EVERY_S = 3600.0
SLOW_MEDIAN_S = 60.0
SLOW_WINDOW = 50
SLOW_WARN_EVERY_S = 600.0

_lock = threading.Lock()
_totals = defaultdict(lambda: [0.0, 0])      # (purpose, model) -> [usd, calls]
_latency = deque(maxlen=SLOW_WINDOW)          # reference-call seconds
_last_report = [time.time()]
_last_slow_warn = [0.0]
_paused_until = [0.0]
PAUSE_S = 600.0
_auto = [0, 0]                               # tool calls answered on the first ask, retried


def usage_body(client) -> dict:
    """`extra_body` fields that make OpenRouter return the call's cost."""
    base = str(getattr(client, "base_url", "") or "")
    return {"usage": {"include": True}} if "openrouter" in base else {}


def request_body(client) -> dict:
    """`extra_body` for the validator's own calls: cost accounting and provider routing."""
    body = usage_body(client)
    if not body:
        return {}
    from alpharidge_ai import config
    prefs = {}
    for key, name in (("ignore", "REFERENCE_PROVIDER_IGNORE"), ("order", "REFERENCE_PROVIDER_ORDER")):
        names = [p.strip() for p in str(getattr(config, name, "") or "").split(",") if p.strip()]
        if names:
            prefs[key] = names
    if "order" in prefs:
        prefs["allow_fallbacks"] = True
    cap = [p.strip() for p in str(getattr(config, "REFERENCE_PROVIDER_MAX_PRICE", "") or "").split(",")]
    try:
        if len(cap) == 2:
            prefs["max_price"] = {"prompt": float(cap[0]), "completion": float(cap[1])}
    except ValueError:
        pass
    if prefs:
        body["provider"] = prefs
    return body


def _cost(response) -> float:
    usage = getattr(response, "usage", None)
    value = getattr(usage, "cost", None)
    if value is None:
        value = (getattr(usage, "model_extra", None) or {}).get("cost")
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def record(purpose: str, model: str, response=None, seconds: float = None) -> None:
    """Account one call. `seconds` feeds the slow-reference alert."""
    now = time.time()
    with _lock:
        entry = _totals[(purpose, model or "")]
        entry[0] += _cost(response) if response is not None else 0.0
        entry[1] += 1
        if seconds is not None:
            _latency.append(float(seconds))
        slow = (len(_latency) >= SLOW_WINDOW // 2
                and statistics.median(_latency) > SLOW_MEDIAN_S
                and now - _last_slow_warn[0] >= SLOW_WARN_EVERY_S)
        if slow:
            _last_slow_warn[0] = now
            median = statistics.median(_latency)
        due = now - _last_report[0] >= REPORT_EVERY_S
        if due:
            rows = sorted(_totals.items(), key=lambda kv: -kv[1][0])
            _totals.clear()
            span = now - _last_report[0]
            _last_report[0] = now
    if slow:
        bt.logging.warning(f"[LLM_SPEND] reference calls are slow: median {median:.0f}s "
                           f"over the last {len(_latency)} calls")
    if due:
        total = sum(v[0] for _, v in rows)
        with _lock:
            answered, retried = _auto
            _auto[0] = _auto[1] = 0
        bt.logging.info(f"[LLM_SPEND] ${total:.3f} over {span / 3600:.1f}h: " + "; ".join(
            f"{p} {m.split('/')[-1]} ${v[0]:.3f} ({v[1]} calls)" for (p, m), v in rows)
            + f"; tool calls {answered} first ask, {retried} retried")


def is_key_limit(error) -> bool:
    """An OpenRouter reply saying the key is out of credit or over its limit."""
    status = getattr(error, "status_code", None)
    text = str(error).lower()
    return status in (402, 403) and ("limit" in text or "credit" in text)


def pause(error) -> None:
    """Stop the validator's own LLM calls for a while; retrying cannot succeed."""
    with _lock:
        first = time.time() >= _paused_until[0]
        _paused_until[0] = time.time() + PAUSE_S
    if first:
        bt.logging.error(f"[LLM_SPEND] API key limit reached, pausing validator LLM calls for "
                         f"{PAUSE_S / 60:.0f} min: {error}")


def paused() -> bool:
    return time.time() < _paused_until[0]


def auto_tools() -> bool:
    """Whether the validator's own-model calls first ask without forcing the tool."""
    from alpharidge_ai import config
    return bool(getattr(config, "REFERENCE_TOOL_AUTO", True))


def tool_choice(name: str, auto: bool):
    return "auto" if auto else {"type": "function", "function": {"name": name}}


def ask_for_tool(prompt: str, name: str, auto: bool) -> str:
    return f"{prompt}\n\nAnswer only by calling {name}." if auto else prompt


def note_auto(answered: bool) -> None:
    with _lock:
        _auto[0 if answered else 1] += 1
