"""
Per-cycle pilot metrics for adaptive dispatch (RFC 2026-06-28, Component 6).

Lightweight counters accumulated at the dispatch/validation/reclaim hook points,
emitted as a single parseable log line per weight cycle (then reset) so the
dashboard can scrape the pilot signals at fixed pricing:
distinct miners scored, completion %, accept-vs-ack-fail, timeout rate, and the
per-miner window-size distribution. (Burn is already logged by calculate_weights
as `total_percent_needed`.)
"""

import statistics
from typing import Dict, List, Set


class AdaptiveDispatchMetrics:
    def __init__(self):
        self._counts: Dict[str, int] = {}
        self._scored: Set[str] = set()

    def incr(self, key: str, n: int = 1) -> None:
        self._counts[key] = self._counts.get(key, 0) + n

    def mark_scored(self, hotkey: str) -> None:
        if hotkey:
            self._scored.add(hotkey)

    def reset(self) -> None:
        self._counts = {}
        self._scored = set()

    @staticmethod
    def _pct(num: int, den: int) -> float:
        return (100.0 * num / den) if den else 0.0

    def format_line(self, window_values: List[float], live: int, on_cooldown: int) -> str:
        c = self._counts
        dispatched = c.get("dispatched", 0)
        valid = c.get("valid", 0)
        wv = sorted(float(w) for w in window_values)
        if wv:
            wmin, wmax, wmed, wmean = wv[0], wv[-1], statistics.median(wv), sum(wv) / len(wv)
        else:
            wmin = wmax = wmed = wmean = 0.0
        parts = [
            "[ADAPTIVE_METRICS]",
            f"distinct_scored={len(self._scored)}",
            f"dispatched={dispatched}",
            f"ack_ok={c.get('ack_ok', 0)}",
            f"ack_fail={c.get('ack_fail', 0)}",
            f"valid={valid}",
            f"invalid={c.get('invalid', 0)}",
            f"timeout={c.get('timeout', 0)}",
            f"completion_pct={self._pct(valid, dispatched):.1f}",
            f"ackfail_pct={self._pct(c.get('ack_fail', 0), dispatched):.1f}",
            f"timeout_pct={self._pct(c.get('timeout', 0), dispatched):.1f}",
            f"window_min={wmin:.2f}",
            f"window_med={wmed:.2f}",
            f"window_mean={wmean:.2f}",
            f"window_max={wmax:.2f}",
            f"window_n={len(wv)}",
            f"live={live}",
            f"on_cooldown={on_cooldown}",
        ]
        return " ".join(parts)
