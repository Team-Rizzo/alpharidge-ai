"""
Per-cycle dispatch metrics for adaptive dispatch (RFC 2026-06-28).

Lightweight counters accumulated at the dispatch/validation/reclaim hook points,
emitted as a single parseable log line per weight cycle (then reset) so the
dashboard can scrape the dispatch signals:
distinct miners scored, completion %, accept-vs-ack-fail, timeout rate, the
per-miner window-size distribution, and the ack-latency distribution (how long
the miner takes to ack a send — `dendrite.process_time` — which reveals whether
the send semaphore is being held across slow acks). (Burn is already logged by
calculate_weights as `total_percent_needed`.)
"""

import statistics
from typing import Dict, List, Set

# Cap retained ack samples per cycle so the list can't grow without bound.
_MAX_ACK_SAMPLES = 10000


class AdaptiveDispatchMetrics:
    def __init__(self):
        self._counts: Dict[str, int] = {}
        self._scored: Set[str] = set()
        self._ack_latencies: List[float] = []

    def incr(self, key: str, n: int = 1) -> None:
        self._counts[key] = self._counts.get(key, 0) + n

    def mark_scored(self, hotkey: str) -> None:
        if hotkey:
            self._scored.add(hotkey)

    def record_ack(self, latency_s) -> None:
        """Record a successful send-ack round-trip (dendrite.process_time, seconds)."""
        if latency_s is not None and len(self._ack_latencies) < _MAX_ACK_SAMPLES:
            self._ack_latencies.append(float(latency_s))

    def reset(self) -> None:
        self._counts = {}
        self._scored = set()
        self._ack_latencies = []

    @staticmethod
    def _pct(num: int, den: int) -> float:
        return (100.0 * num / den) if den else 0.0

    @staticmethod
    def _pctile(sorted_xs: List[float], q: float) -> float:
        if not sorted_xs:
            return 0.0
        i = min(len(sorted_xs) - 1, int(q * len(sorted_xs)))
        return sorted_xs[i]

    def format_line(self, window_values: List[float], live: int, on_cooldown: int) -> str:
        c = self._counts
        dispatched = c.get("dispatched", 0)
        valid = c.get("valid", 0)
        wv = sorted(float(w) for w in window_values)
        if wv:
            wmin, wmax, wmed, wmean = wv[0], wv[-1], statistics.median(wv), sum(wv) / len(wv)
        else:
            wmin = wmax = wmed = wmean = 0.0
        al = sorted(self._ack_latencies)
        parts = [
            "[ADAPTIVE_METRICS]",
            f"distinct_scored={len(self._scored)}",
            f"dispatched={dispatched}",
            f"ack_ok={c.get('ack_ok', 0)}",
            f"ack_fail={c.get('ack_fail', 0)}",
            f"valid={valid}",
            f"invalid={c.get('invalid', 0)}",
            f"incomplete={c.get('incomplete', 0)}",
            f"timeout={c.get('timeout', 0)}",
            f"completion_pct={self._pct(valid, dispatched):.1f}",
            f"ackfail_pct={self._pct(c.get('ack_fail', 0), dispatched):.1f}",
            f"timeout_pct={self._pct(c.get('timeout', 0), dispatched):.1f}",
            f"window_min={wmin:.2f}",
            f"window_med={wmed:.2f}",
            f"window_mean={wmean:.2f}",
            f"window_max={wmax:.2f}",
            f"window_n={len(wv)}",
            f"ack_p50={self._pctile(al, 0.50):.2f}",
            f"ack_p95={self._pctile(al, 0.95):.2f}",
            f"ack_n={len(al)}",
            f"live={live}",
            f"on_cooldown={on_cooldown}",
        ]
        return " ".join(parts)


class DispatchHealth:
    """Dispatch volume against this validator's own recent baseline.

    Counts batches and article cycles in hourly buckets and reports when the last hour
    falls below `min_fraction` of the median hour, once enough history exists.
    """

    def __init__(self, bucket_s: float = 3600.0, history: int = 24, min_history: int = 6,
                 repeat_s: float = 1800.0):
        self._bucket_s = float(bucket_s)
        self._history = int(history)
        self._min_history = int(min_history)
        self._repeat_s = float(repeat_s)
        self._buckets: Dict[int, Dict[str, int]] = {}
        self._last_alert = float("-inf")

    def note(self, kind: str, now: float, n: int = 1) -> None:
        b = int(now // self._bucket_s)
        counts = self._buckets.setdefault(b, {})
        counts[kind] = counts.get(kind, 0) + n
        for old in [k for k in self._buckets if k < b - self._history]:
            del self._buckets[old]

    def check(self, now: float, min_fraction: float) -> List[str]:
        """Alerts for the hour that just ended, at most once per `repeat_s`."""
        current = int(now // self._bucket_s)
        last = current - 1
        past = [k for k in self._buckets if k < last]
        if len(past) < self._min_history or now - self._last_alert < self._repeat_s:
            return []
        out = []
        for kind in ("batches", "cycles"):
            base = statistics.median(self._buckets[k].get(kind, 0) for k in past)
            got = self._buckets.get(last, {}).get(kind, 0)
            if base > 0 and got < min_fraction * base:
                out.append(f"[DISPATCH_HEALTH] {kind} last hour {got} vs typical {base:.0f} "
                           f"({100.0 * got / base:.0f}%)")
        if out:
            self._last_alert = now
        return out
