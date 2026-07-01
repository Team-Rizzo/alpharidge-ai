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
        self._timed_out: Set[str] = set()
        self._ack_latencies: List[float] = []

    def incr(self, key: str, n: int = 1) -> None:
        self._counts[key] = self._counts.get(key, 0) + n

    def mark_scored(self, hotkey: str) -> None:
        if hotkey:
            self._scored.add(hotkey)

    def mark_timeout(self, hotkey: str) -> None:
        """Distinct miners with >=1 lease timeout this cycle. `timeout` counts
        events (one per miner per reclaim cycle, summed); this counts the unique
        set — the depth tripwire is whether that set BROADENS past the existing
        small cluster (~10) as windows open, which the event count alone hides."""
        if hotkey:
            self._timed_out.add(hotkey)

    def record_ack(self, latency_s) -> None:
        """Record a successful send-ack round-trip (dendrite.process_time, seconds)."""
        if latency_s is not None and len(self._ack_latencies) < _MAX_ACK_SAMPLES:
            self._ack_latencies.append(float(latency_s))

    def reset(self) -> None:
        self._counts = {}
        self._scored = set()
        self._timed_out = set()
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

    def format_line(self, window_values: List[float], live: int, on_cooldown: int,
                    val_backlog: int = 0) -> str:
        c = self._counts
        dispatched = c.get("dispatched", 0)
        valid = c.get("valid", 0)
        wv = sorted(float(w) for w in window_values)
        if wv:
            wmin, wmax, wmed, wmean = wv[0], wv[-1], statistics.median(wv), sum(wv) / len(wv)
        else:
            wmin = wmax = wmed = wmean = 0.0
        al = sorted(self._ack_latencies)
        # Window-depth breadth: how many live miners actually crossed into a
        # depth slot. The allocator's slot limit is int(window), so window>=2.0
        # is the first tick a miner can hold a 2nd batch. Under coverage every
        # window sits at ~1 and these are all 0; they are the primary signal that
        # the concurrency ramp is buying depth (and across how many miners, not
        # just the single one window_max reflects).
        window_ge2 = sum(1 for w in wv if w >= 2.0)
        window_ge3 = sum(1 for w in wv if w >= 3.0)
        window_ge4 = sum(1 for w in wv if w >= 4.0)
        parts = [
            "[ADAPTIVE_METRICS]",
            f"distinct_scored={len(self._scored)}",
            f"dispatched={dispatched}",
            f"depth_dispatched={c.get('depth_dispatched', 0)}",
            f"ack_ok={c.get('ack_ok', 0)}",
            f"ack_fail={c.get('ack_fail', 0)}",
            f"valid={valid}",
            f"invalid={c.get('invalid', 0)}",
            f"incomplete={c.get('incomplete', 0)}",
            f"timeout={c.get('timeout', 0)}",
            f"timeout_miners={len(self._timed_out)}",
            f"completion_pct={self._pct(valid, dispatched):.1f}",
            f"val_backlog={val_backlog}",
            f"ackfail_pct={self._pct(c.get('ack_fail', 0), dispatched):.1f}",
            f"timeout_pct={self._pct(c.get('timeout', 0), dispatched):.1f}",
            f"window_min={wmin:.2f}",
            f"window_med={wmed:.2f}",
            f"window_mean={wmean:.2f}",
            f"window_max={wmax:.2f}",
            f"window_ge2={window_ge2}",
            f"window_ge3={window_ge3}",
            f"window_ge4={window_ge4}",
            f"window_n={len(wv)}",
            f"ack_p50={self._pctile(al, 0.50):.2f}",
            f"ack_p95={self._pctile(al, 0.95):.2f}",
            f"ack_n={len(al)}",
            f"live={live}",
            f"on_cooldown={on_cooldown}",
        ]
        return " ".join(parts)
