import math
import time
import bittensor as bt
from typing import Dict, Set, Tuple


BACKOFF_SCHEDULE = [30, 60, 120, 300, 600]  # seconds
CONSECUTIVE_FAILURES_BEFORE_COOLDOWN = 10
MAX_INFLIGHT_PER_MINER = 4


def _cfg(name, default):
    """Read an adaptive-dispatch knob from config lazily — avoids an import cycle
    and lets remote-config updates take effect without a restart."""
    try:
        from alpharidge_ai import config
        return getattr(config, name, default)
    except Exception:
        return default


class MinerCooldownTracker:
    """
    Tracks miner dispatch failures with exponential backoff, and limits
    concurrent in-flight dispatches per miner to avoid overwhelming healthy ones.

    Adaptive dispatch (RFC 2026-06-28): when constructed with ``adaptive=True`` AND
    ``config.ADAPTIVE_DISPATCH_ENABLED`` is on, the static per-miner in-flight cap is
    replaced by a dynamic congestion *window* that grows on clean, on-time completion
    and shrinks on invalid / timeout. Only the article tracker is adaptive; tweet and
    telegram trackers stay static. With the flag off (or adaptive=False) behaviour is
    identical to before.
    """

    def __init__(self, adaptive: bool = False):
        # {hotkey: (consecutive_fails, cooldown_level, cooldown_until)}
        self._state: Dict[str, Tuple[int, int, float]] = {}
        self._inflight: Dict[str, int] = {}

        # ---- Adaptive dispatch state (RFC 2026-06-28) ----
        self._adaptive = adaptive
        self._window: Dict[str, float] = {}       # per-miner congestion window
        self._consec_to: Dict[str, int] = {}      # consecutive lease timeouts (non-response)
        self._covered_ep: Dict[str, int] = {}     # last epoch given a coverage batch
        self._cap: float = None                   # per-tick anti-monopoly cap; None => from config

    # ---- Adaptive knobs (read live so remote-config updates apply) ----

    def _window_min(self) -> float:
        return float(_cfg("DISPATCH_WINDOW_MIN", 1))

    def _grow(self) -> float:
        return float(_cfg("DISPATCH_WINDOW_GROW", 1.0))

    def _shrink(self) -> float:
        return float(_cfg("DISPATCH_WINDOW_SHRINK", 0.5))

    def _chronic_n(self) -> int:
        return int(_cfg("DISPATCH_CHRONIC_TIMEOUT_N", 5))

    def _late_threshold_s(self) -> float:
        # A valid push-back slower than this fraction of the lease freezes growth.
        return float(_cfg("DISPATCH_LATE_FRACTION", 0.6)) * float(_cfg("SCORING_LEASE_TTL_SECONDS", 900))

    def _adaptive_active(self) -> bool:
        return self._adaptive and bool(_cfg("ADAPTIVE_DISPATCH_ENABLED", False))

    def _get_window(self, hotkey: str) -> float:
        return self._window.get(hotkey, self._window_min())

    def _effective_cap(self) -> float:
        if self._cap is not None:
            return self._cap
        budget = float(_cfg("VALIDATOR_MINER_QUERY_CONCURRENCY", 8))
        return max(self._window_min(), float(_cfg("DISPATCH_WINDOW_CAP_PCT", 0.15)) * budget)

    def set_cap(self, cap: float) -> None:
        """Allocator sets the anti-monopoly cap each tick: cap_pct * total in-flight budget."""
        self._cap = max(self._window_min(), float(cap))

    # ---- In-flight tracking ----

    def try_acquire(self, hotkey: str) -> bool:
        """Returns True if the miner has capacity for another dispatch.

        Static limit (``MAX_INFLIGHT_PER_MINER``) unless this is the adaptive tracker
        and the flag is on, in which case the per-miner window governs.
        """
        count = self._inflight.get(hotkey, 0)
        if self._adaptive_active():
            limit = max(1, int(math.floor(self._get_window(hotkey))))
        else:
            limit = MAX_INFLIGHT_PER_MINER
        if count >= limit:
            return False
        self._inflight[hotkey] = count + 1
        return True

    def release(self, hotkey: str) -> None:
        count = self._inflight.get(hotkey, 0)
        if count > 1:
            self._inflight[hotkey] = count - 1
        elif hotkey in self._inflight:
            del self._inflight[hotkey]

    def inflight(self, hotkey: str) -> int:
        return self._inflight.get(hotkey, 0)

    def window(self, hotkey: str) -> float:
        return self._get_window(hotkey)

    def window_values(self) -> list:
        """Current per-miner window sizes (for pilot metrics)."""
        return list(self._window.values())

    # ---- Adaptive window updates (RFC 2026-06-28) ----
    #
    # These update ONLY the window + the chronic-timeout counter. They do NOT touch
    # in-flight: under adaptive dispatch in-flight is reconciled from the article
    # store's PROCESSING set each cycle (reconcile_inflight), which is leak-proof by
    # construction — a missed or duplicated completion event cannot strand the
    # counter, and the per-article store can't desync the per-batch window. (The
    # static path still uses release() at the ack.)

    def record_timely_valid(self, hotkey: str, latency_s: float) -> None:
        """Valid push-back: reset the chronic counter and grow the window — but only
        if the completion was comfortably on-time. A valid-but-slow return means the
        miner is at capacity, so we freeze (hold) the window rather than grow. This is
        what finds capacity *without* ramping into a timeout (objective 8)."""
        self._consec_to[hotkey] = 0
        if latency_s is not None and latency_s <= self._late_threshold_s():
            w = self._get_window(hotkey)
            self._window[hotkey] = min(self._effective_cap(), w + self._grow() / max(w, 1e-9))

    def record_invalid(self, hotkey: str) -> None:
        """Returned-but-invalid: shrink the window. Resets the chronic counter — the
        miner *responded* (it is alive); bad quality is the integrity gate's job, not
        the non-response counter's."""
        self._consec_to[hotkey] = 0
        self._window[hotkey] = max(self._window_min(), self._get_window(hotkey) * self._shrink())

    def record_timeout(self, hotkey: str) -> bool:
        """A reclaim cycle in which this miner had ≥1 lease timeout: shrink the window
        and advance the consecutive-timeout counter. Call once per hotkey per reclaim
        cycle (not per article). Returns True when chronic (>= DISPATCH_CHRONIC_TIMEOUT_N
        consecutive) — the caller then applies the integrity penalty/broadcast. On
        chronic it also drops the miner into the exponential-backoff cooldown and resets
        the counter (clean slate for a fair re-probe after cooldown)."""
        self._window[hotkey] = max(self._window_min(), self._get_window(hotkey) * self._shrink())
        n = self._consec_to.get(hotkey, 0) + 1
        if n >= self._chronic_n():
            self._consec_to[hotkey] = 0
            self.escalate_to_cooldown(hotkey)
            return True
        self._consec_to[hotkey] = n
        return False

    # ---- Coverage tracking (used by the allocator, PR 4) ----

    def covered_epoch(self, hotkey: str) -> int:
        return self._covered_ep.get(hotkey, -1)

    def mark_covered(self, hotkey: str, epoch: int) -> None:
        self._covered_ep[hotkey] = int(epoch)

    # ---- Reconciliation (anti-leak; RFC Component 2) ----

    def reconcile_inflight(self, counts: Dict[str, int]) -> None:
        """Rebuild in-flight counts from an authoritative source (articles still in
        PROCESSING status). Because release moved off the synchronous ack path, a lost
        push-back + a missed reclaim would otherwise leak the counter upward and
        silently shrink a miner's window to zero. Call this periodically."""
        self._inflight = {hk: int(c) for hk, c in counts.items() if c and c > 0}

    # ---- Cooldown tracking ----

    def record_failure(self, hotkey: str) -> None:
        consec, level, _ = self._state.get(hotkey, (0, 0, 0.0))
        consec += 1

        if consec < CONSECUTIVE_FAILURES_BEFORE_COOLDOWN:
            self._state[hotkey] = (consec, level, 0.0)
            return

        level = min(level + 1, len(BACKOFF_SCHEDULE))
        backoff_idx = min(level - 1, len(BACKOFF_SCHEDULE) - 1)
        cooldown_secs = BACKOFF_SCHEDULE[backoff_idx]
        self._state[hotkey] = (consec, level, time.time() + cooldown_secs)
        bt.logging.info(
            f"[COOLDOWN] Miner {hotkey[:12]}.. {consec} consecutive failures, "
            f"cooldown for {cooldown_secs}s (level {level})"
        )

    def escalate_to_cooldown(self, hotkey: str) -> None:
        """Force the next exponential-backoff cooldown step immediately. Used by
        chronic-timeout escalation (a separate, faster signal than record_failure's
        consecutive-dispatch-failure count); repeat escalations lengthen the cooldown."""
        consec, level, _ = self._state.get(hotkey, (0, 0, 0.0))
        level = min(level + 1, len(BACKOFF_SCHEDULE))
        backoff_idx = min(level - 1, len(BACKOFF_SCHEDULE) - 1)
        cooldown_secs = BACKOFF_SCHEDULE[backoff_idx]
        self._state[hotkey] = (consec, level, time.time() + cooldown_secs)
        bt.logging.info(
            f"[COOLDOWN] Miner {hotkey[:12]}.. chronic timeout escalation, "
            f"cooldown for {cooldown_secs}s (level {level})"
        )

    def record_success(self, hotkey: str) -> None:
        if hotkey in self._state:
            _, level, _ = self._state[hotkey]
            if level > 0:
                bt.logging.info(f"[COOLDOWN] Miner {hotkey[:12]}.. recovered, clearing cooldown")
            del self._state[hotkey]

    def is_on_cooldown(self, hotkey: str) -> bool:
        entry = self._state.get(hotkey)
        if entry is None:
            return False
        _, _, cooldown_until = entry
        return cooldown_until > 0 and time.time() < cooldown_until

    def get_cooled_down_hotkeys(self) -> Set[str]:
        now = time.time()
        return {hk for hk, (_, _, until) in self._state.items() if until > 0 and now < until}

    def prune(self, active_hotkeys: Set[str]) -> None:
        stale = [hk for hk in self._state if hk not in active_hotkeys]
        for hk in stale:
            del self._state[hk]
        stale_inflight = [hk for hk in self._inflight if hk not in active_hotkeys]
        for hk in stale_inflight:
            del self._inflight[hk]
        # Adaptive dispatch maps follow the same lifecycle.
        for d in (self._window, self._consec_to, self._covered_ep):
            for hk in [h for h in d if h not in active_hotkeys]:
                del d[hk]
        if stale:
            bt.logging.debug(f"[COOLDOWN] Pruned {len(stale)} stale hotkey(s)")

    def stats(self) -> Tuple[int, int]:
        """Returns (total_tracked, currently_on_cooldown)."""
        now = time.time()
        on_cooldown = sum(1 for _, (_, _, until) in self._state.items() if until > 0 and now < until)
        return len(self._state), on_cooldown
