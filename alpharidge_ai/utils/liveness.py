"""
Liveness roster for adaptive dispatch (RFC 2026-06-28).

Answers one question cheaply and off the dispatch path: "is this miner reachable
right now?" A miner is considered live if it has been *seen* within
``LIVENESS_TTL_S`` — where "seen" means either a successful push-back of results
or a positive response to a background ``IsAlive`` heartbeat sweep. Both sources
feed a single last-seen timestamp, so if the heartbeat sweep ever stalls, miners
simply age out after the TTL rather than being pinned alive forever.

This is deliberately separate from:
  - cooldown / integrity (``MinerCooldownTracker._state``) — "is it trusted?"
  - the in-flight window (``MinerCooldownTracker`` capacity) — "how much work?"

Liveness only gates *eligibility* for selection, and only when adaptive dispatch
is enabled. Keyed by hotkey so it survives UID reassignment.
"""

import time
from typing import Dict, List, Optional, Iterable, Tuple


def _default_ttl() -> float:
    # Imported lazily so the roster can be constructed/tested without pulling in
    # the full config module (and its network deps).
    from alpharidge_ai import config
    return float(getattr(config, "LIVENESS_TTL_S", 120))


class LivenessRoster:
    def __init__(self, clock=time.monotonic, ttl_s: Optional[float] = None):
        # hotkey -> last time seen alive (push-back OR heartbeat), in clock() units
        self._last_seen: Dict[str, float] = {}
        self._clock = clock
        # When None, the TTL is read live from config (so remote-config updates
        # apply without a restart). A concrete value is used mainly by tests.
        self._ttl_s = ttl_s

    def _ttl(self) -> float:
        return self._ttl_s if self._ttl_s is not None else _default_ttl()

    def mark_seen(self, hotkey: str) -> None:
        """Record a successful push-back from a miner. Safe to call on any path."""
        if hotkey:
            self._last_seen[hotkey] = self._clock()

    def update_from_heartbeat(self, alive_hotkeys: Iterable[str]) -> None:
        """Refresh last-seen for every hotkey the latest IsAlive sweep found alive."""
        now = self._clock()
        for hk in (alive_hotkeys or ()):
            if hk:
                self._last_seen[hk] = now

    def is_alive(self, hotkey: str) -> bool:
        if not hotkey:
            return False
        last = self._last_seen.get(hotkey)
        if last is None:
            return False
        return (self._clock() - last) <= self._ttl()

    def live_uids(self, metagraph) -> List[int]:
        """UIDs of currently-live miners, mapped fresh from the metagraph hotkeys."""
        return [uid for uid, hk in enumerate(metagraph.hotkeys) if self.is_alive(hk)]

    def prune(self, active_hotkeys: Iterable[str]) -> None:
        """Drop hotkeys no longer in the metagraph (e.g. deregistered)."""
        active = set(active_hotkeys)
        for hk in [h for h in self._last_seen if h not in active]:
            del self._last_seen[hk]

    def stats(self) -> Tuple[int, int]:
        """(tracked_hotkeys, currently_live)."""
        now = self._clock()
        ttl = self._ttl()
        live = sum(1 for t in self._last_seen.values() if (now - t) <= ttl)
        return len(self._last_seen), live
