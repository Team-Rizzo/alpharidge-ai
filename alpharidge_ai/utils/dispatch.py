"""
Coverage-then-depth article allocator (RFC 2026-06-28).

Pure assignment logic, separated from the validator so it can be tested in
isolation. Given the live miner UIDs and a window-aware tracker, it returns
``(uid, batch_index)`` assignments:

  1. Coverage pass — every live miner that has not been covered this epoch and
     has a free window slot gets exactly one batch (the coverage floor — every live
     miner is scored each epoch).
  2. Depth pass — remaining batches go to miners with window headroom, highest
     window first, round-robin so a single miner can't drain the queue before
     others get a turn.

It is intentionally **fully read-only** on the tracker — coverage is marked by the
caller on *actual dispatch* (not here), and the real per-miner reservation
(``try_acquire``/``release``) stays in the validator's dispatch coroutine, so a
pending-task-cap truncation can neither leak a reserved slot nor mark a miner
covered without sending it work. ``provisional`` mirrors what those acquisitions
will be so we don't assign a miner more than ``floor(window)`` within one tick.
"""

import random
from typing import Dict, List, Optional, Sequence, Tuple


def _slot_limit(tracker, hotkey: str) -> int:
    """How many batches a UID may hold at once this tick.

    Under earned rations the cap comes from what the UID has delivered, not from the
    adaptive window, so a UID whose ration exceeds one batch can hold several.
    """
    # Only when a ration is actually in force. The source is installed at startup and
    # returns None until its switch is published, so its presence says nothing.
    ration = getattr(tracker, "ration_for", None)
    if ration is not None and ration(hotkey) is not None:
        return max(1, int(tracker.batches_per_epoch(hotkey)))
    # Selection and reservation must reach the same number, so ask the tracker for its
    # limit rather than reconstructing it from one of the inputs.
    limit = getattr(tracker, "inflight_limit", None)
    if limit is not None:
        return max(1, int(limit(hotkey)))
    return max(1, int(tracker.window(hotkey)))


def coverage_depth_select(
    live_uids: Sequence[int],
    hotkeys: Sequence[str],
    tracker,
    epoch: int,
    n_batches: int,
    rng: Optional[random.Random] = None,
) -> List[Tuple[int, int]]:
    # Serve in a randomized order so priority does not track UID: when a tick has
    # fewer batches than uncovered targets the front of the list is served first,
    # and window ties in the depth pass break by list order. Selection is a local,
    # non-consensus choice, so a fresh shuffle each tick is safe.
    order = list(live_uids)
    (rng or random).shuffle(order)

    provisional: Dict[str, int] = {}

    def has_slot(uid: int) -> bool:
        hk = hotkeys[uid]
        return tracker.inflight(hk) + provisional.get(hk, 0) < _slot_limit(tracker, hk)

    def take(uid: int) -> None:
        hk = hotkeys[uid]
        provisional[hk] = provisional.get(hk, 0) + 1

    assignments: List[Tuple[int, int]] = []
    bi = 0

    # Coverage pass.
    for uid in order:
        if bi >= n_batches:
            break
        hk = hotkeys[uid]
        if tracker.covered_epoch(hk) < epoch and has_slot(uid):
            assignments.append((uid, bi))
            take(uid)
            bi += 1

    # Depth pass.
    if bi < n_batches:
        depth_order = sorted(order, key=lambda u: tracker.window(hotkeys[u]), reverse=True)
        while bi < n_batches:
            progressed = False
            for uid in depth_order:
                if bi >= n_batches:
                    break
                if has_slot(uid):
                    assignments.append((uid, bi))
                    take(uid)
                    bi += 1
                    progressed = True
            if not progressed:
                break  # every live window is full; remaining batches retry next tick

    return assignments


# ---- Credit allocator ---------------------------------------------------------------

CARRY_MAX_BATCHES = 2.0


def credit_select(
    live_uids: Sequence[int],
    hotkeys: Sequence[str],
    tracker,
    n_batches: int,
    *,
    weight_of=None,
    cap: Optional[float] = None,
    floor_frac: float = 0.0,
    eligible=None,
    carry_max: float = CARRY_MAX_BATCHES,
) -> List[Tuple[int, int]]:
    """Assign this tick's batches by turn owed rather than by draw.

    Every eligible miner is owed a share of each tick; whoever is owed most is served
    first, and an unserved turn stays owed, bounded. `weight_of` scales a miner's share
    (default: equal), `cap` bounds it as a multiple of the average, and `floor_frac`
    reserves a slice for miners outside `eligible`, so one that has stopped delivering
    can still be re-measured.

    Read-only on the tracker apart from the credit it keeps: the reservation stays with
    the validator's dispatch coroutine, as in coverage_depth_select.
    """
    if n_batches <= 0 or not live_uids:
        return []

    uids = [u for u in live_uids if 0 <= u < len(hotkeys)]
    if not uids:
        return []
    ok = [u for u in uids if eligible is None or eligible(hotkeys[u])]
    held = [u for u in uids if u not in set(ok)]

    reserved = min(len(held), int(round(max(0.0, floor_frac) * n_batches))) if held else 0
    share_batches = n_batches - reserved
    if not ok:
        reserved, share_batches = min(len(held), n_batches), 0

    weights = {}
    for u in ok:
        try:
            w = float(weight_of(hotkeys[u])) if weight_of else 1.0
        except Exception:
            w = 1.0
        weights[u] = max(0.0, w)
    total = sum(weights.values())
    if total <= 0.0:
        weights = {u: 1.0 for u in ok}
        total = float(len(ok)) or 1.0
    if cap and cap > 0 and ok:
        ceiling = float(cap) * total / len(ok)
        weights = {u: min(w, ceiling) for u, w in weights.items()}
        total = sum(weights.values()) or 1.0

    for u in ok:
        tracker.add_credit(hotkeys[u], share_batches * weights[u] / total, carry_max)

    pending: Dict[str, int] = {}

    def free(uid: int) -> bool:
        hk = hotkeys[uid]
        return tracker.inflight(hk) + pending.get(hk, 0) < _slot_limit(tracker, hk)

    assignments: List[Tuple[int, int]] = []

    def serve(uid: int, debit: bool) -> None:
        if debit:
            tracker.spend_credit(hotkeys[uid], 1.0, carry_max)
        pending[hotkeys[uid]] = pending.get(hotkeys[uid], 0) + 1
        assignments.append((uid, len(assignments)))

    # Most owed first; hotkey breaks ties so priority never tracks UID. Every batch goes
    # out: the owed are served first, then the same order fills what is left. Serving
    # early is debited, so a turn taken now is one not owed later.
    order = sorted(ok, key=lambda x: (-tracker.credit(hotkeys[x]), hotkeys[x]))
    for uid in order:
        while len(assignments) < share_batches and tracker.credit(hotkeys[uid]) >= 1.0:
            if not free(uid):
                break
            serve(uid, True)
    while len(assignments) < share_batches:
        progressed = False
        for uid in order:
            if len(assignments) >= share_batches:
                break
            if free(uid):
                serve(uid, True)
                progressed = True
        if not progressed:
            break

    # The exploration slice, on the same credit so it rotates on its own.
    if reserved and held:
        for uid in held:
            tracker.add_credit(hotkeys[uid], reserved / len(held), carry_max)
        left = reserved
        for uid in sorted(held, key=lambda x: (-tracker.credit(hotkeys[x]), hotkeys[x])):
            if left <= 0 or len(assignments) >= n_batches:
                break
            if free(uid):
                serve(uid, True)
                left -= 1

    return assignments


class ShadowCredit:
    """A tracker view that keeps its own credit, for running credit dispatch alongside.

    Everything else (in-flight, slot limits, coverage) reads through to the real
    tracker, so the shadow sees the same constraints without touching its state.
    """

    def __init__(self, tracker, credit_state: Dict[str, float]):
        self._tracker = tracker
        self._credit = credit_state

    def __getattr__(self, name):
        return getattr(self._tracker, name)

    def credit(self, hotkey: str) -> float:
        return float(self._credit.get(hotkey, 0.0))

    def add_credit(self, hotkey: str, amount: float, carry_max: float) -> None:
        if amount > 0.0:
            self._credit[hotkey] = min(float(carry_max), self.credit(hotkey) + float(amount))

    def spend_credit(self, hotkey: str, amount: float, carry_max: float = None) -> None:
        floor = -float(CARRY_MAX_BATCHES if carry_max is None else carry_max)
        self._credit[hotkey] = max(floor, self.credit(hotkey) - float(amount))
