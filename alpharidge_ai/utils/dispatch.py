"""
Coverage-then-depth article allocator (RFC 2026-06-28).

Pure assignment logic, separated from the validator so it can be tested in
isolation. Given the live miner UIDs and a window-aware tracker, it returns
``(uid, batch_index)`` assignments:

  1. Coverage pass — every live miner that has not been covered this epoch and
     has a free window slot gets exactly one batch (the coverage floor that puts
     every live miner across κ).
  2. Depth pass — remaining batches go to miners with window headroom, highest
     window first, round-robin so a single miner can't drain the queue before
     others get a turn.

It is intentionally read-only on the tracker except for ``mark_covered`` — the
real per-miner reservation (``try_acquire``/``release``) stays in the validator's
dispatch coroutine, so a pending-task-cap truncation can never leak a reserved
slot. ``provisional`` mirrors what those acquisitions will be so we don't assign a
miner more than ``floor(window)`` within one tick.
"""

from typing import Dict, List, Sequence, Tuple


def _slot_limit(tracker, hotkey: str) -> int:
    return max(1, int(tracker.window(hotkey)))


def coverage_depth_select(
    live_uids: Sequence[int],
    hotkeys: Sequence[str],
    tracker,
    epoch: int,
    n_batches: int,
) -> List[Tuple[int, int]]:
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
    for uid in live_uids:
        if bi >= n_batches:
            break
        hk = hotkeys[uid]
        if tracker.covered_epoch(hk) < epoch and has_slot(uid):
            assignments.append((uid, bi))
            take(uid)
            tracker.mark_covered(hk, epoch)
            bi += 1

    # Depth pass.
    if bi < n_batches:
        depth_order = sorted(live_uids, key=lambda u: tracker.window(hotkeys[u]), reverse=True)
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
