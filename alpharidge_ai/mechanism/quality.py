"""A coldkey's quality: a running score over its decided claims, in the order they were
decided: when each was paid or rejected (a scoop held for its window is decided when it is
confirmed or closed, a claim waiting in its line when it is placed).

    q <- DECAY * q + (1 - DECAY) * x,   x = 1 for a paid claim, 0 for a junk reject

Every other outcome (a known event, a late find, a check that failed on the fetching side)
leaves q as it is. A coldkey starts at START.

Pay: each paid claim's points are scaled by the pay multiplier of its coldkey's q just before
the claim was decided, locked to the claim: its later rows (the settle, a clawback) carry the
same multiplier, and later outcomes never change it. None at PAY_ZERO, full at PAY_FULL,
linear between. The check budget scales by q now: none at ZERO, full at FULL.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Dict, Iterable, Mapping, Optional, Sequence

HALF_LIFE = 20
DECAY = 0.5 ** (1 / HALF_LIFE)
START = 0.50
ZERO, FULL = 0.15, 0.25
PAY_ZERO, PAY_FULL = 0.10, 0.60
# Weights are scaled for each epoch whose points were signed from this time on, by claims
# decided from it; set at release.
QUALITY_FROM: Optional[datetime] = None
# An epoch is signed before a validator reads it, so one read this long before QUALITY_FROM
# by the validator's clock cannot be scaled, and the API is not asked. The margin covers clock
# differences.
ASK_BEFORE = timedelta(days=1)
# Rejects that say the claim does not match its page, and an old story sent as a scoop. A
# judge_unclear whose judge found none of the types for the claim counts as junk too (the API
# serves it so).
JUNK = ("judge_reject", "not_an_event", "unsupported", "page_type", "scoop_stale")
# Leaf kinds of a paid claim that carry its locked multiplier.
LOCKED_KINDS = ("provisional", "settle", "clawback")


def scores(outcomes: Iterable[Sequence]) -> Dict[str, float]:
    """Each owner's q after its (owner, passed, ...) outcomes, in decision order."""
    q: Dict[str, float] = {}
    for owner, passed, *_ in outcomes:
        q[owner] = DECAY * q.get(owner, START) + (1 - DECAY) * (1.0 if passed else 0.0)
    return q


def locked(outcomes: Iterable[Sequence]) -> Dict[str, float]:
    """Each paid claim's pay multiplier, by claim id, from its owner's q just before it, over
    (owner, passed, claim_id) outcomes in decision order."""
    q: Dict[str, float] = {}
    out: Dict[str, float] = {}
    for owner, passed, claim_id in outcomes:
        before = q.get(owner, START)
        if passed:
            out[str(claim_id)] = pay_multiplier(before)
        q[owner] = DECAY * before + (1 - DECAY) * (1.0 if passed else 0.0)
    return out


def may_be_active(now: datetime) -> bool:
    """Whether an epoch read at `now` can have been signed from QUALITY_FROM."""
    return QUALITY_FROM is not None and now >= QUALITY_FROM - ASK_BEFORE


def active(until_ms: Optional[int]) -> bool:
    """Whether an epoch is scaled: its points were signed (the until_ms the API serves with
    its outcomes) at or after QUALITY_FROM. Every validator reads the same signing time, so
    all turn quality on at the same epoch."""
    return QUALITY_FROM is not None and until_ms is not None and int(until_ms) >= from_ms()


def from_ms() -> int:
    return int(QUALITY_FROM.timestamp() * 1000) if QUALITY_FROM is not None else 0


def multiplier(q: float) -> float:
    """The check budget's multiplier at q."""
    return min(1.0, max(0.0, (q - ZERO) / (FULL - ZERO)))


def pay_multiplier(q: float) -> float:
    return min(1.0, max(0.0, (q - PAY_ZERO) / (PAY_FULL - PAY_ZERO)))


def worth(leaf: Mapping, locks: Mapping[str, float]) -> float:
    """A leaf's points, times its claim's locked multiplier when it is a paid claim's row.
    Rows of a claim decided before the start, debt, penalties and carry are not scaled."""
    v = float(leaf["points_awarded"])
    cid = leaf.get("claim_id")
    if leaf.get("validator_verdict") in LOCKED_KINDS and cid is not None:
        v *= locks.get(str(cid), 1.0)
    return v


def scaled(leaves: Iterable[Mapping], locks: Mapping[str, float]) -> Dict[str, float]:
    """Per-hotkey window points, each leaf at its worth, summed exactly (math.fsum) so that
    any leaf order gives the same values."""
    parts: Dict[str, list] = {}
    for leaf in leaves:
        parts.setdefault(str(leaf["miner_hotkey"]), []).append(worth(leaf, locks))
    out = {hk: math.fsum(v) for hk, v in parts.items()}
    return {hk: v for hk, v in out.items() if v != 0}


def unlocked(leaves: Iterable[Mapping], locks: Mapping[str, float]) -> int:
    """Paid claims received from QUALITY_FROM whose rows have no lock. Such a claim was decided
    after its receipt, so from the start, and should have one; scaled() pays it 1."""
    start = from_ms()
    out = set()
    for leaf in leaves:
        cid = leaf.get("claim_id")
        if (leaf.get("validator_verdict") in LOCKED_KINDS and cid is not None
                and str(cid) not in locks and leaf.get("received_ms") is not None
                and int(leaf["received_ms"]) >= start):
            out.add(str(cid))
    return len(out)
