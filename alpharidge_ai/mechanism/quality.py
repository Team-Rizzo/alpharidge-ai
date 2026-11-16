"""A coldkey's quality: a running score over its decided claims, in the order they were
decided.

    q <- DECAY * q + (1 - DECAY) * x,   x = 1 for a paid claim, 0 for a junk reject

Every other outcome (a known event, a late find, a check that failed on the fetching side)
leaves q as it is. A coldkey starts at START.

Pay: each paid claim's points are scaled by the pay multiplier of its coldkey's q just before
the claim was decided, locked to the claim: its later rows (the settle, a clawback) carry the
same multiplier, and later outcomes never change it. None at PAY_ZERO, full at PAY_FULL,
linear between. The check budget scales by q now: none at ZERO, full at FULL.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, Iterable, Mapping, Optional, Sequence

HALF_LIFE = 50
DECAY = 0.5 ** (1 / HALF_LIFE)
START = 0.30
ZERO, FULL = 0.15, 0.25
PAY_ZERO, PAY_FULL = 0.10, 0.90
# Weights are scaled from this time on, by claims judged from it; set at release.
QUALITY_FROM: Optional[datetime] = None
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


def active(now: datetime) -> bool:
    return QUALITY_FROM is not None and now >= QUALITY_FROM


def from_ms() -> int:
    return int(QUALITY_FROM.timestamp() * 1000) if QUALITY_FROM is not None else 0


def multiplier(q: float) -> float:
    """The check budget's multiplier at q."""
    return min(1.0, max(0.0, (q - ZERO) / (FULL - ZERO)))


def pay_multiplier(q: float) -> float:
    return min(1.0, max(0.0, (q - PAY_ZERO) / (PAY_FULL - PAY_ZERO)))


def scaled(leaves: Iterable[Mapping], locks: Mapping[str, float]) -> Dict[str, float]:
    """Per-hotkey window points with each paid claim's rows times its locked multiplier.
    Rows of a claim decided before the start, debt, penalties and carry are not scaled."""
    out: Dict[str, float] = {}
    for leaf in leaves:
        v = float(leaf["points_awarded"])
        cid = leaf.get("claim_id")
        if leaf.get("validator_verdict") in LOCKED_KINDS and cid is not None:
            v *= locks.get(str(cid), 1.0)
        hk = str(leaf["miner_hotkey"])
        out[hk] = out.get(hk, 0.0) + v
    return {hk: v for hk, v in out.items() if v != 0}
