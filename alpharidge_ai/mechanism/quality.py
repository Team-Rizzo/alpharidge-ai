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
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, Mapping, Optional, Sequence

HALF_LIFE = 20
DECAY = 0.5 ** (1 / HALF_LIFE)
START = 0.50
ZERO, FULL = 0.15, 0.25
PAY_ZERO, PAY_FULL = 0.10, 0.60
# Weights are scaled for each epoch whose points were signed from the start on, by claims
# decided from it. The start is EL_QUALITY_FROM_MS in the environment when set (the API's own
# setting, or a local override), else the one the API serves on /config/subnet
# (`set_quality_switch`), else QUALITY_FROM here. None of them set: quality is off.
QUALITY_FROM: Optional[datetime] = None
# The start the API last served, in ms; None until one is served.
_served_ms: Optional[int] = None
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


class QualitySettingError(ValueError):
    """A quality start that is not a time in ms."""


def _as_ms(raw) -> int:
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        raise QualitySettingError(f"quality start {raw!r} is not a time in ms")
    try:
        value = float(raw)
    except ValueError:
        raise QualitySettingError(f"quality start {raw!r} is not a time in ms")
    if value != int(value) or value <= 0:
        raise QualitySettingError(f"quality start {raw!r} is not a time in ms")
    return int(value)


def _env_ms() -> Optional[int]:
    raw = os.getenv("EL_QUALITY_FROM_MS", "").strip()
    return None if raw in ("", "0") else _as_ms(raw)


def start_ms() -> Optional[int]:
    """The start in force, in ms, or None when quality is off."""
    env = _env_ms()
    if env is not None:
        return env
    if _served_ms is not None:
        return _served_ms
    return int(QUALITY_FROM.timestamp() * 1000) if QUALITY_FROM is not None else None


def parse_quality_switch(raw) -> Optional[int]:
    """{"from_ms": start} as start; None for no switch. A start that is not a positive whole
    number of ms raises QualitySettingError."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise QualitySettingError(f"quality switch {raw!r} is not an object")
    return _as_ms(raw.get("from_ms"))


def set_quality_switch(raw) -> Optional[int]:
    """Apply a served quality switch (None clears it). A bad one raises QualitySettingError and
    the last known one stays. Returns the start now in force."""
    global _served_ms
    _served_ms = parse_quality_switch(raw)
    return start_ms()


def may_be_active(now: datetime) -> bool:
    """Whether an epoch read at `now` can have been signed from the start."""
    start = start_ms()
    if start is None:
        return False
    return now >= datetime.fromtimestamp(start / 1000, tz=timezone.utc) - ASK_BEFORE


def active(until_ms: Optional[int]) -> bool:
    """Whether an epoch is scaled: its points were signed (the until_ms the API serves with
    its outcomes) at or after the start. Every validator reads the same signing time, so
    all turn quality on at the same epoch."""
    start = start_ms()
    return start is not None and until_ms is not None and int(until_ms) >= start


def from_ms() -> int:
    start = start_ms()
    return 0 if start is None else start


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
