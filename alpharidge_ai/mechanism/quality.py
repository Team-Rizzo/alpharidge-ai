"""A coldkey's quality: a running score over its decided claims, in the order they were
decided, that scales its weight.

    q <- DECAY * q + (1 - DECAY) * x,   x = 1 for a paid claim, 0 for a junk reject

Every other outcome (a known event, a late find, a check that failed on the fetching side)
leaves q as it is. A coldkey starts at START; its weight is full from FULL and none at ZERO,
linear between.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, Iterable, Mapping, Optional, Tuple

HALF_LIFE = 50
DECAY = 0.5 ** (1 / HALF_LIFE)
START = 0.30
ZERO, FULL = 0.15, 0.25
# Weights are scaled from this time on, by claims judged from it; set at release.
QUALITY_FROM: Optional[datetime] = None
# Rejects that say the claim does not match its page.
JUNK = ("judge_reject", "not_an_event", "unsupported", "page_type")


def scores(outcomes: Iterable[Tuple[str, bool]]) -> Dict[str, float]:
    """Each owner's q after its (owner, passed) outcomes, in decision order."""
    q: Dict[str, float] = {}
    for owner, passed in outcomes:
        q[owner] = DECAY * q.get(owner, START) + (1 - DECAY) * (1.0 if passed else 0.0)
    return q


def active(now: datetime) -> bool:
    return QUALITY_FROM is not None and now >= QUALITY_FROM


def from_ms() -> int:
    return int(QUALITY_FROM.timestamp() * 1000) if QUALITY_FROM is not None else 0


def multiplier(q: float) -> float:
    return min(1.0, max(0.0, (q - ZERO) / (FULL - ZERO)))


def scaled(points: Mapping[str, float], coldkey_of: Mapping[str, str],
           q: Mapping[str, float]) -> Dict[str, float]:
    """Per-hotkey points times the multiplier of the hotkey's coldkey (START when unscored)."""
    out = {}
    for hk, v in points.items():
        v = float(v) * multiplier(q.get(coldkey_of.get(hk, hk), START))
        if v > 0:
            out[hk] = v
    return out
