"""The separate measurements reputation is built from, and how they combine.

Each channel keeps its own running score. Reputation is their weighted mean, so the
mix between channels is set by published weights rather than by how many observations
each happens to produce.
"""
from __future__ import annotations

import math
from typing import Dict, Mapping

LEGACY = "legacy"
TRIAGE = "triage"
FLOOR = "floor"
AUDIT = "audit"
KEEPER = "keeper"
GRADED = "graded"

# Wire codes. Observations travel as numbers, so a channel is sent as its code; an
# observation without one is a legacy observation.
CODES: Dict[str, int] = {LEGACY: 0, TRIAGE: 1, FLOOR: 2, AUDIT: 3, KEEPER: 4, GRADED: 5}
NAMES: Dict[int, str] = {code: name for name, code in CODES.items()}
CHANNELS = tuple(CODES)

DEFAULT_WEIGHTS: Dict[str, float] = {
    LEGACY: 1.0, TRIAGE: 1.0, FLOOR: 1.0, AUDIT: 2.0, KEEPER: 0.5, GRADED: 1.0,
}

# A channel reaches full weight once it holds about one half-life of observations.
DEFAULT_ALPHA = 0.03
WARMUP = math.ceil(math.log(2) / DEFAULT_ALPHA)


def warmup(alpha: float = None) -> int:
    a = DEFAULT_ALPHA if not alpha or alpha <= 0.0 else float(alpha)
    return max(1, math.ceil(math.log(2) / min(a, 1.0)))


def code_of(name: str) -> int:
    return CODES[name]


def name_of(code) -> str:
    """The channel for a wire code, or "" when the code is not one we know."""
    try:
        value = float(code)
    except (TypeError, ValueError):
        return ""
    if value != int(value):
        return ""
    return NAMES.get(int(value), "")


def combine(channels: Mapping[str, Mapping], weights: Mapping[str, float],
            prior: float, alphas: Mapping[str, float] = None) -> float:
    """Weighted mean of the channel scores, each phased in over its first observations."""
    total = 0.0
    mass = 0.0
    for name, st in (channels or {}).items():
        n = int(st.get("n", 0))
        ramp = warmup((alphas or {}).get(name))
        w = float(weights.get(name, 0.0)) * min(1.0, n / ramp)
        if w <= 0.0:
            continue
        total += w
        mass += w * float(st.get("r", prior))
    return mass / total if total > 0.0 else float(prior)
