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
AUDIT_V2 = "audit_v2"

# Wire codes. Observations travel as numbers, so a channel is sent as its code; an
# observation without one is a legacy observation.
CODES: Dict[str, int] = {LEGACY: 0, TRIAGE: 1, FLOOR: 2, AUDIT: 3, KEEPER: 4, GRADED: 5,
                         AUDIT_V2: 6}
NAMES: Dict[int, str] = {code: name for name, code in CODES.items()}
CHANNELS = tuple(CODES)

DEFAULT_WEIGHTS: Dict[str, float] = {
    LEGACY: 1.0, TRIAGE: 1.0, FLOOR: 1.0, AUDIT: 2.0, KEEPER: 0.5, GRADED: 1.0,
    AUDIT_V2: 0.0,
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
            prior: float, alphas: Mapping[str, float] = None,
            defaults: Mapping[str, float] = None) -> float:
    """Weighted mean of the channel scores.

    A channel with a default always counts at its full weight: it reads as the default
    until it has data and moves to its own measurement over its warm-up. A channel without
    one is phased in by weight instead.
    """
    channels = channels or {}
    defaults = defaults or {}
    total = 0.0
    mass = 0.0
    for name in sorted(set(channels) | set(defaults)):
        w = float(weights.get(name, 0.0))
        if w <= 0.0:
            continue
        st = channels.get(name) or {}
        n = int(st.get("n", 0))
        ramp = min(1.0, n / warmup((alphas or {}).get(name)))
        measured = float(st.get("r", prior))
        if name in defaults:
            total += w
            mass += w * (ramp * measured + (1.0 - ramp) * float(defaults[name]))
        elif ramp > 0.0:
            total += w * ramp
            mass += w * ramp * measured
    return mass / total if total > 0.0 else float(prior)
