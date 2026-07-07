"""Reputation integration and emission gate.

reputation: substantive-weighted recency EMA of per-article graded scores, in [0,1].
emission:   steep logistic floor of reputation, multiplied by volume elsewhere.

Params are injected (defaults match the validated config); the caller reads served
config and passes them in.
"""
from __future__ import annotations
import math

# defaults — override from served config
ALPHA = 0.03           # EMA step; half-life ~ ln(2)/alpha samples
PRIOR = 0.5            # cold-start value (below the floor => new state earns ~0 until proven)
MIDPOINT = 0.59       # logistic centre; calibrate on the live distribution before gating
GAIN = 100.0          # logistic steepness


def update(prev, score, weight, alpha=ALPHA):
    """Single-step EMA update. `weight` scales the step (substantive samples move more).
    prev/score in [0,1]; returns the new reputation."""
    a = min(1.0, alpha * max(weight, 0.0))
    return (1.0 - a) * prev + a * score


def replay(scored, alpha=ALPHA, prior=PRIOR):
    """Recompute reputation from an ordered sequence of (score, weight). For backfill /
    verification; production updates incrementally with update()."""
    r = prior
    for s, w in scored:
        r = update(r, s, w, alpha)
    return r


def gate(reputation, midpoint=MIDPOINT, gain=GAIN):
    """Emission multiplier in (0,1). Steep => ~0 below the floor, ~1 above; the caller
    multiplies this by volume, so a sub-floor value earns ~0 at any volume."""
    return 1.0 / (1.0 + math.exp(-gain * (reputation - midpoint)))
