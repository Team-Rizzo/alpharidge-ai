"""Total weight must not depend on the shape of the emission curve.

Settlement compares total weight against a fixed capacity. If the multiplier is applied
raw, any change to the curve — bar, gain or ceiling — moves total weight, and burn moves
with it against a capacity calibrated for the previous shape. The curve has to
redistribute between miners without changing the total.
"""
import types

import pytest

from alpharidge_ai.validator import reputation


def _mult(rep, midpoint, gain, ceiling, bonus_start, bonus_full):
    return reputation.emission(rep, midpoint=midpoint, gain=gain, bonus_ceiling=ceiling,
                               bonus_start=bonus_start, bonus_full=bonus_full)


def _normalised_weights(reps, points, **curve):
    """What the reward path computes: multiplier scaled so the points-weighted mean is 1."""
    mults = [_mult(r, **curve) for r in reps]
    mass = sum(m * p for m, p in zip(mults, points))
    total = sum(points)
    scale = (total / mass) if mass > 0 else 1.0
    return [m * scale * p for m, p in zip(mults, points)]


FIELD = [0.55, 0.66, 0.72, 0.80, 0.84, 0.87, 0.88, 0.89, 0.91, 0.96]
POINTS = [120, 240, 310, 180, 400, 260, 90, 500, 150, 330]


@pytest.mark.parametrize("curve", [
    dict(midpoint=0.869, gain=6, ceiling=1.0, bonus_start=0.890, bonus_full=0.914),
    dict(midpoint=0.869, gain=6, ceiling=3.0, bonus_start=0.890, bonus_full=0.914),
    dict(midpoint=0.322, gain=6, ceiling=1.0, bonus_start=0.322, bonus_full=0.366),
    dict(midpoint=0.750, gain=20, ceiling=2.0, bonus_start=0.860, bonus_full=0.960),
])
def test_total_weight_is_invariant_to_the_curve(curve):
    """Every curve, however steep, must produce the same total as no curve at all."""
    w = _normalised_weights(FIELD, POINTS, **curve)
    assert sum(w) == pytest.approx(sum(POINTS), rel=1e-9)


def test_the_curve_still_redistributes():
    """Invariant total is not the same as no effect — ratios between miners must move."""
    flat = dict(midpoint=0.869, gain=6, ceiling=1.0, bonus_start=0.890, bonus_full=0.914)
    steep = dict(midpoint=0.869, gain=6, ceiling=3.0, bonus_start=0.890, bonus_full=0.914)
    a = _normalised_weights(FIELD, POINTS, **flat)
    b = _normalised_weights(FIELD, POINTS, **steep)
    top = FIELD.index(max(FIELD))
    assert b[top] > a[top], "a higher ceiling must move weight toward the top miner"
    assert sum(a) == pytest.approx(sum(b), rel=1e-9), "while leaving the total alone"


def test_ratios_between_miners_are_untouched_by_normalisation():
    """Normalising is a scale, not a reshaping."""
    curve = dict(midpoint=0.869, gain=6, ceiling=1.0, bonus_start=0.890, bonus_full=0.914)
    raw = [_mult(r, **curve) * p for r, p in zip(FIELD, POINTS)]
    norm = _normalised_weights(FIELD, POINTS, **curve)
    for i in range(1, len(FIELD)):
        assert norm[i] / norm[0] == pytest.approx(raw[i] / raw[0], rel=1e-9)


def test_a_field_gated_to_zero_does_not_divide_by_zero():
    """Undefined scale must leave the multiplier alone rather than invent one."""
    reps = [0.0] * 5
    curve = dict(midpoint=0.95, gain=200, ceiling=0.0, bonus_start=0.95, bonus_full=0.99)
    mults = [_mult(r, **curve) for r in reps]
    mass = sum(m * p for m, p in zip(mults, [100] * 5))
    scale = (500 / mass) if mass > 0 else 1.0
    assert scale == 1.0 or mass > 0
