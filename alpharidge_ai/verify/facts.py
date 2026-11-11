"""Fact matching: whether two facts state the same quantity for the same field."""
from __future__ import annotations

import re

from alpharidge_ai.verify import floor

_WS = re.compile(r"\s+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _spaced(name) -> str:
    """snake_case and camelCase written as words."""
    return _CAMEL.sub(" ", str(name or "")).replace("_", " ")


def _field(item) -> str:
    return _WS.sub(" ", _spaced(getattr(item, "field", None)).strip().lower())


def _values_match(a: float, b: float) -> bool:
    return abs(a - b) / max(abs(b), 1e-9) <= floor.VALUE_REL_TOLERANCE


def same_quantity(left, right) -> bool:
    """Same unit class and same magnitude, with any scale in the unit applied."""
    try:
        lv, rv = float(getattr(left, "value")), float(getattr(right, "value"))
    except (TypeError, ValueError):
        return False
    lu, ru = getattr(left, "unit", None), getattr(right, "unit", None)
    if floor.unit_class(lu) != floor.unit_class(ru):
        return False
    return _values_match(lv * floor.unit_magnitude(lu), rv * floor.unit_magnitude(ru))


def claims_match(left, right) -> bool:
    """Same field, same unit class, same magnitude."""
    return _field(left) == _field(right) and same_quantity(left, right)
