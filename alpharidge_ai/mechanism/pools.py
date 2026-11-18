"""Points per hotkey over the ledger window.

Every leaf counts at its own points, whatever its category; the only scaling is settle() over
the total capacity. Points are summed per hotkey and may be negative; for weights a hotkey's
negative balance is netted against the other hotkeys of its coldkey (the signed coldkey at
receipt, so a deregistered hotkey's debt still counts), and only then floored at zero. Debt
never lapses: when the epoch leaving the window would lift a coldkey's negative balance, the
unpaid part comes back as a carry leaf in the next epoch (see carried). Once quality scales
the weights, that balance is valued as they value it (quality.worth).

Pure and shared by the API and validators, so both reach the same numbers from the leaves.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, List, Mapping, Optional

from alpharidge_ai.ledger.attestation import LEAF_V2

# Used when no signed mechanism profile is in force.
WINDOW_EPOCHS = 216
CAPACITY = 900.0
# The share of a claim's points paid at verification; the rest settles later.
PROVISIONAL_SHARE = 0.40


def category_of(leaf: Mapping) -> str:
    return str(leaf.get("categorical_key") or "").split("|", 1)[0]


def _floor6(x: float) -> float:
    """Rounded down to 6 dp, so per-hotkey rounding never adds up to more than the total.
    Float noise below 1e-9 is dropped first, so a sum that lands just under a step keeps it."""
    return math.floor(round(x * 1e6, 3)) / 1e6


def window_points(leaves: Iterable[Mapping]) -> Dict[str, float]:
    """Points per hotkey, rounded down to 6 dp. Hotkeys at zero are left out."""
    totals: Dict[str, float] = {}
    for leaf in leaves:
        hk = str(leaf["miner_hotkey"])
        totals[hk] = totals.get(hk, 0.0) + float(leaf["points_awarded"])
    return {hk: _floor6(v) for hk, v in sorted(totals.items()) if _floor6(v) != 0}


def coldkeys(leaves: Iterable[Mapping]) -> Dict[str, str]:
    """Each hotkey's coldkey at receipt, from its latest leaf that names one."""
    latest: Dict[str, tuple] = {}
    for leaf in leaves:
        ck = leaf.get("coldkey")
        if not ck or leaf.get("leaf_version") != LEAF_V2:
            continue
        hk, at = str(leaf["miner_hotkey"]), int(leaf.get("received_ms") or 0)
        if hk not in latest or at >= latest[hk][0]:
            latest[hk] = (at, str(ck))
    return {hk: ck for hk, (_, ck) in latest.items()}


def _points(leaf: Mapping) -> float:
    return float(leaf["points_awarded"])


def carried(prev: Iterable[Mapping], leaving: Iterable[Mapping],
            worth: Optional[Callable[[Mapping], float]] = None) -> List[dict]:
    """Carry entries for the epoch after the window prev, of which leaving is the oldest
    epoch: for each coldkey (a hotkey without one is its own) whose balance over prev is
    below zero, the part of it that leaving's rows hold, as one negative entry on the
    hotkey and category whose leaving rows are lowest. Each leaf counts at worth(leaf),
    its points when worth is None. Worths are summed exactly (math.fsum), so any leaf order
    gives the same entries; raw points are summed in leaf order, as released."""
    add = _in_order if worth is None else math.fsum
    # The released total over a coldkey's hotkeys used the builtin sum (compensated from
    # Python 3.12), kept for raw points.
    total = sum if worth is None else math.fsum
    worth = worth or _points
    prev = list(prev)
    ck = coldkeys(prev)
    parts: Dict[str, List[float]] = {}
    for leaf in prev:
        g = ck.get(str(leaf["miner_hotkey"]), str(leaf["miner_hotkey"]))
        parts.setdefault(g, []).append(worth(leaf))
    balance = {g: add(v) for g, v in parts.items()}
    hk_parts: Dict[str, Dict[str, List[float]]] = {}
    cat_parts: Dict[str, Dict[str, List[float]]] = {}
    for leaf in leaving:
        hk = str(leaf["miner_hotkey"])
        g, cat, pts = ck.get(hk, hk), category_of(leaf), worth(leaf)
        hk_parts.setdefault(g, {}).setdefault(hk, []).append(pts)
        cat_parts.setdefault(g, {}).setdefault(cat, []).append(pts)
    out = []
    for g in sorted(hk_parts):
        by_hk = {hk: add(v) for hk, v in hk_parts[g].items()}
        by_cat = {cat: add(v) for cat, v in cat_parts[g].items()}
        amount = _floor6(min(-balance.get(g, 0.0), -total(by_hk.values())))
        if amount <= 0:
            continue
        hk = min(by_hk, key=lambda h: (by_hk[h], h))
        cat = min(by_cat, key=lambda c: (by_cat[c], c))
        out.append({"miner_hotkey": hk, "coldkey": ck.get(hk), "category": cat,
                    "points_awarded": -amount})
    return out


def _in_order(values: Iterable[float]) -> float:
    """A sum in the given order, from 0.0."""
    total = 0.0
    for v in values:
        total += v
    return total


def net_by_coldkey(points: Mapping[str, float],
                   coldkey_of: Mapping[str, str]) -> Dict[str, float]:
    """Per-hotkey points for weights: each coldkey's total, when above zero, shared across its
    positive hotkeys in proportion. A hotkey without a coldkey is its own."""
    groups: Dict[str, list] = {}
    for hk in sorted(points):
        groups.setdefault(coldkey_of.get(hk, hk), []).append(hk)
    out: Dict[str, float] = {}
    for hks in groups.values():
        total = sum(float(points[hk]) for hk in hks)
        positive = sum(max(0.0, float(points[hk])) for hk in hks)
        if total <= 0 or positive <= 0:
            continue
        for hk in hks:
            v = _floor6(max(0.0, float(points[hk])) * total / positive)
            if v > 0:
                out[hk] = v
    return out
