"""Points per hotkey over the ledger window.

Every leaf counts at its own points, whatever its category; the only scaling is settle() over
the total capacity. Points are summed per hotkey and may be negative; for weights a hotkey's
negative balance is netted against the other hotkeys of its coldkey (the signed coldkey at
receipt, so a deregistered hotkey's debt still counts), and only then floored at zero. Debt
never lapses: when the epoch leaving the window would lift a coldkey's negative balance, the
unpaid part comes back as a carry leaf in the next epoch (see carried).

Pure and shared by the API and validators, so both reach the same numbers from the leaves.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping

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


def carried(prev: Iterable[Mapping], leaving: Iterable[Mapping]) -> List[dict]:
    """Carry entries for the epoch after the window prev, of which leaving is the oldest
    epoch: for each coldkey (a hotkey without one is its own) whose raw balance over prev
    is below zero, the part of it that leaving's rows hold, as one negative entry on the
    hotkey and category whose leaving rows are lowest."""
    prev = list(prev)
    ck = coldkeys(prev)
    balance: Dict[str, float] = {}
    for leaf in prev:
        g = ck.get(str(leaf["miner_hotkey"]), str(leaf["miner_hotkey"]))
        balance[g] = balance.get(g, 0.0) + float(leaf["points_awarded"])
    out_hk: Dict[str, Dict[str, float]] = {}
    out_cat: Dict[str, Dict[str, float]] = {}
    for leaf in leaving:
        hk = str(leaf["miner_hotkey"])
        g, cat, pts = ck.get(hk, hk), category_of(leaf), float(leaf["points_awarded"])
        by_hk, by_cat = out_hk.setdefault(g, {}), out_cat.setdefault(g, {})
        by_hk[hk] = by_hk.get(hk, 0.0) + pts
        by_cat[cat] = by_cat.get(cat, 0.0) + pts
    out = []
    for g in sorted(out_hk):
        amount = _floor6(min(-balance.get(g, 0.0), -sum(out_hk[g].values())))
        if amount <= 0:
            continue
        hk = min(out_hk[g], key=lambda h: (out_hk[g][h], h))
        cat = min(out_cat[g], key=lambda c: (out_cat[g][c], c))
        out.append({"miner_hotkey": hk, "coldkey": ck.get(hk), "category": cat,
                    "points_awarded": -amount})
    return out


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
