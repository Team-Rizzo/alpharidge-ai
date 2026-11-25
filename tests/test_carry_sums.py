"""Scaled balances sum to the same value in any leaf order; raw carry is unchanged."""

import math
import random
from typing import Dict

from alpharidge_ai.mechanism import quality
from alpharidge_ai.mechanism.pools import _floor6, carried, category_of, coldkeys


def _carried_7f732ab(prev, leaving):
    """pools.carried as released in 7f732ab (raw points, summed in leaf order)."""
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


def _leaves(rng, n):
    hks = [f"h{i}" for i in range(rng.randint(1, 6))]
    cks = {h: rng.choice(["c1", "c2", None]) for h in hks}
    out = []
    for _ in range(n):
        h = rng.choice(hks)
        leaf = {"miner_hotkey": h, "points_awarded": round(rng.uniform(-30, 12), 6),
                "categorical_key": rng.choice(["finance|event", "technology|scoop"]),
                "validator_verdict": rng.choice(["provisional", "settle", "debt", "clawback"]),
                "claim_id": str(rng.randint(1, 40)), "received_ms": rng.randint(0, 10 ** 6)}
        if cks[h]:
            leaf.update(coldkey=cks[h], leaf_version=2)
        out.append(leaf)
    return out


def test_raw_carry_is_bit_identical_to_the_released_rule():
    rng = random.Random(7)
    for _ in range(3000):
        prev = _leaves(rng, rng.randint(0, 40))
        leaving = prev[:rng.randint(0, len(prev))]
        assert carried(prev, leaving) == _carried_7f732ab(prev, leaving)
        assert carried(prev, leaving, None) == _carried_7f732ab(prev, leaving)


def test_a_scaled_carry_is_the_same_in_any_leaf_order():
    """The API lists the window by ledger id, validators epoch by epoch; scaled values summed
    in order could land on either side of a 1e-6 step."""
    rng = random.Random(11)
    for _ in range(1500):
        prev = _leaves(rng, rng.randint(1, 40))
        leaving = prev[:rng.randint(1, len(prev))]
        locks = {str(i): rng.uniform(0.0, 1.0) for i in range(1, 41)}
        worth = lambda leaf: quality.worth(leaf, locks)
        want = carried(prev, leaving, worth)
        for _ in range(3):
            p, lv = prev[:], leaving[:]
            rng.shuffle(p)
            rng.shuffle(lv)
            assert carried(p, lv, worth) == want


def test_the_reviewers_near_step_case_agrees():
    def leaf(kind, pts, cid, cat="finance|event"):
        return {"miner_hotkey": "hk", "coldkey": "ck", "leaf_version": 2, "received_ms": 1,
                "validator_verdict": kind, "categorical_key": cat, "points_awarded": pts,
                "claim_id": cid}
    rng = random.Random(5)
    for _ in range(20000):
        d = -round(rng.uniform(20, 80), 6)
        p1, p2 = round(rng.uniform(1, 12), 6), round(rng.uniform(1, 12), 6)
        m1 = rng.uniform(0.2, 1.0)
        target = -(rng.randint(1_000_000, 9_000_000) + 0.9995) / 1e6
        m2 = (target - d - p1 * m1) / p2
        if not 0 < m2 < 1:
            continue
        leaves = [leaf("debt", d, "9", "finance|scoop"), leaf("provisional", p1, "1"),
                  leaf("provisional", p2, "2")]
        worth = lambda x, locks={"1": m1, "2": m2}: quality.worth(x, locks)
        assert (carried(leaves, leaves[:1], worth)
                == carried(leaves[1:] + leaves[:1], leaves[:1], worth))


def test_scaled_points_are_the_same_in_any_order():
    rng = random.Random(3)
    for _ in range(500):
        leaves = _leaves(rng, rng.randint(1, 60))
        locks = {str(i): rng.uniform(0.0, 1.0) for i in range(1, 41)}
        want = quality.scaled(leaves, locks)
        rng.shuffle(leaves)
        assert quality.scaled(leaves, locks) == want
        assert all(math.isfinite(v) for v in want.values())
