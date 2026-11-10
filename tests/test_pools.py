import math

from alpharidge_ai.mechanism.pools import net_by_coldkey, window_points
from alpharidge_ai.mechanism.settlement import settle


def leaf(hk, cat, pts, i=0):
    return {"resource_type": "ledger", "resource_id": str(i), "miner_hotkey": hk,
            "validator_verdict": "provisional", "categorical_key": f"{cat}|event",
            "points_awarded": pts}


def test_a_flood_in_one_category_never_caps_another_below_capacity():
    calm = [leaf("a", "sports", 10.0)]
    flood = [leaf(f"x{i}", "finance", 100.0, i) for i in range(8)]
    alone = settle(window_points(calm), 1000.0).shares
    busy = settle(window_points(calm + flood), 1000.0)
    assert busy.shares["a"] == alone["a"] == 10.0 / 1000.0
    assert busy.shares["x0"] == 100.0 / 1000.0
    assert math.isclose(busy.burn, 1 - 810.0 / 1000.0)


def test_over_capacity_every_category_scales_together():
    leaves = [leaf("a", "sports", 300.0), leaf("b", "finance", 900.0, 1),
              leaf("c", "politics", 800.0, 2)]
    got = settle(window_points(leaves), 1000.0)
    assert got.burn == 0.0
    assert {k: round(v, 9) for k, v in got.shares.items()} == {"a": 0.15, "b": 0.45, "c": 0.4}


def test_negatives_net_by_coldkey_and_any_category_counts():
    got = window_points([leaf("a", "weather", 5.0), leaf("b", "sports", 2.0),
                         leaf("b", "sports", -3.0)])
    assert got == {"a": 5.0, "b": -1.0}
    points = {"a": 3.0, "b": -1.0, "c": 1.0, "d": -2.0}
    assert net_by_coldkey(points, {"a": "x", "b": "x", "c": "y", "d": "y"}) == {"a": 2.0}
    assert net_by_coldkey(points, {}) == {"a": 3.0, "c": 1.0}


def test_split_rounding_never_adds():
    whole = window_points([leaf("a", "sports", 0.500001)])
    split = window_points([leaf("b", "sports", 0.2500005), leaf("c", "sports", 0.2500005, 1)])
    assert sum(split.values()) <= sum(whole.values())
    parts = [leaf("a", "sports", 1.987375), leaf("a", "sports", 2.364371, 1)]
    assert window_points(parts) == {"a": 4.351746}
