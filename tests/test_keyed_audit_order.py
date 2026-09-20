"""The audit budget is spent on articles the key picks, not on the order they arrived in."""

import random
import types

from alpharidge_ai.analyzer.scoring import _keyed_order
from alpharidge_ai.oracle.runner import Auditor


def batch(ids):
    return [types.SimpleNamespace(id=i) for i in ids]


def auditor(key=b"k" * 32):
    return Auditor(key, lambda block: None)


IDS = list(range(100, 140))


def test_the_order_is_the_keys_not_the_batchs():
    a = auditor()
    ordered = [x.id for x in _keyed_order(a, batch(IDS))]
    assert ordered != IDS


def test_shuffling_the_batch_does_not_change_what_is_audited():
    a = auditor()
    first = [x.id for x in _keyed_order(a, batch(IDS))]
    for seed in range(5):
        shuffled = IDS[:]
        random.Random(seed).shuffle(shuffled)
        assert [x.id for x in _keyed_order(a, batch(shuffled))] == first


def test_the_first_slots_are_stable_under_permutation():
    a = auditor()
    cap = 4
    keep = {x.id for x in _keyed_order(a, batch(IDS))[:cap]}
    for seed in range(5):
        shuffled = IDS[:]
        random.Random(seed).shuffle(shuffled)
        assert {x.id for x in _keyed_order(a, batch(shuffled))[:cap]} == keep


def test_two_validators_order_differently():
    one = [x.id for x in _keyed_order(auditor(b"a" * 32), batch(IDS))]
    two = [x.id for x in _keyed_order(auditor(b"b" * 32), batch(IDS))]
    assert one != two


def test_an_unreadable_article_does_not_break_the_pass():
    a = auditor()
    rows = batch(IDS[:3]) + [types.SimpleNamespace(id="not-an-id")]
    assert len(_keyed_order(a, rows)) == 4


def test_an_auditor_without_a_key_orders_nothing_away():
    broken = types.SimpleNamespace(order_key=lambda aid: 1 / 0)
    rows = batch(IDS[:3])
    assert [x.id for x in _keyed_order(broken, rows)] == IDS[:3]


def test_the_order_key_is_stable_and_bounded():
    a = auditor()
    for aid in IDS:
        value = a.order_key(aid)
        assert 0.0 <= value < 1.0
        assert value == a.order_key(aid)
