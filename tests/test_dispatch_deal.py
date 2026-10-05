"""Dealing a tick's articles across its batches."""
import alpharidge_ai.config as config
from alpharidge_ai.utils.dispatch import deal


def test_equal_sizes_deal_round_robin():
    assert deal(list(range(6)), [3, 3]) == [[0, 2, 4], [1, 3, 5]]


def test_sends_the_same_items_as_a_contiguous_cut():
    items = list(range(60))
    sizes = [24, 26, 7]
    out = deal(items, sizes)
    assert [len(b) for b in out] == sizes
    assert sorted(x for b in out for x in b) == items[:sum(sizes)]


def test_a_full_batch_drops_out_and_the_rest_keep_dealing():
    assert deal(list(range(7)), [2, 5]) == [[0, 2], [1, 3, 4, 5, 6]]


def test_more_capacity_than_items_leaves_batches_short():
    out = deal(list(range(5)), [4, 4])
    assert out == [[0, 2, 4], [1, 3]]


def test_no_items_or_no_sizes():
    assert deal([], [3, 3]) == [[], []]
    assert deal([1, 2], []) == []
    assert deal([1, 2], [0, 2]) == [[], [1, 2]]


def test_switch_off_cuts_contiguous_runs(monkeypatch):
    from neurons.validator import Validator
    monkeypatch.setattr(config, "DISPATCH_DEAL", False, raising=False)
    assert Validator._cut_batches(list(range(7)), [3, 4]) == [[0, 1, 2], [3, 4, 5, 6]]
    monkeypatch.setattr(config, "DISPATCH_DEAL", True, raising=False)
    assert Validator._cut_batches(list(range(7)), [3, 4]) == [[0, 2, 4], [1, 3, 5, 6]]


def test_served_config_can_switch_it():
    assert "DISPATCH_DEAL" in config._REMOTE_CONFIG_KEYS
