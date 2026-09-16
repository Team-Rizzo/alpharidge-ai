"""Anything measured from the reputation store must be measured over the paid field.

The store keeps every hotkey it has ever seen and never dropped one, so a median taken
over it describes a population several times the size of the field being paid.
"""
import pytest

from alpharidge_ai.validator import emission_params as ep
from alpharidge_ai.validator.reputation_store import ReputationStore


SNAP = {
    "live_a": {"r": 0.90, "n": 500},
    "live_b": {"r": 0.80, "n": 500},
    "live_c": {"r": 0.70, "n": 500},
    "gone_x": {"r": 0.10, "n": 500},   # deregistered, frozen low
    "gone_y": {"r": 0.12, "n": 500},
    "gone_z": {"r": 0.14, "n": 500},
}
REGISTERED = ["live_a", "live_b", "live_c"]


def test_the_median_is_taken_over_registered_hotkeys_only():
    assert ep.live_median(SNAP, 100, REGISTERED) == pytest.approx(0.80)


def test_departed_hotkeys_would_otherwise_move_it():
    """Six entries against three registered miners moves the median by 0.39."""
    everything = [k for k in SNAP]
    assert ep.live_median(SNAP, 100, everything) == pytest.approx(0.41, abs=0.02)


def test_under_observed_hotkeys_are_still_excluded():
    snap = dict(SNAP, live_d={"r": 0.99, "n": 3})
    assert ep.live_median(snap, 100, REGISTERED + ["live_d"]) == pytest.approx(0.80)


def test_an_empty_registered_set_yields_nothing_rather_than_the_store():
    """A metagraph read that failed must not silently fall back to every hotkey."""
    assert ep.live_median(SNAP, 100, []) is None


# ---- pruning ----------------------------------------------------------------------

def _store(tmp_path, state):
    s = ReputationStore(path=tmp_path / "rep.json")
    s.state = dict(state)
    return s


def test_a_departed_hotkey_is_dropped_once_it_has_gone_quiet(tmp_path):
    s = _store(tmp_path, {"live_a": {"r": 0.9, "n": 5, "e": 1000},
                          "gone_x": {"r": 0.1, "n": 5, "e": 100}})
    assert s.prune_unregistered(["live_a"], epoch=5000, grace_epochs=100) == 1
    assert set(s.state) == {"live_a"}


def test_a_departed_hotkey_keeps_its_record_during_the_grace_period(tmp_path):
    """A record that does not survive a brief absence is not a record."""
    s = _store(tmp_path, {"live_a": {"r": 0.9, "n": 5, "e": 1000},
                          "gone_x": {"r": 0.1, "n": 5, "e": 1000}})
    assert s.prune_unregistered(["live_a"], epoch=1050, grace_epochs=100) == 0
    assert set(s.state) == {"live_a", "gone_x"}


def test_a_registered_hotkey_is_never_dropped_however_quiet(tmp_path):
    s = _store(tmp_path, {"live_a": {"r": 0.9, "n": 5, "e": 1}})
    assert s.prune_unregistered(["live_a"], epoch=99999, grace_epochs=1) == 0
    assert set(s.state) == {"live_a"}


def test_an_unreadable_metagraph_prunes_nothing(tmp_path):
    """An empty registered set means the read failed, not that every miner left."""
    s = _store(tmp_path, {"live_a": {"r": 0.9, "n": 5, "e": 1}})
    assert s.prune_unregistered([], epoch=99999, grace_epochs=1) == 0
    assert set(s.state) == {"live_a"}


def test_state_with_no_epoch_stamp_is_treated_as_current(tmp_path):
    """Entries written before the stamp existed must not all vanish on the first pass."""
    s = _store(tmp_path, {"old": {"r": 0.5, "n": 5}})
    assert s.prune_unregistered(["someone_else"], epoch=99999, grace_epochs=1) == 0
    assert set(s.state) == {"old"}
