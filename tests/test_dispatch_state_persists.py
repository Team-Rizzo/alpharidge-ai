"""Per-miner dispatch and cooldown state must survive a process restart.

Dispatch throughput is batch size times concurrent slots, and both are per-miner
state. Escalation counters are carried too, so restarting does not reset them.
"""
import json
from pathlib import Path

import pytest

from alpharidge_ai import config
from alpharidge_ai.utils.cooldown import MinerCooldownTracker


@pytest.fixture
def at(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DISPATCH_STATE_LOCATION", str(tmp_path / "d.json"), raising=False)
    return tmp_path / "d.json"


def _populated():
    c = MinerCooldownTracker(adaptive=True)
    c._batch_size = {"a": 24.0, "b": 18.0}
    c._window = {"a": 3.5, "b": 2.0}
    c._consec_fail = {"a": 2}
    c._inv_level = {"b": 1}
    c._state = {"a": (3, 1, 1.0e9)}
    return c


def test_throughput_state_survives(at):
    _populated().save()
    c = MinerCooldownTracker(adaptive=True); c.load()
    assert c._batch_size == {"a": 24.0, "b": 18.0}
    assert c._window == {"a": 3.5, "b": 2.0}


def test_escalation_state_survives(at):
    """Escalation counters carry across a restart."""
    _populated().save()
    c = MinerCooldownTracker(adaptive=True); c.load()
    assert c._consec_fail == {"a": 2}
    assert c._inv_level == {"b": 1}
    assert c._state["a"] == (3, 1, 1.0e9)


def test_live_leases_do_not_survive(at):
    """Live leases must start empty."""
    c = _populated(); c._inflight = {"a": 3}; c.save()
    d = MinerCooldownTracker(adaptive=True); d.load()
    assert d._inflight == {}


def test_a_cold_start_is_not_an_error(at):
    c = MinerCooldownTracker(adaptive=True)
    c.load()                      # no file
    assert c._batch_size == {}


def test_a_corrupt_file_is_not_an_error(at):
    at.write_text("{ this is not json")
    c = MinerCooldownTracker(adaptive=True)
    c.load()
    assert c._batch_size == {}


def test_save_is_atomic(at):
    """Writes go via a temp file."""
    _populated().save()
    assert at.exists()
    assert not at.with_suffix(".tmp").exists()
    json.loads(at.read_text())
