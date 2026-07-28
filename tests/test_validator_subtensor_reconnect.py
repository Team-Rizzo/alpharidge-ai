"""Unit tests for the main-loop reconnect that clears a wedged substrate websocket.

A half-finished recv leaves the connection raising ConcurrencyError on every chain
call. Because `self.block` is the first statement in the main loop's try block, that
stops the loop before concurrent_forward() and dispatch goes silent until the socket
is replaced -- retrying it cannot help.
"""

import types

import pytest

from alpharidge_ai.base.validator import BaseValidatorNeuron


class FakeSubtensor:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class ConcreteValidator(BaseValidatorNeuron):
    """BaseValidatorNeuron is abstract; run() is what's under test here."""

    async def forward(self, synapse=None):
        return synapse


def make_validator(monkeypatch, fail_times, reconnect_raises=False):
    """A stand-in exercising only run()'s error handling, with the chain calls faked.

    `fail_times` is how many consecutive iterations raise before the loop succeeds;
    the loop is stopped on the first success via should_exit.
    """
    v = ConcreteValidator.__new__(ConcreteValidator)
    v.step = 0
    v.should_exit = False
    v.config = types.SimpleNamespace(
        mock=False,
        neuron=types.SimpleNamespace(axon_off=True),
        netuid=45,
        subtensor=types.SimpleNamespace(chain_endpoint="ws://test"),
    )
    v.axon = None
    v.subtensor = FakeSubtensor()
    v.metagraph = types.SimpleNamespace(sync=lambda subtensor=None: None)

    state = {"calls": 0, "rebuilds": 0, "forwards": 0}

    # `block` is a property on the class, so patch it there rather than on the instance.
    # run() reads it once before entering the loop; only calls after that are iterations.
    def fake_block(self):
        state["calls"] += 1
        if 1 < state["calls"] <= fail_times + 1:
            raise Exception("ConcurrencyError: cannot call recv while another thread")
        return 8721521

    monkeypatch.setattr(type(v), "block", property(fake_block), raising=False)
    monkeypatch.setattr(v, "sync", lambda: None, raising=False)

    def fake_forward():
        state["forwards"] += 1
        v.should_exit = True  # stop the loop after one clean pass

    v.loop = types.SimpleNamespace(run_until_complete=lambda coro: None)
    v.concurrent_forward = fake_forward
    # concurrent_forward() is called through run_until_complete; invoke it directly.
    v.loop.run_until_complete = lambda coro: coro

    def fake_subtensor_ctor(config=None):
        state["rebuilds"] += 1
        if reconnect_raises:
            raise RuntimeError("endpoint unreachable")
        return FakeSubtensor()

    monkeypatch.setattr("bittensor.Subtensor", fake_subtensor_ctor)
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setattr(
        "alpharidge_ai.base.validator.time", types.SimpleNamespace(sleep=lambda s: None)
    )

    v._verify_axon_reachable = lambda: None
    return v, state


def test_no_reconnect_below_threshold(monkeypatch):
    """A short blip is transient; the connection must not be torn down for it."""
    v, state = make_validator(monkeypatch, fail_times=BaseValidatorNeuron.SUBTENSOR_RECONNECT_AFTER - 1)
    v.run()
    assert state["rebuilds"] == 0
    assert state["forwards"] == 1


def test_reconnects_once_streak_hits_threshold(monkeypatch):
    """A sustained streak means the socket is wedged -- rebuild it."""
    v, state = make_validator(monkeypatch, fail_times=BaseValidatorNeuron.SUBTENSOR_RECONNECT_AFTER)
    old = v.subtensor
    v.run()
    assert state["rebuilds"] == 1
    assert old.closed, "the wedged connection should be closed once replaced"
    assert v.subtensor is not old


def test_reconnect_retried_on_each_further_streak(monkeypatch):
    """An endpoint that is genuinely down keeps being retried, not given up on."""
    n = BaseValidatorNeuron.SUBTENSOR_RECONNECT_AFTER
    v, state = make_validator(monkeypatch, fail_times=3 * n)
    v.run()
    assert state["rebuilds"] == 3


def test_failed_rebuild_keeps_previous_connection(monkeypatch):
    """If the replacement cannot be built, don't close what we still have."""
    v, state = make_validator(
        monkeypatch, fail_times=BaseValidatorNeuron.SUBTENSOR_RECONNECT_AFTER, reconnect_raises=True
    )
    old = v.subtensor
    v.run()
    assert state["rebuilds"] == 1
    assert not old.closed
    assert v.subtensor is old


def test_streak_resets_after_a_clean_pass(monkeypatch):
    """Failures separated by successful iterations must not accumulate into a rebuild."""
    v, state = make_validator(monkeypatch, fail_times=0)
    n = BaseValidatorNeuron.SUBTENSOR_RECONNECT_AFTER

    calls = {"i": 0}

    # Discounting run()'s pre-loop read, fail n-1 times, succeed, then fail n-1 more:
    # never a streak of n, so no rebuild should ever be triggered.
    def flaky_block(self):
        calls["i"] += 1
        i = calls["i"] - 1  # iteration index; 0 is the pre-loop read
        if 1 <= i < n or n < i < 2 * n:
            raise Exception("ConcurrencyError: cannot call recv")
        return 8721521

    monkeypatch.setattr(type(v), "block", property(flaky_block), raising=False)

    seen = {"forwards": 0}

    def fake_forward():
        seen["forwards"] += 1
        if seen["forwards"] == 2:
            v.should_exit = True

    v.concurrent_forward = fake_forward
    v.run()
    assert state["rebuilds"] == 0
