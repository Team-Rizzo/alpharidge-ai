"""A failed quality read is tried again within the epoch, as a failed pull is; a carry that
does not match reads the quality answer again."""

import asyncio
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from tests.test_validator_loop import LEAVES, api, loop, signed_points

START = datetime(2026, 12, 1, tzinfo=timezone.utc)
T = int(START.timestamp() * 1000)


@pytest.fixture
def on(monkeypatch):
    from alpharidge_ai.mechanism import quality
    from alpharidge_ai.validator import loop as loop_module
    monkeypatch.setattr(quality, "QUALITY_FROM", START)
    monkeypatch.setattr(loop_module, "_now", lambda: START + timedelta(hours=1))
    errors = []
    monkeypatch.setattr(loop_module.bt.logging, "error", errors.append)
    return errors


def test_a_failed_quality_read_is_tried_again_on_the_next_tick(on):
    calls = []

    def q(epoch):
        calls.append(epoch)
        if len(calls) == 1:
            return httpx.Response(503, json={"detail": "busy"})
        return {"owners": ["hk1"], "until_ms": T + 10, "outcomes": [[0, 1, "1"]]}

    lp = loop(api(signed_points(), quality=q))
    assert asyncio.run(lp.tick()) is None
    assert [e for e in on if "ALARM" in e and "quality unavailable" in e]
    weights = asyncio.run(lp.tick())
    assert weights is not None and calls == [7, 7] and lp.accepted_epoch == 7


def test_a_failed_carry_quality_read_is_tried_again_on_the_next_tick(on, monkeypatch):
    # Epoch 4 leaves the window at 7, so the carry into 7 asks for epoch 6's locks.
    debt = {"resource_type": "ledger", "resource_id": "9", "miner_hotkey": "hk1",
            "validator_verdict": "debt", "categorical_key": "finance|scoop",
            "points_awarded": -1.0, "claim_id": "9"}
    paid = {**debt, "resource_id": "5", "validator_verdict": "provisional",
            "categorical_key": "finance|event", "points_awarded": 5.0, "claim_id": "5"}
    monkeypatch.setitem(LEAVES, 4, [debt])
    monkeypatch.setitem(LEAVES, 5, [paid])
    calls = []

    def q(epoch):
        calls.append(epoch)
        if epoch == 6 and calls.count(6) == 1:
            return httpx.Response(503, json={"detail": "busy"})
        return {"owners": ["hk1"], "until_ms": T + epoch, "outcomes": [[0, 1, "5"]]}

    lp = loop(api(lambda e: signed_points(epoch=e), quality=q))
    assert asyncio.run(lp.tick()) is None
    assert [e for e in on if "ALARM" in e and "quality" in e]
    assert asyncio.run(lp.tick()) is not None
    assert calls.count(6) == 2 and lp.accepted_epoch == 7


def test_a_carry_mismatch_drops_the_cached_quality_answer(on, monkeypatch):
    """The carry into 7 is valued with epoch 6's locks. A first answer that leaves the debt
    uncarried is read again when the pull retries."""
    debt = {"resource_type": "ledger", "resource_id": "9", "miner_hotkey": "hk1",
            "validator_verdict": "debt", "categorical_key": "finance|scoop",
            "points_awarded": -10.0, "claim_id": "9"}
    paid = {**debt, "resource_id": "5", "validator_verdict": "provisional",
            "categorical_key": "finance|event", "points_awarded": 20.0, "claim_id": "5"}
    carry = {"resource_type": "ledger", "resource_id": "10", "miner_hotkey": "hk1",
             "validator_verdict": "carry", "categorical_key": "finance|carry",
             "points_awarded": -10.0, "claim_id": None}
    monkeypatch.setitem(LEAVES, 4, [debt])
    monkeypatch.setitem(LEAVES, 5, [paid])
    monkeypatch.setitem(LEAVES, 7, LEAVES[7] + [carry])
    junk = [[0, 0, 100 + i] for i in range(120)]
    calls = []

    def q(epoch):
        calls.append(epoch)
        if epoch == 6 and calls.count(6) == 1:
            # A stale answer: claim 5 at full pay, so nothing would carry.
            return {"owners": ["hk1"], "until_ms": T + epoch, "outcomes": [[0, 1, "5"]]}
        return {"owners": ["hk1"], "until_ms": T + epoch, "outcomes": junk + [[0, 1, "5"]]}

    lp = loop(api(lambda e: signed_points(epoch=e), quality=q))
    weights = asyncio.run(lp.tick())
    assert calls.count(6) == 2
    assert weights is not None and weights.tolist() == pytest.approx([0.0, 0.0, 0.1, 0.9])
