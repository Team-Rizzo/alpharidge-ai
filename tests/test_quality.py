import pytest

from alpharidge_ai.mechanism import quality


def test_a_run_of_junk_takes_the_weight_and_passes_bring_it_back():
    q = quality.scores([("ck", True)] * 5 + [("ck", False)] * 100)
    assert q["ck"] < quality.ZERO and quality.multiplier(q["ck"]) == 0.0
    back = quality.scores([("ck", True)] * 5 + [("ck", False)] * 100 + [("ck", True)] * 40)
    assert quality.multiplier(back["ck"]) == pytest.approx(1.0, abs=0.15)


def test_half_life_and_the_bands():
    q = quality.scores([("ck", False)] * quality.HALF_LIFE)
    assert q["ck"] == pytest.approx(quality.START / 2)
    assert quality.multiplier(0.2) == pytest.approx(0.5)
    assert quality.pay_multiplier(0.5) == pytest.approx(0.5)
    assert quality.pay_multiplier(0.9) == 1.0 and quality.pay_multiplier(0.1) == 0.0


def test_each_paid_claim_is_locked_at_the_q_before_it():
    outcomes = [("ck", True, i) for i in range(300)] + [("ck", False, 1000 + i) for i in range(200)]
    outcomes += [("ck", True, 2000), ("new", True, 3000)]
    locks = quality.locked(outcomes)
    q = quality.START
    for i in range(10):
        assert locks[str(i)] == pytest.approx(quality.pay_multiplier(q))
        q = quality.DECAY * q + (1 - quality.DECAY)
    # A clean run climbs to full pay; the junk wave after it leaves those claims as they were.
    assert locks["299"] == 1.0 and locks["2000"] == 0.0
    assert locks["3000"] == pytest.approx(quality.pay_multiplier(quality.START))
    assert not any(str(1000 + i) in locks for i in range(200))
    assert quality.locked(outcomes[:300]) == {k: v for k, v in locks.items() if int(k) < 300}


def test_a_claims_rows_carry_its_lock_and_other_rows_are_not_scaled():
    leaves = [{"miner_hotkey": "a", "validator_verdict": "provisional", "claim_id": "1", "points_awarded": 4.0},
              {"miner_hotkey": "a", "validator_verdict": "settle", "claim_id": "1", "points_awarded": 6.0},
              {"miner_hotkey": "b", "validator_verdict": "provisional", "claim_id": "2", "points_awarded": 2.0},
              {"miner_hotkey": "b", "validator_verdict": "clawback", "claim_id": "2", "points_awarded": -2.0},
              {"miner_hotkey": "c", "validator_verdict": "provisional", "claim_id": "3", "points_awarded": 1.0},
              {"miner_hotkey": "c", "validator_verdict": "debt", "claim_id": "4", "points_awarded": -0.5},
              {"miner_hotkey": "d", "validator_verdict": "carry", "claim_id": None, "points_awarded": -1.0}]
    out = quality.scaled(leaves, {"1": 0.5, "2": 0.25})
    assert out == {"a": pytest.approx(5.0), "c": pytest.approx(0.5), "d": pytest.approx(-1.0)}


def test_the_start_is_off_until_set(monkeypatch):
    from datetime import datetime, timezone
    now = datetime(2026, 12, 1, tzinfo=timezone.utc)
    assert not quality.active(now) and quality.from_ms() == 0
    monkeypatch.setattr(quality, "QUALITY_FROM", now)
    assert quality.active(now) and quality.from_ms() == int(now.timestamp() * 1000)
