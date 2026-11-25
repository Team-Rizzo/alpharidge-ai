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
    assert quality.pay_multiplier(0.35) == pytest.approx(0.5)
    assert quality.pay_multiplier(0.6) == 1.0 and quality.pay_multiplier(0.1) == 0.0


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


def test_the_start_is_off_until_set_and_goes_by_the_epochs_signing(monkeypatch):
    from datetime import datetime, timedelta, timezone
    now = datetime(2026, 12, 1, tzinfo=timezone.utc)
    assert not quality.may_be_active(now) and not quality.active(10 ** 13)
    assert quality.from_ms() == 0
    monkeypatch.setattr(quality, "QUALITY_FROM", now)
    start = int(now.timestamp() * 1000)
    assert quality.from_ms() == start
    # The epoch's signing time decides, not the clock that asks.
    assert quality.active(start) and not quality.active(start - 1) and not quality.active(None)
    # Well before the start an epoch cannot have been signed from it: no need to ask.
    assert not quality.may_be_active(now - quality.ASK_BEFORE - timedelta(seconds=1))
    assert quality.may_be_active(now - quality.ASK_BEFORE) and quality.may_be_active(now)


def test_paid_claims_received_from_the_start_without_a_lock_are_counted(monkeypatch):
    from datetime import datetime, timezone
    start = datetime(2026, 12, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(quality, "QUALITY_FROM", start)
    t = quality.from_ms()

    def leaf(cid, kind, received_ms, pts=1.0):
        return {"miner_hotkey": "a", "validator_verdict": kind, "claim_id": cid,
                "points_awarded": pts, "received_ms": received_ms}

    leaves = [leaf("1", "provisional", t - 1), leaf("1", "settle", t - 1),
              leaf("2", "provisional", t), leaf("2", "settle", t),
              leaf("3", "provisional", t + 5), leaf("4", "provisional", t + 9),
              leaf("5", "debt", t + 9, -1.0),
              {"miner_hotkey": "a", "validator_verdict": "provisional", "claim_id": "6",
               "points_awarded": 1.0}]
    # Claim 1 was received before the start; 3 has its lock; 5 is debt; 6 has no receipt.
    assert quality.unlocked(leaves, {"3": 0.5}) == 2
    assert quality.scaled(leaves, {"3": 0.5}) == {"a": pytest.approx(5.5)}


def _row(epoch, kind, pts, cid=None, hk="a"):
    return {"epoch": epoch, "miner_hotkey": hk, "validator_verdict": kind, "claim_id": cid,
            "categorical_key": "finance|scoop", "points_awarded": pts, "coldkey": "ck",
            "leaf_version": 2, "received_ms": epoch}


def _carry(rows, epoch, window, worth=None):
    """The carry leaves into `epoch`, as the API writes them."""
    from alpharidge_ai.mechanism.pools import carried
    prev = [x for x in rows if epoch - window <= x["epoch"] < epoch]
    leaving = [x for x in rows if x["epoch"] == epoch - window]
    return [{**c, "epoch": epoch, "validator_verdict": "carry", "claim_id": None,
             "categorical_key": f"{c['category']}|carry"}
            for c in (carried(prev, leaving, worth) if worth else carried(prev, leaving))]


def _scaled_window(rows, epoch, window, locks):
    from alpharidge_ai.mechanism.pools import net_by_coldkey
    inside = [x for x in rows if epoch - window < x["epoch"] <= epoch]
    return sum(quality.scaled(inside, locks).values()), net_by_coldkey(
        quality.scaled(inside, locks), {"a": "ck"})


def test_raw_carry_forgives_debt_that_quality_left_unpaid():
    # A junk-heavy coldkey misses a scoop (-10) and is paid 20 at lock 0 the next epoch; once
    # the debt epoch leaves the 3-epoch window, an honest claim pays 5 at lock 1.
    locks = {"1": 0.0, "2": 1.0}
    rows = [_row(0, "debt", -10.0, "9"), _row(1, "provisional", 20.0, "1")]
    assert _scaled_window(rows, 2, 3, locks) == (-10.0, {})
    raw = rows + _carry(rows, 3, 3) + [_row(3, "provisional", 5.0, "2")]
    # Raw, the coldkey's window is +10, so nothing carries and the debt is gone.
    assert _carry(rows, 3, 3) == []
    assert _scaled_window(raw, 3, 3, locks) == (5.0, {"a": 5.0})
    # Valued as the weights value it, the debt carries and the honest pay goes to it.
    worth = lambda leaf: quality.worth(leaf, locks)
    fixed = rows + _carry(rows, 3, 3, worth) + [_row(3, "provisional", 5.0, "2")]
    assert [c["points_awarded"] for c in _carry(rows, 3, 3, worth)] == [-10.0]
    assert _scaled_window(fixed, 3, 3, locks) == (-5.0, {})


def test_raw_carry_charges_debt_the_scaled_window_already_covered():
    # A clawback whose paid row left earlier (lock 0.25), a debt and a full-pay claim.
    locks = {"3": 0.25, "4": 1.0}
    rows = [_row(0, "debt", -1.0, "9"), _row(1, "clawback", -8.0, "3"),
            _row(2, "provisional", 6.0, "4")]
    assert _scaled_window(rows, 2, 3, locks)[0] == pytest.approx(3.0)
    assert [c["points_awarded"] for c in _carry(rows, 3, 3)] == [-1.0]
    worth = lambda leaf: quality.worth(leaf, locks)
    assert _carry(rows, 3, 3, worth) == []
