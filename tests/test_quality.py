import pytest

from alpharidge_ai.mechanism import quality


def test_a_run_of_junk_takes_the_weight_and_passes_bring_it_back():
    q = quality.scores([("ck", True)] * 5 + [("ck", False)] * 100)
    assert q["ck"] < quality.ZERO and quality.multiplier(q["ck"]) == 0.0
    back = quality.scores([("ck", True)] * 5 + [("ck", False)] * 100 + [("ck", True)] * 40)
    assert quality.multiplier(back["ck"]) == pytest.approx(1.0, abs=0.15)


def test_a_steady_honest_miner_keeps_full_weight():
    q = quality.scores([("ck", i % 3 == 0) for i in range(600)])
    assert quality.multiplier(q["ck"]) == 1.0


def test_half_life_and_the_linear_band():
    q = quality.scores([("ck", False)] * quality.HALF_LIFE)
    assert q["ck"] == pytest.approx(quality.START / 2)
    assert quality.multiplier(0.2) == pytest.approx(0.5)
    assert quality.multiplier(0.9) == 1.0 and quality.multiplier(0.1) == 0.0


def test_points_scale_by_the_coldkey_and_an_unscored_one_is_full():
    out = quality.scaled({"a": 4.0, "b": 2.0, "c": 1.0}, {"a": "ck", "b": "ck"},
                         {"ck": 0.225})
    assert out == {"a": pytest.approx(3.0), "b": pytest.approx(1.5), "c": 1.0}


def test_the_start_is_off_until_set(monkeypatch):
    from datetime import datetime, timezone
    now = datetime(2026, 12, 1, tzinfo=timezone.utc)
    assert not quality.active(now) and quality.from_ms() == 0
    monkeypatch.setattr(quality, "QUALITY_FROM", now)
    assert quality.active(now) and quality.from_ms() == int(now.timestamp() * 1000)
