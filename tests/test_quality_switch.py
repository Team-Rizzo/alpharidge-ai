"""The quality start: off by default, switched on by the start the API serves, a local
EL_QUALITY_FROM_MS winning over it."""
from datetime import datetime, timedelta, timezone

import pytest

from alpharidge_ai.mechanism import quality
from alpharidge_ai.validator.settings_check import apply_quality_switch

T = 1_795_100_000_000


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.setattr(quality, "_served_ms", None)
    monkeypatch.setattr(quality, "QUALITY_FROM", None)
    monkeypatch.delenv("EL_QUALITY_FROM_MS", raising=False)


def test_off_until_a_start_is_served():
    assert quality.start_ms() is None
    assert quality.from_ms() == 0
    assert not quality.active(T + 1)
    assert not quality.may_be_active(datetime.now(timezone.utc))


def test_a_served_start_turns_quality_on_from_it():
    assert apply_quality_switch({"from_ms": T}) == T
    assert quality.from_ms() == T
    assert not quality.active(T - 1)
    assert quality.active(T)
    start = datetime.fromtimestamp(T / 1000, tz=timezone.utc)
    assert not quality.may_be_active(start - quality.ASK_BEFORE - timedelta(seconds=1))
    assert quality.may_be_active(start - quality.ASK_BEFORE)


def test_a_bad_start_is_refused_and_the_last_one_stays():
    apply_quality_switch({"from_ms": T})
    for bad in ({"from_ms": -5}, {"from_ms": "soon"}, {"from_ms": 1.5}, {"from_ms": True}, [T]):
        assert apply_quality_switch(bad) == T
    assert quality.start_ms() == T


def test_serving_none_clears_it():
    apply_quality_switch({"from_ms": T})
    assert apply_quality_switch(None) is None
    assert not quality.active(T + 1)


def test_a_local_start_wins_over_the_served_one(monkeypatch):
    apply_quality_switch({"from_ms": T})
    monkeypatch.setenv("EL_QUALITY_FROM_MS", str(T + 5))
    assert quality.start_ms() == T + 5
    monkeypatch.setenv("EL_QUALITY_FROM_MS", "0")
    assert quality.start_ms() == T


def test_the_code_start_applies_when_nothing_is_served(monkeypatch):
    start = datetime.fromtimestamp(T / 1000, tz=timezone.utc)
    monkeypatch.setattr(quality, "QUALITY_FROM", start)
    assert quality.start_ms() == T
    apply_quality_switch({"from_ms": T + 7})
    assert quality.start_ms() == T + 7
