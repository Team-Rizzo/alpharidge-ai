from datetime import date, datetime, timezone

from alpharidge_ai.schema import EventType
from alpharidge_ai.schema import types
from alpharidge_ai.validator.scan import Scoop, held_claim, second_mark
from alpharidge_ai.verify import floor

RECEIVED = datetime(2026, 10, 13, 12, 0, tzinfo=timezone.utc)
BEFORE, AFTER = datetime(2026, 10, 13, 11, 0, tzinfo=timezone.utc), datetime(
    2026, 10, 13, 13, 0, tzinfo=timezone.utc)


def test_the_premium_needs_no_trusted_page_out_before_the_receipt():
    assert types.scoop_multiple(AFTER, RECEIVED) == types.SCOOP_PREMIUM
    assert types.scoop_multiple(BEFORE, RECEIVED) == types.EVENT_PAY_X
    assert types.scoop_multiple(AFTER, RECEIVED, first_seen=BEFORE) == types.EVENT_PAY_X
    assert types.scoop_multiple(AFTER, RECEIVED, first_seen=AFTER) == types.SCOOP_PREMIUM
    assert types.scoop_multiple(AFTER, RECEIVED, contested=True) == types.EVENT_PAY_X


def test_the_defaults_are_todays_pay_and_the_settings_scale_it(monkeypatch):
    assert (types.EVENT_PAY_X, types.SCOOP_PREMIUM, types.SCOOP_MISS_DEBT_X) == (1, 5, 2)
    assert types.event_price("high") == 12 and types.event_price("none") is None
    debt = types.scoop_debt("merger_acquisition", date(2026, 10, 13), RECEIVED)
    monkeypatch.setattr(types, "EVENT_PAY_X", 0.25)
    monkeypatch.setattr(types, "SCOOP_PREMIUM", 10.0)
    monkeypatch.setattr(types, "SCOOP_MISS_DEBT_X", 1.0)
    assert types.event_price("high") == 3
    assert types.scoop_multiple(BEFORE, RECEIVED) == 0.25
    assert types.scoop_multiple(AFTER, RECEIVED) == 10
    assert types.scoop_debt("merger_acquisition", date(2026, 10, 13), RECEIVED) == debt / 2


def _scoop(**kw):
    base = dict(id=1, event_type=EventType("merger_acquisition"), stage=None,
                event_date=date(2026, 10, 13), received_at=RECEIVED,
                subjects=("Beta Systems", "Beta Systems Inc.", "Acme"), ciks=(), facts=())
    base.update(kw)
    return Scoop(**base)


def _mark(scoop, text):
    return second_mark(scoop, "Beta Systems", floor.normalize(text).text, text)


def test_a_found_page_must_show_one_more_mark_than_the_subject():
    s = _scoop()
    assert not _mark(s, "Beta Systems Inc. opened a new office on Tuesday.")
    assert _mark(s, "Acme said Beta Systems would join it.")
    assert _mark(s, "Beta Systems agreed to be bought on Tuesday.") is False
    assert _mark(s, "The acquisition of Beta Systems closed.")
    assert _mark(_scoop(tickers=("BETA",)), "Shares of Beta Systems (BETA) rose.")
    assert not _mark(_scoop(tickers=("ICE",)), "Beta Systems said ice storms hit.")
    assert _mark(_scoop(facts=({"field": "deal_value", "value": 1.2},)),
                 "Beta Systems sold itself for $1.2 billion.")
    assert held_claim(s, "https://wire.example.com/x",
                      "Beta Systems Inc. opened a new office on Tuesday.") is None
