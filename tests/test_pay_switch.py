from datetime import date, datetime, timedelta, timezone

import pytest
import requests

from alpharidge_ai import config
from alpharidge_ai.schema import Claim
from alpharidge_ai.schema import types
from alpharidge_ai.validator import reverify, settings_check

START = datetime(2026, 11, 20, 12, 0, tzinfo=timezone.utc)
START_MS = int(START.timestamp() * 1000)
SWITCH = {"pay_table": "new", "from_ms": START_MS}
BEFORE, AFTER = START - timedelta(minutes=1), START + timedelta(minutes=1)
CLAIM = Claim(
    kind="event", headline="Acme to acquire Beta Systems", event_type="merger_acquisition",
    category="finance", event_date="2026-11-20",
    actors=[{"id_scheme": "other", "id": "a", "role": "acquirer", "name": "Acme Inc."},
            {"id_scheme": "other", "id": "b", "role": "target", "name": "Beta Systems"}],
    facts=[], source_url="https://news.example.org/acme-beta",
    quote="Acme Inc. agreed on Tuesday to acquire Beta Systems, the companies said.")


@pytest.fixture(autouse=True)
def _no_switch(monkeypatch):
    monkeypatch.setattr(types, "_switch", None)
    for k in ("EL_PAY_TABLE", "EL_PAY_TABLE_FROM_MS", "EL_EVENT_PAY_X", "EL_SCOOP_PREMIUM",
              "EL_SCOOP_MISS_DEBT_X"):
        monkeypatch.delenv(k, raising=False)


def _item(received):
    """A paid event claim's re-verify item as the API prices it: at the claim's receipt."""
    base = types.event_price("high", received)
    return {"route": "event", "points": 0.4 * base, "factors": {"base": base, "T": 1.0, "A": 1.0},
            "judge": {}, "published_at": None}


def test_until_a_switch_is_served_every_claim_prices_at_todays_table():
    assert types.pay_table() == "today"
    assert types.event_price("high", AFTER) == 12
    assert types.scoop_multiple(AFTER + timedelta(hours=1), AFTER) == 5
    assert types.scoop_debt("merger_acquisition", date(2026, 11, 20), AFTER) == -24


def test_a_claim_prices_by_the_table_at_its_receipt():
    types.set_pay_switch(SWITCH)
    assert types.event_price("high", BEFORE) == 12 and types.event_price("high", AFTER) == 3
    assert types.scoop_multiple(BEFORE + timedelta(hours=1), BEFORE) == 5
    assert types.scoop_multiple(AFTER + timedelta(hours=1), AFTER) == 10
    assert types.scoop_debt("merger_acquisition", date(2026, 11, 20), BEFORE) == -24
    assert types.scoop_debt("merger_acquisition", date(2026, 11, 20), AFTER) == -12
    assert types.pay_settings(AFTER)["SCOOP_PREMIUM"] == 10


def test_an_epoch_that_straddles_the_start_is_priced_alike_by_the_api_and_a_validator():
    types.set_pay_switch(SWITCH)
    items = [(_item(BEFORE), BEFORE), (_item(AFTER), AFTER)]
    assert [i["factors"]["base"] for i, _ in items] == [12, 3]
    assert all(reverify.paid_right(i, CLAIM, r) for i, r in items)
    # A validator that has not read the switch yet prices the later claim at today's table;
    # hence the start is set well ahead of the hourly config read.
    types.set_pay_switch(None)
    assert reverify.paid_right(items[0][0], CLAIM, BEFORE)
    assert not reverify.paid_right(items[1][0], CLAIM, AFTER)


@pytest.mark.parametrize("bad", [
    {"pay_table": "double", "from_ms": START_MS}, {"pay_table": "new"},
    {"pay_table": "new", "from_ms": "soon"}, {"pay_table": "new", "from_ms": -1},
    {"pay_table": "new", "from_ms": True}, {"pay_table": "new", "from_ms": 1.5}, "new"])
def test_a_bad_switch_is_refused_and_the_last_one_stays(bad):
    types.set_pay_switch(SWITCH)
    with pytest.raises(types.PaySettingError):
        types.set_pay_switch(bad)
    assert types.pay_switch() == ("new", START_MS)


def test_a_local_setting_wins(monkeypatch):
    types.set_pay_switch(SWITCH)
    monkeypatch.setenv("EL_SCOOP_PREMIUM", "7")
    assert types.scoop_multiple(AFTER + timedelta(hours=1), AFTER) == 7
    assert types.event_price("high", AFTER) == 3
    monkeypatch.setenv("EL_PAY_TABLE", "today")
    monkeypatch.setenv("EL_PAY_TABLE_FROM_MS", str(START_MS))
    assert types.event_price("high", AFTER) == 12


def _serve(monkeypatch, payload=None, fail=False):
    class _Response:
        def raise_for_status(self):
            if fail:
                raise requests.HTTPError("503")

        def json(self):
            return payload

    monkeypatch.setattr(requests, "get", lambda *a, **kw: _Response())
    monkeypatch.setattr(config, "MINER_API_URL", "http://test.invalid")
    monkeypatch.setattr(config, "_build_auth_headers", lambda *a: {})
    monkeypatch.setattr(config, "_remote_config_last_fetch", 0.0)
    monkeypatch.setattr(settings_check, "_differs", None)


def test_a_config_read_applies_the_served_switch_and_a_failed_read_keeps_it(monkeypatch):
    _serve(monkeypatch, {"config": {}, "pay_switch": SWITCH})
    config.refresh_remote_config(force=True)
    assert types.pay_switch() == ("new", START_MS)
    _serve(monkeypatch, fail=True)
    config.refresh_remote_config(force=True)
    assert types.pay_switch() == ("new", START_MS)
    _serve(monkeypatch, {"config": {}, "pay_switch": {"pay_table": "x", "from_ms": 1}})
    config.refresh_remote_config(force=True)
    assert types.pay_switch() == ("new", START_MS)
    _serve(monkeypatch, {"config": {}})
    config.refresh_remote_config(force=True)
    assert types.pay_switch() is None
