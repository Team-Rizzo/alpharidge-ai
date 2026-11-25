import bittensor as bt
import requests

from alpharidge_ai import config
from alpharidge_ai.validator import settings_check

TODAY = {"EVENT_PAY_X": 1.0, "SCOOP_PREMIUM": 5.0, "SCOOP_MISS_DEBT_X": 2.0,
         "SCOOP_DEBT_TRUST": 1.0}


def test_the_own_pay_line_names_every_value():
    assert settings_check.pay_line(TODAY) == (
        "[CONFIG] pay settings: EVENT_PAY_X=1 SCOOP_PREMIUM=5 SCOOP_MISS_DEBT_X=2 "
        "SCOOP_DEBT_TRUST=1 (the API and every validator must use the same values)")


def test_matching_pay_raises_no_alarm():
    assert settings_check.pay_alarm(dict(TODAY), TODAY) is None
    # JSON may carry whole numbers as ints; an older API may not serve the debt trust.
    assert settings_check.pay_alarm({"EVENT_PAY_X": 1, "SCOOP_PREMIUM": 5,
                                     "SCOOP_MISS_DEBT_X": 2}, TODAY) is None


def test_differing_pay_raises_an_alarm_naming_each_value():
    alarm = settings_check.pay_alarm({**TODAY, "EVENT_PAY_X": 0.25, "SCOOP_PREMIUM": "x"},
                                     TODAY)
    assert alarm.startswith("[CONFIG] ALARM pay settings differ from the API's: ")
    assert "EVENT_PAY_X here 1, API 0.25" in alarm and "SCOOP_PREMIUM here 5, API x" in alarm
    assert "SCOOP_MISS_DEBT_X here" not in alarm
    # Pay settings never reach the epoch's points: what they break is re-verify's price
    # check and the scoop debt check.
    assert "Every epoch" not in alarm and "points_mismatch" not in alarm
    assert "re-verify" in alarm and "unreachable" in alarm and "unbound_debt" in alarm


def _serve(monkeypatch, payload):
    class _Response:
        def raise_for_status(self):
            pass

        def json(self):
            return payload

    monkeypatch.setattr(requests, "get", lambda *a, **kw: _Response())
    monkeypatch.setattr(config, "MINER_API_URL", "http://test.invalid")
    monkeypatch.setattr(config, "_build_auth_headers", lambda *a: {})
    monkeypatch.setattr(config, "_remote_config_last_fetch", 0.0)
    logged = []
    for level in ("error", "warning", "info"):
        monkeypatch.setattr(bt.logging, level,
                            lambda msg, *a, _level=level, **kw: logged.append((_level, msg)))
    return logged


def test_each_config_read_compares_the_served_pay(monkeypatch):
    monkeypatch.delenv("EL_EVENT_PAY_X", raising=False)
    monkeypatch.setattr(settings_check, "_differs", None)
    logged = _serve(monkeypatch, {"config": {}, "pay": {**TODAY, "EVENT_PAY_X": 0.25}})
    config.refresh_remote_config(force=True)
    assert [m for lvl, m in logged if lvl == "error" and "[CONFIG] ALARM pay settings" in m]
    assert settings_check.pay_differs()

    logged = _serve(monkeypatch, {"config": {}, "pay": dict(TODAY)})
    config.refresh_remote_config(force=True)
    assert not [m for _, m in logged if "ALARM" in m]
    assert settings_check.pay_differs() is None


def test_an_api_that_serves_no_pay_is_warned_about_not_alarmed(monkeypatch):
    monkeypatch.setattr(settings_check, "_differs", "old alarm")
    logged = _serve(monkeypatch, {"config": {}})
    config.refresh_remote_config(force=True)
    assert not [m for _, m in logged if "ALARM" in m]
    assert [m for lvl, m in logged if lvl == "warning" and "serves no pay settings" in m]
    # Nothing to compare: re-verify checks prices as usual.
    assert settings_check.pay_differs() is None
