import os
import shutil
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

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


PAY_VARS = ("EL_EVENT_PAY_X", "EL_SCOOP_PREMIUM", "EL_SCOOP_MISS_DEBT_X")


def _no_pay_env():
    return {k: v for k, v in os.environ.items() if not k.startswith("EL_")}


def _clear_pay(monkeypatch):
    for k in PAY_VARS:
        monkeypatch.delenv(k, raising=False)


def test_the_defaults_are_todays_pay():
    out = subprocess.run(
        [sys.executable, "-c", "from alpharidge_ai.schema import types as t; "
         "print(t.EVENT_PAY_X, t.SCOOP_PREMIUM, t.SCOOP_MISS_DEBT_X, t.event_price('high'))"],
        env=_no_pay_env(), capture_output=True, text=True, check=True).stdout.split()
    assert [float(x) for x in out] == [1.0, 5.0, 2.0, 12.0]


def test_the_pay_settings_are_served_whole(monkeypatch):
    _clear_pay(monkeypatch)
    monkeypatch.setenv("EL_EVENT_PAY_X", "0.25")
    assert types.pay_settings() == {"EVENT_PAY_X": 0.25, "SCOOP_PREMIUM": 5.0,
                                    "SCOOP_MISS_DEBT_X": 2.0, "SCOOP_DEBT_TRUST": 1.0}


def test_the_settings_scale_pay(monkeypatch):
    _clear_pay(monkeypatch)
    assert (types.EVENT_PAY_X, types.SCOOP_PREMIUM, types.SCOOP_MISS_DEBT_X) == (1, 5, 2)
    assert types.event_price("high") == 12 and types.event_price("none") is None
    debt = types.scoop_debt("merger_acquisition", date(2026, 10, 13), RECEIVED)
    monkeypatch.setenv("EL_EVENT_PAY_X", "0.25")
    monkeypatch.setenv("EL_SCOOP_PREMIUM", "10")
    monkeypatch.setenv("EL_SCOOP_MISS_DEBT_X", "1")
    assert (types.EVENT_PAY_X, types.SCOOP_PREMIUM, types.SCOOP_MISS_DEBT_X) == (0.25, 10, 1)
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


def _subnet_copy(tmp_path, env_file: str = ""):
    """A copy of the package with a .vali_env beside it, as a validator checkout has."""
    shutil.copytree(Path(types.__file__).resolve().parents[2] / "alpharidge_ai",
                    tmp_path / "alpharidge_ai", ignore=shutil.ignore_patterns("__pycache__"))
    (tmp_path / ".vali_env").write_text(env_file)
    env = _no_pay_env()
    env["PYTHONPATH"] = str(tmp_path)
    # The copy reads its env file as a validator checkout does, whatever the suite sets.
    env.pop("ALPHARIDGE_SKIP_ENV_FILES", None)
    return env


def test_pay_values_in_the_vali_env_file_apply_whichever_module_is_imported_first(tmp_path):
    env = _subnet_copy(tmp_path,
                       "EL_EVENT_PAY_X=0.25\nEL_SCOOP_PREMIUM=10\nEL_SCOOP_MISS_DEBT_X=1\n")
    show = "print(t.EVENT_PAY_X, t.SCOOP_PREMIUM, t.SCOOP_MISS_DEBT_X, t.event_price('high'))"
    config = "from alpharidge_ai import config"
    schema = "from alpharidge_ai.schema import types as t"
    for imports in (f"{config}; {schema}", f"{schema}; {config}"):
        out = subprocess.run([sys.executable, "-c", f"{imports}; {show}"], env=env,
                             capture_output=True, text=True, check=True, cwd=tmp_path).stdout
        assert [float(x) for x in out.split()[-4:]] == [0.25, 10.0, 1.0, 3.0], imports


def test_importing_the_schema_loads_no_env_file(tmp_path):
    # The API imports the schema and must keep its own .env, never the subnet's .vali_env.
    env = _subnet_copy(tmp_path, "EL_EVENT_PAY_X=0.25\nMINER_API_URL=http://vali.test\n")
    env.pop("MINER_API_URL", None)
    out = subprocess.run(
        [sys.executable, "-c", "import os; import alpharidge_ai.schema; "
         "from alpharidge_ai.schema import types as t; "
         "print(os.getenv('EL_EVENT_PAY_X'), os.getenv('MINER_API_URL'), t.EVENT_PAY_X)"],
        env=env, capture_output=True, text=True, check=True, cwd=tmp_path).stdout
    assert out.split() == ["None", "None", "1.0"]


@pytest.mark.parametrize("var,raw", [
    ("EL_EVENT_PAY_X", ""), ("EL_EVENT_PAY_X", "1,5"), ("EL_SCOOP_PREMIUM", "nan"),
    ("EL_SCOOP_MISS_DEBT_X", "-1"), ("EL_SCOOP_PREMIUM", "inf"), ("EL_EVENT_PAY_X", "abc"),
])
def test_a_bad_pay_setting_stops_with_its_name(monkeypatch, var, raw):
    _clear_pay(monkeypatch)
    monkeypatch.setenv(var, raw)
    with pytest.raises(types.PaySettingError) as e:
        types.pay_settings()
    assert var in str(e.value) and repr(raw) in str(e.value)
    # Pricing that reads the value stops too, rather than with a bare float() error.
    use = {"EL_EVENT_PAY_X": lambda: types.event_price("high"),
           "EL_SCOOP_PREMIUM": lambda: types.scoop_multiple(AFTER, RECEIVED),
           "EL_SCOOP_MISS_DEBT_X": lambda: types.scoop_debt("merger_acquisition",
                                                             date(2026, 10, 13), RECEIVED)}
    with pytest.raises(types.PaySettingError):
        use[var]()


def test_a_premium_below_the_event_rate_stops(monkeypatch):
    _clear_pay(monkeypatch)
    monkeypatch.setenv("EL_EVENT_PAY_X", "2")
    monkeypatch.setenv("EL_SCOOP_PREMIUM", "1")
    with pytest.raises(types.PaySettingError, match="EL_SCOOP_PREMIUM"):
        types.pay_settings()
    # Spaces around a value and zero pay are allowed.
    monkeypatch.setenv("EL_EVENT_PAY_X", " 0 ")
    monkeypatch.setenv("EL_SCOOP_MISS_DEBT_X", "0")
    assert types.pay_settings()["EVENT_PAY_X"] == 0.0


def test_the_suite_reads_no_checkout_env_file_and_no_shell_pay():
    # conftest makes the suite hermetic, whatever the caller exports.
    assert os.environ.get("ALPHARIDGE_SKIP_ENV_FILES") == "1"
    assert not [k for k in os.environ if k.startswith("EL_")]
