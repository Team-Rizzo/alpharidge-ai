"""Credit dispatch as the validator drives it: the switch, who is owed, and shadow mode."""

import types

import pytest

from alpharidge_ai import config
from alpharidge_ai.utils.dispatch import ShadowCredit, credit_select
from neurons.validator import Validator
from tests.test_credit_dispatch import HOTKEYS, UIDS, Tracker


def validator(tracker=None, **flags):
    v = types.SimpleNamespace(
        _article_cooldown=tracker or Tracker(),
        _reputation_store=types.SimpleNamespace(reputation=lambda hk: 0.6,
                                                samples=lambda hk: 100),
        _mechanism_profile=types.SimpleNamespace(resolve=lambda block: None),
        block=1_000,
        _shadow_credit_state={}, _shadow_served={}, _live_served={}, _shadow_ticks=0)
    for name in ("_dispatch_weight", "_dispatch_eligible", "_credit_args",
                 "_credit_assign", "_shadow_credit", "_log_shadow_spread"):
        setattr(v, name, types.MethodType(getattr(Validator, name), v))
    for k, val in flags.items():
        setattr(config, k, val)
    return v


@pytest.fixture(autouse=True)
def _restore():
    keys = ["DISPATCH_MODE", "DISPATCH_CREDIT_SHADOW", "DISPATCH_CREDIT_WEIGHT_K",
            "DISPATCH_CREDIT_RECENCY_S", "DISPATCH_CREDIT_TRIAL_EPOCHS",
            "REPUTATION_SCORING_ENABLED"]
    before = {k: getattr(config, k, None) for k in keys}
    yield
    for k, v in before.items():
        if v is not None:
            setattr(config, k, v)


# ---- the switch -------------------------------------------------------------------

def test_it_is_off_by_default():
    assert config.DISPATCH_MODE == "coverage"
    assert config.DISPATCH_CREDIT_SHADOW is False
    assert config.DISPATCH_CREDIT_WEIGHT_K == 0.0


def test_the_settings_are_served():
    for key in ("DISPATCH_MODE", "DISPATCH_CREDIT_SHADOW", "DISPATCH_CREDIT_WEIGHT_K",
                "DISPATCH_CREDIT_CAP", "DISPATCH_CREDIT_FLOOR_PCT",
                "DISPATCH_CREDIT_RECENCY_S", "DISPATCH_CREDIT_TRIAL_EPOCHS"):
        assert key in config._REMOTE_CONFIG_KEYS


def test_dispatch_settings_are_not_consensus_keys():
    """Dispatch is each validator's own choice; only settlement needs agreement."""
    assert "DISPATCH_MODE" not in config._CONSENSUS_KEYS


# ---- who is owed a share ----------------------------------------------------------

def test_a_miner_returning_work_is_owed_a_share():
    tracker = Tracker()
    tracker.last_valid = {"hk000": 0.0}
    v = validator(tracker, DISPATCH_CREDIT_RECENCY_S=7200)
    tracker.delivered_since = lambda hk, s: hk == "hk000"
    tracker.starting_up = lambda hk, e, t: False
    assert Validator._dispatch_eligible(v, epoch=10)("hk000")
    assert not Validator._dispatch_eligible(v, epoch=10)("hk001")


def test_a_miner_that_has_not_returned_yet_still_gets_its_start():
    tracker = Tracker()
    tracker.delivered_since = lambda hk, s: False
    tracker.starting_up = lambda hk, e, t: hk == "hk002"
    v = validator(tracker, DISPATCH_CREDIT_TRIAL_EPOCHS=3)
    assert Validator._dispatch_eligible(v, epoch=10)("hk002")
    assert not Validator._dispatch_eligible(v, epoch=10)("hk003")


# ---- weights ----------------------------------------------------------------------

def test_no_weighting_by_default():
    v = validator(DISPATCH_CREDIT_WEIGHT_K=0.0, REPUTATION_SCORING_ENABLED=True)
    assert Validator._dispatch_weight(v) is None


def test_weighting_is_off_while_reputation_scoring_is():
    v = validator(DISPATCH_CREDIT_WEIGHT_K=1.0, REPUTATION_SCORING_ENABLED=False)
    assert Validator._dispatch_weight(v) is None


def test_a_weight_reads_the_multiplier():
    v = validator(DISPATCH_CREDIT_WEIGHT_K=1.0, REPUTATION_SCORING_ENABLED=True)
    weight = Validator._dispatch_weight(v)
    assert weight is not None
    value = weight("hk000")
    assert value >= 0.0


def test_an_unreadable_store_still_dispatches():
    v = validator(DISPATCH_CREDIT_WEIGHT_K=1.0, REPUTATION_SCORING_ENABLED=True)
    v._reputation_store = types.SimpleNamespace(
        reputation=lambda hk: (_ for _ in ()).throw(RuntimeError("cold")),
        samples=lambda hk: 0)
    weight = Validator._dispatch_weight(v)
    assert weight("hk000") == 1.0


# ---- shadow mode ------------------------------------------------------------------

def test_shadow_keeps_its_credit_off_the_real_tracker():
    real = Tracker()
    state = {}
    shadow = ShadowCredit(real, state)
    credit_select(UIDS, HOTKEYS, shadow, 40)
    assert state and not real.credits


def test_shadow_sees_the_real_constraints():
    real = Tracker()
    real.inflights = {hk: 1 for hk in HOTKEYS}          # everything busy
    shadow = ShadowCredit(real, {})
    assert credit_select(UIDS, HOTKEYS, shadow, 40) == []


def test_shadow_logs_and_changes_nothing(monkeypatch):
    lines = []
    import neurons.validator as nv
    monkeypatch.setattr(nv.bt.logging, "debug", lambda m: lines.append(m))
    monkeypatch.setattr(nv.bt.logging, "info", lambda m: lines.append(m))
    tracker = Tracker()
    tracker.delivered_since = lambda hk, s: True
    tracker.starting_up = lambda hk, e, t: False
    v = validator(tracker, DISPATCH_CREDIT_SHADOW=True)
    Validator._shadow_credit(v, UIDS, HOTKEYS, 5, 40, [(1, 0)])
    assert not tracker.credits
    assert v._shadow_credit_state
    assert any("credit shadow" in line for line in lines)


def test_shadow_is_silent_when_off(monkeypatch):
    lines = []
    import neurons.validator as nv
    monkeypatch.setattr(nv.bt.logging, "debug", lambda m: lines.append(m))
    v = validator(DISPATCH_CREDIT_SHADOW=False)
    Validator._shadow_credit(v, UIDS, HOTKEYS, 5, 40, [])
    assert not lines


def test_a_broken_shadow_does_not_disturb_dispatch_but_says_so(monkeypatch):
    lines = []
    import neurons.validator as nv
    monkeypatch.setattr(nv, "credit_select",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(nv.bt.logging, "debug", lambda m: lines.append(m))
    v = validator(DISPATCH_CREDIT_SHADOW=True)
    Validator._shadow_credit(v, UIDS, HOTKEYS, 5, 40, [])   # must not raise
    assert any("shadow failed" in line for line in lines)


def test_the_spread_report_compares_both(monkeypatch):
    lines = []
    import neurons.validator as nv
    monkeypatch.setattr(nv.bt.logging, "info", lambda m: lines.append(m))
    v = validator()
    v._live_served = {hk: (10 if i < 5 else 0) for i, hk in enumerate(HOTKEYS)}
    v._shadow_served = {hk: 2 for hk in HOTKEYS}
    v._shadow_ticks = 180
    Validator._log_shadow_spread(v, UIDS, HOTKEYS)
    assert any("spread live" in line and "credit" in line for line in lines)
    assert v._shadow_ticks == 0 and not v._live_served
