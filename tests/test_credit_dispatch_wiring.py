"""Credit dispatch as the validator drives it: the switch, who is owed, and shadow mode."""

import types

import pytest

from alpharidge_ai import config
from alpharidge_ai.utils.dispatch import CARRY_MAX_BATCHES, ShadowCredit, credit_select
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
    for name in ("_dispatch_eligible", "_credit_args",
                 "_credit_assign", "_shadow_credit", "_log_shadow_spread"):
        setattr(v, name, types.MethodType(getattr(Validator, name), v))
    for k, val in flags.items():
        setattr(config, k, val)
    return v


@pytest.fixture(autouse=True)
def _restore():
    keys = ["DISPATCH_MODE", "DISPATCH_CREDIT_SHADOW", "REPUTATION_SCORING_ENABLED",
            "ADAPTIVE_BATCH_SIZE_ENABLED", "MINER_BATCH_SIZE"]
    before = {k: getattr(config, k, None) for k in keys}
    yield
    for k, v in before.items():
        if v is not None:
            setattr(config, k, v)


# ---- the switch -------------------------------------------------------------------

def test_it_is_off_by_default():
    assert config.DISPATCH_MODE == "coverage"
    assert config.DISPATCH_CREDIT_SHADOW is False


def test_only_the_two_switches_are_settings():
    """The allocator's own constants are not knobs an operator can turn."""
    assert "DISPATCH_MODE" in config._REMOTE_CONFIG_KEYS
    assert "DISPATCH_CREDIT_SHADOW" in config._REMOTE_CONFIG_KEYS
    assert not [k for k in config._REMOTE_CONFIG_KEYS
                if k.startswith("DISPATCH_CREDIT_") and k != "DISPATCH_CREDIT_SHADOW"]


def test_dispatch_settings_are_not_consensus_keys():
    """Dispatch is each validator's own choice; only settlement needs agreement."""
    assert "DISPATCH_MODE" not in config._CONSENSUS_KEYS


# ---- who is owed a share ----------------------------------------------------------

def test_a_miner_returning_work_is_owed_a_share():
    tracker = Tracker()
    tracker.last_valid = {"hk000": 0.0}
    v = validator(tracker)
    tracker.delivered_since = lambda hk, s: hk == "hk000"
    tracker.starting_up = lambda hk, e, t: False
    assert Validator._dispatch_eligible(v, epoch=10)("hk000")
    assert not Validator._dispatch_eligible(v, epoch=10)("hk001")


def test_a_miner_that_has_not_returned_yet_still_gets_its_start():
    tracker = Tracker()
    tracker.delivered_since = lambda hk, s: False
    tracker.starting_up = lambda hk, e, t: hk == "hk002"
    v = validator(tracker)
    assert Validator._dispatch_eligible(v, epoch=10)("hk002")
    assert not Validator._dispatch_eligible(v, epoch=10)("hk003")


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


def test_shadow_counts_articles_not_turns(monkeypatch):
    import neurons.validator as nv
    monkeypatch.setattr(nv.bt.logging, "debug", lambda m: None)
    tracker = Tracker()
    tracker.delivered_since = lambda hk, s: True
    tracker.starting_up = lambda hk, e, t: False
    v = validator(tracker, DISPATCH_CREDIT_SHADOW=True, MINER_BATCH_SIZE=24,
                  DISPATCH_MODE="coverage")
    Validator._shadow_credit(v, UIDS, HOTKEYS, 5, 10, [(1, 0)])
    assert sum(v._shadow_served.values()) == 10 * 24
    assert not v._live_served


def test_live_is_counted_from_what_was_sent():
    v = validator(DISPATCH_CREDIT_SHADOW=True, DISPATCH_MODE="coverage")
    v.metagraph = types.SimpleNamespace(hotkeys=HOTKEYS)
    v._count_live_served = types.MethodType(Validator._count_live_served, v)
    v._count_live_served([(0, ["a"] * 32), (1, ["a"] * 7), (0, ["a"] * 3)])
    assert v._live_served == {HOTKEYS[0]: 35, HOTKEYS[1]: 7}
    config.DISPATCH_CREDIT_SHADOW = False
    v._count_live_served([(0, ["a"] * 32)])
    assert v._live_served[HOTKEYS[0]] == 35


def test_rations_left_unapplied_are_called_out(monkeypatch):
    lines = []
    import neurons.validator as nv
    monkeypatch.setattr(nv.bt.logging, "warning", lambda m: lines.append(m))
    v = types.SimpleNamespace()
    v._warn_rations_bypassed = types.MethodType(Validator._warn_rations_bypassed, v)
    config.DISPATCH_MODE, config.ADAPTIVE_BATCH_SIZE_ENABLED = "coverage", True
    v._warn_rations_bypassed()
    assert not lines
    config.ADAPTIVE_BATCH_SIZE_ENABLED = False
    v._warn_rations_bypassed()
    v._warn_rations_bypassed()
    assert len(lines) == 1 and "ADAPTIVE_BATCH_SIZE_ENABLED is off" in lines[0]


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


# ---- turns that never reach a miner ------------------------------------------------

def _refunder(mode="credit"):
    tracker = Tracker()
    v = types.SimpleNamespace(_article_cooldown=tracker,
                              metagraph=types.SimpleNamespace(hotkeys=HOTKEYS))
    v._refund_credit = types.MethodType(Validator._refund_credit, v)
    config.DISPATCH_MODE = mode
    return v, tracker


def test_an_unsent_turn_is_given_back():
    v, tracker = _refunder()
    tracker.credits["hk000"] = 0.0
    v._refund_credit(0)
    assert tracker.credit("hk000") == pytest.approx(1.0)


def test_nothing_is_refunded_while_credit_is_off():
    v, tracker = _refunder(mode="coverage")
    v._refund_credit(0)
    assert not tracker.credits


def test_a_refund_for_an_unknown_uid_is_harmless():
    v, tracker = _refunder()
    v._refund_credit(9_999)
    assert not tracker.credits


def test_refunds_respect_the_carry_bound():
    v, tracker = _refunder()
    for _ in range(10):
        v._refund_credit(0)
    assert tracker.credit("hk000") <= CARRY_MAX_BATCHES


def test_contradictory_settings_are_called_out(monkeypatch):
    lines = []
    import neurons.validator as nv
    monkeypatch.setattr(nv.bt.logging, "warning", lambda m: lines.append(m))
    v = types.SimpleNamespace()
    v._warn_credit_preconditions = types.MethodType(Validator._warn_credit_preconditions, v)
    v._warn_credit_unreachable = types.MethodType(Validator._warn_credit_unreachable, v)
    config.ADAPTIVE_BATCH_SIZE_ENABLED = True
    v._warn_credit_preconditions()
    v._warn_credit_preconditions()          # once only
    v._warn_credit_unreachable()
    assert sum("adaptive batch size" in line for line in lines) == 1
    assert any("ADAPTIVE_DISPATCH_ENABLED is off" in line for line in lines)


def test_too_large_a_batch_is_called_out(monkeypatch):
    lines = []
    import neurons.validator as nv
    monkeypatch.setattr(nv.bt.logging, "warning", lambda m: lines.append(m))
    v = types.SimpleNamespace()
    v._warn_credit_preconditions = types.MethodType(Validator._warn_credit_preconditions, v)
    config.ADAPTIVE_BATCH_SIZE_ENABLED = False
    config.MINER_BATCH_SIZE = 32
    v._warn_credit_preconditions()
    assert not lines
    config.MINER_BATCH_SIZE = 40
    v._warn_credit_preconditions()
    v._warn_credit_preconditions()
    assert sum("MINER_BATCH_SIZE=40" in line for line in lines) == 1
    config.MINER_BATCH_SIZE = 16
