"""Unit tests for the adaptive-dispatch congestion window in MinerCooldownTracker."""

import time

import pytest

import alpharidge_ai.config as config
from alpharidge_ai.utils.cooldown import MinerCooldownTracker, MAX_INFLIGHT_PER_MINER


@pytest.fixture
def adaptive_on(monkeypatch):
    monkeypatch.setattr(config, "ADAPTIVE_DISPATCH_ENABLED", True, raising=False)
    # Pin knobs so tests are independent of remote-config drift.
    monkeypatch.setattr(config, "DISPATCH_WINDOW_MIN", 1, raising=False)
    monkeypatch.setattr(config, "DISPATCH_WINDOW_GROW", 1.0, raising=False)
    monkeypatch.setattr(config, "DISPATCH_WINDOW_SHRINK", 0.5, raising=False)
    monkeypatch.setattr(config, "DISPATCH_LATE_FRACTION", 0.6, raising=False)
    monkeypatch.setattr(config, "DISPATCH_CHRONIC_TIMEOUT_N", 5, raising=False)
    monkeypatch.setattr(config, "SCORING_LEASE_TTL_SECONDS", 900, raising=False)
    yield


# ---- Backward compatibility (flag off / non-adaptive) ----

def test_static_cap_when_not_adaptive():
    """A non-adaptive tracker always uses MAX_INFLIGHT_PER_MINER, regardless of flag."""
    t = MinerCooldownTracker(adaptive=False)
    for _ in range(MAX_INFLIGHT_PER_MINER):
        assert t.try_acquire("hk")
    assert not t.try_acquire("hk")  # 5th blocked


def test_adaptive_tracker_static_when_flag_off(monkeypatch):
    monkeypatch.setattr(config, "ADAPTIVE_DISPATCH_ENABLED", False, raising=False)
    t = MinerCooldownTracker(adaptive=True)
    for _ in range(MAX_INFLIGHT_PER_MINER):
        assert t.try_acquire("hk")
    assert not t.try_acquire("hk")  # identical to today


# ---- Window gating (flag on) ----

def test_window_gates_acquire(adaptive_on):
    t = MinerCooldownTracker(adaptive=True)
    t.set_cap(100)
    # window defaults to 1 -> only one in-flight
    assert t.try_acquire("hk")
    assert not t.try_acquire("hk")
    # growing the window raises the limit; record_* do NOT release (inflight is
    # reconciled from the store, not decremented by completion events).
    t.record_timely_valid("hk", latency_s=10)   # window 1 -> 2.0, inflight unchanged
    assert t.window("hk") == pytest.approx(2.0)
    assert t.try_acquire("hk")       # inflight 2 (1 < 2)
    assert not t.try_acquire("hk")   # floor(window)=2 reached
    # In-flight is reconciled from the article store's PROCESSING set, which is what
    # frees slots as work completes.
    t.reconcile_inflight({"hk": 1})
    assert t.try_acquire("hk")       # 1 < 2 again


def test_record_methods_do_not_touch_inflight(adaptive_on):
    t = MinerCooldownTracker(adaptive=True)
    t.set_cap(100)
    t.try_acquire("hk")              # inflight 1
    t.record_timely_valid("hk", latency_s=10)
    t.record_invalid("hk")
    t.record_timeout("hk")
    assert t.inflight("hk") == 1     # unchanged by completion events


# ---- Grow / freeze / shrink ----

def test_on_time_grows_window(adaptive_on):
    t = MinerCooldownTracker(adaptive=True)
    t.set_cap(100)
    t.record_timely_valid("hk", latency_s=10)   # 1 -> 2.0
    assert t.window("hk") == pytest.approx(2.0)
    t.record_timely_valid("hk", latency_s=10)   # 2 -> 2.5
    assert t.window("hk") == pytest.approx(2.5)


def test_slow_but_valid_freezes_window(adaptive_on):
    """A valid push-back slower than LATE_FRACTION*lease must NOT grow the window."""
    t = MinerCooldownTracker(adaptive=True)
    t.set_cap(100)
    t.record_timely_valid("hk", latency_s=10)    # grow to 2.0
    late = 0.6 * 900 + 1                          # just over the late threshold
    t.record_timely_valid("hk", latency_s=late)  # frozen
    assert t.window("hk") == pytest.approx(2.0)


def test_cap_bounds_growth(adaptive_on):
    t = MinerCooldownTracker(adaptive=True)
    t.set_cap(2.0)
    for _ in range(20):
        t.record_timely_valid("hk", latency_s=10)
    assert t.window("hk") == pytest.approx(2.0)   # never exceeds cap


def test_invalid_shrinks_window(adaptive_on):
    t = MinerCooldownTracker(adaptive=True)
    t.set_cap(100)
    for _ in range(3):
        t.record_timely_valid("hk", latency_s=10)
    w_before = t.window("hk")
    t.record_invalid("hk")
    assert t.window("hk") == pytest.approx(w_before * 0.5)


def test_window_never_below_min(adaptive_on):
    t = MinerCooldownTracker(adaptive=True)
    for _ in range(10):
        t.record_invalid("hk")
    assert t.window("hk") == pytest.approx(1.0)   # W_MIN floor


# ---- consec_to rules (resolved decisions) ----

def test_timeout_increments_then_chronic_escalates(adaptive_on):
    t = MinerCooldownTracker(adaptive=True)
    for i in range(4):
        assert t.record_timeout("hk") is False
    # 5th consecutive timeout -> chronic escalation
    assert t.record_timeout("hk") is True
    assert t.is_on_cooldown("hk")           # dropped into backoff cooldown
    assert t._consec_to["hk"] == 0          # counter reset on escalation


def test_valid_resets_consec_to(adaptive_on):
    t = MinerCooldownTracker(adaptive=True)
    for _ in range(4):
        t.record_timeout("hk")
    t.record_timely_valid("hk", latency_s=10)   # resets streak
    for _ in range(4):
        assert t.record_timeout("hk") is False  # needs 5 fresh consecutive
    assert not t.is_on_cooldown("hk")


def test_invalid_resets_consec_to(adaptive_on):
    """Decision: an invalid is a *response* (alive) — it must reset the non-response streak."""
    t = MinerCooldownTracker(adaptive=True)
    for _ in range(4):
        t.record_timeout("hk")
    t.record_invalid("hk")                       # alive, resets streak
    assert t._consec_to["hk"] == 0
    for _ in range(4):
        assert t.record_timeout("hk") is False
    assert not t.is_on_cooldown("hk")


# ---- Reconciliation & coverage ----

def test_reconcile_inflight_rebuilds_counts(adaptive_on):
    t = MinerCooldownTracker(adaptive=True)
    t.try_acquire("a"); t.try_acquire("a"); t.try_acquire("b")
    # Authoritative source says 'a' has 1 outstanding, 'b' has 0 (leak repaired).
    t.reconcile_inflight({"a": 1, "b": 0, "c": 3})
    assert t.inflight("a") == 1
    assert t.inflight("b") == 0
    assert t.inflight("c") == 3


def test_snapshot_reports_per_hotkey_state(adaptive_on):
    t = MinerCooldownTracker(adaptive=True)
    t.set_cap(100)
    t.try_acquire("hk")
    t.record_timely_valid("hk", latency_s=10)   # window 1 -> 2.0, inflight stays 1
    t.mark_covered("hk", 5)
    for _ in range(5):
        t.record_timeout("dead")                 # 5th escalates -> cooldown
    snap = t.snapshot()
    assert "hk" in snap and "dead" in snap
    assert snap["hk"]["window"] == pytest.approx(2.0)
    assert snap["hk"]["inflight"] == 1
    assert snap["hk"]["covered_epoch"] == 5
    assert snap["hk"]["on_cooldown"] is False
    assert snap["dead"]["on_cooldown"] is True
    assert snap["dead"]["cooldown_remaining_s"] > 0


def test_coverage_tracking():
    t = MinerCooldownTracker(adaptive=True)
    assert t.covered_epoch("hk") == -1
    t.mark_covered("hk", 42)
    assert t.covered_epoch("hk") == 42


def test_prune_clears_adaptive_state(adaptive_on):
    t = MinerCooldownTracker(adaptive=True)
    t.record_timely_valid("keep", latency_s=10)
    t.record_timely_valid("drop", latency_s=10)
    t.mark_covered("drop", 5)
    t.prune({"keep"})
    assert "drop" not in t._window
    assert "drop" not in t._covered_ep
    assert "keep" in t._window


# ---- Faithfulness-based stub cooldown (2026-07-09) ----

@pytest.fixture
def faith_knobs(adaptive_on, monkeypatch):
    monkeypatch.setattr(config, "DISPATCH_COOLDOWN_FAITHFULNESS_FLOOR", 0.5, raising=False)
    monkeypatch.setattr(config, "DISPATCH_CONSEC_INVALID_N", 10, raising=False)
    monkeypatch.setattr(config, "DISPATCH_INVALID_COOLDOWN_FIRST_S", 60, raising=False)
    monkeypatch.setattr(config, "DISPATCH_INVALID_COOLDOWN_MAX_S", 600, raising=False)
    monkeypatch.setattr(config, "DISPATCH_COOLDOWN_SHADOW_MODE", True, raising=False)
    yield


def test_grounded_batches_never_trip(faith_knobs):
    t = MinerCooldownTracker(adaptive=True)
    for _ in range(30):
        t.record_faithfulness("hk", 0.8)
    assert "hk" not in t.get_cooled_down_hotkeys()
    assert t._consec_inv.get("hk", 0) == 0


def test_none_faith_is_noop(faith_knobs):
    t = MinerCooldownTracker(adaptive=True)
    t.record_faithfulness("hk", None)
    assert "hk" not in t._consec_inv and "hk" not in t._last_faith


def test_shadow_mode_counts_but_does_not_enforce(faith_knobs, monkeypatch):
    monkeypatch.setattr(config, "DISPATCH_COOLDOWN_SHADOW_MODE", True, raising=False)
    t = MinerCooldownTracker(adaptive=True)
    for _ in range(10):
        t.record_faithfulness("hk", 0.30)
    assert "hk" not in t.get_cooled_down_hotkeys()   # not enforced
    assert "hk" not in t._inv_until                    # no park written in shadow
    assert t._inv_level.get("hk") == 1                # but the trip was recorded


def test_enforce_mode_parks_at_first_s(faith_knobs, monkeypatch):
    monkeypatch.setattr(config, "DISPATCH_COOLDOWN_SHADOW_MODE", False, raising=False)
    t = MinerCooldownTracker(adaptive=True)
    for i in range(9):
        t.record_faithfulness("hk", 0.30)
        assert "hk" not in t.get_cooled_down_hotkeys()
    t.record_faithfulness("hk", 0.30)                 # 10th
    assert "hk" in t.get_cooled_down_hotkeys()
    assert "hk" not in t._state                        # park lives in _inv_until, not _state
    assert t._inv_level["hk"] == 1
    assert 58 <= (t._inv_until["hk"] - time.time()) <= 61


def test_enforce_escalation_survives_reprobe_ack(faith_knobs, monkeypatch):
    monkeypatch.setattr(config, "DISPATCH_COOLDOWN_SHADOW_MODE", False, raising=False)
    t = MinerCooldownTracker(adaptive=True)
    for _ in range(10):
        t.record_faithfulness("hk", 0.30)             # trip 1 -> level 1
    t.record_success("hk")                            # ack; must not reset level or park
    for _ in range(10):
        t.record_faithfulness("hk", 0.30)             # trip 2
    assert t._inv_level["hk"] == 2
    assert 598 <= (t._inv_until["hk"] - time.time()) <= 601


def test_park_is_not_sprung_by_record_success(faith_knobs, monkeypatch):
    """The durability fix: an enforced park lives in _inv_until, which record_success (the
    ack-path / passing-batch call that del's _state) must NOT clear."""
    monkeypatch.setattr(config, "DISPATCH_COOLDOWN_SHADOW_MODE", False, raising=False)
    t = MinerCooldownTracker(adaptive=True)
    for _ in range(10):
        t.record_faithfulness("hk", 0.30)
    assert "hk" in t.get_cooled_down_hotkeys()
    for _ in range(20):                               # concurrent in-flight acks / passing batches
        t.record_success("hk")
    assert "hk" in t.get_cooled_down_hotkeys()         # park still held
    assert "hk" in t._inv_until


def test_grounded_batch_self_heals_park(faith_knobs, monkeypatch):
    monkeypatch.setattr(config, "DISPATCH_COOLDOWN_SHADOW_MODE", False, raising=False)
    t = MinerCooldownTracker(adaptive=True)
    for _ in range(10):
        t.record_faithfulness("hk", 0.30)
    assert "hk" in t.get_cooled_down_hotkeys()
    t.record_faithfulness("hk", 0.9)                  # a grounded in-flight batch returns
    assert "hk" not in t.get_cooled_down_hotkeys()     # park cleared
    assert "hk" not in t._inv_until and t._inv_level.get("hk", 0) == 0


def test_expired_park_is_dispatchable(faith_knobs, monkeypatch):
    monkeypatch.setattr(config, "DISPATCH_COOLDOWN_SHADOW_MODE", False, raising=False)
    t = MinerCooldownTracker(adaptive=True)
    for _ in range(10):
        t.record_faithfulness("hk", 0.30)
    t._inv_until["hk"] = time.time() - 1              # simulate expiry
    assert "hk" not in t.get_cooled_down_hotkeys()


def test_grounded_batch_resets_streak(faith_knobs):
    t = MinerCooldownTracker(adaptive=True)
    for _ in range(9):
        t.record_faithfulness("hk", 0.30)
    t.record_faithfulness("hk", 0.9)                  # grounded -> reset
    assert t._consec_inv.get("hk") == 0 and t._inv_level.get("hk") == 0
    for _ in range(9):
        t.record_faithfulness("hk", 0.30)
    assert "hk" not in t.get_cooled_down_hotkeys()    # needs a full fresh N


def test_legit_dip_then_grounded_never_parks(faith_knobs):
    t = MinerCooldownTracker(adaptive=True)
    for _ in range(50):
        t.record_faithfulness("hk", 0.30)
        t.record_faithfulness("hk", 0.8)
    assert "hk" not in t.get_cooled_down_hotkeys()


def test_faithfulness_floor_is_configurable(faith_knobs, monkeypatch):
    monkeypatch.setattr(config, "DISPATCH_COOLDOWN_FAITHFULNESS_FLOOR", 0.7, raising=False)
    t = MinerCooldownTracker(adaptive=True)
    t.record_faithfulness("hk", 0.6)                  # 0.6 < 0.7 -> flagged
    assert t._consec_inv.get("hk") == 1


def test_prune_clears_faith_state(faith_knobs):
    t = MinerCooldownTracker(adaptive=True)
    for _ in range(5):
        t.record_faithfulness("drop", 0.30)
    t.prune({"keep"})
    assert "drop" not in t._consec_inv
    assert "drop" not in t._inv_level
    assert "drop" not in t._last_faith


def test_snapshot_exposes_faith_fields(faith_knobs):
    t = MinerCooldownTracker(adaptive=True)
    for _ in range(4):
        t.record_faithfulness("hk", 0.30)
    snap = t.snapshot()
    assert snap["hk"]["consec_inv"] == 4
    assert "inv_level" in snap["hk"]
    assert snap["hk"]["last_faith"] == 0.3
