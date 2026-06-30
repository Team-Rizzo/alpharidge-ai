"""Unit tests for the adaptive-dispatch congestion window in MinerCooldownTracker."""

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


# ---- Capacity-class backoff (DISPATCH_CAPACITY_SHRINK; freeze option) ----

def test_capacity_shrink_defaults_to_integrity_shrink(adaptive_on, monkeypatch):
    """Unset/default DISPATCH_CAPACITY_SHRINK -> capacity backoff == the 0.5 shrink,
    i.e. behavior is unchanged for timeout and incomplete-analysis."""
    monkeypatch.setattr(config, "DISPATCH_CAPACITY_SHRINK", 0.5, raising=False)
    t = MinerCooldownTracker(adaptive=True)
    t.set_cap(100)
    for _ in range(3):
        t.record_timely_valid("hk", latency_s=10)
    w = t.window("hk")
    t.record_timeout("hk")
    assert t.window("hk") == pytest.approx(w * 0.5)        # lease-timeout
    t.set_cap(100)
    for _ in range(3):
        t.record_timely_valid("hk2", latency_s=10)
    w2 = t.window("hk2")
    t.record_capacity_backoff("hk2")
    assert t.window("hk2") == pytest.approx(w2 * 0.5)      # incomplete analysis


def test_capacity_freeze_holds_window(adaptive_on, monkeypatch):
    """DISPATCH_CAPACITY_SHRINK=1.0 freezes the window on capacity signals so a
    transient timeout/incomplete doesn't collapse a depth window — while a genuine
    integrity invalid still takes the full 0.5 shrink."""
    monkeypatch.setattr(config, "DISPATCH_CAPACITY_SHRINK", 1.0, raising=False)
    t = MinerCooldownTracker(adaptive=True)
    t.set_cap(100)
    for _ in range(3):
        t.record_timely_valid("hk", latency_s=10)
    w = t.window("hk")
    assert w > 2.0
    t.record_timeout("hk")
    assert t.window("hk") == pytest.approx(w)              # frozen, not shrunk
    t.record_capacity_backoff("hk")
    assert t.window("hk") == pytest.approx(w)              # frozen
    t.record_invalid("hk")                                 # integrity still shrinks
    assert t.window("hk") == pytest.approx(w * 0.5)


def test_capacity_freeze_still_escalates_chronic(adaptive_on, monkeypatch):
    """Freezing the window must NOT disable chronic-timeout escalation — non-response
    still counts toward the cooldown threshold even when the window is held."""
    monkeypatch.setattr(config, "DISPATCH_CAPACITY_SHRINK", 1.0, raising=False)
    t = MinerCooldownTracker(adaptive=True)
    for _ in range(4):
        assert t.record_timeout("hk") is False
    assert t.record_timeout("hk") is True                 # 5th = chronic


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
