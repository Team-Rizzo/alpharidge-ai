"""Unit tests for the coverage-then-depth article allocator (pure logic)."""

import pytest

import alpharidge_ai.config as config
from alpharidge_ai.utils.cooldown import MinerCooldownTracker
from alpharidge_ai.utils.dispatch import coverage_depth_select


@pytest.fixture(autouse=True)
def adaptive_on(monkeypatch):
    monkeypatch.setattr(config, "ADAPTIVE_DISPATCH_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "DISPATCH_WINDOW_MIN", 1, raising=False)
    monkeypatch.setattr(config, "DISPATCH_WINDOW_GROW", 1.0, raising=False)
    monkeypatch.setattr(config, "DISPATCH_WINDOW_SHRINK", 0.5, raising=False)
    monkeypatch.setattr(config, "DISPATCH_LATE_FRACTION", 0.6, raising=False)
    monkeypatch.setattr(config, "SCORING_LEASE_TTL_SECONDS", 900, raising=False)
    yield


def make_tracker(hotkeys, windows=None):
    """A tracker with optional preset windows (via on-time completions)."""
    t = MinerCooldownTracker(adaptive=True)
    t.set_cap(100)
    if windows:
        for hk, target in windows.items():
            # grow toward `target` with on-time completions
            for _ in range(200):
                if t.window(hk) >= target:
                    break
                t.record_timely_valid(hk, latency_s=10)
    return t


HOTKEYS = [f"hk{i}" for i in range(5)]


def test_coverage_gives_every_live_miner_one_batch():
    t = make_tracker(HOTKEYS)
    live = [0, 1, 2, 3, 4]
    # plenty of batches; coverage should touch each live miner exactly once first
    assignments = coverage_depth_select(live, HOTKEYS, t, epoch=1, n_batches=5)
    covered = [uid for uid, _ in assignments]
    assert sorted(covered) == [0, 1, 2, 3, 4]
    # each marked covered for this epoch
    for hk in HOTKEYS:
        assert t.covered_epoch(hk) == 1


def test_coverage_floor_before_depth():
    # window of hk0 is large, but coverage must still reach everyone first
    t = make_tracker(HOTKEYS, windows={"hk0": 5})
    t.set_cap(100)
    live = [0, 1, 2, 3, 4]
    assignments = coverage_depth_select(live, HOTKEYS, t, epoch=1, n_batches=5)
    # exactly one batch each on the coverage pass (5 batches, 5 miners)
    assert sorted(uid for uid, _ in assignments) == [0, 1, 2, 3, 4]


def test_depth_favours_higher_window():
    t = make_tracker(HOTKEYS, windows={"hk0": 4, "hk1": 2})
    t.set_cap(100)
    live = [0, 1, 2, 3, 4]
    # 5 coverage + extra depth batches
    assignments = coverage_depth_select(live, HOTKEYS, t, epoch=1, n_batches=12)
    counts = {}
    for uid, _ in assignments:
        counts[uid] = counts.get(uid, 0) + 1
    # hk0 (window 4) takes more than the window-1 miners
    assert counts[0] == 4
    assert counts[1] == 2
    assert counts[2] == 1 and counts[3] == 1 and counts[4] == 1


def test_already_covered_skips_coverage_but_depth_still_flows():
    t = make_tracker(HOTKEYS, windows={"hk0": 3})
    t.set_cap(100)
    live = [0, 1, 2, 3, 4]
    # everyone already covered this epoch
    for hk in HOTKEYS:
        t.mark_covered(hk, 7)
    assignments = coverage_depth_select(live, HOTKEYS, t, epoch=7, n_batches=3)
    # No coverage owed -> straight to depth. Depth is round-robin in window order,
    # so 3 batches go one-each to the top-3 miners (hk0 first); not all to hk0.
    assert len(assignments) == 3
    uids = [uid for uid, _ in assignments]
    assert len(set(uids)) == 3        # round-robin spread, not greedy-fill
    assert 0 in uids                  # highest window served first


def test_window_full_miner_excluded():
    t = make_tracker(HOTKEYS)
    # hk0 window 1, already 1 in-flight -> no slot
    t.try_acquire("hk0")
    live = [0, 1]
    assignments = coverage_depth_select(live, HOTKEYS, t, epoch=1, n_batches=4)
    assigned_uids = [uid for uid, _ in assignments]
    assert 0 not in assigned_uids        # full, skipped
    assert 1 in assigned_uids


def test_batch_indices_are_unique_and_in_range():
    t = make_tracker(HOTKEYS, windows={"hk0": 3, "hk1": 3})
    t.set_cap(100)
    live = [0, 1, 2, 3, 4]
    n = 9
    assignments = coverage_depth_select(live, HOTKEYS, t, epoch=1, n_batches=n)
    idxs = [bi for _, bi in assignments]
    assert sorted(idxs) == list(range(len(assignments)))
    assert all(0 <= bi < n for bi in idxs)


def test_no_live_miners_returns_empty():
    t = make_tracker(HOTKEYS)
    assert coverage_depth_select([], HOTKEYS, t, epoch=1, n_batches=5) == []


def test_unassigned_when_all_windows_full():
    t = make_tracker(HOTKEYS)
    live = [0, 1]
    # both at window 1, fill them
    t.try_acquire("hk0")
    t.try_acquire("hk1")
    assignments = coverage_depth_select(live, HOTKEYS, t, epoch=1, n_batches=5)
    assert assignments == []   # nothing assignable; batches retry next tick
