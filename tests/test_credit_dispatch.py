"""Credit dispatch: work is handed out by turn owed, not by draw."""

import statistics as st

import pytest

from alpharidge_ai.utils.dispatch import CARRY_MAX_BATCHES, credit_select


class Tracker:
    """The parts of the cooldown tracker the allocator touches."""

    def __init__(self, limit=1):
        self.credits = {}
        self.inflights = {}
        self.covered = {}
        self.limit = limit

    def credit(self, hk):
        return self.credits.get(hk, 0.0)

    def add_credit(self, hk, amount, carry_max):
        self.credits[hk] = min(carry_max, self.credit(hk) + amount)

    def spend_credit(self, hk, amount, carry_max=CARRY_MAX_BATCHES):
        self.credits[hk] = max(-carry_max, self.credit(hk) - amount)

    def inflight(self, hk):
        return self.inflights.get(hk, 0)

    def inflight_limit(self, hk):
        return self.limit

    def covered_epoch(self, hk):
        return self.covered.get(hk, -1)


HOTKEYS = [f"hk{i:03d}" for i in range(240)]
UIDS = list(range(240))


def run(ticks, n_batches, tracker=None, **kw):
    """Serve `ticks` ticks and return batches per hotkey."""
    tracker = tracker or Tracker()
    got = {hk: 0 for hk in HOTKEYS}
    for _ in range(ticks):
        for uid, _bi in credit_select(UIDS, HOTKEYS, tracker, n_batches, **kw):
            got[HOTKEYS[uid]] += 1
    return got, tracker


def test_everyone_gets_the_same_share_over_time():
    got, _ = run(ticks=120, n_batches=80)
    served = list(got.values())
    assert min(served) >= 30 and max(served) - min(served) <= 2


def test_it_beats_a_draw_on_evenness():
    got, _ = run(ticks=30, n_batches=80)
    served = [v for v in got.values()]
    spread = st.pstdev(served) / (st.mean(served) or 1)
    assert spread < 0.15


def test_every_batch_is_handed_out():
    tracker = Tracker(limit=4)
    got, _ = run(ticks=10, n_batches=80, tracker=tracker)
    assert sum(got.values()) == 800


def test_nothing_is_assigned_without_batches_or_miners():
    assert credit_select(UIDS, HOTKEYS, Tracker(), 0) == []
    assert credit_select([], HOTKEYS, Tracker(), 10) == []


def test_a_miner_that_misses_a_turn_is_served_first_next():
    tracker = Tracker()
    tracker.inflights["hk000"] = 1          # busy this tick
    credit_select(UIDS, HOTKEYS, tracker, 80)
    assert tracker.credit("hk000") > 0.0
    tracker.inflights["hk000"] = 0
    assigned = {uid for uid, _ in credit_select(UIDS, HOTKEYS, tracker, 80)}
    assert 0 in assigned


def test_credit_cannot_bank_a_burst():
    tracker = Tracker()
    for hk in HOTKEYS:
        tracker.inflights[hk] = 1           # nobody can take anything
    for _ in range(50):
        credit_select(UIDS, HOTKEYS, tracker, 80)
    assert max(tracker.credits.values()) <= CARRY_MAX_BATCHES


def test_quality_weighting_scales_the_share():
    weights = {hk: (2.0 if hk == "hk000" else 1.0) for hk in HOTKEYS}
    got, _ = run(ticks=200, n_batches=80, weight_of=weights.get)
    others = st.median(v for hk, v in got.items() if hk != "hk000")
    assert 1.7 <= got["hk000"] / others <= 2.3


def test_the_cap_bounds_the_best_miner():
    weights = {hk: (10.0 if hk == "hk000" else 1.0) for hk in HOTKEYS}
    got, _ = run(ticks=200, n_batches=80, weight_of=weights.get, cap=2.0)
    others = st.median(v for hk, v in got.items() if hk != "hk000")
    assert got["hk000"] / others <= 2.6


def test_without_weights_it_is_plain_even_dispatch():
    weighted, _ = run(ticks=60, n_batches=80, weight_of=lambda hk: 1.0)
    plain, _ = run(ticks=60, n_batches=80)
    assert weighted == plain


def test_a_broken_weight_does_not_break_dispatch():
    def explode(hk):
        raise RuntimeError("no reputation")
    got, _ = run(ticks=20, n_batches=80, weight_of=explode)
    assert sum(got.values()) == 1600


def test_idle_miners_keep_a_slice_so_they_can_be_re_measured():
    working = set(HOTKEYS[:200])
    got, _ = run(ticks=60, n_batches=80, eligible=working.__contains__, floor_frac=0.05)
    idle = [v for hk, v in got.items() if hk not in working]
    assert min(idle) > 0
    assert sum(idle) < 0.1 * sum(got.values())


def test_with_no_floor_idle_miners_get_nothing():
    working = set(HOTKEYS[:200])
    got, _ = run(ticks=20, n_batches=80, eligible=working.__contains__)
    assert all(v == 0 for hk, v in got.items() if hk not in working)


def test_when_nobody_qualifies_the_work_still_goes_out():
    got, _ = run(ticks=5, n_batches=80, eligible=lambda hk: False, floor_frac=0.05)
    assert sum(got.values()) == 400


def test_priority_does_not_track_uid():
    """Ties break on hotkey, so a low UID must not be served first every tick."""
    tracker = Tracker()
    firsts = []
    for _ in range(20):
        assigned = credit_select(UIDS, HOTKEYS, tracker, 5)
        firsts.append(assigned[0][0] if assigned else None)
    assert len(set(firsts)) > 1


def test_the_same_state_gives_the_same_answer():
    a = credit_select(UIDS, HOTKEYS, Tracker(), 40)
    b = credit_select(UIDS, HOTKEYS, Tracker(), 40)
    assert a == b


@pytest.mark.parametrize("limit", [1, 2, 4])
def test_a_miner_is_never_given_more_than_it_can_hold(limit):
    tracker = Tracker(limit=limit)
    for _ in range(20):
        counts = {}
        for uid, _bi in credit_select(UIDS, HOTKEYS, tracker, 400):
            counts[uid] = counts.get(uid, 0) + 1
        assert max(counts.values(), default=0) <= limit
