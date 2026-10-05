"""The dispatch-health alert."""
from alpharidge_ai.utils.dispatch_metrics import DispatchHealth

H = 3600.0


def _fill(h, hours, per_hour, start=0):
    for i in range(start, start + hours):
        h.note("batches", i * H + 1, per_hour)
        h.note("cycles", i * H + 1, 45)


def test_quiet_until_enough_history():
    h = DispatchHealth()
    _fill(h, 3, 300)
    assert h.check(4 * H + 5, 0.5) == []


def test_alerts_when_last_hour_drops_below_half():
    h = DispatchHealth()
    _fill(h, 8, 300)
    h.note("batches", 8 * H + 1, 100)
    h.note("cycles", 8 * H + 1, 45)
    msgs = h.check(9 * H + 5, 0.5)
    assert len(msgs) == 1 and "batches last hour 100 vs typical 300" in msgs[0]


def test_normal_hour_is_quiet():
    h = DispatchHealth()
    _fill(h, 9, 300)
    assert h.check(9 * H + 5, 0.5) == []


def test_repeats_at_most_every_half_hour():
    h = DispatchHealth()
    _fill(h, 8, 300)
    assert h.check(9 * H + 5, 0.5)            # hour 8 empty -> alert
    assert h.check(9 * H + 600, 0.5) == []
    assert h.check(9 * H + 1900, 0.5)


def test_old_buckets_are_dropped():
    h = DispatchHealth(history=24)
    _fill(h, 40, 10)
    assert len(h._buckets) <= 25
