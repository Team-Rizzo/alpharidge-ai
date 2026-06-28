"""Unit tests for the adaptive-dispatch liveness roster."""

from alpharidge_ai.utils.liveness import LivenessRoster


class FakeClock:
    """Manually-advanced monotonic clock."""
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


class FakeMetagraph:
    def __init__(self, hotkeys):
        self.hotkeys = hotkeys


def test_mark_seen_alive_within_ttl_then_ages_out():
    clk = FakeClock()
    r = LivenessRoster(clock=clk, ttl_s=120)
    r.mark_seen("hk1")
    assert r.is_alive("hk1")

    clk.advance(119)
    assert r.is_alive("hk1")          # still within TTL

    clk.advance(2)                    # now 121 > 120
    assert not r.is_alive("hk1")      # aged out


def test_unseen_hotkey_is_not_alive():
    r = LivenessRoster(clock=FakeClock(), ttl_s=120)
    assert not r.is_alive("never-seen")


def test_empty_or_none_hotkey_safe():
    r = LivenessRoster(clock=FakeClock(), ttl_s=120)
    r.mark_seen("")          # no-op, must not raise
    r.mark_seen(None)        # no-op, must not raise
    assert not r.is_alive("")
    assert not r.is_alive(None)


def test_heartbeat_marks_alive_and_ages_out_if_sweep_stalls():
    clk = FakeClock()
    r = LivenessRoster(clock=clk, ttl_s=120)
    r.update_from_heartbeat(["a", "b", ""])   # "" ignored
    assert r.is_alive("a") and r.is_alive("b")

    # Sweep stalls (no further heartbeats): both age out after the TTL rather
    # than being pinned alive forever.
    clk.advance(121)
    assert not r.is_alive("a")
    assert not r.is_alive("b")


def test_pushback_refreshes_heartbeat_timestamp():
    clk = FakeClock()
    r = LivenessRoster(clock=clk, ttl_s=120)
    r.update_from_heartbeat(["a"])
    clk.advance(100)
    r.mark_seen("a")          # push-back refreshes
    clk.advance(100)          # 100 since refresh, < TTL
    assert r.is_alive("a")


def test_live_uids_maps_through_metagraph():
    clk = FakeClock()
    r = LivenessRoster(clock=clk, ttl_s=120)
    mg = FakeMetagraph(["hk0", "hk1", "hk2", "hk3"])
    r.mark_seen("hk1")
    r.mark_seen("hk3")
    assert r.live_uids(mg) == [1, 3]


def test_prune_drops_inactive_hotkeys():
    r = LivenessRoster(clock=FakeClock(), ttl_s=120)
    r.mark_seen("keep")
    r.mark_seen("drop")
    r.prune({"keep"})
    tracked, _ = r.stats()
    assert tracked == 1
    assert r.is_alive("keep")
    assert not r.is_alive("drop")


def test_stats_counts_tracked_and_live():
    clk = FakeClock()
    r = LivenessRoster(clock=clk, ttl_s=120)
    r.mark_seen("a")
    r.mark_seen("b")
    clk.advance(121)          # both stale now
    r.mark_seen("c")          # fresh
    tracked, live = r.stats()
    assert tracked == 3
    assert live == 1
