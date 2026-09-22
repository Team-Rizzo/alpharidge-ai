"""Speed as a dispatch weight, paid only while quality keeps up with the field."""

import math
import random
import types

import pytest

from alpharidge_ai import config
from alpharidge_ai.utils.cooldown import MinerCooldownTracker
from alpharidge_ai.utils.dispatch import SPEED_SPAN, SPEED_WINDOW_S, speed_weights
from neurons.validator import Validator

NOW = 1_800_000_000.0
SPA, Q, SD = 2.0, 0.5, 0.3


def field(n=40, fast=None, slow=None, worse=None, drop=0.2, audits=100, seed=3):
    """n miners at SPA s/article and quality ~Q; `fast`/`slow`/`worse` are overrides."""
    rng = random.Random(seed)
    t = MinerCooldownTracker()
    hks = [f"hk{i:02d}" for i in range(n)]
    for i, hk in enumerate(hks):
        spa = SPA
        if fast and hk in fast:
            spa = SPA / fast[hk]
        if slow and hk in slow:
            spa = SPA * slow[hk]
        for k in range(20):
            t.record_speed(hk, spa * 24, 24, when=NOW - 3600 * k)
        base = Q - (drop if worse and hk in worse else 0.0)
        for k in range(audits):
            t.record_quality(hk, min(1.0, max(0.0, rng.gauss(base, SD))),
                             when=NOW - 600 * k)
    return t, hks


def test_off_means_no_weights():
    t, hks = field()
    assert speed_weights(t, hks, 0.0, 3.0, NOW) == {}


def test_faster_earns_more_within_the_band():
    t, hks = field(fast={"hk00": 2.0}, slow={"hk01": 2.0})
    w = speed_weights(t, hks, 0.10, 3.0, NOW)
    assert w["hk00"] == pytest.approx(1.10)
    assert w["hk01"] == pytest.approx(0.90)
    assert w["hk05"] == pytest.approx(1.0)


def test_a_small_edge_earns_a_small_bonus():
    t, hks = field(fast={"hk00": 1.05})
    w = speed_weights(t, hks, 0.10, 3.0, NOW)
    assert w["hk00"] == pytest.approx(1 + 0.10 * math.log(1.05) / SPEED_SPAN)
    assert w["hk00"] < 1.02


def test_fast_but_worse_is_held_at_the_bottom():
    t, hks = field(fast={"hk00": 2.0}, worse={"hk00"}, drop=0.25)
    w = speed_weights(t, hks, 0.10, 3.0, NOW)
    assert w["hk00"] == pytest.approx(0.90)
    assert t.gated_out("hk00")


def test_honest_miners_pass_the_gate():
    t, hks = field(n=60)
    speed_weights(t, hks, 0.10, 3.0, NOW)
    assert sum(t.gated_out(hk) for hk in hks) <= 1


def test_the_gate_reopens_only_with_margin():
    t, hks = field(fast={"hk00": 2.0})
    t.set_gated_out("hk00", True)
    t._quality["hk00"].clear()
    for k in range(100):
        t.record_quality("hk00", Q - SD * 2.5 / 10, when=NOW - 600 * k)   # z ~ -2.5
    assert speed_weights(t, hks, 0.10, 3.0, NOW)["hk00"] == pytest.approx(0.90)
    t._quality["hk00"].clear()
    for k in range(100):
        t.record_quality("hk00", Q, when=NOW - 600 * k)
    assert speed_weights(t, hks, 0.10, 3.0, NOW)["hk00"] == pytest.approx(1.10)
    assert not t.gated_out("hk00")


def test_too_little_history_is_neutral():
    t, hks = field(audits=20)
    assert speed_weights(t, hks, 0.10, 3.0, NOW) == {}


def test_old_samples_do_not_count():
    t, hks = field()
    t._speed["hk00"].clear()
    for k in range(20):
        t.record_speed("hk00", 1.0 * 24, 24, when=NOW - SPEED_WINDOW_S - 60 - k)
    assert "hk00" not in speed_weights(t, hks, 0.10, 3.0, NOW)


def test_small_batches_are_not_timed():
    t = MinerCooldownTracker()
    t.record_speed("hk", 3.0, 3, when=NOW)
    t.record_speed("hk", None, 24, when=NOW)
    assert t.speed_samples("hk", 0) == []


def test_samples_and_gate_survive_a_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DISPATCH_STATE_LOCATION", str(tmp_path / "d.json"),
                        raising=False)
    t, hks = field(n=3)
    t.set_gated_out("hk01", True)
    t.save()
    again = MinerCooldownTracker()
    again.load()
    assert again.speed_samples("hk00", 0) == t.speed_samples("hk00", 0)
    assert again.quality_samples("hk02", 0) == t.quality_samples("hk02", 0)
    assert again.gated_out("hk01") and not again.gated_out("hk00")


def test_departed_miners_are_forgotten():
    t, hks = field(n=3)
    t.set_gated_out("hk01", True)
    t.prune({"hk00"})
    assert t.speed_samples("hk01", 0) == [] and not t.gated_out("hk01")


# ---- the validator side ------------------------------------------------------------

@pytest.fixture
def restore_config():
    before = (config.DISPATCH_SPEED_BONUS, config.DISPATCH_QUALITY_GATE_Z)
    yield
    config.DISPATCH_SPEED_BONUS, config.DISPATCH_QUALITY_GATE_Z = before


def _validator(tracker, hks):
    v = types.SimpleNamespace(_article_cooldown=tracker,
                              metagraph=types.SimpleNamespace(hotkeys=hks))
    for name in ("_dispatch_eligible", "_credit_args"):
        setattr(v, name, types.MethodType(getattr(Validator, name), v))
    return v


def test_the_bonus_is_off_by_default():
    assert config.DISPATCH_SPEED_BONUS == 0.0
    assert "DISPATCH_SPEED_BONUS" in config._REMOTE_CONFIG_KEYS
    assert "DISPATCH_SPEED_BONUS" not in config._CONSENSUS_KEYS


def test_credit_args_carry_the_weight_only_when_on(restore_config, monkeypatch):
    t, hks = field(fast={"hk00": 2.0})
    v = _validator(t, hks)
    config.DISPATCH_SPEED_BONUS = 0.0
    assert "weight_of" not in v._credit_args(1)
    config.DISPATCH_SPEED_BONUS = 0.10
    monkeypatch.setattr("neurons.validator.time.time", lambda: NOW)
    monkeypatch.setattr("neurons.validator.bt.logging.info", lambda m: None)
    weight_of = v._credit_args(1)["weight_of"]
    assert weight_of("hk00") == pytest.approx(1.10)
    assert weight_of("never-seen") == 1.0


def test_audited_pool_scores_feed_the_gate():
    from alpharidge_ai.oracle.runner import Observation
    fake = types.SimpleNamespace(_article_cooldown=MinerCooldownTracker())
    obs = [Observation(article_id=1, score=0.3, weight=1.0, path="pool", score_v2=0.5),
           Observation(article_id=2, score=0.6, weight=0.3, path="keeper")]
    Validator._log_audit(fake, "hk", obs, live=False)
    assert fake._article_cooldown.quality_samples("hk", 0) == [0.5]
