"""A channel with little data reads as its published default, at full weight."""

import math

import pytest

from alpharidge_ai.mechanism import channels as ch
from alpharidge_ai.mechanism import profile as mp
from alpharidge_ai.validator import reputation as rep
from alpharidge_ai.validator.reputation_store import ReputationStore
from tests.test_profile_client import valid

WEIGHTS = {ch.AUDIT: 3.0, ch.TRIAGE: 1.0, ch.FLOOR: 0.5, ch.KEEPER: 0.0,
           ch.GRADED: 0.0, ch.LEGACY: 0.0}
DEFAULTS = {ch.AUDIT: 0.34, ch.TRIAGE: 0.97, ch.FLOOR: 0.93}
STOCK = (3 * 0.34 + 0.97 + 0.5 * 0.93) / 4.5


def gate(r, mid=0.456, gain=44.7):
    return 1.0 / (1.0 + math.exp(-gain * (r - mid)))


def test_a_miner_with_no_data_reads_as_the_defaults():
    assert ch.combine({}, WEIGHTS, 0.5, None, DEFAULTS) == pytest.approx(STOCK)


def test_a_thin_history_no_longer_crushes_an_honest_miner():
    chans = {ch.AUDIT: {"r": 0.25, "n": 7},
             ch.TRIAGE: {"r": 0.97, "n": 2},
             ch.FLOOR: {"r": 0.95, "n": 3}}
    without = ch.combine(chans, WEIGHTS, 0.5)
    with_defaults = ch.combine(chans, WEIGHTS, 0.5, None, DEFAULTS)
    assert gate(without) < 0.05
    assert gate(with_defaults) > 0.9


def test_a_warm_channel_reads_as_its_own_measurement():
    chans = {name: {"r": 0.1, "n": 10_000} for name in DEFAULTS}
    assert ch.combine(chans, WEIGHTS, 0.5, None, DEFAULTS) == pytest.approx(0.1)


def test_the_blend_moves_with_the_observation_count():
    half = ch.WARMUP // 2
    chans = {ch.AUDIT: {"r": 0.0, "n": half}}
    value = ch.combine(chans, {ch.AUDIT: 1.0}, 0.5, None, {ch.AUDIT: 0.4})
    ramp = half / ch.WARMUP
    assert value == pytest.approx((1 - ramp) * 0.4)


def test_a_channel_without_a_default_is_phased_in_as_before():
    chans = {ch.AUDIT: {"r": 0.0, "n": ch.WARMUP}, ch.TRIAGE: {"r": 1.0, "n": 1}}
    weights = {ch.AUDIT: 1.0, ch.TRIAGE: 1.0}
    expected = ch.combine(chans, weights, 0.5)
    assert ch.combine(chans, weights, 0.5, None, {}) == pytest.approx(expected)


def test_zero_weight_channels_are_ignored_even_with_a_default():
    value = ch.combine({}, {ch.AUDIT: 1.0, ch.KEEPER: 0.0}, 0.5, None,
                       {ch.AUDIT: 0.3, ch.KEEPER: 1.0})
    assert value == pytest.approx(0.3)


def test_the_store_applies_defaults_and_rederives(tmp_path):
    store = ReputationStore(path=tmp_path / "rep.json")
    store.set_channel_weights(WEIGHTS)
    store.record_local(1, "me", "m", 1, 0.0, 1.0, channel=ch.AUDIT)
    store.finalize(1)
    before = store.reputation("m")
    store.set_channel_weights(WEIGHTS, None, DEFAULTS)
    assert store.reputation("m") > before
    assert store.reputation("never-seen") == pytest.approx(STOCK)


def test_without_defaults_an_unseen_hotkey_reads_the_prior(tmp_path):
    from alpharidge_ai.validator.reputation_store import _prior
    store = ReputationStore(path=tmp_path / "rep.json")
    assert store.reputation("never-seen") == pytest.approx(_prior())


# ---- the profile ------------------------------------------------------------------

def _raw(version):
    raw = valid()
    raw["schema_version"] = version
    raw["emission"]["channel_defaults"] = dict(DEFAULTS)
    return raw


def test_defaults_need_schema_1_4():
    for version in ("1.2.0", "1.3.0"):
        with pytest.raises(mp.ProfileError):
            mp.parse(_raw(version))
    assert mp.parse(_raw("1.4.0")).emission.defaults() == DEFAULTS


def test_schema_1_4_still_accepts_the_1_3_fields():
    raw = _raw("1.4.0")
    raw["emission"]["channel_weights"] = {ch.AUDIT: 3.0}
    raw["oracle"]["grader_models"][0]["scale"] = 0.9
    assert mp.parse(raw)


def test_no_defaults_means_none_are_applied():
    assert mp.parse(valid()).emission.defaults() == {}


@pytest.mark.parametrize("bad", [{"nonsense": 0.5}, {ch.AUDIT: -0.1}, {ch.AUDIT: 1.1},
                                 [0.5]])
def test_bad_defaults_are_rejected(bad):
    raw = _raw("1.4.0")
    raw["emission"]["channel_defaults"] = bad
    with pytest.raises(mp.ProfileError):
        mp.parse(raw)
