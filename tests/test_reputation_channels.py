"""Reputation is a fixed-weight mix of separate channels."""

import json
import random

import pytest

from alpharidge_ai.mechanism import channels as ch
from alpharidge_ai.mechanism import profile as mp
from alpharidge_ai.validator.reputation_store import ReputationStore
from tests.test_profile_client import valid

WEIGHTS = {ch.TRIAGE: 1.0, ch.AUDIT: 3.0, ch.FLOOR: 0.0, ch.KEEPER: 0.0,
           ch.GRADED: 0.0, ch.LEGACY: 0.0}


def _store(tmp_path, weights=WEIGHTS):
    store = ReputationStore(path=tmp_path / "rep.json")
    store.set_channel_weights(weights)
    return store


def _rec(store, epoch, aid, score, channel, target="m", sender="me", weight=1.0):
    store.record_local(epoch, sender, target, aid, score, weight, channel=channel)


def test_channels_on_the_same_article_do_not_overwrite_each_other(tmp_path):
    store = _store(tmp_path)
    _rec(store, 1, 7, 1.0, ch.TRIAGE)
    _rec(store, 1, 7, 0.0, ch.AUDIT)
    store.finalize(1, alpha=0.5)
    c = store.state["m"]["c"]
    assert set(c) == {ch.TRIAGE, ch.AUDIT}
    assert c[ch.TRIAGE]["r"] > c[ch.AUDIT]["r"]


def test_the_mix_does_not_depend_on_how_many_observations_each_channel_makes(tmp_path):
    store = _store(tmp_path)
    epoch = 1
    for i in range(400):
        _rec(store, epoch, i, 1.0, ch.TRIAGE)
    for i in range(40):
        _rec(store, epoch, 10_000 + i, 0.0, ch.AUDIT)
    store.finalize(epoch, alpha=0.5)
    # Ten times as many triage observations, yet audit carries its published share.
    assert store.reputation("m") == pytest.approx(1.0 * 1 / 4 + 0.0 * 3 / 4, abs=1e-6)


def test_a_young_channel_is_phased_in(tmp_path):
    store = _store(tmp_path)
    for i in range(200):
        _rec(store, 1, i, 1.0, ch.TRIAGE)
    _rec(store, 1, 9_999, 0.0, ch.AUDIT)
    store.finalize(1, alpha=1.0)
    w_audit = 3.0 * 1 / ch.WARMUP
    assert store.reputation("m") == pytest.approx(1.0 / (1.0 + w_audit))


def test_samples_count_every_channel(tmp_path):
    store = _store(tmp_path)
    _rec(store, 1, 1, 1.0, ch.TRIAGE)
    _rec(store, 1, 1, 1.0, ch.AUDIT)
    _rec(store, 1, 2, 1.0, ch.FLOOR)
    store.finalize(1)
    assert store.samples("m") == 3


def test_peer_rows_carry_their_channel(tmp_path):
    store = _store(tmp_path)
    store.ingest("peer", 1, {"m": [
        [1, 0.2, 1.0, ch.code_of(ch.AUDIT)],
        [2, 0.9, 1.0],                       # legacy row
        [3, 0.5, 1.0, 99],                   # unknown channel
        [4, 0.5, 1.0, 1.5],                  # not a channel code
    ]}, seq=1)
    store.finalize(1)
    assert set(store.state["m"]["c"]) == {ch.AUDIT, ch.LEGACY}
    assert store.samples("m") == 2


def test_rows_we_send_are_readable_by_an_older_peer(tmp_path):
    store = _store(tmp_path)
    _rec(store, 1, 5, 0.4, ch.TRIAGE, weight=2.0)
    (row,) = store.export(1, "me")["m"]
    assert [float(x) for x in row][:3] == [5.0, 0.4, 2.0]
    assert ch.name_of(row[3]) == ch.TRIAGE


def test_rows_fit_the_broadcast_message():
    from alpharidge_ai.protocol import ValidatorReputationObs
    store = ReputationStore()
    store.record_local(1, "me", "m", 5, 0.4, 1.0, channel=ch.KEEPER)
    msg = ValidatorReputationObs(epoch=1, observations=store.export(1, "me"),
                                 sender_hotkey="me", seq=1)
    assert ch.name_of(msg.observations["m"][0][3]) == ch.KEEPER


def test_state_from_before_channels_carries_on_unchanged(tmp_path):
    path = tmp_path / "rep.json"
    path.write_text(json.dumps({"state": {"m": {"r": 0.83, "n": 250, "e": 9}},
                                "finalized": [], "obs": {}, "last_seen_seq": {}}))
    store = ReputationStore(path=path)
    store.load()
    assert store.reputation("m") == 0.83
    assert store.state["m"]["c"] == {ch.LEGACY: {"r": 0.83, "n": 250}}
    store.set_channel_weights(dict(ch.DEFAULT_WEIGHTS, **{ch.TRIAGE: 1.0}))
    assert store.reputation("m") == pytest.approx(0.83)


def test_retiring_the_legacy_channel_removes_its_influence(tmp_path):
    path = tmp_path / "rep.json"
    path.write_text(json.dumps({"state": {"m": {"r": 0.2, "n": 250}},
                                "finalized": [], "obs": {}, "last_seen_seq": {}}))
    store = ReputationStore(path=path)
    store.load()
    for i in range(100):
        store.record_local(1, "me", "m", i, 1.0, 1.0, channel=ch.TRIAGE)
    store.finalize(1, alpha=1.0)
    assert store.reputation("m") < 1.0
    store.set_channel_weights(dict(ch.DEFAULT_WEIGHTS, **{ch.LEGACY: 0.0}))
    assert store.reputation("m") == pytest.approx(1.0)


def test_channels_survive_a_restart(tmp_path):
    store = _store(tmp_path)
    _rec(store, 1, 1, 0.3, ch.AUDIT)
    _rec(store, 2, 2, 0.7, ch.TRIAGE)
    store.finalize(1)
    store.save()
    again = ReputationStore(path=tmp_path / "rep.json")
    again.load()
    again.set_channel_weights(WEIGHTS)
    assert again.state["m"]["c"] == store.state["m"]["c"]
    again.finalize(2)
    assert set(again.state["m"]["c"]) == {ch.AUDIT, ch.TRIAGE}


def test_arrival_order_does_not_change_the_result(tmp_path):
    rows = [(i, random.Random(i).random(), 1.0,
             ch.code_of([ch.TRIAGE, ch.AUDIT, ch.FLOOR][i % 3])) for i in range(60)]
    results = []
    for seed in range(3):
        store = _store(tmp_path / str(seed))
        (tmp_path / str(seed)).mkdir()
        shuffled = rows[:]
        random.Random(seed).shuffle(shuffled)
        store.ingest("a", 1, {"m": shuffled[:30]}, seq=1)
        store.ingest("b", 1, {"m": shuffled[30:]}, seq=1)
        store.finalize(1)
        results.append(store.state["m"])
    assert results[0] == results[1] == results[2]


def test_new_weights_rederive_every_reputation(tmp_path):
    store = _store(tmp_path)
    for i in range(50):
        _rec(store, 1, i, 1.0, ch.TRIAGE)
        _rec(store, 1, i, 0.0, ch.AUDIT)
    store.finalize(1, alpha=1.0)
    before = store.reputation("m")
    store.set_channel_weights(dict(WEIGHTS, **{ch.AUDIT: 1.0}))
    assert store.reputation("m") > before


# ---- the profile ------------------------------------------------------------------

def test_profile_defaults_the_weights():
    assert mp.parse(valid()).emission.weights() == ch.DEFAULT_WEIGHTS


def test_profile_overrides_named_weights_only():
    raw = valid()
    raw["emission"]["channel_weights"] = {ch.AUDIT: 4.0, ch.LEGACY: 0.0}
    weights = mp.parse(raw).emission.weights()
    assert weights[ch.AUDIT] == 4.0 and weights[ch.LEGACY] == 0.0
    assert weights[ch.TRIAGE] == ch.DEFAULT_WEIGHTS[ch.TRIAGE]


@pytest.mark.parametrize("bad", [
    {"nonsense": 1.0},
    {ch.AUDIT: -1.0},
    {ch.AUDIT: 101.0},
    {name: 0.0 for name in ch.CHANNELS},
    [1, 2],
])
def test_profile_rejects_bad_weights(bad):
    raw = valid()
    raw["emission"]["channel_weights"] = bad
    with pytest.raises(mp.ProfileError):
        mp.parse(raw)


# ---- the validator routes each source to its channel -------------------------------

def test_audit_paths_go_to_their_own_channels():
    import neurons.validator as vm
    from alpharidge_ai.oracle.runner import Observation

    recorded = []

    class Fake:
        _log_audit = vm.Validator._log_audit

        def _record_observations(self, hotkey, observations, channel):
            recorded.extend((channel, o) for o in observations)

    obs = [Observation(1, 0.4, 1.0, "pool"), Observation(2, 0.9, 0.5, "keeper")]
    Fake()._log_audit("hk-0123456789ab", obs, live=True)
    assert (ch.AUDIT, (1, 0.4, 1.0)) in recorded
    assert (ch.KEEPER, (2, 0.9, 0.5)) in recorded
    recorded.clear()
    Fake()._log_audit("hk-0123456789ab", obs, live=False)
    assert not recorded


# ---- per-channel steps --------------------------------------------------------------

def test_a_channel_can_move_more_slowly(tmp_path):
    store = _store(tmp_path)
    store.set_channel_weights(WEIGHTS, {ch.AUDIT: 0.1, ch.TRIAGE: 0.5})
    _rec(store, 1, 1, 1.0, ch.TRIAGE)
    _rec(store, 1, 1, 1.0, ch.AUDIT)
    store.finalize(1, alpha=0.5)
    c = store.state["m"]["c"]
    from alpharidge_ai.validator.reputation_store import _prior
    prior = _prior()
    assert c[ch.TRIAGE]["r"] == pytest.approx(prior + 0.5 * (1.0 - prior))
    assert c[ch.AUDIT]["r"] == pytest.approx(prior + 0.1 * (1.0 - prior))


def test_unnamed_channels_use_the_epoch_alpha(tmp_path):
    fast = _store(tmp_path / "a")
    slow = _store(tmp_path / "b")
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    slow.set_channel_weights(WEIGHTS, {ch.AUDIT: 0.01})
    for store in (fast, slow):
        _rec(store, 1, 1, 1.0, ch.TRIAGE)
        store.finalize(1, alpha=0.5)
    assert fast.state["m"]["c"][ch.TRIAGE] == slow.state["m"]["c"][ch.TRIAGE]


def test_a_slower_channel_is_phased_in_over_its_own_half_life():
    assert ch.warmup(0.03) == ch.WARMUP
    assert ch.warmup(0.015) > ch.WARMUP
    chans = {ch.TRIAGE: {"r": 1.0, "n": 1000}, ch.AUDIT: {"r": 0.0, "n": ch.WARMUP}}
    weights = {ch.TRIAGE: 1.0, ch.AUDIT: 1.0}
    fast = ch.combine(chans, weights, 0.5)
    slow = ch.combine(chans, weights, 0.5, {ch.AUDIT: 0.015})
    assert slow > fast


def test_changing_the_steps_rederives_reputation(tmp_path):
    store = _store(tmp_path)
    for i in range(30):
        _rec(store, 1, i, 1.0, ch.TRIAGE)
        _rec(store, 1, i, 0.0, ch.AUDIT)
    store.finalize(1, alpha=0.5)
    before = store.reputation("m")
    store.set_channel_weights(WEIGHTS, {ch.AUDIT: 0.001})
    assert store.reputation("m") != before


def test_profile_reads_channel_alphas():
    raw = valid()
    raw["emission"]["channel_alphas"] = {ch.AUDIT: 0.015}
    alphas = mp.parse(raw).emission.alphas()
    assert alphas[ch.AUDIT] == 0.015
    assert alphas[ch.TRIAGE] == raw["emission"]["ema_alpha"]
    assert mp.parse(valid()).emission.alphas()[ch.AUDIT] == valid()["emission"]["ema_alpha"]


@pytest.mark.parametrize("bad", [{"nonsense": 0.1}, {ch.AUDIT: 0.0}, {ch.AUDIT: 1.5}, [0.1]])
def test_profile_rejects_bad_channel_alphas(bad):
    raw = valid()
    raw["emission"]["channel_alphas"] = bad
    with pytest.raises(mp.ProfileError):
        mp.parse(raw)
