"""The audit_v2 channel: the rematch rule, its profile gating, and state it must not lose."""

import types

import pytest

from alpharidge_ai.mechanism import channels as ch
from alpharidge_ai.mechanism import profile as mp
from alpharidge_ai.models.article_intelligence import NumericClaim
from alpharidge_ai.oracle import audit
from alpharidge_ai.oracle.audit import Adjudication
from alpharidge_ai.oracle.runner import Observation
from alpharidge_ai.utils.cooldown import MinerCooldownTracker
from alpharidge_ai.validator.reputation_store import ReputationStore
from tests.test_profile_client import valid


def claim(name, value, unit="USD", confidence=0.9):
    return NumericClaim(metric_name=name, value=value, unit=unit, confidence=confidence)


GOLD = [claim("quarterly revenue", 1.2e9), claim("operating margin", 12.0, "percent")]


def first(gold=GOLD, valid_keys=()):
    return Adjudication(grader_keys={("g", j) for j in range(len(gold))},
                        valid=set(valid_keys))


# ---- matching ---------------------------------------------------------------------

def test_the_exact_rule_is_unchanged():
    assert audit.claims_match(claim("Quarterly  Revenue", 1.2e9), GOLD[0])
    assert not audit.claims_match(claim("revenue", 1.2e9), GOLD[0])
    assert not audit.claims_match(claim("quarterly revenue", 1.3e9), GOLD[0])


def test_a_differently_worded_name_now_matches():
    got = audit.rematch([claim("revenue in the quarter", 1.2e9)], GOLD, first(), {0})
    assert got.miner_keys == [("g", 0)]


def test_the_unit_class_still_gates():
    got = audit.rematch([claim("operating margin", 12.0, "USD")], GOLD, first(), {0})
    assert got.miner_keys == [("m", 0)]


def test_an_unrelated_name_does_not_match():
    got = audit.rematch([claim("said on tuesday", 1.2e9)], GOLD, first(), {0})
    assert got.miner_keys == [("m", 0)]


def test_a_name_stuffed_with_the_sentence_does_not_match():
    stuffed = claim("the company said on tuesday that quarterly revenue for the period "
                    "rose sharply compared with analyst expectations and last year", 1.2e9)
    got = audit.rematch([stuffed], GOLD, first(), {0})
    assert got.miner_keys == [("m", 0)]


def test_accents_and_case_do_not_matter():
    a = types.SimpleNamespace(metric_name="Précio Médio")
    b = types.SimpleNamespace(metric_name="preco medio")
    assert audit.names_overlap(a, b)


def test_exact_names_are_settled_before_loose_ones():
    gold = [claim("revenue", 5.0), claim("net revenue", 5.0)]
    subs = [claim("net revenue growth", 5.0), claim("revenue", 5.0)]
    got = audit.rematch(subs, gold, first(gold), {0, 1})
    assert got.miner_keys == [("g", 1), ("g", 0)]


def test_only_gold_the_article_states_can_be_matched():
    held = Adjudication(grader_keys={("g", 1)})
    got = audit.rematch([claim("quarterly revenue", 1.2e9)], GOLD, held, {0})
    assert got.miner_keys == [("m", 0)]
    assert got.grader_keys == {("g", 1)}


# ---- extras -----------------------------------------------------------------------

def test_verified_extras_are_credited_up_to_the_budget():
    extras = [claim(f"figure {i}", 100.0 + i) for i in range(6)]
    got = audit.rematch(extras, GOLD, first(), set(range(6)))
    credited = sorted(i for _, i in got.valid)
    assert credited == [0, 1, 2]


def test_the_budget_grows_with_the_reference():
    gold = [claim(f"gold {j}", float(j + 1)) for j in range(5)]
    extras = [claim(f"figure {i}", 100.0 + i) for i in range(7)]
    got = audit.rematch(extras, gold, first(gold), set(range(7)))
    assert len(got.valid) == 5


def test_adjudicated_claims_are_reused_not_re_asked():
    got = audit.rematch([claim("figure", 7.0)], GOLD, first(valid_keys={("m", 0)}), set())
    assert ("m", 0) in got.valid
    assert got.residual == [0]


def test_an_unverified_claim_stays_unsupported():
    got = audit.rematch([claim("figure", 7.0)], GOLD, first(), set())
    assert got.valid == set()


def test_rematch_keeps_every_first_pass_match(monkeypatch):
    monkeypatch.setattr(audit, "grounded_grader_claims", lambda claims, text: {0, 1})
    subs = [claim("quarterly revenue", 1.2e9), claim("margin, operating", 12.0, "percent")]
    one = audit.adjudicate(subs, GOLD, {0, 1}, "", None)
    two = audit.rematch(subs, GOLD, one, {0, 1})
    assert one.miner_keys[0] == ("g", 0)
    firsts = {k for k in one.miner_keys if k[0] == "g"}
    assert firsts <= {k for k in two.miner_keys if k[0] == "g"}


# ---- the profile ------------------------------------------------------------------

def _raw(version, **emission):
    raw = valid()
    raw["schema_version"] = version
    raw["emission"].update(emission)
    return raw


@pytest.mark.parametrize("field", ["channel_weights", "channel_alphas", "channel_defaults"])
def test_the_channel_needs_schema_1_5(field):
    value = {"channel_weights": 1.0, "channel_alphas": 0.03, "channel_defaults": 0.4}[field]
    with pytest.raises(mp.ProfileError, match="1.5.0"):
        mp.parse(_raw("1.4.0", **{field: {ch.AUDIT_V2: value}}))
    assert mp.parse(_raw("1.5.0", **{field: {ch.AUDIT_V2: value}}))


def test_the_scale_needs_schema_1_5():
    raw = _raw("1.4.0")
    raw["oracle"]["grader_models"][0]["audit_v2_scale"] = 0.8
    with pytest.raises(mp.ProfileError):
        mp.parse(raw)
    raw["schema_version"] = "1.5.0"
    assert mp.parse(raw).oracle.grader_models[0].audit_v2_scale == 0.8


def test_the_scale_defaults_to_the_audit_scale():
    raw = _raw("1.5.0")
    raw["oracle"]["grader_models"][0]["scale"] = 0.9
    assert mp.parse(raw).oracle.grader_models[0].audit_v2_scale == 0.9


def test_an_older_profile_leaves_the_channel_unweighted():
    assert mp.parse(_raw("1.4.0")).emission.weights()[ch.AUDIT_V2] == 0.0


def test_recording_starts_with_schema_1_5():
    assert not mp.records(None, ch.AUDIT_V2)
    assert not mp.records(mp.parse(_raw("1.4.0")), ch.AUDIT_V2)
    assert mp.records(mp.parse(_raw("1.5.0")), ch.AUDIT_V2)
    assert mp.records(mp.parse(_raw("1.4.0")), ch.AUDIT)


# ---- the store --------------------------------------------------------------------

def test_the_channel_travels_and_applies(tmp_path):
    store = ReputationStore(path=tmp_path / "rep.json")
    store.record_local(10, "me", "hk", 1, 0.6, 1.0, channel=ch.AUDIT_V2)
    wire = ReputationStore.wire_payload(store.export(10, "me"))
    peer = ReputationStore(path=tmp_path / "peer.json")
    assert peer.ingest("me", 10, wire, seq=10)[0]
    peer.finalize(10)
    assert peer.state["hk"]["c"][ch.AUDIT_V2]["n"] == 1


def test_at_zero_weight_it_does_not_move_reputation(tmp_path):
    store = ReputationStore(path=tmp_path / "rep.json")
    store.set_channel_weights({ch.AUDIT: 1.0, ch.AUDIT_V2: 0.0})
    store.record_local(10, "me", "hk", 1, 0.2, 1.0, channel=ch.AUDIT)
    store.finalize(10)
    before = store.reputation("hk")
    store.record_local(11, "me", "hk", 2, 1.0, 1.0, channel=ch.AUDIT_V2)
    store.finalize(11)
    assert store.reputation("hk") == pytest.approx(before)


def test_swapping_weights_reads_the_new_channel(tmp_path):
    store = ReputationStore(path=tmp_path / "rep.json")
    for epoch in range(10, 60):
        store.record_local(epoch, "me", "hk", epoch, 0.2, 1.0, channel=ch.AUDIT)
        store.record_local(epoch, "me", "hk", epoch, 0.5, 1.0, channel=ch.AUDIT_V2)
        store.finalize(epoch)
    store.set_channel_weights({ch.AUDIT: 0.0, ch.AUDIT_V2: 3.0})
    assert store.reputation("hk") == pytest.approx(0.5)
    store.set_channel_weights({ch.AUDIT: 3.0, ch.AUDIT_V2: 0.0})
    assert store.reputation("hk") == pytest.approx(0.2)


def test_state_for_an_unknown_channel_survives_a_load_and_save(tmp_path):
    path = tmp_path / "rep.json"
    path.write_text(
        '{"state": {"hk": {"r": 0.5, "n": 3, "e": 9, "c": {'
        '"audit": {"r": 0.4, "n": 2, "m": 0.2, "d": 0.5}, '
        '"later_channel": {"r": 0.7, "n": 1, "m": 0.7, "d": 1.0}}}}, '
        '"finalized": [], "obs": {}, "last_seen_seq": {}}')
    store = ReputationStore(path=path)
    store.load()
    store.save()
    again = ReputationStore(path=path)
    again.load()
    assert again.state["hk"]["c"]["later_channel"]["r"] == 0.7
    assert again.state["hk"]["c"]["audit"]["r"] == 0.4


def test_malformed_channel_state_is_skipped_not_fatal(tmp_path):
    path = tmp_path / "rep.json"
    path.write_text('{"state": {"hk": {"r": 0.5, "n": 3, "c": {'
                    '"audit": {"r": 0.4, "n": 2}, "junk": "x"}}}}')
    store = ReputationStore(path=path)
    store.load()
    assert set(store.state["hk"]["c"]) == {"audit"}


# ---- the validator ----------------------------------------------------------------

def _validator(schema):
    from neurons.validator import Validator
    recorded = []
    fake = types.SimpleNamespace(
        block=100,
        _mechanism_profile=types.SimpleNamespace(
            resolve=lambda block: mp.parse(_raw(schema))),
        _record_observations=lambda hk, obs, channel: recorded.append((channel, obs)),
        _article_cooldown=MinerCooldownTracker())
    return Validator, fake, recorded


OBS = [Observation(article_id=1, score=0.3, weight=1.0, path="pool", score_v2=0.5),
       Observation(article_id=2, score=0.6, weight=0.3, path="keeper")]


def test_the_validator_records_the_channel_under_1_5():
    Validator, fake, recorded = _validator("1.5.0")
    Validator._log_audit(fake, "hk", OBS, live=True)
    by = dict(recorded)
    assert by[ch.AUDIT_V2] == [(1, 0.5, 1.0)]
    assert by[ch.AUDIT] == [(1, 0.3, 1.0)]
    assert by[ch.KEEPER] == [(2, 0.6, 0.3)]


def test_the_validator_does_not_record_it_earlier():
    Validator, fake, recorded = _validator("1.4.0")
    Validator._log_audit(fake, "hk", OBS, live=True)
    assert ch.AUDIT_V2 not in dict(recorded)


def test_nothing_is_recorded_in_shadow():
    Validator, fake, recorded = _validator("1.5.0")
    Validator._log_audit(fake, "hk", OBS, live=False)
    assert recorded == []
