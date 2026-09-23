"""Relevance premium: bounded by the field's keep rate and by the relevance audit."""

import random
import time
import types

import pytest

from alpharidge_ai import config
from alpharidge_ai.utils.cooldown import RELEVANCE_HALF_LIFE_S, MinerCooldownTracker
from alpharidge_ai.validator import triage_audit
from alpharidge_ai.triage import LABEL_IRRELEVANT, LABEL_RELEVANT
from alpharidge_ai.validator.triage_grader import (RELEVANCE_REF_CAP, TriageConfig, grade_batch,
                                                   relevance_audit_factor, relevance_factors)
from tests.test_triage import make_item, never_relevant, stage_junk
from neurons.validator import Validator

NOW = 1_800_000_000.0


def field(rate=0.15, n=40, seen=200, extra=None):
    t = MinerCooldownTracker()
    hks = [f"hk{i:02d}" for i in range(n)]
    for hk in hks:
        t.record_relevance(hk, round(rate * seen), seen, when=NOW)
    for hk, (kept, s) in (extra or {}).items():
        t.record_relevance(hk, kept, s, when=NOW)
        hks.append(hk)
    return t, hks


def test_off_means_no_factors():
    t, hks = field(extra={"x": (160, 200)})
    assert relevance_factors(t, hks, 0.0, NOW) == {}


def test_a_field_near_its_rate_is_untouched():
    t, hks = field(extra={"a": (50, 200)})          # 25% against a 15% field, bound 30%
    assert relevance_factors(t, hks, 2.0, NOW) == {}


def test_keeping_far_above_the_field_is_paid_at_the_bound():
    t, hks = field(extra={"x": (492, 600)})         # 82%
    f = relevance_factors(t, hks, 2.0, NOW)["x"]
    rate = (492 + 48 * 0.15) / (600 + 48)
    assert f == pytest.approx(0.30 / rate)
    assert f * rate == pytest.approx(0.30)           # premium as if keeping at the bound


def test_prior_weight():
    t, hks = field(extra={"new": (20, 24)})
    f = relevance_factors(t, hks, 2.0, NOW)["new"]
    assert 0.7 < f < 1.0


def test_the_field_reference_is_capped():
    t, hks = field(rate=0.6)
    t.record_relevance("y", 190, 200, when=NOW); hks.append("y")
    f = relevance_factors(t, hks, 2.0, NOW)
    assert "y" in f and f["y"] == pytest.approx(2.0 * RELEVANCE_REF_CAP / ((190 + 48 * RELEVANCE_REF_CAP) / 248))


def test_counts_decay_and_persist(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DISPATCH_STATE_LOCATION", str(tmp_path / "d.json"), raising=False)
    t = MinerCooldownTracker()
    t.record_relevance("hk", 10, 20, when=NOW)
    k, n = t.relevance("hk", NOW + RELEVANCE_HALF_LIFE_S)
    assert (k, n) == pytest.approx((5.0, 10.0))
    t.save()
    again = MinerCooldownTracker(); again.load()
    assert again.relevance("hk", NOW) == pytest.approx((10.0, 20.0))
    again.prune(set())
    assert again.relevance("hk", NOW) == (0.0, 0.0)


# ---- the validator side ------------------------------------------------------------

def _article(aid, content="x" * 2500):
    return types.SimpleNamespace(id=aid, content=content, analysis=None,
                                 model_copy=lambda update=None: _article(aid, content))


def _validator(factor):
    paid = []
    v = types.SimpleNamespace()
    v._k_for = lambda aid: 1
    v._has_full_analysis = lambda a: True
    v._attribute_pay = lambda per, payout, record=True: paid.append((dict(per), payout))
    v._triage_only_analysis = lambda a: None
    v._miner_reward = types.SimpleNamespace(add_reward=lambda hk, n: None)
    v._article_store = types.SimpleNamespace(
        update_article=lambda *a, **k: None, add_article=lambda *a, **k: None,
        set_processed=lambda *a, **k: None, reset_to_unprocessed=lambda *a, **k: None,
        mark_rewarded=lambda *a, **k: None, is_rewarded=lambda *a, **k: False)
    v._article_cooldown = MinerCooldownTracker()
    v._relevance_factor = lambda hk: factor
    v._buffer_variants = lambda *a, **k: None
    return v, paid


def _res(ids, canaries=()):
    return types.SimpleNamespace(relevant_ids=list(ids), borderline_valuable_ids=[],
                                 borderline_discard_ids=[], retire_candidate_ids=[],
                                 canary_ids=list(canaries))


def test_the_premium_scales_and_the_attribution_matches():
    full, paid_full = _validator(1.0)
    half, paid_half = _validator(0.5)
    batch = [_article(i) for i in range(1, 5)]
    Validator._apply_triage_outcome(full, batch, "hk", _res([1, 2]), set(), {}, full_push=True)
    Validator._apply_triage_outcome(half, batch, "hk", _res([1, 2]), set(), {}, full_push=True)
    fee = TriageConfig().fee_points
    assert paid_full[0][0][1] == pytest.approx(fee + 6 * 3)
    assert paid_half[0][0][1] == pytest.approx(fee + 0.5 * 6 * 3)


def test_the_keep_rate_is_recorded_without_canaries():
    v, _ = _validator(1.0)
    batch = [_article(i) for i in range(1, 6)]
    Validator._apply_triage_outcome(v, batch, "hk", _res([1, 2, 5], canaries=[5]), set(), {},
                                    full_push=True)
    assert v._article_cooldown.relevance("hk") == pytest.approx((2.0, 4.0), abs=1e-3)


def test_verification_lane_premium():
    v, paid = _validator(0.5)
    batch = [_article(i) for i in range(1, 6)]
    Validator._apply_verification_outcome(v, batch, "hk", _res([1, 2, 5], canaries=[5]), set())
    assert v._article_cooldown.relevance("hk") == pytest.approx((2.0, 4.0), abs=1e-3)
    fee = TriageConfig().fee_points
    assert paid[0][0][1] == pytest.approx(fee + 0.5 * 6 * 3)


def test_the_bound_is_a_served_setting_on_by_default():
    assert config.TRIAGE_RELEVANCE_BOUND == 2.0
    assert "TRIAGE_RELEVANCE_BOUND" in config._REMOTE_CONFIG_KEYS


# ---- the relevance audit -----------------------------------------------------------

def test_the_audit_asks_for_room_and_needs_only_the_verdict():
    assert triage_audit._TOOL["function"]["parameters"]["required"] == ["relevant", "confidence"]
    seen = {}

    class Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    seen.update(kw)
                    call = types.SimpleNamespace(function=types.SimpleNamespace(
                        arguments='{"relevant": false, "confidence": 0.9}'))
                    msg = types.SimpleNamespace(tool_calls=[call])
                    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])
    assert triage_audit.TriageAuditor(Client, "m").relevance_verdict("t", "b") is False
    assert seen["max_tokens"] >= 1000


def test_a_failed_audit_is_visible(monkeypatch):
    lines = []
    monkeypatch.setattr(triage_audit.bt.logging, "warning", lambda m: lines.append(m))

    class Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    raise RuntimeError("boom")
    assert triage_audit.TriageAuditor(Client, "m").relevance_verdict("t", "b") is None
    assert any("verdict unavailable" in l for l in lines)


# ---- the relevance audit of claims -------------------------------------------------

def test_full_pay_within_tolerance():
    assert relevance_audit_factor(0, 0) == 1.0
    assert relevance_audit_factor(40, 2) == 1.0
    assert relevance_audit_factor(3, 1) > 0.8


def test_rejected_claims_cut_the_premium_steeply():
    assert 0 < relevance_audit_factor(40, 14) < relevance_audit_factor(40, 8) < 0.7
    assert relevance_audit_factor(40, 30) == 0.0


def test_premium_per_article_falls_with_rejections():
    import math
    p0, p1, n = 0.04, 0.70, 26

    def mean_factor(p):
        return sum(math.comb(n, k) * p ** k * (1 - p) ** (n - k) * relevance_audit_factor(n, k)
                   for k in range(n + 1))
    assert mean_factor(p0) > 0.97
    for share in (0.1, 0.25, 0.5, 0.8):
        assert mean_factor((1 - share) * p0 + share * p1) / (1 - share) < 1.0


def test_the_tolerance_is_a_served_setting():
    assert config.TRIAGE_RELEVANCE_AUDIT_TOL == 0.06
    assert "TRIAGE_RELEVANCE_AUDIT_TOL" in config._REMOTE_CONFIG_KEYS
    assert relevance_audit_factor(40, 4) < 1.0 == relevance_audit_factor(40, 4, tol=0.10)


def _claims(n_relevant, n_irrelevant=2):
    return ([make_item(i, LABEL_RELEVANT) for i in range(1, n_relevant + 1)]
            + [make_item(100 + i, LABEL_IRRELEVANT, "non_economic") for i in range(n_irrelevant)])


def _grade(items, det=never_relevant, irrelevant=lambda it: True, n=1, canaries=None):
    asked = []

    def llm_irrelevant(it):
        asked.append(it["article_id"])
        return irrelevant(it)
    res = grade_batch(items, canaries or {}, det, lambda i: False, stage_junk,
                      random.Random(0), TriageConfig(), enforced=True,
                      llm_irrelevant=llm_irrelevant, audit_relevant_n=n)
    return res, asked


def test_a_rejected_claim_is_unpaid_and_raises_an_event():
    res, asked = _grade(_claims(3))
    assert res.relevance_audited == 1 and len(asked) == 1
    assert res.false_positive_ids == asked
    assert ("soft", "triage_false_positive", asked[0]) in [
        (e.kind, e.code, e.article_id) for e in res.events]


def test_only_relevance_claims_outside_canaries_are_audited():
    res, asked = _grade(_claims(2), n=5, canaries={1: ("neg", False)})
    assert asked == [2] and res.relevance_audited == 1


def test_deterministic_relevance_path():
    res, asked = _grade(_claims(2), det=lambda it: True)
    assert res.relevance_audited == 1 and asked == [] and res.false_positive_ids == []


def test_an_upheld_claim_and_the_off_switch():
    res, _ = _grade(_claims(2), irrelevant=lambda it: False)
    assert res.relevance_audited == 1 and res.false_positive_ids == []
    res, asked = _grade(_claims(2), n=0)
    assert res.relevance_audited == 0 and asked == []


def test_audit_counts_decay_and_persist(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DISPATCH_STATE_LOCATION", str(tmp_path / "d.json"), raising=False)
    t = MinerCooldownTracker()
    t.record_relevance_audit("hk", 1, 1, when=NOW)
    t.record_relevance_audit("hk", 1, 0, when=NOW)
    assert t.relevance_audit("hk", NOW + RELEVANCE_HALF_LIFE_S) == pytest.approx((1.0, 0.5))
    t.save()
    again = MinerCooldownTracker(); again.load()
    assert again.relevance_audit("hk", NOW) == pytest.approx((2.0, 1.0))


def test_the_validator_combines_both_factors(monkeypatch):
    monkeypatch.setattr(config, "TRIAGE_RELEVANCE_BOUND", 2.0, raising=False)
    now = time.time()
    t, hks = field(extra={"x": (492, 600)})
    for hk in hks:
        t.record_relevance(hk, 0, 0, when=now)
    t._relevance = {hk: [k, n, now] for hk, (k, n) in
                    ((hk, t.relevance(hk, NOW)) for hk in hks)}
    t.record_relevance_audit("x", 30, 6, when=now)
    t.record_relevance_audit("hk01", 30, 6, when=now)
    t.record_relevance_audit("hk02", 30, 1, when=now)
    v = types.SimpleNamespace(_article_cooldown=t, metagraph=types.SimpleNamespace(hotkeys=hks))
    bound = relevance_factors(t, hks, 2.0, now)["x"]
    assert Validator._relevance_factor(v, "x") == pytest.approx(
        bound * relevance_audit_factor(30, 6))
    assert Validator._relevance_factor(v, "hk01") == pytest.approx(relevance_audit_factor(30, 6))
    assert Validator._relevance_factor(v, "hk02") == 1.0


def test_the_audit_is_a_served_setting_on_by_default():
    assert config.TRIAGE_RELEVANCE_AUDIT_N == 1
    assert "TRIAGE_RELEVANCE_AUDIT_N" in config._REMOTE_CONFIG_KEYS


def test_the_audit_confidence_is_about_the_answer():
    desc = triage_audit._TOOL["function"]["parameters"]["properties"]["confidence"]["description"]
    assert "Not the probability" in desc
