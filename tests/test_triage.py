"""Unit tests for the triage schema (alpharidge_ai.triage) and the validator
triage grader (alpharidge_ai.validator.triage_grader).

Run with: pytest tests/test_triage.py -v
No LLM, no network — grading logic is pure with injected oracles.
"""
import random

import pytest

from alpharidge_ai.triage import (
    LABEL_BORDERLINE,
    LABEL_IRRELEVANT,
    LABEL_RELEVANT,
    build_proof_of_read,
    build_triage_record,
    deterministic_relevant,
    extract_triage,
    verify_proof_of_read,
)
from alpharidge_ai.validator.triage_grader import (
    CanaryPool,
    TriageConfig,
    fp_soft_event,
    grade_batch,
)

TITLE = "Acme Corp beats earnings expectations"
BODY = ("Acme Corp ($ACME) reported quarterly revenue of $4.2 billion, "
        "beating analyst expectations. Shares rose 6% in after-hours trading. "
        "The company raised full-year guidance citing strong demand.")


def make_item(aid, label=None, reason=None, good_proof=True, title=TITLE, body=BODY,
              extra_analysis=None):
    analysis = dict(extra_analysis or {})
    if label is not None:
        analysis["triage"] = build_triage_record(label, reason)
    if good_proof:
        analysis["proof_of_read"] = build_proof_of_read(title, body)
    elif good_proof is False and label is not None:
        analysis["proof_of_read"] = {"content_hash": "0" * 64, "word_count": 1}
    return {"article_id": aid, "title": title, "body": body,
            "analysis_data": analysis or None}


def never_relevant(_item):
    return False


def stage_junk(_item):
    return "irrelevant"


def stage_ambiguous(_item):
    return "borderline"


def llm_says(mapping, default=False):
    return lambda item: mapping.get(item["article_id"], default)


CFG = TriageConfig()
RNG = random.Random


# ---------------------------------------------------------------- schema ----

class TestSchema:
    def test_proof_of_read_roundtrip(self):
        proof = build_proof_of_read(TITLE, BODY)
        assert verify_proof_of_read(proof, TITLE, BODY)

    def test_proof_of_read_rejects_tamper(self):
        proof = build_proof_of_read(TITLE, BODY)
        assert not verify_proof_of_read(proof, TITLE, BODY + " extra")
        assert not verify_proof_of_read({**proof, "word_count": proof["word_count"] + 1}, TITLE, BODY)
        assert not verify_proof_of_read(None, TITLE, BODY)
        assert not verify_proof_of_read({}, TITLE, BODY)

    def test_triage_record_validation(self):
        rec = build_triage_record(LABEL_IRRELEVANT, "non_economic", 0.8)
        assert rec["label"] == LABEL_IRRELEVANT and rec["confidence"] == 0.8
        with pytest.raises(ValueError):
            build_triage_record("maybe")
        with pytest.raises(ValueError):
            build_triage_record(LABEL_IRRELEVANT)  # reason required

    def test_extract_triage_paths(self):
        ok, err = extract_triage({"triage": build_triage_record(LABEL_RELEVANT)})
        assert ok and err is None
        none_rec, err = extract_triage({"assets": []})     # pre-v3 payload
        assert none_rec is None and err is None
        assert extract_triage(None) == (None, None)
        _, err = extract_triage({"triage": "yes"})
        assert err == "triage_not_object"
        _, err = extract_triage({"triage": {"label": "spam"}})
        assert err == "triage_bad_label"
        _, err = extract_triage({"triage": {"label": LABEL_IRRELEVANT}})
        assert err == "triage_missing_reason"

    def test_deterministic_relevant(self):
        assert deterministic_relevant([{"resolved_via": "keyword"}])
        assert not deterministic_relevant([{"resolved_via": "ner"}])
        assert not deterministic_relevant([])
        assert not deterministic_relevant(None)


# ---------------------------------------------------------------- grading ----

class TestGradeBatch:
    def test_no_triage_data_grace_before_enforcement(self):
        items = [{"article_id": i, "title": TITLE, "body": BODY,
                  "analysis_data": {"assets": []}} for i in range(3)]
        res = grade_batch(items, {}, never_relevant, lambda i: None, stage_junk, RNG(0), CFG,
                          enforced=False)
        assert res.grace and not res.events and not res.proof_failures
        assert res.observations(CFG, clean_article_id=0) == []

    def test_no_triage_data_fails_proof_when_enforced(self):
        items = [{"article_id": i, "title": TITLE, "body": BODY,
                  "analysis_data": {"assets": []}} for i in range(3)]
        res = grade_batch(items, {}, never_relevant, lambda i: None, stage_junk, RNG(0), CFG,
                          enforced=True)
        assert set(res.proof_failures) == {0, 1, 2}
        obs = res.observations(CFG, clean_article_id=0)
        assert all(s == 0.0 for _, s, _ in obs)
        assert {aid for aid, _, w in obs if w == CFG.hard_weight} == {0, 1, 2}

    def test_malformed_triage_never_gets_grace(self):
        # A present-but-broken record is not "pre-triage": graded even unenforced.
        items = [{"article_id": 1, "title": TITLE, "body": BODY,
                  "analysis_data": {"triage": {"label": "spam"}}}]
        res = grade_batch(items, {}, never_relevant, lambda i: None, stage_junk, RNG(0), CFG,
                          enforced=False)
        assert not res.grace
        assert ("soft", "triage_malformed") in [(e.kind, e.code) for e in res.events]

    def test_clean_batch_single_positive_observation(self):
        items = [make_item(1, LABEL_RELEVANT),
                 make_item(2, LABEL_IRRELEVANT, "non_economic"),
                 make_item(3, LABEL_IRRELEVANT, "non_economic")]
        res = grade_batch(items, {}, never_relevant, lambda i: False, stage_junk, RNG(0), CFG, enforced=True)
        assert not res.events and not res.proof_failures
        assert res.relevant_ids == [1]
        assert res.observations(CFG, clean_article_id=1) == [(1, 1.0, CFG.clean_weight)]
        # un-contradicted irrelevant claims become retire candidates
        assert set(res.retire_candidate_ids) == {2, 3}

    def test_proof_of_read_failure_is_hard(self):
        items = [make_item(1, LABEL_RELEVANT, good_proof=False),
                 make_item(2, LABEL_IRRELEVANT, "non_economic")]
        res = grade_batch(items, {}, never_relevant, lambda i: False, stage_junk, RNG(0), CFG, enforced=True)
        assert res.proof_failures == [1]
        # A proof-of-read failure must reach the reputation stream, not just
        # the penalty path — it is the strongest evidence the miner never read.
        assert (1, 0.0, CFG.hard_weight) in res.observations(CFG, clean_article_id=1)

    def test_pos_canary_missed_deterministic_is_hard(self):
        items = [make_item(1, LABEL_IRRELEVANT, "non_economic"),
                 make_item(2, LABEL_RELEVANT)]
        res = grade_batch(items, {1: ("pos", True)}, never_relevant,
                          lambda i: False, stage_junk, RNG(0), CFG, enforced=True)
        assert [(e.kind, e.code) for e in res.events] == [("hard", "canary_pos_missed")]

    def test_pos_canary_missed_llm_labeled_is_soft(self):
        items = [make_item(1, LABEL_IRRELEVANT, "non_economic"),
                 make_item(2, LABEL_RELEVANT)]
        res = grade_batch(items, {1: ("pos", False)}, never_relevant,
                          lambda i: False, stage_junk, RNG(0), CFG, enforced=True)
        assert [(e.kind, e.code) for e in res.events] == [("soft", "canary_pos_missed")]

    def test_neg_canary_flagged_is_soft_and_pos_canary_kept_is_clean(self):
        items = [make_item(1, LABEL_RELEVANT),                       # neg canary flagged
                 make_item(2, LABEL_RELEVANT),                       # pos canary correctly kept
                 make_item(3, LABEL_IRRELEVANT, "non_economic")]
        res = grade_batch(items, {1: ("neg", False), 2: ("pos", True)},
                          never_relevant, lambda i: False, stage_junk, RNG(0), CFG, enforced=True)
        assert [(e.kind, e.code) for e in res.events] == [("soft", "canary_neg_flagged")]

    def test_audit_deterministic_false_negative_is_hard(self):
        items = [make_item(1, LABEL_IRRELEVANT, "non_economic"),
                 make_item(2, LABEL_RELEVANT)]
        res = grade_batch(items, {}, lambda i: i["article_id"] == 1,
                          lambda i: False, stage_junk, RNG(0), CFG, enforced=True)
        assert [(e.kind, e.code) for e in res.events] == \
            [("hard", "false_negative_deterministic")]
        assert 1 not in res.retire_candidate_ids

    def test_audit_llm_false_negative_is_always_soft(self):
        items = [make_item(1, LABEL_IRRELEVANT, "non_economic")]
        res = grade_batch(items, {}, never_relevant, lambda i: True, stage_junk, RNG(0), CFG, enforced=True)
        assert [(e.kind, e.code) for e in res.events] == [("soft", "false_negative_llm")]

    def test_audit_llm_unavailable_no_event(self):
        items = [make_item(1, LABEL_IRRELEVANT, "non_economic")]
        res = grade_batch(items, {}, never_relevant, lambda i: None, stage_junk, RNG(0), CFG, enforced=True)
        assert not res.events

    def test_malformed_record_on_v3_miner_is_soft_and_auditable(self):
        items = [make_item(1, LABEL_RELEVANT),
                 {"article_id": 2, "title": TITLE, "body": BODY,
                  "analysis_data": {"triage": {"label": "spam"},
                                    "proof_of_read": build_proof_of_read(TITLE, BODY)}}]
        res = grade_batch(items, {}, lambda i: i["article_id"] == 2,
                          lambda i: False, stage_junk, RNG(0), CFG, enforced=True)
        codes = sorted(e.code for e in res.events)
        assert "triage_malformed" in codes
        # treated as irrelevant claim -> audited -> deterministic FN caught
        assert "false_negative_deterministic" in codes

    def test_borderline_cap_excess_becomes_irrelevant_claim(self):
        cfg = TriageConfig(borderline_cap=1, audit_irrelevant_n=0)
        items = [make_item(1, LABEL_BORDERLINE), make_item(2, LABEL_BORDERLINE),
                 make_item(3, LABEL_RELEVANT)]
        res = grade_batch(items, {}, never_relevant, lambda i: False, stage_junk, RNG(0), cfg,
                          enforced=True)
        assert res.borderline_ids == [1]
        assert 2 in res.retire_candidate_ids   # capped -> irrelevant claim
        assert res.relevant_ids == [3]

    def test_fp_soft_event(self):
        e = fp_soft_event(7)
        assert (e.kind, e.code, e.article_id) == ("soft", "triage_false_positive", 7)


# ---------------------------------------------------------------- miner stage ----

class TestTriageStage:
    @pytest.fixture(scope="class")
    def stage(self):
        from alpharidge_ai.analyzer.asset_extractor import AssetExtractor
        from alpharidge_ai.analyzer.triage_stage import TriageStage
        return TriageStage(AssetExtractor())

    def test_asset_article_is_relevant_r1(self, stage):
        title = "Apple beats earnings expectations"
        body = ("Apple Inc ($AAPL) reported quarterly revenue of $94 billion, "
                "beating analyst expectations. Shares rose 6% after hours.")
        rec, proof, matches = stage.evaluate(title, body)
        assert rec["label"] == "relevant" and matches
        assert verify_proof_of_read(proof, title, body)

    def test_macro_with_named_economy_is_relevant_r2(self, stage):
        rec, _, _ = stage.evaluate(
            "Central bank surprises with rate cut",
            "The central bank of Brazil announced a surprise interest rate cut "
            "on Thursday, citing slowing inflation across the economy.")
        assert rec["label"] == "relevant"

    def test_macro_without_economy_is_borderline(self, stage):
        rec, _, _ = stage.evaluate(
            "Officials debate inflation outlook",
            "Rising inflation remains a concern for households, officials said, "
            "though no specific policy response was announced.")
        assert rec["label"] == "borderline"

    def test_general_news_is_irrelevant(self, stage):
        rec, _, _ = stage.evaluate(
            "Local choir wins regional competition",
            "The community choir took first place at the regional festival on "
            "Saturday, delighting a crowd of hundreds with folk songs.")
        assert rec["label"] == "irrelevant" and rec["reason_code"] == "non_economic"

    def test_pronoun_us_does_not_trigger_economy(self, stage):
        # Regression: case-insensitive "us" matched the pronoun and inflated
        # the relevant rate from ~15% to ~46% on real data.
        rec, _, _ = stage.evaluate(
            "Join us for the summer recipe special",
            "Come cook with us: this tariff-free guacamole brings the whole "
            "family together for movie night.")
        assert rec["label"] != "relevant"


# ---------------------------------------------------------------- canaries ----

class TestCanaryPool:
    def test_draw_and_exposure_cap(self):
        cfg = TriageConfig(canary_max_exposures=2)
        clock = {"t": 1000.0}
        pool = CanaryPool(cfg, now=lambda: clock["t"])
        pool.add(11, "pos", deterministic=True)
        rng = RNG(0)
        assert pool.draw("pos", rng) == 11
        assert pool.draw("pos", rng) == 11
        assert pool.draw("pos", rng) is None      # exposure cap reached
        assert pool.draw("neg", rng) is None

    def test_ttl_expiry(self):
        cfg = TriageConfig(canary_ttl_s=100)
        clock = {"t": 0.0}
        pool = CanaryPool(cfg, now=lambda: clock["t"])
        pool.add(5, "neg", deterministic=False)
        assert pool.size("neg") == 1
        clock["t"] = 101.0
        assert pool.size("neg") == 0
        assert pool.draw("neg", RNG(0)) is None

    def test_label_of(self):
        pool = CanaryPool(CFG, now=lambda: 0.0)
        pool.add(9, "pos", deterministic=True)
        assert pool.label_of(9) == ("pos", True)
        assert pool.label_of(10) is None

    def test_draw_restricted_to_available_never_burns_dead_exposures(self):
        cfg = TriageConfig(canary_max_exposures=1)
        pool = CanaryPool(cfg, now=lambda: 0.0)
        pool.add(5, "pos", deterministic=True)
        pool.add(6, "pos", deterministic=True)
        rng = random.Random(0)
        assert pool.draw("pos", rng, available={6}) == 6      # 5 not injectable
        assert pool.draw("pos", rng, available={6}) is None   # 6 spent; 5 intact
        assert pool.draw("pos", rng, available={5, 6}) == 5
