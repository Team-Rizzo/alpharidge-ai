"""End-to-end triage loop: dispatch -> canary injection -> miner triage ->
validator grading -> reputation observations -> pay.

Exercises the real Validator methods (_inject_canaries, _grade_triage,
_record_triage_observations, _apply_triage_outcome) against the real
TriageStage and CanaryPool, with only the neuron's I/O surface faked. This is
the test that would catch a wiring mistake between the pieces, which the unit
tests cannot see.

Run with: pytest tests/test_triage_e2e.py -v
"""
import random
import types

import pytest

from alpharidge_ai import config
from alpharidge_ai.utils.api_models import NewsArticleForScoring
from alpharidge_ai.validator.triage_grader import CanaryPool, TriageConfig

import neurons.validator as validator_module

ASSET_ARTICLE = (
    "Apple beats earnings expectations",
    "Apple Inc ($AAPL) reported quarterly revenue of $94 billion, beating "
    "analyst expectations. Shares rose 6% in after-hours trading as the company "
    "raised full-year guidance citing strong iPhone demand.",
)
JUNK_ARTICLE = (
    "Local choir wins regional competition",
    "The community choir took first place at the regional festival on Saturday, "
    "delighting a crowd of hundreds with a program of traditional folk songs "
    "and a well-received encore.",
)
AMBIGUOUS_ARTICLE = (
    "Officials debate inflation outlook",
    "Rising inflation remains a concern for households, officials said, "
    "though no specific policy response was announced.",
)
MACRO_ARTICLE = (
    "Brazil central bank cuts rates",
    "The central bank of Brazil announced a surprise interest rate cut on "
    "Thursday, citing slowing inflation. Economists expect further easing.",
)


def stage_junk(_item):
    return "irrelevant"


def article(aid, pair):
    title, content = pair
    return NewsArticleForScoring(id=aid, url=f"http://x/{aid}", title=title,
                                 content=content, source="test")


class FakeStore:
    def __init__(self):
        self.processed, self.reset, self.rewarded = set(), set(), set()
        self.updated = {}

    def update_article(self, article_id, art):
        self.updated[int(article_id)] = art

    def add_article(self, art, **kw):
        self.updated[int(art.id)] = art

    def set_processed(self, article_id):
        self.processed.add(int(article_id))

    def reset_to_unprocessed(self, article_id):
        self.reset.add(int(article_id))

    def is_rewarded(self, article_id):
        return int(article_id) in self.rewarded

    def mark_rewarded(self, article_id):
        self.rewarded.add(int(article_id))


class FakeReward:
    def __init__(self):
        self.points = 0

    def add_reward(self, hotkey, reward):
        self.points += int(reward)


class HarnessValidator:
    """Real triage methods, faked I/O surface."""

    _triage_cfg = validator_module.Validator._triage_cfg
    _det_relevant_item = validator_module.Validator._det_relevant_item
    _stage_label_item = validator_module.Validator._stage_label_item
    _confirm_clearly_irrelevant = validator_module.Validator._confirm_clearly_irrelevant
    _get_triage_stage = validator_module.Validator._get_triage_stage
    _llm_relevant_item = validator_module.Validator._llm_relevant_item
    _get_triage_auditor = validator_module.Validator._get_triage_auditor
    _feed_pos_canaries = validator_module.Validator._feed_pos_canaries
    _inject_canaries = validator_module.Validator._inject_canaries
    _grade_triage = validator_module.Validator._grade_triage
    _record_triage_observations = validator_module.Validator._record_triage_observations
    _apply_triage_outcome = validator_module.Validator._apply_triage_outcome
    _mint_neg_canaries = validator_module.Validator._mint_neg_canaries
    _has_full_analysis = staticmethod(validator_module.Validator._has_full_analysis)
    _triage_only_analysis = staticmethod(validator_module.Validator._triage_only_analysis)
    _k_for = validator_module.Validator._k_for

    def __init__(self):
        self._canary_pool = CanaryPool(TriageConfig())
        self._canary_articles = {}
        self._article_k = {}
        self._triage_extractor = None
        self._triage_auditor = None
        self._triage_stage = None
        self._article_intel_analyzer = None
        self._article_store = FakeStore()
        self._miner_reward = FakeReward()
        self.observations = []

    def _record_observations(self, hotkey, observations):
        self.observations.extend(observations)


@pytest.fixture(scope="module")
def stage():
    from alpharidge_ai.analyzer.asset_extractor import AssetExtractor
    from alpharidge_ai.analyzer.triage_stage import TriageStage
    return TriageStage(AssetExtractor())


@pytest.fixture
def triage_on():
    """Historical name: triage has no off switch anymore. Kept so the test
    signatures read naturally ('with triage on...')."""
    yield


def mine(stage, articles, strategy="honest"):
    """Run the real miner triage stage over a dispatch batch."""
    from alpharidge_ai.triage import build_proof_of_read, build_triage_record
    out = []
    for art in articles:
        rec, proof, _ = stage.evaluate(art.title, art.content)
        if strategy == "lazy":
            rec = build_triage_record("irrelevant", "non_economic")
        elif strategy == "spam":
            rec = build_triage_record("relevant")
        elif strategy == "no_read":
            proof = {"content_hash": "0" * 64, "word_count": 1}
        analysis = types.SimpleNamespace(
            analysis_data={"schema_version": 3, "triage": rec, "proof_of_read": proof})
        out.append(art.model_copy(update={"analysis": analysis}))
    return out


class TestEndToEnd:
    def test_honest_batch_pays_and_files_articles(self, stage, triage_on):
        v = HarnessValidator()
        batch = [article(1, ASSET_ARTICLE), article(2, JUNK_ARTICLE),
                 article(3, MACRO_ARTICLE), article(4, JUNK_ARTICLE)]
        returned = mine(stage, batch)

        res = v._grade_triage(returned, batch, "hk1")
        assert res is not None
        assert not res.events and not res.proof_failures
        assert set(res.relevant_ids) == {1, 3}
        assert set(res.retire_candidate_ids) == {2, 4}

        v._record_triage_observations("hk1", res, returned)
        assert v.observations == [(1, 1.0, 1.0)]   # one clean-batch observation

        v._apply_triage_outcome(returned, "hk1", res, fp_ids=set())
        # fee round(0.2*4)=1, plus 6x length-weight(1, short fixtures) per relevant
        assert v._miner_reward.points == 1 + 6 * 2
        assert v._article_store.processed == {1, 2, 3, 4}
        assert v._article_store.rewarded == {1, 3}
        assert not v._article_store.reset

    def test_lazy_miner_caught_by_deterministic_audit(self, stage, triage_on):
        v = HarnessValidator()
        batch = [article(1, ASSET_ARTICLE), article(2, JUNK_ARTICLE),
                 article(3, JUNK_ARTICLE)]
        returned = mine(stage, batch, strategy="lazy")

        res = v._grade_triage(returned, batch, "hk2")
        codes = [(e.kind, e.code) for e in res.events]
        assert ("hard", "false_negative_deterministic") in codes

        v._record_triage_observations("hk2", res, returned)
        assert all(score == 0.0 for _, score, _ in v.observations)
        assert any(w == 2.0 for *_, w in v.observations)   # hard weight

        # The asset article is never retired despite being claimed irrelevant.
        v._apply_triage_outcome(returned, "hk2", res, fp_ids=set())
        assert 1 in v._article_store.reset
        assert 1 not in v._article_store.processed

    def test_no_read_miner_fails_proof(self, stage, triage_on):
        v = HarnessValidator()
        batch = [article(1, ASSET_ARTICLE), article(2, JUNK_ARTICLE)]
        returned = mine(stage, batch, strategy="no_read")
        res = v._grade_triage(returned, batch, "hk3")
        assert set(res.proof_failures) == {1, 2}
        v._record_triage_observations("hk3", res, returned)
        assert v.observations and all(s == 0.0 for _, s, _ in v.observations)

    def test_spam_miner_earns_nothing_for_junk(self, stage, triage_on):
        v = HarnessValidator()
        batch = [article(1, ASSET_ARTICLE), article(2, JUNK_ARTICLE),
                 article(3, JUNK_ARTICLE)]
        returned = mine(stage, batch, strategy="spam")
        res = v._grade_triage(returned, batch, "hk4")
        assert set(res.relevant_ids) == {1, 2, 3}
        # Reference analysis refutes the two junk articles.
        v._apply_triage_outcome(returned, "hk4", res, fp_ids={2, 3})
        assert v._miner_reward.points == 1 + 6   # fee + only the real one

    def test_pre_triage_miner_grace_then_enforced(self, stage, triage_on, monkeypatch):
        v = HarnessValidator()
        batch = [article(1, ASSET_ARTICLE), article(2, JUNK_ARTICLE)]
        legacy = [a.model_copy(update={"analysis": types.SimpleNamespace(
            analysis_data={"schema_version": 2, "title": a.title})}) for a in batch]
        monkeypatch.setattr(config, "TRIAGE_ENFORCED", False, raising=False)
        res = v._grade_triage(legacy, batch, "hk5")
        assert res.grace and not res.proof_failures
        monkeypatch.setattr(config, "TRIAGE_ENFORCED", True, raising=False)
        res2 = v._grade_triage(legacy, batch, "hk5")
        assert set(res2.proof_failures) == {1, 2}

    def test_triage_has_no_off_switch(self, stage):
        v = HarnessValidator()
        returned = mine(stage, [article(1, ASSET_ARTICLE)])
        assert v._grade_triage(returned, [article(1, ASSET_ARTICLE)], "hk6") is not None


class TestExploitResistance:
    """Regressions for exploits found in adversarial review of this branch."""

    def test_forged_article_text_cannot_launder_an_asset_article(self, stage, triage_on):
        # Miner echoes back junk text under the real article's id, with a
        # proof-of-read computed over its own forgery. Grading must audit OUR
        # copy, so the gazetteer still sees the asset and the proof still fails.
        v = HarnessValidator()
        sent = [article(1, ASSET_ARTICLE), article(2, JUNK_ARTICLE),
                article(3, JUNK_ARTICLE)]
        forged = article(1, JUNK_ARTICLE)
        returned = mine(stage, [forged] + sent[1:], strategy="lazy")

        res = v._grade_triage(returned, sent, "evil")
        assert 1 in res.proof_failures
        assert 1 not in res.retire_candidate_ids

        # Even with a valid proof, the gazetteer audit runs on our copy.
        from alpharidge_ai.triage import build_proof_of_read, build_triage_record
        good_proof = build_proof_of_read(*ASSET_ARTICLE)
        returned[0] = forged.model_copy(update={"analysis": types.SimpleNamespace(
            analysis_data={"schema_version": 3,
                           "triage": build_triage_record("irrelevant", "non_economic"),
                           "proof_of_read": good_proof})})
        res2 = v._grade_triage(returned, sent, "evil")
        assert not res2.proof_failures
        assert ("hard", "false_negative_deterministic") in [
            (e.kind, e.code) for e in res2.events]

    def test_borderline_is_not_a_free_pass_on_deterministic_assets(self, stage, triage_on):
        from alpharidge_ai.triage import build_proof_of_read, build_triage_record
        v = HarnessValidator()
        batch = [article(1, ASSET_ARTICLE), article(2, JUNK_ARTICLE)]
        returned = [a.model_copy(update={"analysis": types.SimpleNamespace(
            analysis_data={"schema_version": 3,
                           "triage": build_triage_record("borderline"),
                           "proof_of_read": build_proof_of_read(a.title, a.content)})})
            for a in batch]
        res = v._grade_triage(returned, batch, "hk")
        assert ("hard", "false_negative_deterministic") in [
            (e.kind, e.code) for e in res.events]
        # Borderline never retires an article either.
        assert not res.retire_candidate_ids

    def test_borderline_does_not_evade_a_positive_canary(self, stage, triage_on):
        from alpharidge_ai.triage import build_proof_of_read, build_triage_record
        v = HarnessValidator()
        canary = article(10, ASSET_ARTICLE)
        v._canary_pool.add(10, "pos", deterministic=True)
        v._canary_articles[10] = canary
        returned = [canary.model_copy(update={"analysis": types.SimpleNamespace(
            analysis_data={"schema_version": 3,
                           "triage": build_triage_record("borderline"),
                           "proof_of_read": build_proof_of_read(*ASSET_ARTICLE)})})]
        res = v._grade_triage(returned, [canary], "hk")
        assert ("hard", "canary_pos_missed") in [(e.kind, e.code) for e in res.events]

    def test_adverse_finding_survives_merge_with_quality_observation(self, stage, triage_on):
        # The reputation store keeps only the first observation per article id,
        # so a favourable quality score must not mask a triage finding on it.
        v = HarnessValidator()
        batch = [article(1, ASSET_ARTICLE), article(2, JUNK_ARTICLE)]
        returned = mine(stage, batch, strategy="spam")
        res = v._grade_triage(returned, batch, "hk")
        res.events.append(validator_module.fp_soft_event(2))
        v._record_triage_observations("hk", res, returned,
                                      graded_observations=[(2, 0.9, 2.0)])
        assert dict((aid, s) for aid, s, _ in v.observations)[2] == 0.0
        assert len([o for o in v.observations if o[0] == 2]) == 1

    def test_gazetteer_overrides_reference_irrelevant_verdict(self, stage, triage_on):
        # Live incident 2026-07-24: the reference LLM missed a ticker in
        # market-report spam, minted it as a negative canary, and honest
        # gazetteer-based miners were soft-punished for keeping it. The
        # deterministic gazetteer must veto LLM 'clearly irrelevant' verdicts
        # before they mint canaries or false-positive charges.
        v = HarnessValidator()
        asset_art, junk_art = article(1, ASSET_ARTICLE), article(2, JUNK_ARTICLE)
        sent_by_id = {1: asset_art, 2: junk_art}
        confirmed = v._confirm_clearly_irrelevant(
            [(1, True), (2, True), (2, False)], sent_by_id)
        assert confirmed == {2}   # asset article vetoed despite the LLM verdict

    def test_reference_stage_vetoes_foreign_asset_articles(self, stage, triage_on):
        # Second live incident: a leveraged-funds story on foreign-listed
        # equities had no gazetteer hit AND a reference miss, so it minted as a
        # negative canary and honest miners were punished for keeping it. The
        # shipped TriageStage must also concur before ground truth is minted —
        # its R2 branch catches what the asset gazetteer does not.
        v = HarnessValidator()
        macro_art = article(3, MACRO_ARTICLE)   # TriageStage says relevant, no gazetteer asset
        confirmed = v._confirm_clearly_irrelevant([(3, True)], {3: macro_art})
        assert confirmed == set()

    def test_canary_readd_does_not_refresh_ttl_or_exposures(self):
        cfg = TriageConfig(canary_max_exposures=2)
        clock = {"t": 0.0}
        pool = CanaryPool(cfg, now=lambda: clock["t"])
        pool.add(5, "pos", deterministic=True)
        rng = random.Random(0)
        pool.draw("pos", rng)
        pool.add(5, "pos", deterministic=True)   # recirculated through the pool
        pool.draw("pos", rng)
        assert pool.draw("pos", rng) is None     # exposure cap still reachable


@pytest.fixture
def canaries_certain(monkeypatch, triage_on):
    """Make injection deterministic so canary tests aren't RNG-flaky."""
    monkeypatch.setattr(validator_module.TRIAGE_CFG, "canary_pos_rate", 1.0)
    monkeypatch.setattr(validator_module.TRIAGE_CFG, "canary_neg_rate", 0.0)


class TestCanaryFlow:
    def test_pos_canaries_fed_from_gazetteer_and_injected(self, stage, canaries_certain):
        v = HarnessValidator()
        incoming = [article(10, ASSET_ARTICLE), article(11, JUNK_ARTICLE)]
        v._feed_pos_canaries(incoming)
        assert v._canary_pool.size("pos") == 1        # only the asset article
        assert v._canary_pool.label_of(10) == ("pos", True)

        batch = [article(1, JUNK_ARTICLE), article(2, JUNK_ARTICLE),
                 article(3, JUNK_ARTICLE)]
        injected = v._inject_canaries(batch, random.Random(0))
        assert 10 in injected
        assert any(int(a.id) == 10 for a in batch)
        assert len(batch) == 3                         # swap, not append

    def test_lazy_miner_trips_injected_pos_canary(self, stage, canaries_certain):
        v = HarnessValidator()
        v._feed_pos_canaries([article(10, ASSET_ARTICLE)])
        batch = [article(1, JUNK_ARTICLE), article(2, JUNK_ARTICLE),
                 article(3, JUNK_ARTICLE)]
        v._inject_canaries(batch, random.Random(0))
        returned = mine(stage, batch, strategy="lazy")

        res = v._grade_triage(returned, batch, "hk7")
        assert ("hard", "canary_pos_missed") in [(e.kind, e.code) for e in res.events]
        # Canaries are graded only — never re-stored or re-paid.
        v._apply_triage_outcome(returned, "hk7", res, fp_ids=set())
        assert 10 not in v._article_store.processed
        assert 10 not in v._article_store.rewarded
        assert 10 in v._article_store.reset   # returned to the pool, not leased

    def test_honest_miner_passes_injected_canary(self, stage, canaries_certain):
        v = HarnessValidator()
        v._feed_pos_canaries([article(10, ASSET_ARTICLE)])
        batch = [article(1, JUNK_ARTICLE), article(2, JUNK_ARTICLE),
                 article(3, JUNK_ARTICLE)]
        v._inject_canaries(batch, random.Random(0))
        res = v._grade_triage(mine(stage, batch), batch, "hk8")
        assert not res.events


class TestDefectFixes:
    """Regressions for the 2026-08-11 defect plan (D1-D5)."""

    def test_d1_retired_article_stores_triage_only(self, stage, triage_on):
        # Stored retire records carry only the triage record and proof.
        from alpharidge_ai.triage import build_proof_of_read, build_triage_record
        v = HarnessValidator()
        junk = article(2, JUNK_ARTICLE)
        smuggled = junk.model_copy(update={"analysis": types.SimpleNamespace(
            analysis_data={
                "triage": build_triage_record("irrelevant", "non_economic"),
                "proof_of_read": build_proof_of_read(*JUNK_ARTICLE),
                "event_fingerprint": {"content_hash": "x", "event_type": "other"},
                "topic_signature": {"primary_sector_id": 13},
                "title_embedding": [0.1] * 384,
            })})
        res = v._grade_triage([smuggled], [junk], "hk")
        # A payload beside a non-relevant label is an event; article resets.
        assert ("soft", "analysis_on_nonrelevant") in [(e.kind, e.code) for e in res.events]
        assert 2 not in res.retire_candidate_ids
        v._apply_triage_outcome([smuggled], "hk", res, fp_ids=set(),
                                sent_by_id={2: junk})
        assert 2 not in v._article_store.updated
        assert 2 in v._article_store.reset

        # A clean irrelevant claim retires with the rebuilt triage-only record.
        clean = mine(stage, [junk])
        res2 = v._grade_triage(clean, [junk], "hk")
        assert 2 in res2.retire_candidate_ids
        v._apply_triage_outcome(clean, "hk", res2, fp_ids=set(), sent_by_id={2: junk})
        data = v._article_store.updated[2].analysis.analysis_data
        assert data["triage"]["label"] == "irrelevant"
        assert data["proof_of_read"]
        assert "event_fingerprint" not in data and "topic_signature" not in data

    def test_d4_fee_floor_small_batch(self, stage, triage_on):
        v = HarnessValidator()
        batch = [article(2, JUNK_ARTICLE)]
        returned = mine(stage, batch)
        res = v._grade_triage(returned, batch, "hk")
        v._apply_triage_outcome(returned, "hk", res, fp_ids=set(),
                                sent_by_id={2: batch[0]})
        assert v._miner_reward.points == 1   # int(round(0.2*1)) == 0 before

    def test_d4_analyzed_pos_canary_is_paid(self, stage, triage_on):
        from alpharidge_ai.triage import build_proof_of_read, build_triage_record
        v = HarnessValidator()
        canary = article(10, ASSET_ARTICLE)
        v._canary_pool.add(10, "pos", deterministic=True)
        v._canary_articles[10] = canary
        analysed = canary.model_copy(update={"analysis": types.SimpleNamespace(
            analysis_data={
                "triage": build_triage_record("relevant"),
                "proof_of_read": build_proof_of_read(*ASSET_ARTICLE),
                "event_fingerprint": {"content_hash": "y"},
            })})
        res = v._grade_triage([analysed], [canary], "hk")
        v._apply_triage_outcome([analysed], "hk", res, fp_ids=set())
        assert 10 not in v._article_store.processed   # graded, never stored
        assert v._miner_reward.points == 6             # round(0.2 fee + 6)

    def test_d5_declining_triage_is_an_integrity_failure(self):
        # No triage data -> no proofs -> integrity failures.
        from alpharidge_ai.validator.triage_grader import grade_batch
        items = [{"article_id": 7, "title": "t", "body": "b",
                  "analysis_data": {"assets": []}},
                 {"article_id": 8, "title": "t", "body": "b",
                  "analysis_data": {"assets": []}}]
        cfg = TriageConfig()
        res = grade_batch(items, {}, lambda i: False, lambda i: None, stage_junk,
                          random.Random(0), cfg, enforced=True)
        assert set(res.proof_failures) == {7, 8}
        obs = res.observations(cfg, clean_article_id=7)
        assert all(s == 0.0 for _, s, _ in obs)
        hard_ids = {aid for aid, _, w in obs if w == cfg.hard_weight}
        assert hard_ids == {7, 8}

    def test_d5_analysis_on_nonrelevant_label_costs_something(self):
        from alpharidge_ai.triage import build_proof_of_read, build_triage_record
        from alpharidge_ai.validator.triage_grader import grade_batch
        item = {"article_id": 3, "title": "t", "body": "some body text here",
                "analysis_data": {
                    "triage": build_triage_record("irrelevant", "non_economic"),
                    "proof_of_read": build_proof_of_read("t", "some body text here"),
                    "event_fingerprint": {"content_hash": "z"},
                }}
        rng = random.Random(0)
        res = grade_batch([item], {}, lambda i: False, lambda i: None, stage_junk, rng,
                          TriageConfig(), enforced=False)
        assert ("soft", "analysis_on_nonrelevant") in [(e.kind, e.code) for e in res.events]

    def test_d2_neg_mint_requires_both_framings(self, stage, triage_on):
        class FakeAuditor:
            def __init__(self, verdicts):
                self.verdicts = verdicts
                self.calls = []
            def relevance_verdict(self, title, body, framing="strict"):
                self.calls.append(framing)
                return self.verdicts.get(framing)
        junk = article(20, JUNK_ARTICLE)

        v = HarnessValidator()
        v._triage_auditor = FakeAuditor({"strict": False, "editorial": False})
        v._mint_neg_canaries([junk])
        assert v._canary_pool.size("neg") == 1   # both framings concur
        assert set(v._triage_auditor.calls) == {"strict", "editorial"}

        v2 = HarnessValidator()
        v2._triage_auditor = FakeAuditor({"strict": False, "editorial": None})
        v2._mint_neg_canaries([junk])
        assert v2._canary_pool.size("neg") == 0  # one framing unsure: no mint

        v3 = HarnessValidator()
        v3._triage_auditor = FakeAuditor({"strict": False, "editorial": False})
        asset = article(21, ASSET_ARTICLE)
        v3._mint_neg_canaries([asset])
        assert v3._canary_pool.size("neg") == 0  # stage says relevant: never a neg
        assert v3._triage_auditor.calls == []    # LLM not even consulted


class TestBorderlineLane:
    """Borderline articles: analyze, then flag valuable/not_valuable."""

    def _flagged(self, art, pair, flag, extra=None):
        from alpharidge_ai.triage import build_proof_of_read, build_triage_record
        rec = build_triage_record("borderline")
        rec["flag"] = flag
        data = {"triage": rec, "proof_of_read": build_proof_of_read(*pair)}
        data.update(extra or {})
        return art.model_copy(update={"analysis": types.SimpleNamespace(
            analysis_data=data)})

    def test_valuable_flag_is_kept_and_paid(self, stage, triage_on):
        v = HarnessValidator()
        art = article(1, MACRO_ARTICLE)
        returned = [self._flagged(art, MACRO_ARTICLE, "valuable", {
            "event_fingerprint": {"content_hash": "x"},
            "economic_data": [{"event_name": "rate decision"}],
        })]
        res = v._grade_triage(returned, [art], "hk")
        assert res.borderline_valuable_ids == [1] and not res.events
        v._apply_triage_outcome(returned, "hk", res, fp_ids=set(), sent_by_id={1: art})
        assert 1 in v._article_store.processed
        assert v._article_store.updated[1].analysis.analysis_data.get("economic_data")
        assert v._miner_reward.points == 6            # round(0.2 fee + 6)

    def test_discard_flag_stores_triage_only_unpaid(self, stage, triage_on):
        # Genuinely ambiguous article (the reference stage also says borderline).
        v = HarnessValidator()
        art = article(2, AMBIGUOUS_ARTICLE)
        returned = [self._flagged(art, AMBIGUOUS_ARTICLE, "not_valuable", {
            "event_fingerprint": {"content_hash": "y"},
            "topic_signature": {"primary_sector_symbol": "OTHER"},
        })]
        res = v._grade_triage(returned, [art], "hk")
        assert res.borderline_discard_ids == [2] and not res.events
        v._apply_triage_outcome(returned, "hk", res, fp_ids=set(), sent_by_id={2: art})
        assert 2 in v._article_store.processed
        data = v._article_store.updated[2].analysis.analysis_data
        assert "event_fingerprint" not in data          # stored triage-only
        assert v._miner_reward.points == 1               # fee floor only: discards are unpaid

    def test_borderline_on_plain_junk_is_unwarranted_and_unpaid(self, stage, triage_on):
        # Farming shape: junk labeled borderline, analyzed, honestly discarded.
        # The reference stage saw no ambiguity, so the claim costs reputation
        # and earns nothing beyond the fee.
        v = HarnessValidator()
        art = article(5, JUNK_ARTICLE)
        returned = [self._flagged(art, JUNK_ARTICLE, "not_valuable", {
            "event_fingerprint": {"content_hash": "q"},
            "topic_signature": {"primary_sector_symbol": "OTHER"},
        })]
        res = v._grade_triage(returned, [art], "hk")
        assert ("soft", "borderline_unwarranted") in [(e.kind, e.code) for e in res.events]
        assert not res.borderline_discard_ids
        v._apply_triage_outcome(returned, "hk", res, fp_ids=set(), sent_by_id={5: art})
        assert v._miner_reward.points == 1              # fee only
        assert 5 in v._article_store.reset

    def test_flag_contradicting_own_analysis_is_flagged_and_resets(self, stage, triage_on):
        v = HarnessValidator()
        art = article(3, JUNK_ARTICLE)
        # Claims not_valuable while the analysis shows economic data.
        returned = [self._flagged(art, JUNK_ARTICLE, "not_valuable", {
            "event_fingerprint": {"content_hash": "z"},
            "economic_data": [{"event_name": "cpi"}],
        })]
        res = v._grade_triage(returned, [art], "hk")
        assert ("soft", "borderline_flag_mismatch") in [(e.kind, e.code) for e in res.events]
        assert not res.borderline_discard_ids
        v._apply_triage_outcome(returned, "hk", res, fp_ids=set(), sent_by_id={3: art})
        assert 3 in v._article_store.reset and 3 not in v._article_store.processed

    def test_unflagged_borderline_still_bounces(self, stage, triage_on):
        v = HarnessValidator()
        art = article(4, JUNK_ARTICLE)
        returned = mine(stage, [art], strategy="honest")
        from alpharidge_ai.triage import build_triage_record
        rec = build_triage_record("borderline")
        returned[0].analysis.analysis_data["triage"] = rec
        res = v._grade_triage(returned, [art], "hk")
        assert res.borderline_ids == [4]
        v._apply_triage_outcome(returned, "hk", res, fp_ids=set(), sent_by_id={4: art})
        assert 4 in v._article_store.reset

    def test_valuable_borderline_joins_deep_validation(self, stage, triage_on):
        from alpharidge_ai.triage import analysis_indicates_value
        assert analysis_indicates_value({"economic_data": [{"event_name": "gdp"}]})
        assert analysis_indicates_value({"topic_signature": {"primary_sector_symbol": "MACRO"}})
        assert not analysis_indicates_value({"topic_signature": {"primary_sector_symbol": "OTHER"}})
        assert not analysis_indicates_value(None)


class TestOverlap:
    """Overlap: split-pot pay, verification registry, variant capture."""

    def _bind(self, v):
        for name in ("_register_verification", "_pop_verification",
                     "_prune_verification", "_buffer_variants",
                     "_apply_verification_outcome"):
            setattr(v, name, getattr(validator_module.Validator, name).__get__(v))
        v._verification_pending = {}
        v._variant_buffer = []
        return v

    def test_enforced_primary_pay_is_split(self, stage, triage_on, monkeypatch):
        v = HarnessValidator()
        batch = [article(1, ASSET_ARTICLE), article(2, JUNK_ARTICLE),
                 article(3, MACRO_ARTICLE), article(4, JUNK_ARTICLE)]
        returned = mine(stage, batch)
        res = v._grade_triage(returned, batch, "hk")
        monkeypatch.setattr(config, "TRIAGE_ENFORCED", True, raising=False)
        v._apply_triage_outcome(returned, "hk", res, fp_ids=set(),
                                sent_by_id={int(a.id): a for a in batch})
        # k=3: fee 1/3 + two relevant at 6*1/3 each -> round(4.33) = 4
        assert v._miner_reward.points == 4

    def test_unenforced_pay_unchanged(self, stage, triage_on, monkeypatch):
        v = HarnessValidator()
        batch = [article(1, ASSET_ARTICLE), article(2, JUNK_ARTICLE),
                 article(3, MACRO_ARTICLE), article(4, JUNK_ARTICLE)]
        returned = mine(stage, batch)
        res = v._grade_triage(returned, batch, "hk")
        monkeypatch.setattr(config, "TRIAGE_ENFORCED", False, raising=False)
        v._apply_triage_outcome(returned, "hk", res, fp_ids=set(),
                                sent_by_id={int(a.id): a for a in batch})
        assert v._miner_reward.points == 1 + 6 * 2   # k=1: original rates

    def test_verification_outcome_pays_split_and_buffers_variants(self, stage, triage_on):
        from alpharidge_ai.triage import build_proof_of_read, build_triage_record
        v = self._bind(HarnessValidator())
        art = article(1, ASSET_ARTICLE)
        analysed = art.model_copy(update={"analysis": types.SimpleNamespace(
            analysis_data={"triage": build_triage_record("relevant"),
                           "proof_of_read": build_proof_of_read(*ASSET_ARTICLE),
                           "event_fingerprint": {"content_hash": "v"}})})
        res = v._grade_triage([analysed], [art], "hk")
        import time as _t
        v._article_k["1"] = (3, _t.time())            # dispatched to 3 assignees
        v._apply_verification_outcome([analysed], "hk", res, fp_ids=set())
        assert v._miner_reward.points == 2            # round(0.2/3 + 6/3)
        assert len(v._variant_buffer) == 1
        assert v._variant_buffer[0]["miner_hotkey"] == "hk"
        assert not v._article_store.processed and not v._article_store.reset

    def test_verification_registry_roundtrip_and_ttl(self, stage, triage_on, monkeypatch):
        v = self._bind(HarnessValidator())
        art = article(9, JUNK_ARTICLE)
        v._register_verification("hkv", [art])
        assert v._pop_verification("hkv", "9").id == 9
        assert v._pop_verification("hkv", "9") is None   # single use
        v._register_verification("hkv", [art])
        monkeypatch.setattr(validator_module.TRIAGE_CFG, "verification_ttl_s", -1)
        v._prune_verification()
        assert v._pop_verification("hkv", "9") is None   # expired
