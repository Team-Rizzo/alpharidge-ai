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
MACRO_ARTICLE = (
    "Brazil central bank cuts rates",
    "The central bank of Brazil announced a surprise interest rate cut on "
    "Thursday, citing slowing inflation. Economists expect further easing.",
)


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
    _llm_relevant_item = validator_module.Validator._llm_relevant_item
    _get_triage_auditor = validator_module.Validator._get_triage_auditor
    _feed_pos_canaries = validator_module.Validator._feed_pos_canaries
    _inject_canaries = validator_module.Validator._inject_canaries
    _grade_triage = validator_module.Validator._grade_triage
    _record_triage_observations = validator_module.Validator._record_triage_observations
    _apply_triage_outcome = validator_module.Validator._apply_triage_outcome

    def __init__(self):
        self._canary_pool = CanaryPool(TriageConfig())
        self._canary_articles = {}
        self._triage_extractor = None
        self._triage_auditor = None
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
def triage_on(monkeypatch):
    monkeypatch.setattr(config, "TRIAGE_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "TRIAGE_AUDIT_LLM_ENABLED", False, raising=False)
    monkeypatch.setattr(config, "TRIAGE_FEE_POINTS", 0.2, raising=False)
    monkeypatch.setattr(config, "TRIAGE_REL_POINT_MULT", 5, raising=False)


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

        res = v._grade_triage(returned, "hk1")
        assert res is not None and not res.v2_grace
        assert not res.events and not res.proof_failures
        assert set(res.relevant_ids) == {1, 3}
        assert set(res.retire_candidate_ids) == {2, 4}

        v._record_triage_observations("hk1", res, returned)
        assert v.observations == [(1, 1.0, 1.0)]   # one clean-batch observation

        v._apply_triage_outcome(returned, "hk1", res, fp_ids=set())
        # fee round(0.2*4)=1, plus 5x length-weight(1, short fixtures) per relevant
        assert v._miner_reward.points == 1 + 5 * 2
        assert v._article_store.processed == {1, 2, 3, 4}
        assert v._article_store.rewarded == {1, 3}
        assert not v._article_store.reset

    def test_lazy_miner_caught_by_deterministic_audit(self, stage, triage_on):
        v = HarnessValidator()
        batch = [article(1, ASSET_ARTICLE), article(2, JUNK_ARTICLE),
                 article(3, JUNK_ARTICLE)]
        returned = mine(stage, batch, strategy="lazy")

        res = v._grade_triage(returned, "hk2")
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
        res = v._grade_triage(returned, "hk3")
        assert set(res.proof_failures) == {1, 2}
        v._record_triage_observations("hk3", res, returned)
        assert v.observations and all(s == 0.0 for _, s, _ in v.observations)

    def test_spam_miner_earns_nothing_for_junk(self, stage, triage_on):
        v = HarnessValidator()
        batch = [article(1, ASSET_ARTICLE), article(2, JUNK_ARTICLE),
                 article(3, JUNK_ARTICLE)]
        returned = mine(stage, batch, strategy="spam")
        res = v._grade_triage(returned, "hk4")
        assert set(res.relevant_ids) == {1, 2, 3}
        # Reference analysis refutes the two junk articles.
        v._apply_triage_outcome(returned, "hk4", res, fp_ids={2, 3})
        assert v._miner_reward.points == 1 + 5   # fee + only the real one

    def test_v2_miner_takes_grace_path(self, stage, triage_on):
        v = HarnessValidator()
        batch = [article(1, ASSET_ARTICLE), article(2, JUNK_ARTICLE)]
        legacy = [a.model_copy(update={"analysis": types.SimpleNamespace(
            analysis_data={"schema_version": 2, "title": a.title})}) for a in batch]
        res = v._grade_triage(legacy, "hk5")
        assert res.v2_grace and not res.events

    def test_triage_disabled_is_inert(self, stage, monkeypatch):
        monkeypatch.setattr(config, "TRIAGE_ENABLED", False, raising=False)
        v = HarnessValidator()
        returned = mine(stage, [article(1, ASSET_ARTICLE)])
        assert v._grade_triage(returned, "hk6") is None


@pytest.fixture
def canaries_certain(monkeypatch, triage_on):
    """Make injection deterministic so canary tests aren't RNG-flaky."""
    monkeypatch.setattr(config, "TRIAGE_CANARY_POS_RATE", 1.0, raising=False)
    monkeypatch.setattr(config, "TRIAGE_CANARY_NEG_RATE", 0.0, raising=False)


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

        res = v._grade_triage(returned, "hk7")
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
        res = v._grade_triage(mine(stage, batch), "hk8")
        assert not res.events
