"""The reference an audit is scored against comes from the model drawn for the article."""

import types

from alpharidge_ai.analyzer import scoring
from alpharidge_ai.analyzer.article_intelligence_analyzer import ArticleIntelligenceAnalyzer
from alpharidge_ai.mechanism import profile as mp
from alpharidge_ai.oracle.runner import Auditor, Observation
from tests.test_keyed_audit_pass import NUMBERS, _article
from tests.test_profile_client import valid


def _profile(models):
    raw = valid()
    raw["oracle"]["keyed_rate_pool"] = 1.0
    raw["oracle"]["grader_models"] = models
    return mp.parse(raw)


MODELS = [{"id": "model-a", "weight": 0.6}, {"id": "model-b", "weight": 0.4}]


def test_the_reference_model_is_the_drawn_grader():
    profile = _profile(MODELS)
    auditor = Auditor(b"k", lambda block: profile)
    for article_id in range(50):
        choice = auditor._selector.select(
            article_id, NUMBERS, pool_tiers=profile.oracle.pool_tiers,
            keyed_rate_pool=1.0, keyed_rate_keeper=0.0,
            grader_models=profile.oracle.grader_models)
        assert auditor.reference_model(article_id, 0) == choice.grader_model


def test_both_models_are_drawn():
    profile = _profile(MODELS)
    auditor = Auditor(b"k", lambda block: profile)
    drawn = {auditor.reference_model(i, 0) for i in range(200)}
    assert drawn == {"model-a", "model-b"}


def test_no_profile_means_the_analyzers_own_model():
    assert Auditor(b"k", lambda block: None).reference_model(1, 0) == ""


def test_the_llm_call_uses_the_model_it_is_given():
    seen = []

    def create(**kwargs):
        seen.append(kwargs["model"])
        return types.SimpleNamespace(choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(tool_calls=None))])

    analyzer = ArticleIntelligenceAnalyzer.__new__(ArticleIntelligenceAnalyzer)
    analyzer.model = "default"
    analyzer.client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    analyzer._llm_call("p", {}, "t", "model-b")
    analyzer._llm_call("p", {}, "t")
    assert seen == ["model-b", "default"]


class DrawingAuditor:
    def __init__(self, model):
        self.model = model

    def selects(self, article_id, text, block):
        return True

    def reference_model(self, article_id, block):
        return self.model

    def audit(self, article_id, text, miner_intel, grader_intel, result, block):
        return Observation(article_id=article_id, score=0.5, weight=1.0, path="pool",
                           grader_model=getattr(grader_intel, "model", ""))


class ModelAnalyzer:
    model = "default"

    def __init__(self):
        self.models = []

    def analyze(self, model=None, **kwargs):
        self.models.append(model or self.model)
        return types.SimpleNamespace(numeric_claims=[], quotes=[],
                                     model=model or self.model)


def _run(monkeypatch, auditor, analyzer, size=6):
    from tests.test_floor_gating import _payload, TEXT, TITLE

    monkeypatch.setattr(scoring, "_cfg_get",
                        lambda k, d=None: 2 if k == "AUDIT_MAX_PER_BATCH" else d)
    blob = _payload([{"metric_name": "revenue", "value": 1.2e9, "unit": "USD",
                      "confidence": 0.9}])
    batch = [_article(i, blob, TEXT) for i in range(1, size + 1)]
    for a in batch:
        a.title = TITLE
    monkeypatch.setattr(scoring, "validate_article_intelligence",
                        lambda m, v: (True, 1.0, {}))
    monkeypatch.setattr(scoring, "_summary_agreement", lambda m, v: 1.0)
    _, result = scoring.validate_miner_article_intelligence_batch(
        batch, analyzer, sample_size=1, auditor=auditor, block=0)
    return result


def test_every_audit_is_scored_against_the_drawn_model(monkeypatch):
    analyzer = ModelAnalyzer()
    result = _run(monkeypatch, DrawingAuditor("model-b"), analyzer)
    observed = result["audit_observations"]
    assert observed
    assert {o.grader_model for o in observed} == {"model-b"}


def test_the_sample_analysis_is_reused_when_the_draw_matches(monkeypatch):
    analyzer = ModelAnalyzer()
    _run(monkeypatch, DrawingAuditor("default"), analyzer)
    assert analyzer.models.count("default") == len(analyzer.models)
    assert len(analyzer.models) == 1 + 2


def test_the_reference_run_is_extraction_only(monkeypatch):
    analyzer = ModelAnalyzer()
    seen = []
    original = analyzer.analyze

    def record(**kwargs):
        seen.append(kwargs.get("reference"))
        return original(**kwargs)

    analyzer.analyze = record
    _run(monkeypatch, DrawingAuditor("model-b"), analyzer)
    assert seen[0] is None          # the acceptance analysis
    assert seen[1:] and all(seen[1:])
