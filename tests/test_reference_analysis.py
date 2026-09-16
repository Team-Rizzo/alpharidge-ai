"""The audit reference is extraction only, and one result per article, model and text."""

import json
import types

import pytest

from alpharidge_ai.analyzer.article_intelligence_analyzer import ArticleIntelligenceAnalyzer

TITLE = "Acme lifts guidance"
BODY = ("Acme Corp said on Tuesday that quarterly revenue rose to $1.2 billion, "
        "up 12.5% from a year earlier, and raised its full-year outlook.")

EXTRACT = {
    "content_type": "earnings", "sentiment": "bullish", "sentiment_score": 0.5,
    "numeric_claims": [{"metric_name": "revenue", "value": 1.2e9, "unit": "USD",
                        "confidence": 0.9}],
    "quotes": [], "additional_tickers": [], "economic_data": [],
}


class Client:
    def __init__(self):
        self.tools = []
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        name = kwargs["tool_choice"]["function"]["name"]
        self.tools.append(name)
        args = EXTRACT if name == "extract_and_classify" else {
            "headline": "h", "one_liner": "o", "context_paragraph": "c"}
        call = types.SimpleNamespace(function=types.SimpleNamespace(
            arguments=json.dumps(args)))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(tool_calls=[call]))])


@pytest.fixture(scope="module")
def analyzer():
    return ArticleIntelligenceAnalyzer(api_key="x", llm_base="http://127.0.0.1:9",
                                       enable_refined=False, enable_flair=False)


@pytest.fixture
def client(analyzer):
    stub = Client()
    analyzer.client = stub
    analyzer._reference_cache = type(analyzer._reference_cache)()
    return stub


def _run(analyzer, article_id=1, body=BODY, **kwargs):
    return analyzer.analyze(article_id=article_id, url="u", title=TITLE, source="s",
                            published="2026-09-16T00:00:00Z", content=body, **kwargs)


def test_a_reference_run_skips_the_summary_call(analyzer, client):
    result = _run(analyzer, reference=True)
    assert result is not None
    assert client.tools == ["extract_and_classify"]
    assert [c.metric_name for c in result.numeric_claims] == ["revenue"]


def test_a_normal_run_still_summarises(analyzer, client):
    _run(analyzer)
    assert client.tools == ["extract_and_classify", "reason_and_summarize"]


def test_the_reference_is_computed_once_per_article(analyzer, client):
    first = _run(analyzer, reference=True)
    second = _run(analyzer, reference=True)
    assert client.tools == ["extract_and_classify"]
    assert second.numeric_claims == first.numeric_claims
    assert second is not first


def test_different_text_is_a_different_reference(analyzer, client):
    _run(analyzer, reference=True)
    _run(analyzer, reference=True, body=BODY + " Shares rose 4%.")
    assert client.tools == ["extract_and_classify"] * 2


def test_a_different_article_is_a_different_reference(analyzer, client):
    _run(analyzer, article_id=1, reference=True)
    _run(analyzer, article_id=2, reference=True)
    assert client.tools == ["extract_and_classify"] * 2


def test_normal_runs_are_never_cached(analyzer, client):
    _run(analyzer)
    _run(analyzer)
    assert client.tools.count("extract_and_classify") == 2


def test_a_run_for_a_hotkey_is_never_cached(analyzer, client):
    _run(analyzer, reference=True, miner_hotkey="hk")
    _run(analyzer, reference=True, miner_hotkey="hk")
    assert client.tools.count("extract_and_classify") == 2


def test_each_model_has_its_own_reference(analyzer, client):
    _run(analyzer, reference=True, model="model-a")
    _run(analyzer, reference=True, model="model-b")
    _run(analyzer, reference=True, model="model-a")
    assert client.tools == ["extract_and_classify"] * 2
