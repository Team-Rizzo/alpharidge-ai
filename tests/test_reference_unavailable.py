"""A validator-side call that returns nothing usable is never graded against."""

import json
import types

import pytest

from alpharidge_ai import config
from alpharidge_ai.analyzer import scoring
from alpharidge_ai.analyzer.article_intelligence_analyzer import (ArticleIntelligenceAnalyzer,
                                                                  ReferenceUnavailable)


def _reply(args=None, finish="tool_calls"):
    calls = None if args is None else [types.SimpleNamespace(
        function=types.SimpleNamespace(arguments=args))]
    return types.SimpleNamespace(provider="p1", choices=[types.SimpleNamespace(
        finish_reason=finish, message=types.SimpleNamespace(tool_calls=calls))])


def _analyzer(replies, seen=None):
    queue = list(replies)

    def create(**kwargs):
        if seen is not None:
            seen.append(kwargs)
        r = queue.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    a = ArticleIntelligenceAnalyzer.__new__(ArticleIntelligenceAnalyzer)
    a.model = "m"
    a.client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    return a


def test_a_miner_call_is_unchanged():
    seen = []
    assert _analyzer([_reply(None)], seen)._llm_call("p", {}, "t") == {}
    assert len(seen) == 1
    assert _analyzer([RuntimeError("boom")])._llm_call("p", {}, "t") == {}


def test_a_validator_call_retries_once_then_raises():
    seen = []
    with pytest.raises(ReferenceUnavailable):
        _analyzer([_reply(None), _reply(None)], seen)._llm_call("p", {}, "t", strict=True)
    assert len(seen) == 2


def test_a_cut_off_reply_is_not_a_reference():
    good = json.dumps({"x": 1})
    a = _analyzer([_reply('{"x": 1', finish="length"), _reply(good)])
    assert a._llm_call("p", {}, "t", strict=True) == {"x": 1}
    with pytest.raises(ReferenceUnavailable):
        _analyzer([_reply('{"x": ', "stop"), _reply('{"x": ', "stop")])._llm_call(
            "p", {}, "t", strict=True)


def _routed(replies, seen):
    a = _analyzer(replies, seen)
    a.client.base_url = "https://openrouter.ai/api/v1"
    return a


def test_providers_can_be_excluded_for_validator_calls(monkeypatch):
    monkeypatch.setattr(config, "REFERENCE_PROVIDER_IGNORE", "BadCo, Other")
    monkeypatch.setattr(config, "REFERENCE_PROVIDER_ORDER", "")
    seen = []
    _routed([_reply("{}")], seen)._llm_call("p", {}, "t", strict=True)
    assert seen[0]["extra_body"]["provider"] == {"ignore": ["BadCo", "Other"]}
    seen.clear()
    _routed([_reply("{}")], seen)._llm_call("p", {}, "t")
    assert "extra_body" not in seen[0]


def test_routing_only_goes_to_openrouter(monkeypatch):
    monkeypatch.setattr(config, "REFERENCE_PROVIDER_ORDER", "GoodCo")
    seen = []
    _analyzer([_reply("{}")], seen)._llm_call("p", {}, "t", strict=True)
    assert "extra_body" not in seen[0]


def test_the_served_order_applies(monkeypatch):
    monkeypatch.delenv("OVERRIDE_REFERENCE_PROVIDER_ORDER", raising=False)
    monkeypatch.setattr(config, "REFERENCE_PROVIDER_ORDER", "")
    cast, attr = config._REMOTE_CONFIG_KEYS["REFERENCE_PROVIDER_ORDER"]
    monkeypatch.setattr(config, attr, cast("GoodCo,Other"))
    seen = []
    _routed([_reply("{}")], seen)._llm_call("p", {}, "t", strict=True)
    assert seen[0]["extra_body"]["provider"]["order"] == ["GoodCo", "Other"]
    assert "REFERENCE_PROVIDER_ORDER" not in config._CONSENSUS_KEYS


def test_the_grader_uses_the_same_routing(monkeypatch):
    monkeypatch.setattr(config, "REFERENCE_PROVIDER_ORDER", "GoodCo")
    from alpharidge_ai.utils import llm_spend
    client = types.SimpleNamespace(base_url="https://openrouter.ai/api/v1")
    body = llm_spend.request_body(client)
    assert body["provider"] == {"order": ["GoodCo"], "allow_fallbacks": True} and body["usage"]
    assert llm_spend.request_body(types.SimpleNamespace(base_url="https://llm.chutes.ai/v1")) == {}


class _Refs:
    """Validator analyzer whose reference fails for the listed article ids."""
    model = "m"

    def __init__(self, failing):
        self.failing = set(failing)

    def analyze(self, article_id=None, **kwargs):
        if int(article_id) in self.failing:
            return None
        return types.SimpleNamespace(numeric_claims=[], quotes=[], assets=[], economic_data=[])


def _validate(monkeypatch, failing, verdict=(True, 1.0, {})):
    from tests.test_floor_gating import _payload, TEXT, TITLE
    from tests.test_keyed_audit_pass import _article
    monkeypatch.setattr(scoring, "validate_article_intelligence", lambda m, v: verdict)
    monkeypatch.setattr(scoring, "_summary_agreement", lambda m, v: 1.0)
    blob = _payload([{"metric_name": "revenue", "value": 1.2e9, "unit": "USD", "confidence": 0.9}])
    batch = [_article(i, blob, TEXT) for i in (1, 2)]
    for a in batch:
        a.title = TITLE
    return scoring.validate_miner_article_intelligence_batch(batch, _Refs(failing), sample_size=2)


def test_a_sample_without_a_reference_is_skipped(monkeypatch):
    ok, result = _validate(monkeypatch, failing=[1])
    assert ok and result["skipped_samples"] == 1 and result["total_sampled"] == 1
    assert not result["discrepancies"]


def test_no_reference_at_all_gives_no_verdict(monkeypatch):
    ok, result = _validate(monkeypatch, failing=[1, 2], verdict=(False, 0.1, {}))
    assert not ok and result["no_verdict"] and result["skipped_samples"] == 2
    assert not result["discrepancies"]


def test_sample_replacement(monkeypatch):
    from tests.test_floor_gating import _payload, TEXT, TITLE
    from tests.test_keyed_audit_pass import _article
    monkeypatch.setattr(scoring, "validate_article_intelligence", lambda m, v: (True, 1.0, {}))
    monkeypatch.setattr(scoring, "_summary_agreement", lambda m, v: 1.0)
    blob = _payload([{"metric_name": "revenue", "value": 1.2e9, "unit": "USD", "confidence": 0.9}])
    batch = [_article(i, blob, TEXT) for i in (1, 2)]
    for a in batch:
        a.title = TITLE
    refs = _Refs(failing=[])
    first = []

    def analyze(article_id=None, **kw):
        if not first:
            first.append(article_id)
            return None
        return types.SimpleNamespace(numeric_claims=[], quotes=[], assets=[], economic_data=[])
    refs.analyze = analyze
    ok, result = scoring.validate_miner_article_intelligence_batch(batch, refs, sample_size=1)
    assert ok and result["skipped_samples"] == 1 and result["total_sampled"] == 1


def test_a_real_miss_still_fails(monkeypatch):
    ok, result = _validate(monkeypatch, failing=[1], verdict=(False, 0.3, {}))
    assert not ok and result["discrepancies"][0]["reason"] == "validation_failed"


def test_malformed_list_items_do_not_break_the_fact_sheet():
    a = ArticleIntelligenceAnalyzer.__new__(ArticleIntelligenceAnalyzer)
    ner = types.SimpleNamespace(resolved_entities=[], sentence_sentiments=[])
    call1 = {"quotes": ["a bare quote", {"speaker": "CEO", "text": "we grew"}],
             "economic_data": ["not an object", {"event_name": "CPI", "actual_value": 3.1}]}
    sheet = a._build_fact_sheet("t", "s", None, {"symbol": "S"}, call1, ner, [])
    assert 'Quote: ?: "a bare quote"' in sheet and "CPI: 3.1" in sheet


def test_a_preferred_provider_keeps_fallbacks(monkeypatch):
    monkeypatch.setattr(config, "REFERENCE_PROVIDER_ORDER", "GoodCo")
    monkeypatch.setattr(config, "REFERENCE_PROVIDER_IGNORE", "")
    seen = []
    _routed([_reply("{}")], seen)._llm_call("p", {}, "t", strict=True)
    assert seen[0]["extra_body"]["provider"] == {"order": ["GoodCo"], "allow_fallbacks": True}


def test_routing_is_served_and_can_be_overridden_locally(monkeypatch):
    from tests.test_consensus_keys import _serve
    monkeypatch.setattr(config, "REFERENCE_PROVIDER_ORDER", "", raising=False)
    monkeypatch.setattr(config, "REFERENCE_PROVIDER_IGNORE", "", raising=False)
    monkeypatch.delenv("OVERRIDE_REFERENCE_PROVIDER_ORDER", raising=False)
    _serve(monkeypatch, {"REFERENCE_PROVIDER_ORDER": "A,B", "REFERENCE_PROVIDER_IGNORE": "C"})
    config.refresh_remote_config(force=True)
    assert config.REFERENCE_PROVIDER_ORDER == "A,B" and config.REFERENCE_PROVIDER_IGNORE == "C"
    monkeypatch.setenv("OVERRIDE_REFERENCE_PROVIDER_ORDER", "Local")
    monkeypatch.setattr(config, "_remote_config_last_fetch", 0.0)
    config.refresh_remote_config(force=True)
    assert config.REFERENCE_PROVIDER_ORDER == "Local"
