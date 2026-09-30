"""An exhausted API key pauses the validator's own LLM calls instead of retrying."""

import types

import pytest

from alpharidge_ai.analyzer.article_intelligence_analyzer import ReferenceUnavailable
from alpharidge_ai.utils import llm_spend
from tests import test_reference_unavailable as ru


class KeyError403(Exception):
    status_code = 403

    def __str__(self):
        return "Error code: 403 - Key limit exceeded (monthly limit)"


class Credits402(Exception):
    status_code = 402

    def __str__(self):
        return "Error code: 402 - Insufficient credits"


class Other403(Exception):
    status_code = 403

    def __str__(self):
        return "Error code: 403 - Forbidden"


def test_what_counts_as_a_key_limit():
    assert llm_spend.is_key_limit(KeyError403()) and llm_spend.is_key_limit(Credits402())
    assert not llm_spend.is_key_limit(Other403())
    assert not llm_spend.is_key_limit(RuntimeError("limit"))


def test_a_key_limit_is_not_retried_and_pauses_later_calls():
    seen = []
    with pytest.raises(ReferenceUnavailable):
        ru._analyzer([KeyError403(), ru._reply("{}")], seen)._llm_call("p", {}, "t", strict=True)
    assert len(seen) == 1 and llm_spend.paused()
    seen.clear()
    with pytest.raises(ReferenceUnavailable):
        ru._analyzer([ru._reply("{}")], seen)._llm_call("p", {}, "t", strict=True)
    assert seen == []


def test_a_miner_side_call_is_unaffected():
    llm_spend.pause(KeyError403())
    assert ru._analyzer([ru._reply('{"x": 1}')])._llm_call("p", {}, "t") == {"x": 1}


def test_the_pause_ends():
    llm_spend.pause(KeyError403())
    llm_spend._paused_until[0] = 0.0
    assert not llm_spend.paused()


def test_triage_and_grader_calls_stop_while_paused():
    from alpharidge_ai.oracle.grader import Grader
    from alpharidge_ai.validator.triage_audit import TriageAuditor
    calls = []

    def create(**kw):
        calls.append(kw)
        raise KeyError403()
    client = types.SimpleNamespace(base_url="x", chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(create=create)))
    aud = TriageAuditor(client, "m")
    assert aud.relevance_verdict("t", "b") is None and llm_spend.paused() and len(calls) == 1
    assert aud.relevance_verdict("t", "b") is None and len(calls) == 1
    g = Grader(client)
    assert g._call("m", "p", {"function": {"name": "n"}}) is None and len(calls) == 1


def _one_of_two(monkeypatch):
    from alpharidge_ai.analyzer import scoring
    from tests.test_floor_gating import _payload, TEXT, TITLE
    from tests.test_keyed_audit_pass import _article
    monkeypatch.setattr(scoring, "validate_article_intelligence", lambda m, v: (True, 1.0, {}))
    monkeypatch.setattr(scoring, "_summary_agreement", lambda m, v: 1.0)
    blob = _payload([{"metric_name": "revenue", "value": 1.2e9, "unit": "USD", "confidence": 0.9}])
    batch = [_article(i, blob, TEXT) for i in (1, 2)]
    for a in batch:
        a.title = TITLE
    return scoring.validate_miner_article_intelligence_batch(batch, ru._Refs(failing=[1, 2]), sample_size=1)


def test_a_failed_sample_is_replaced_normally(monkeypatch):
    ok, result = _one_of_two(monkeypatch)
    assert result["skipped_samples"] == 2 and result["no_verdict"]


def test_no_replacement_samples_while_paused(monkeypatch):
    llm_spend.pause(KeyError403())
    ok, result = _one_of_two(monkeypatch)
    assert result["skipped_samples"] == 1 and result["no_verdict"] and not ok
