"""The validator's own LLM spend is summarised by purpose and model."""

import types

from alpharidge_ai.utils import llm_spend


def _resp(cost):
    return types.SimpleNamespace(usage=types.SimpleNamespace(cost=cost))


def test_cost_is_asked_for_only_from_openrouter():
    assert llm_spend.usage_body(types.SimpleNamespace(base_url="https://openrouter.ai/api/v1/")) == \
        {"usage": {"include": True}}
    assert llm_spend.usage_body(types.SimpleNamespace(base_url="https://llm.example/v1")) == {}
    assert llm_spend.usage_body(types.SimpleNamespace()) == {}


def test_an_hourly_summary_and_a_slow_warning(monkeypatch):
    lines = []
    monkeypatch.setattr(llm_spend.bt.logging, "info", lambda m: lines.append(m))
    monkeypatch.setattr(llm_spend.bt.logging, "warning", lambda m: lines.append(m))
    llm_spend._totals.clear(); llm_spend._latency.clear()
    llm_spend._last_report[0] = __import__("time").time(); llm_spend._last_slow_warn[0] = 0.0
    monkeypatch.setattr(llm_spend, "REPORT_EVERY_S", 1e9)
    for _ in range(30):
        llm_spend.record("reference", "x/model-a", _resp(0.001), seconds=90)
    assert any("reference calls are slow" in l for l in lines)
    monkeypatch.setattr(llm_spend, "REPORT_EVERY_S", 0.0)
    llm_spend.record("grader:adjudicate", "x/model-b", _resp(0.002))
    summary = [l for l in lines if l.startswith("[LLM_SPEND] $")]
    assert summary and "reference model-a $0.030 (30 calls)" in summary[0]
