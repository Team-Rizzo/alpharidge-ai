"""Own-model validator calls ask for the tool first and force it only on a retry."""

import types

import pytest

from alpharidge_ai import config
from alpharidge_ai.utils import llm_spend
from alpharidge_ai.validator.triage_audit import TriageAuditor
from tests import test_reference_unavailable as ru

FORCED = {"type": "function", "function": {"name": "t"}}


@pytest.fixture(autouse=True)
def _auto_on(monkeypatch):
    monkeypatch.setattr(config, "REFERENCE_TOOL_AUTO", True, raising=False)
    monkeypatch.setattr(llm_spend, "_auto", [0, 0])


def test_first_ask_is_not_forced():
    seen = []
    assert ru._analyzer([ru._reply('{"x": 1}')], seen)._llm_call("p", {}, "t", strict=True) == {"x": 1}
    assert len(seen) == 1 and seen[0]["tool_choice"] == "auto"
    assert seen[0]["messages"][0]["content"].endswith("Answer only by calling t.")
    assert llm_spend._auto == [1, 0]


@pytest.mark.parametrize("first", [ru._reply(None), ru._reply('{"x": ')])
def test_a_miss_is_asked_again_with_the_tool_forced(first):
    seen = []
    assert ru._analyzer([first, ru._reply('{"x": 2}')], seen)._llm_call("p", {}, "t", strict=True) == {"x": 2}
    assert [k["tool_choice"] for k in seen] == ["auto", FORCED]
    assert seen[1]["messages"][0]["content"] == "p"
    assert llm_spend._auto == [0, 1]


def test_grader_models_and_miner_calls_stay_forced():
    seen = []
    ru._analyzer([ru._reply('{"x": 1}')], seen)._llm_call("p", {}, "t", model="other/grader", strict=True)
    ru._analyzer([ru._reply('{"x": 1}')], seen)._llm_call("p", {}, "t")
    assert [k["tool_choice"] for k in seen] == [FORCED, FORCED]


def test_the_switch_turns_it_off(monkeypatch):
    monkeypatch.setattr(config, "REFERENCE_TOOL_AUTO", False)
    seen = []
    ru._analyzer([ru._reply('{"x": 1}')], seen)._llm_call("p", {}, "t", strict=True)
    assert seen[0]["tool_choice"] == FORCED and seen[0]["messages"][0]["content"] == "p"


def _judge(replies, seen):
    queue = list(replies)

    def create(**kw):
        seen.append(kw)
        args = queue.pop(0)
        calls = None if args is None else [types.SimpleNamespace(function=types.SimpleNamespace(arguments=args))]
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(tool_calls=calls))])
    client = types.SimpleNamespace(base_url="x", chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    return TriageAuditor(client, "m")


def test_the_judge_asks_first_and_retries_forced():
    seen = []
    assert _judge(['{"relevant": false, "confidence": 0.9}'], seen).relevance_verdict("t", "b") is False
    assert seen[0]["tool_choice"] == "auto"
    seen = []
    assert _judge([None, '{"relevant": true, "confidence": 0.9}'], seen).relevance_verdict("t", "b") is True
    assert seen[0]["tool_choice"] == "auto" and seen[1]["tool_choice"]["type"] == "function"


@pytest.mark.parametrize("value,expected", [('"false"', False), ('"True"', True), ('"maybe"', None), ("1", None)])
def test_judge_verdicts_given_as_text(value, expected):
    assert _judge([f'{{"relevant": {value}, "confidence": 0.9}}'], []).relevance_verdict("t", "b") is expected


def test_the_hourly_summary_reports_retries(monkeypatch):
    lines = []
    monkeypatch.setattr(llm_spend.bt.logging, "info", lambda m, *a, **k: lines.append(m))
    monkeypatch.setattr(llm_spend, "_last_report", [0.0])
    llm_spend.note_auto(True); llm_spend.note_auto(False)
    llm_spend.record("reference:tier3", "m")
    assert "tool calls 1 first ask, 1 retried" in lines[-1]
