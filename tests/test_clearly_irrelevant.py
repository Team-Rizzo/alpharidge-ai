"""An article is only declared irrelevant, for canaries and penalties, beyond doubt."""

import types

from alpharidge_ai.validator import triage_audit
from alpharidge_ai.validator.triage_audit import TriageAuditor, names_a_listing
from alpharidge_ai.validator.triage_grader import TriageConfig, TriageEvent, TriageGradeResult, fp_soft_event


def test_listing_notation():
    assert names_a_listing("", "トーア紡コーポレーション<3204.T>がこの日の取引終了後に")
    assert names_a_listing("", "shares of Acme (NASDAQ: ACME) rose")
    assert names_a_listing("", "HKEX:1211 closed higher")
    assert names_a_listing("$TSLA jumps", "")
    assert not names_a_listing("Local choir wins regional competition", "A crowd of hundreds cheered.")
    assert not names_a_listing("", "It cost $5 at the door")


class _Judge(TriageAuditor):
    def __init__(self, verdicts):
        self.verdicts, self.calls = verdicts, []

    def relevance_verdict(self, title, body, framing="strict"):
        self.calls.append(framing)
        return self.verdicts.get(framing)


def test_both_readings_must_agree():
    assert _Judge({"strict": False, "lenient": False}).clearly_irrelevant("t", "b")
    assert not _Judge({"strict": False, "lenient": True}).clearly_irrelevant("t", "b")
    assert not _Judge({"strict": False, "lenient": None}).clearly_irrelevant("t", "b")
    assert not _Judge({"strict": True, "lenient": False}).clearly_irrelevant("t", "b")


def test_a_listing_is_never_irrelevant_and_costs_no_call():
    j = _Judge({"strict": False, "lenient": False})
    assert not j.clearly_irrelevant("Toabo buyback", "<3204.T> announced a buyback")
    assert j.calls == []


def test_the_lenient_prompt_carries_the_article():
    seen = []

    def create(**kw):
        seen.append(kw)
        call = types.SimpleNamespace(function=types.SimpleNamespace(
            arguments='{"relevant": false, "confidence": 0.9}'))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(tool_calls=[call]))])
    client = types.SimpleNamespace(base_url="https://openrouter.ai/api/v1",
                                   chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    assert TriageAuditor(client, "m").relevance_verdict("A title", "The body text", "lenient") is False
    prompt = seen[0]["messages"][0]["content"]
    assert "A title" in prompt and "The body text" in prompt and "any language" in prompt
    assert seen[0]["extra_body"]["usage"]
    assert set(triage_audit._PROMPTS) == {"strict", "lenient"}


def _score(events):
    res = TriageGradeResult(batch_size=4, events=events)
    return dict((a, s) for a, s, _ in res.observations(TriageConfig(), clean_article_id=1))[1]


def test_a_flagged_canary_does_not_lower_the_score():
    assert _score([TriageEvent("soft", "canary_neg_flagged", 3)]) == 1.0
    assert _score([fp_soft_event(3)]) < 1.0
