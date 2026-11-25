"""The sports wording tried in shadow: asked and logged, never posted."""
import asyncio
import json
import threading

import httpx
import pytest

from alpharidge_ai import config
from alpharidge_ai.judge import prompt, shadow
from alpharidge_ai.judge.client import ScriptedJudge
from alpharidge_ai.schema import canonical_body
from alpharidge_ai.verify.fetch import FetchConfig, Fetcher
from tests.test_judge import claim
from tests.test_validator_loop import JudgeAPI, _deal, _site

SPORT = [
    ("award", {"role": "recipient", "name": "Ana Lee"}, [{"field": "award", "value": "MVP"}]),
    ("injury", {"role": "player", "name": "Ana Lee"}, []),
    ("match_result", {"role": "team", "name": "Reds"}, [{"field": "score", "value": "2-1"}]),
    ("transfer", {"role": "player", "name": "Ana Lee"}, [{"field": "to_team", "value": "Reds"}]),
]


def _sport(event_type, actor, facts):
    return claim(event_type=event_type, category="sports",
                 actors=[{"id_scheme": "other", "id": actor["name"], **actor}], facts=facts)


@pytest.mark.parametrize("event_type,actor,facts", SPORT)
def test_the_shadow_wording_holds_a_sports_type_only_for_sport(event_type, actor, facts):
    c = _sport(event_type, actor, facts)
    live = prompt.judge_questions(c)[0]
    with prompt.wording(shadow.table()):
        q = prompt.judge_questions(c)[0]
    assert "sport" in q["kind"]["criteria"][event_type]
    assert q["event"]["criteria"] != live["event"]["criteria"]
    assert prompt.judge_questions(c)[0] == live and shadow.wanted(c)


def test_live_questions_keep_types_json_while_another_thread_builds_the_shadow_ones():
    c = _sport(*SPORT[0])
    live = prompt.judge_questions(c)[0]
    entered, done = threading.Event(), threading.Event()

    def other():
        with prompt.wording(shadow.table()):
            entered.set()
            done.wait(5)

    t = threading.Thread(target=other)
    t.start()
    entered.wait(5)
    assert prompt.judge_questions(c)[0] == live
    done.set()
    t.join()
    assert prompt.JUDGE_PROMPT_VERSION == "v17"


def test_with_shadow_on_a_validator_posts_the_live_answers_and_logs_both(monkeypatch, tmp_path):
    from alpharidge_ai.validator.judge import JudgeWorker
    log = tmp_path / "shadow.jsonl"
    monkeypatch.setattr(config, "JUDGE_SHADOW", True)
    monkeypatch.setattr(config, "JUDGE_SHADOW_LOG", str(log))
    monkeypatch.setattr(shadow, "wanted", lambda c: True)
    deal = _deal()
    client = JudgeAPI([{"kind": "claim", "claim_id": "1", "key": "k1", "url": deal.source_url,
                        "body": canonical_body(deal).decode(), "at": "2026-10-13T12:40:00Z"}])
    judge_ = ScriptedJudge({"subject": "yes"})
    w = JudgeWorker(client, judge_, Fetcher(FetchConfig(
        public_only=False, domain_rps=1000, retry_delay_s=0),
        transport=httpx.MockTransport(_site)))

    async def run():
        try:
            return await w.run()
        finally:
            await w.fetcher.aclose()

    asyncio.run(run())
    assert judge_.calls == 2 and len(client.posted) == 1
    row = json.loads(log.read_text())
    assert (row["claim_id"], row["live"], row["shadow"]) == ("1", "pass", "pass")
    assert client.posted[0]["answers"] == row["live_answers"]


def test_shadow_is_off_unless_set():
    assert config.JUDGE_SHADOW is False
