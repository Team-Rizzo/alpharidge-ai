"""3.9.15: explicit reasoning setting, canary pruning, peer presence and memory reporting."""
import types

import alpharidge_ai.config as config
from alpharidge_ai.analyzer.relevance_check import RelevanceCheck
from alpharidge_ai.utils import llm_spend


def _client(base="https://openrouter.ai/api/v1"):
    seen = []
    msg = types.SimpleNamespace(content='{"relevant": false}', tool_calls=None)
    resp = types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason="stop")],
                                 usage=None)

    def create(**kw):
        seen.append(kw)
        return resp
    c = types.SimpleNamespace(base_url=base, chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(create=create)))
    return c, seen


def test_reasoning_body():
    c, _ = _client()
    assert llm_spend.reasoning_body(c, "off") == {"reasoning": {"enabled": False}}
    assert llm_spend.reasoning_body(c, "ON") == {"reasoning": {"enabled": True}}
    assert llm_spend.reasoning_body(c, "") == {} and llm_spend.reasoning_body(c, "default") == {}
    other, _ = _client("https://api.example.com/v1")
    assert llm_spend.reasoning_body(other, "off") == {}       # only sent to OpenRouter


def test_request_body_carries_the_setting(monkeypatch):
    monkeypatch.setattr(config, "REFERENCE_PROVIDER_ORDER", "deepinfra/fp8", raising=False)
    c, _ = _client()
    body = llm_spend.request_body(c, "off")
    assert body["reasoning"] == {"enabled": False} and body["usage"] == {"include": True}
    assert body["provider"]["order"] == ["deepinfra/fp8"]
    assert "reasoning" not in llm_spend.request_body(c)


def test_relevance_check_sends_the_setting():
    c, seen = _client()
    RelevanceCheck(c, "m", reasoning="off").verdict("t", "b")
    RelevanceCheck(c, "m", spend_purpose="relevance_judge", reasoning="off").verdict("t", "b")
    RelevanceCheck(c, "m").verdict("t", "b")
    assert [kw.get("extra_body", {}).get("reasoning") for kw in seen] == [
        {"enabled": False}, {"enabled": False}, None]


def test_validator_judge_follows_the_served_setting(monkeypatch):
    import neurons.validator as vm
    v = types.SimpleNamespace(_triage_auditor=types.SimpleNamespace(_client=_client()[0]))
    v._get_triage_auditor = lambda: v._triage_auditor
    monkeypatch.setattr(config, "TRIAGE_JUDGE_MODEL", "m", raising=False)
    monkeypatch.setattr(config, "VALIDATOR_REASONING", "off", raising=False)
    first = vm.Validator._get_relevance_judge(v)
    assert first.reasoning == "off" and vm.Validator._get_relevance_judge(v) is first
    monkeypatch.setattr(config, "VALIDATOR_REASONING", "on", raising=False)
    assert vm.Validator._get_relevance_judge(v).reasoning == "on"


def test_reference_and_miner_calls_carry_their_settings(monkeypatch):
    from alpharidge_ai.analyzer.article_intelligence_analyzer import ArticleIntelligenceAnalyzer
    monkeypatch.setattr(config, "VALIDATOR_REASONING", "off", raising=False)
    monkeypatch.setattr(config, "MINER_REASONING", "on", raising=False)
    monkeypatch.setattr(llm_spend, "record", lambda *a, **k: None)
    c, seen = _client()
    an = object.__new__(ArticleIntelligenceAnalyzer)
    an.client, an.model = c, "m"
    tool = {"type": "function", "function": {"name": "x", "parameters": {}}}
    for strict in (True, False):
        try:
            an._llm_call("p", tool, "x", strict=strict)
        except Exception:
            pass
    assert seen[0]["extra_body"]["reasoning"] == {"enabled": False}     # validator reference
    assert seen[-1]["extra_body"] == {"reasoning": {"enabled": True}}   # miner


def test_settings_are_served_and_consensus():
    for key in ("VALIDATOR_REASONING", "GRADER_REASONING"):
        assert key in config._REMOTE_CONFIG_KEYS and key in config._CONSENSUS_KEYS
    assert "MINER_REASONING" not in config._REMOTE_CONFIG_KEYS
    assert config.VALIDATOR_REASONING == "off" and config.GRADER_REASONING == "off"


def test_spent_canaries_are_pruned_after_their_last_dispatch_returns():
    import random
    from alpharidge_ai.validator.triage_grader import CanaryPool, TriageConfig
    t = [0.0]
    pool = CanaryPool(TriageConfig(), now=lambda: t[0])
    pool.add(1, "pos", True)
    pool.add(2, "pos", True)
    for _ in range(3):
        pool.draw("pos", random.Random(0), available={1})
    t[0] = 600.0
    pool.prune()
    assert pool.label_of(1) is not None                 # spent, but a response may still come back
    t[0] = 600.0 + 1800.0
    pool.prune()
    assert pool.label_of(1) is None and pool.label_of(2) is not None
    t[0] = 6 * 3600.0
    pool.prune()
    assert pool.label_of(2) is None                      # TTL still applies


def test_peer_presence_and_gap_warning():
    from alpharidge_ai.validator.reward_broadcast_store import RewardBroadcastStore
    s = RewardBroadcastStore(path=__import__("pathlib").Path("/nonexistent/x.json"), keep_epochs=50)
    for e in range(100, 124):
        s.by_epoch_by_sender[e] = {"peerA": {1: 5}}
        if e < 120:
            s.by_epoch_by_sender[e]["peerB"] = {1: 5}
    assert s.sender_presence(100, 123) == {"peerA": 24, "peerB": 20}
    assert s.missing_recent(100, 123) == ["peerB"]
    assert s.missing_recent(100, 119) == []


def test_memory_line_reads_smaps_rollup(tmp_path):
    from alpharidge_ai.utils import memory
    p = tmp_path / "smaps_rollup"
    p.write_text("Rss:            33554432 kB\nPss_File:       23068672 kB\nAnonymous:      10485760 kB\n")
    assert memory.read(str(p)) == {"Rss": 33554432, "Pss_File": 23068672, "Anonymous": 10485760}
    assert memory.read(str(tmp_path / "missing")) is None
