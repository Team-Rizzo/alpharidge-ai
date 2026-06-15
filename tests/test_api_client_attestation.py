import pytest
from talisman_ai.utils.api_client import TalismanAPIClient


@pytest.mark.asyncio
async def test_get_attestation_calls_endpoint(monkeypatch):
    client = TalismanAPIClient.__new__(TalismanAPIClient)
    captured = {}

    async def fake_request(method, endpoint, json=None, params=None):
        captured["method"] = method
        captured["endpoint"] = endpoint
        captured["params"] = params
        return {"validator_hotkey": "valiSelf", "epoch": 7, "per_miner_points": {"m1": 2.0},
                "total_points": 2.0, "merkle_root": "abc", "signature": "sig"}

    monkeypatch.setattr(client, "_request", fake_request, raising=False)
    out = await client.get_attestation(epoch=7)
    assert captured["endpoint"] == "/attestation"
    assert captured["params"] == {"epoch": 7}
    assert out["signature"] == "sig"


@pytest.mark.asyncio
async def test_get_verdicts_and_post_report(monkeypatch):
    client = TalismanAPIClient.__new__(TalismanAPIClient)
    calls = []

    async def fake_request(method, endpoint, json=None, params=None):
        calls.append((method, endpoint, json, params))
        if endpoint == "/verdicts":
            return {"validator_hotkey": "A", "epoch": 7, "verdicts": [], "count": 0}
        return {"success": True, "message": "ok", "count": 1}

    monkeypatch.setattr(client, "_request", fake_request, raising=False)
    v = await client.get_verdicts(validator="A", epoch=7)
    assert v["count"] == 0
    r = await client.post_report(accused_hotkey="bad", epoch=7, reason="budget_exceeded", evidence={"x": 1})
    assert r["success"] is True
    assert calls[0] == ("GET", "/verdicts", None, {"validator": "A", "epoch": 7})
    assert calls[1][0:2] == ("POST", "/reports")
