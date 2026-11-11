"""Profile client: signature checks, refresh, activation, and restart behaviour."""

import json

import pytest

import alpharidge_ai.config as config
from alpharidge_ai.validator import profile_client as pc


def valid(version=2, activation_block=10_000, capacity=301_743.0,
          publish_block=None) -> dict:
    return {
        "version": version,
        "publish_block": (activation_block - 400 if publish_block is None
                          else publish_block),
        "activation_block": activation_block,
        "schema_version": "2.0.0",
        "settlement": {"C": capacity},
        "ledger": {"window_epochs": 216},
        "signature": "aa" * 32,
    }


class Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MINER_API_URL", "http://test.invalid", raising=False)
    monkeypatch.setattr(config, "REMOTE_CONFIG_REFRESH_SECONDS", 3600, raising=False)
    monkeypatch.setattr(config, "_build_auth_headers", lambda *a: {}, raising=False)
    return pc.ProfileClient(path=tmp_path / "profile.json", require_signature=False)


def serve(monkeypatch, payload):
    monkeypatch.setattr(pc.requests, "get", lambda *a, **kw: Response(payload))


# ---- refresh ----------------------------------------------------------------------

def test_a_served_profile_is_staged_and_then_activates(client, monkeypatch):
    serve(monkeypatch, {"next": valid()})
    assert client.refresh(block=9_000, force=True)

    assert client.resolve(9_999) is None          # nothing in force yet
    active = client.resolve(10_000)
    assert active is not None and active.version == 2
    assert active.settlement.C == pytest.approx(301_743.0)


def test_a_fetch_failure_leaves_the_running_profile_alone(client, monkeypatch):
    serve(monkeypatch, {"current": valid(version=2, activation_block=400, publish_block=0)})
    client.refresh(block=1_000, force=True)
    assert client.resolve(1_000).version == 2

    def boom(*a, **kw):
        raise RuntimeError("api down")
    monkeypatch.setattr(pc.requests, "get", boom)

    assert not client.refresh(block=2_000, force=True)
    assert client.resolve(2_000).version == 2


def test_a_malformed_profile_is_refused(client, monkeypatch):
    broken = valid()
    broken["settlement"]["C"] = 0.0
    serve(monkeypatch, {"next": broken})
    assert not client.refresh(block=9_000, force=True)
    assert client.resolve(20_000) is None


def test_refresh_is_rate_limited_unless_forced(client, monkeypatch):
    calls = []

    def counted(url, **kw):
        calls.append(url.rsplit("/", 2)[-2:])
        return Response({"next": valid()})
    monkeypatch.setattr(pc.requests, "get", counted)

    client.refresh(block=9_000, force=True)
    client.refresh(block=9_000)
    client.refresh(block=9_000)
    # One config fetch and one profile fetch per refresh.
    assert calls == [["config", "subnet"], ["mechanism", "profile"]]


def test_no_api_url_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MINER_API_URL", "", raising=False)
    client = pc.ProfileClient(path=tmp_path / "p.json", require_signature=False)
    assert not client.refresh(block=1, force=True)


# ---- signatures -------------------------------------------------------------------

def test_a_bad_signature_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MINER_API_URL", "http://test.invalid", raising=False)
    monkeypatch.setattr(config, "_build_auth_headers", lambda *a: {}, raising=False)
    monkeypatch.setattr(pc, "verify_signature", lambda m, s: False)
    serve(monkeypatch, {"next": valid()})

    client = pc.ProfileClient(path=tmp_path / "p.json", require_signature=True)
    assert not client.refresh(block=9_000, force=True)
    assert client.resolve(20_000) is None


def test_a_good_signature_is_accepted(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MINER_API_URL", "http://test.invalid", raising=False)
    monkeypatch.setattr(config, "_build_auth_headers", lambda *a: {}, raising=False)
    seen = {}

    def verify(message, signature):
        seen["message"] = message
        return True
    monkeypatch.setattr(pc, "verify_signature", verify)
    serve(monkeypatch, {"next": valid()})

    client = pc.ProfileClient(path=tmp_path / "p.json", require_signature=True)
    assert client.refresh(block=9_000, force=True)
    assert "signature" not in seen["message"]


def test_signature_check_uses_the_pinned_attestation_key(monkeypatch):
    monkeypatch.setattr(config, "API_ATTESTATION_PUBKEY", "", raising=False)
    assert not pc.verify_signature("message", "aa" * 32)


# ---- persistence ------------------------------------------------------------------

def test_an_accepted_profile_survives_a_restart(client, monkeypatch, tmp_path):
    serve(monkeypatch, {"current": valid(version=3, activation_block=400, publish_block=0)})
    client.refresh(block=1_000, force=True)
    client.resolve(1_000)

    restarted = pc.ProfileClient(path=tmp_path / "profile.json",
                                 require_signature=False)
    restarted.load()
    assert restarted.resolve(1_000).version == 3


def test_a_pending_profile_survives_a_restart(client, monkeypatch, tmp_path):
    serve(monkeypatch, {"next": valid(version=4, activation_block=50_000)})
    client.refresh(block=40_000, force=True)

    restarted = pc.ProfileClient(path=tmp_path / "profile.json",
                                 require_signature=False)
    restarted.load()
    assert restarted.resolve(49_999) is None
    assert restarted.resolve(50_000).version == 4


def test_a_corrupt_store_does_not_crash_startup(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text("{not json")
    client = pc.ProfileClient(path=path, require_signature=False)
    client.load()
    assert client.resolve(1) is None


def test_a_stored_profile_that_no_longer_validates_is_dropped(tmp_path):
    path = tmp_path / "profile.json"
    bad = valid()
    bad["settlement"]["C"] = -5.0
    path.write_text(json.dumps({"current": bad}))
    client = pc.ProfileClient(path=path, require_signature=False)
    client.load()
    assert client.resolve(1) is None


# ---- ordering ---------------------------------------------------------------------

def test_a_replayed_older_version_cannot_displace_a_newer_one(client, monkeypatch):
    serve(monkeypatch, {"current": valid(version=5, activation_block=400, publish_block=0)})
    client.refresh(block=1_000, force=True)
    client.resolve(1_000)

    serve(monkeypatch, {"next": valid(version=4, activation_block=2_000,
                                      capacity=999.0)})
    client._last_fetch = 0
    assert not client.refresh(block=1_000, force=True)
    assert client.resolve(2_000).settlement.C == pytest.approx(301_743.0)


def test_served_config_applies_unless_overridden(monkeypatch):
    monkeypatch.setattr(config, "MINER_API_URL", "http://test.invalid", raising=False)
    monkeypatch.setattr(config, "_build_auth_headers", lambda *a: {}, raising=False)
    monkeypatch.setattr(config, "JUDGE_URL", "")
    monkeypatch.setattr(config, "REVERIFY_FETCH_TIMEOUT_S", 15.0)
    monkeypatch.setenv("OVERRIDE_REVERIFY_FETCH_TIMEOUT_S", "20")
    serve(monkeypatch, {"config": {"EL_JUDGE_URL": "https://judge.test/v1",
                                   "REVERIFY_FETCH_TIMEOUT_S": "5"}})
    config.refresh_remote_config(force=True)
    assert config.JUDGE_URL == "https://judge.test/v1"
    assert config.REVERIFY_FETCH_TIMEOUT_S == 20.0
