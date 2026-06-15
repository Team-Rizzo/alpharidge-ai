import importlib


def test_attestation_config_defaults(monkeypatch):
    monkeypatch.delenv("API_ATTESTATION_PUBKEY", raising=False)
    monkeypatch.setenv("DEEP_VERIFY_SAMPLE_RATE", "0.25")
    from talisman_ai import config
    importlib.reload(config)
    assert config.API_ATTESTATION_PUBKEY == ""
    assert abs(config.DEEP_VERIFY_SAMPLE_RATE - 0.25) < 1e-9
