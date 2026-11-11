"""The per-validator audit key."""

from alpharidge_ai.validator import audit_key


# ---- the key ----------------------------------------------------------------------

def test_a_key_is_generated_and_reused(tmp_path, monkeypatch):
    monkeypatch.delenv("AUDIT_KEY", raising=False)
    path = tmp_path / "key"
    first = audit_key.load(path)
    assert len(first) == audit_key.KEY_BYTES
    assert audit_key.load(path) == first


def test_two_validators_get_different_keys(tmp_path, monkeypatch):
    monkeypatch.delenv("AUDIT_KEY", raising=False)
    assert audit_key.load(tmp_path / "a") != audit_key.load(tmp_path / "b")


def test_an_explicit_key_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIT_KEY", "ab" * 32)
    assert audit_key.load(tmp_path / "key") == bytes.fromhex("ab" * 32)


def test_the_key_file_is_not_world_readable(tmp_path, monkeypatch):
    monkeypatch.delenv("AUDIT_KEY", raising=False)
    path = tmp_path / "key"
    audit_key.load(path)
    assert path.stat().st_mode & 0o077 == 0
