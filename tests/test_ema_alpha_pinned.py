"""The reputation EMA alpha is a consensus value and comes from the profile.

A profile applies at one activation block fleet-wide; served config does not.
"""
import pytest

from alpharidge_ai import config
from alpharidge_ai.mechanism import profile as mp
from alpharidge_ai.validator.reputation_store import ReputationStore


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    """The store's default path is inside the package; keep tests off it."""
    monkeypatch.setattr(config, "REPUTATION_STATE_LOCATION",
                        str(tmp_path / "rep.json"), raising=False)


def test_the_profile_carries_an_ema_alpha():
    assert "ema_alpha" in {f for f in mp.Emission.__dataclass_fields__}


def test_identical_observations_under_different_alphas_diverge():
    """Different alphas produce different EMAs."""
    a, b = ReputationStore(), ReputationStore()
    for s in (a, b):
        s.record_local(5, "me", "miner", 1, 0.9, 1.0)
    a.finalize(5, alpha=0.03)
    b.finalize(5, alpha=0.12)
    ra = a.snapshot()["miner"]["r"]
    rb = b.snapshot()["miner"]["r"]
    assert abs(ra - rb) > 1e-6, "different alphas must produce different EMAs"


def test_identical_observations_under_the_same_alpha_agree():
    a, b = ReputationStore(), ReputationStore()
    for s in (a, b):
        s.record_local(5, "me", "miner", 1, 0.9, 1.0)
        s.finalize(5, alpha=0.03)
    assert a.snapshot()["miner"]["r"] == b.snapshot()["miner"]["r"]


def test_the_call_site_reads_the_profile_not_served_config():
    """The call site must read the profile."""
    import inspect
    from alpharidge_ai.validator import validation_client
    src = inspect.getsource(validation_client.ValidationClient.run)
    i = src.index("_reputation_store.finalize(")
    window = src[max(0, i - 700):i]
    assert "emission.ema_alpha" in window, "finalize must take alpha from the profile"
