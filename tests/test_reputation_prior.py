"""REPUTATION_PRIOR is served config — the store must read it at the use-site.

Before this wiring the served value was fetched and applied to config but never read:
the cold-start seed was the hardcoded reputation.PRIOR, so the lever was inert.
"""

import pytest

import alpharidge_ai.config as config
from alpharidge_ai.validator import reputation as rep
from alpharidge_ai.validator.reputation_store import ReputationStore


@pytest.fixture
def store(tmp_path):
    return ReputationStore(path=tmp_path / "rep.json")


def test_unknown_hotkey_reads_served_prior(store, monkeypatch):
    monkeypatch.setattr(config, "REPUTATION_PRIOR", 0.30, raising=False)
    assert store.reputation("never-seen") == pytest.approx(0.30)


def test_served_prior_change_applies_without_restart(store, monkeypatch):
    """The hourly refresh rebinds config.REPUTATION_PRIOR in place — no restart needed."""
    monkeypatch.setattr(config, "REPUTATION_PRIOR", 0.30, raising=False)
    assert store.reputation("hk") == pytest.approx(0.30)
    monkeypatch.setattr(config, "REPUTATION_PRIOR", 0.65, raising=False)
    assert store.reputation("hk") == pytest.approx(0.65)


def test_finalize_seeds_new_hotkey_from_served_prior(store, monkeypatch):
    """First observation for an unseen target EMAs off the served prior, not rep.PRIOR."""
    monkeypatch.setattr(config, "REPUTATION_PRIOR", 0.20, raising=False)
    store.record_local(7, "self-hk", "target-hk", article_id=1, graded=1.0, weight=1.0)
    store.finalize(7, alpha=0.5)
    # one step of update(): (1-0.5)*0.20 + 0.5*1.0
    assert store.reputation("target-hk") == pytest.approx(0.60)
    assert store.samples("target-hk") == 1


def test_falls_back_to_module_default_when_unset(store, monkeypatch):
    monkeypatch.delattr(config, "REPUTATION_PRIOR", raising=False)
    assert store.reputation("hk") == pytest.approx(rep.PRIOR)


def test_bad_served_value_falls_back(store, monkeypatch):
    """A malformed served value must not crash scoring."""
    monkeypatch.setattr(config, "REPUTATION_PRIOR", "not-a-float", raising=False)
    assert store.reputation("hk") == pytest.approx(rep.PRIOR)


def test_known_hotkey_unaffected_by_prior(store, monkeypatch):
    """Persisted state wins — changing the prior must not rewrite existing reputations."""
    store.state["hk"] = {"r": 0.71, "n": 42}
    monkeypatch.setattr(config, "REPUTATION_PRIOR", 0.10, raising=False)
    assert store.reputation("hk") == pytest.approx(0.71)
