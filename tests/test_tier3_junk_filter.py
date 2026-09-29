"""Tier-3 verdicts."""

import pytest

from alpharidge_ai import config
from alpharidge_ai.analyzer.scoring import validate_article_intelligence
from alpharidge_ai.models.article_intelligence import EntityType, ExtractedEntity
from tests import test_article_intelligence as tai
from tests.test_article_intelligence import _make_intel


@pytest.fixture(autouse=True)
def mode_on(monkeypatch):
    monkeypatch.setattr(config, "TIER3_JUNK_FILTER", True, raising=False)
    monkeypatch.setattr(config, "TIER3_THRESHOLD", 0.99, raising=False)


def _ents(*names):
    return [ExtractedEntity(name=n, entity_type=EntityType.ORGANIZATION) for n in names]


def test_verdict_pass():
    miner = tai.TestHybridValidationContract._noisy_miner()
    ok, comp, details = validate_article_intelligence(miner, _make_intel(narrative_keywords=["fed-policy"]))
    assert ok and comp < 0.99 and details["verdict"] == "pass"


def test_entity_naming_does_not_decide():
    ok, _, details = validate_article_intelligence(
        _make_intel(entities=_ents("United States")), _make_intel(entities=_ents("US")))
    assert ok and details["verdict"] == "pass" and details["tier3"]["junk_score"] < 1.0


def test_mode_switch(monkeypatch):
    monkeypatch.setattr(config, "TIER3_JUNK_FILTER", False, raising=False)
    miner = tai.TestHybridValidationContract._noisy_miner()
    ok, _, details = validate_article_intelligence(miner, _make_intel(narrative_keywords=["fed-policy"]))
    assert not ok and details["verdict"] == "junk"
    assert "TIER3_JUNK_FILTER" in config._REMOTE_CONFIG_KEYS
