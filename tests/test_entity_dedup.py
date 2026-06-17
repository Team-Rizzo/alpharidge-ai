from types import SimpleNamespace

from talisman_ai.analyzer.article_intelligence_analyzer import ArticleIntelligenceAnalyzer
from talisman_ai.models.article_intelligence import EntityRole, Sentiment


def _ent(name, etype="organization", role="mentioned", conf=0.9, sentiment=None):
    return SimpleNamespace(canonical_name=name, entity_type=etype, role=role,
                           ticker=None, sentiment_toward=sentiment, confidence=conf)


def _build(resolved):
    a = ArticleIntelligenceAnalyzer.__new__(ArticleIntelligenceAnalyzer)  # skip __init__ (no models)
    return a._build_entities_from_ner(SimpleNamespace(resolved_entities=resolved))


def test_dedup_merges_repeated_entity():
    out = _build([_ent("Medtronic"), _ent("Medtronic"), _ent("Medtronic"), _ent("Apple Inc")])
    names = [e.name for e in out]
    assert names.count("Medtronic") == 1
    assert "Apple Inc" in names


def test_dedup_is_case_insensitive():
    out = _build([_ent("SpaceX"), _ent("spacex"), _ent("SPACEX")])
    assert len([e for e in out if e.name.lower() == "spacex"]) == 1


def test_dedup_keeps_highest_confidence_and_merges_role():
    out = _build([
        _ent("Medtronic", role="mentioned", conf=0.5),
        _ent("Medtronic", role="subject", conf=0.9),
    ])
    assert len(out) == 1
    # more specific role survives the merge
    assert out[0].role != EntityRole.MENTIONED


def test_dedup_merges_sentiment_signal():
    out = _build([
        _ent("Tesla", conf=0.9, sentiment=None),
        _ent("Tesla", conf=0.5, sentiment="bullish"),
    ])
    assert len(out) == 1
    assert out[0].sentiment_toward == Sentiment.BULLISH
