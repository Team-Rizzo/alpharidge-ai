from types import SimpleNamespace

from talisman_ai.analyzer.article_intelligence_analyzer import ArticleIntelligenceAnalyzer


def _asset(ticker, name, surface_forms=None):
    return SimpleNamespace(ticker=ticker, canonical_name=name, surface_forms=surface_forms)


def _run(resolved_assets, sentences, fallback="neutral"):
    a = ArticleIntelligenceAnalyzer.__new__(ArticleIntelligenceAnalyzer)  # skip __init__
    ner = SimpleNamespace(sentence_sentiments=sentences)
    return a._finbert_asset_sentiments(resolved_assets, ner, fallback_direction=fallback)


def test_direction_from_company_name_not_ticker():
    out = _run(
        [_asset("TSLA", "Tesla")],
        [{"text": "Tesla shares surged 19% after record deliveries.", "sentiment": "bullish", "score": 0.95},
         {"text": "The broader market was quiet otherwise.", "sentiment": "neutral", "score": 0.6}],
    )
    tsla = next(o for o in out if o["ticker"] == "TSLA")
    assert tsla["direction"] == "bullish"


def test_direction_matches_via_surface_form_alias():
    out = _run(
        [_asset("TSLA", "Tesla, Inc.", surface_forms=["tesla"])],
        [{"text": "tesla crushed earnings and the stock jumped.", "sentiment": "bullish", "score": 0.9}],
    )
    assert out[0]["direction"] == "bullish"


def test_fallback_uses_article_sentiment_when_no_mention():
    out = _run(
        [_asset("AAPL", "Apple")],
        [{"text": "An unrelated macro note about bonds.", "sentiment": "bullish", "score": 0.8}],
        fallback="bearish",
    )
    assert out[0]["direction"] == "bearish"
    assert out[0]["confidence"] <= 0.3 + 1e-9  # low confidence on fallback


def test_no_longer_defaults_hard_neutral():
    # The old bug: ticker symbol not in prose -> always neutral. Now fallback wins.
    out = _run([_asset("NVDA", "Nvidia")],
               [{"text": "Nvidia guidance blew past estimates.", "sentiment": "bullish", "score": 0.97}],
               fallback="neutral")
    assert out[0]["direction"] == "bullish"
