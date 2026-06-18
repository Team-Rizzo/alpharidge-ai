"""Tests for target-aware per-asset aspect sentiment (FinABSA scorer).

Fast tests stub the model; the one integration test actually runs FinABSA on the
exact failure the redesign targets (Microsoft slumps while the market rallies in
the SAME sentence) and is skipped if the model can't be loaded offline.
"""
import types
import pytest

from talisman_ai.analyzer.aspect_sentiment import (
    AspectSentimentScorer, vote_to_sentiment, aggregate_votes, score_assets,
    _best_target_form, _mention_sentences, _target_in_sentence,
)
from talisman_ai.analyzer.horizon import _SCALE


# ── confidence calibration: must reward EVIDENCE VOLUME, not just agreement ─────
def _conf(votes):
    return aggregate_votes(votes)[2]

def test_confidence_rewards_more_corroborating_mentions():
    # Same (perfect) consensus, more agreeing mentions -> higher confidence.
    # Without this, one throwaway sentence outranks six corroborating ones.
    assert _conf([1, 1, 1, 1]) > _conf([1])

def test_single_mention_confidence_is_modest():
    # A lone unanimous mention must NOT reach near-certainty.
    assert _conf([1]) <= 0.5

def test_low_consensus_yields_low_confidence():
    # 1 bullish among 6 -> weak minority signal -> low confidence.
    assert _conf([1, 0, 0, 0, 0, 0]) < 0.4

def test_confidence_is_bounded():
    for v in ([1], [1, 1, 1, 1, 1, 1], [1, -1], [0], [-1, -1, -1]):
        assert 0.0 <= _conf(v) <= 1.0


# ── pure logic: vote -> 7-class ────────────────────────────────────────────────
def test_single_decisive_mention_is_not_very():
    # one bearish mention -> bearish, never very_bearish (needs >=2 agreeing)
    assert vote_to_sentiment(net=-1, n=1, agree=1) == ("bearish", 1.0)

def test_two_agreeing_mentions_reach_very():
    assert vote_to_sentiment(net=2, n=2, agree=2)[0] == "very_bullish"

def test_minority_signal_is_slight():
    # 1 bullish among 3 mentions -> slightly_bullish (strength 0.33 < 0.40)
    d, mag = vote_to_sentiment(net=1, n=3, agree=1)
    assert d == "slightly_bullish" and mag < 0.4

def test_conflicting_votes_are_neutral():
    assert vote_to_sentiment(net=0, n=2, agree=1) == ("neutral", 0.0)

def test_no_mentions_neutral():
    assert vote_to_sentiment(net=0, n=0, agree=0) == ("neutral", 0.0)


# ── target form selection + mention matching ───────────────────────────────────
def test_prefers_name_over_bare_ticker():
    assert _best_target_form({"MSFT", "Microsoft"}) == "Microsoft"

def test_mention_matching_is_word_boundaried_and_caseless():
    sents = ["Microsoft fell today.", "Unrelated about cars."]
    assert _mention_sentences(sents, {"microsoft"}) == ["Microsoft fell today."]


def test_target_must_be_present_in_sentence():
    # Regression: the global longest form ("Apple Inc.") may be ABSENT from a
    # given sentence; FinABSA's [TGT] needs a form that actually occurs there,
    # else nothing is marked and it returns a generic/neutral label.
    forms = {"AAPL", "Apple Inc.", "apple", "tim cook"}
    # "Apple Inc." not present, "Apple" is -> pick the present form, not the fallback
    assert _target_in_sentence("Apple reports a blowout quarter.", forms,
                               fallback="Apple Inc.") == "apple"
    # nothing present -> fall back to the global form
    assert _target_in_sentence("The board approved a buyback.", forms,
                               fallback="Apple Inc.") == "Apple Inc."


# ── end-to-end score_assets with a stubbed model (no FinABSA load) ──────────────
class _StubScorer(AspectSentimentScorer):
    """Votes by keyword so we can test aggregation + dict contract offline."""
    def _ensure(self):  # never load a real model
        self._model = object()
    @staticmethod
    def _vote(text):
        t = text.lower()
        if "slumped" in t or "fell" in t or "weak" in t:
            return -1
        if "rallied" in t or "gained" in t or "beat" in t:
            return 1
        return 0
    def label_many(self, pairs, batch_size=16):
        return [self._vote(txt) for txt, _ in pairs]


def _ner(sentences):
    return types.SimpleNamespace(
        sentence_sentiments=[{"text": s} for s in sentences])

def _asset(ticker, name, forms):
    return types.SimpleNamespace(ticker=ticker, canonical_name=name, surface_forms=forms)


def test_score_assets_attributes_per_target():
    sents = [
        "Microsoft stock slumped today after weak guidance.",
        "Meanwhile the broader market rallied to new highs.",
    ]
    ner = _ner(sents)
    assets = [
        _asset("MSFT", "Microsoft", ["Microsoft"]),
        _asset("SPX", "market", ["market"]),
    ]
    out = {d["ticker"]: d for d in score_assets(assets, ner, _StubScorer())}
    # MSFT must be bearish (was neutral under whole-sentence FinBERT)
    assert out["MSFT"]["direction"] == "bearish"
    assert "very" not in out["SPX"]["direction"]
    assert out["SPX"]["direction"] == "bullish"
    # contract: all keys present, horizons present (mirror direction for now)
    for d in out.values():
        assert set(d) >= {"ticker", "direction", "magnitude", "confidence",
                          "short_term", "medium_term", "long_term",
                          "causal_driver", "evidence_spans"}
        # horizons are independent now (horizon.project_horizons), not mirrors —
        # just assert they're valid Sentiment values here.
        for h in ("short_term", "medium_term", "long_term"):
            assert d[h] in _SCALE


def test_coref_attributes_generic_referent_to_single_primary():
    # "The company..." names no asset, but is attributed to the lone primary subject
    ner = _ner(["Acme shares rallied on the news.",
                "The company gained after strong guidance."])
    a = _asset("ACME", "Acme", ["Acme"]); a.is_primary_subject = True
    out = score_assets([a], ner, _StubScorer())[0]
    assert "2 mention" in out["causal_driver"]      # both sentences attributed
    assert out["direction"].endswith("bullish")


def test_coref_off_when_multiple_primaries():
    # With >1 primary subject, "the company" is ambiguous -> no coref expansion
    ner = _ner(["Acme rallied.", "The company gained sharply."])
    a = _asset("ACME", "Acme", ["Acme"]); a.is_primary_subject = True
    b = _asset("BETA", "Beta", ["Beta"]); b.is_primary_subject = True
    res = {d["ticker"]: d for d in score_assets([a, b], ner, _StubScorer())}
    assert "1 mention" in res["ACME"]["causal_driver"]   # only the explicit mention


def test_low_confidence_primary_defers_to_article_sentiment():
    # The Carvana failure: 1 noisy positive vote (the "Bull-Trap" title) among many
    # neutral mentions -> slightly_bullish at ~0 confidence on a bearish article.
    # The primary subject must defer to the article-level read when FinABSA is weak.
    ner = _ner([
        "Acme rallied to a fresh high in early trading.",   # the lone +1 (noisy)
        "I rate Acme a hold here.",                          # 0
        "Acme remains a hold given headwinds.",              # 0
        "Acme trades at a rich multiple.",                   # 0
        "Acme is a hold.",                                   # 0
    ])
    a = _asset("ACME", "Acme", ["Acme"]); a.is_primary_subject = True
    out = score_assets([a], ner, _StubScorer(), fallback_direction="bearish")[0]
    assert out["direction"] == "bearish"            # deferred to article-level
    assert "inconclusive" in out["causal_driver"].lower()

def test_confident_primary_keeps_its_own_direction():
    # When FinABSA IS confident (several agreeing mentions), per-asset direction
    # must win over the article-level fallback — the whole point of FinABSA.
    ner = _ner([
        "Acme slumped after weak guidance.",
        "Acme fell again as shares weakened.",
        "Acme slumped to a new low on weak demand.",
    ])
    a = _asset("ACME", "Acme", ["Acme"]); a.is_primary_subject = True
    out = score_assets([a], ner, _StubScorer(), fallback_direction="bullish")[0]
    assert out["direction"].endswith("bearish")     # not the bullish fallback


def test_score_assets_fallback_when_no_mention():
    ner = _ner(["A sentence about something else entirely."])
    assets = [_asset("MSFT", "Microsoft", ["Microsoft"])]
    out = score_assets(assets, ner, _StubScorer(), fallback_direction="bullish")
    assert out[0]["direction"] == "bullish"
    assert "fell back" in out[0]["causal_driver"]


# ── integration: real FinABSA on the canonical multi-entity failure ────────────
@pytest.mark.slow
def test_finabsa_separates_two_entities_in_one_sentence():
    try:
        scorer = AspectSentimentScorer()
        scorer._ensure()
    except Exception as e:  # model not available offline
        pytest.skip(f"FinABSA unavailable: {e}")
    sent = ("Microsoft stock slumped today after the company issued weak guidance, "
            "even as the broader market rallied.")
    msft = scorer._label(sent, "Microsoft")
    mkt = scorer._label(sent, "market")
    assert msft == -1, "Microsoft should read bearish"
    assert mkt == 1, "market should read bullish (separated from Microsoft)"
