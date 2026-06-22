"""Tolerant per-asset sentiment determinism gate.

Cross-hardware neural inference can't be bit-identical, so the validator's
asset-sentiment determinism check compares on the ORDINAL sentiment ladder with
adjacent-class tolerance and averages across the batch's common assets, rather
than demanding exact 7-class match (which is unachievable, and — at the 1-6
assets a typical article resolves — makes the old 0.9 "tolerance" a de-facto
exact-match gate). A true sign reversal (a miner running a different model) must
still score low.
"""
from types import SimpleNamespace

from alpharidge_ai.analyzer.scoring import (
    _sentiment_agreement, asset_sentiment_agreement, DET_AGREEMENT_THRESHOLD,
)


def _asset(direction, s, m, l):
    return SimpleNamespace(direction=direction, short_term_outlook=s,
                           medium_term_outlook=m, long_term_outlook=l)


# ── _sentiment_agreement: ordinal ladder, adjacent-tolerant ────────────────────
def test_identical_is_full_agreement():
    assert _sentiment_agreement("bullish", "bullish") == 1.0

def test_adjacent_classes_count_as_agreement():
    # inherent jitter at a class boundary -> treated as agreement
    assert _sentiment_agreement("bullish", "slightly_bullish") == 1.0
    assert _sentiment_agreement("neutral", "slightly_bearish") == 1.0

def test_two_steps_is_partial():
    a = _sentiment_agreement("very_bullish", "slightly_bullish")  # dist 2
    assert 0.7 < a < 0.9

def test_full_reversal_scores_zero():
    assert _sentiment_agreement("very_bullish", "very_bearish") == 0.0

def test_sign_reversal_scores_low():
    assert _sentiment_agreement("bullish", "bearish") < 0.5   # dist 4

def test_unknown_labels_fall_back_to_exact():
    assert _sentiment_agreement("weird", "weird") == 1.0
    assert _sentiment_agreement("weird", "bullish") == 0.0


# ── asset_sentiment_agreement: averaged over common assets ─────────────────────
def test_no_common_assets_is_full_agreement():
    agreement, n = asset_sentiment_agreement({}, {"AAPL": _asset("bullish","bullish","bullish","bullish")})
    assert n == 0 and agreement == 1.0

def test_identical_assets_full_agreement():
    a = {"AAPL": _asset("bullish", "bullish", "slightly_bullish", "neutral")}
    agreement, n = asset_sentiment_agreement(a, dict(a))
    assert n == 1 and agreement == 1.0

def test_all_adjacent_jitter_passes_threshold():
    m = {"AAPL": _asset("bullish", "bullish", "slightly_bullish", "neutral")}
    v = {"AAPL": _asset("slightly_bullish", "bullish", "bullish", "slightly_bullish")}
    agreement, _ = asset_sentiment_agreement(m, v)
    assert agreement == 1.0 and agreement >= DET_AGREEMENT_THRESHOLD

def test_one_reversal_among_many_is_diluted_and_passes():
    m = {t: _asset("bullish","bullish","bullish","bullish") for t in ("A","B","C","D","E")}
    v = dict(m)
    v["E"] = _asset("bearish","bearish","bearish","bearish")  # one asset fully reversed
    agreement, n = asset_sentiment_agreement(m, v)
    assert n == 5 and agreement >= DET_AGREEMENT_THRESHOLD

def test_systematic_reversal_fails_threshold():
    m = {t: _asset("bullish","bullish","bullish","bullish") for t in ("A","B","C")}
    v = {t: _asset("bearish","bearish","bearish","bearish") for t in ("A","B","C")}
    agreement, _ = asset_sentiment_agreement(m, v)
    assert agreement < DET_AGREEMENT_THRESHOLD

def test_threshold_is_sane():
    assert 0.5 < DET_AGREEMENT_THRESHOLD < 1.0
