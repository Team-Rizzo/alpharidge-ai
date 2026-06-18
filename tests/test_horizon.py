"""Tests for independent per-asset horizon outlooks (temporal evidence partitioning)."""
import types
import pytest

from talisman_ai.analyzer.horizon import (
    bucket_sentence, decay_one_step, project_horizons,
)
from talisman_ai.analyzer.aspect_sentiment import score_assets, AspectSentimentScorer


# ── decay ──────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("d,expected", [
    ("very_bearish", "bearish"),
    ("bearish", "slightly_bearish"),
    ("slightly_bearish", "neutral"),
    ("neutral", "neutral"),
    ("slightly_bullish", "neutral"),
    ("very_bullish", "bullish"),
])
def test_decay_one_step(d, expected):
    assert decay_one_step(d) == expected


# ── bucketing ───────────────────────────────────────────────────────────────────
def test_bucket_short_cue():
    assert bucket_sentence("Shares slumped today in early trading.") == "short"

def test_bucket_long_cue():
    assert bucket_sentence("The long-term strategy remains intact.") == "long"

def test_bucket_medium_default():
    assert bucket_sentence("The company sells software to enterprises.") == "medium"

def test_bucket_long_outranks_short_on_tie():
    # one short cue + one long cue -> long wins the tie
    assert bucket_sentence("Today the multi-year transformation continues.") == "long"


# ── projection: the headline capability — opposite-sign horizons ─────────────────
def test_opposite_sign_horizons():
    mentions = [
        "The stock cratered today after the warning.",          # short, bearish vote
        "But its long-term roadmap looks stronger than ever.",  # long, bullish vote
    ]
    votes = [-1, +1]
    short, medium, long = project_horizons(mentions, votes, overall_direction="neutral")
    assert short.endswith("bearish")
    assert long.endswith("bullish")


def test_mirror_when_no_temporal_evidence():
    # no short/long cues -> honestly mirror direction, don't fabricate decay
    mentions = ["The company reported results.", "Analysts discussed the figures."]
    votes = [-1, -1]
    assert project_horizons(mentions, votes, "bearish") == ("bearish", "bearish", "bearish")


def test_empty_horizon_backfills_with_decay():
    # only a short-term bearish mention; medium/long absent -> decayed backfill
    mentions = ["Shares fell today."]
    votes = [-1]
    short, medium, long = project_horizons(mentions, votes, "bearish")
    assert short == "bearish"
    assert medium == decay_one_step("bearish")   # slightly_bearish
    assert long == decay_one_step("bearish")


# ── end-to-end through score_assets with a stubbed model ─────────────────────────
class _StubScorer(AspectSentimentScorer):
    def _ensure(self):
        self._model = object()
    @staticmethod
    def _vote(text):
        t = text.lower()
        if "cratered" in t or "fell" in t or "warning" in t:
            return -1
        if "roadmap" in t or "rallied" in t or "strong" in t:
            return 1
        return 0
    def label_many(self, pairs, batch_size=16):
        return [self._vote(txt) for txt, _ in pairs]


def test_score_assets_emits_independent_horizons():
    sents = [
        "Acme shares cratered today on the warning.",
        "Yet Acme's long-term roadmap looks strong.",
    ]
    ner = types.SimpleNamespace(sentence_sentiments=[{"text": s} for s in sents])
    asset = types.SimpleNamespace(ticker="ACME", canonical_name="Acme",
                                  surface_forms=["Acme"])
    out = score_assets([asset], ner, _StubScorer())[0]
    assert out["short_term"].endswith("bearish")
    assert out["long_term"].endswith("bullish")
    # the field is now informative: horizons are not all equal
    assert len({out["short_term"], out["long_term"]}) == 2
