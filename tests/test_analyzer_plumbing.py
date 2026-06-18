"""Pure-helper tests for analyzer plumbing fixes (no model load):

- count_entity_mentions: real per-entity mention_count (was hardcoded 1)
- sanitize_forward_event_date: drop fabricated PAST dates (a "forward" event
  cannot predate publication) and null-ish junk.
"""
from types import SimpleNamespace

from talisman_ai.analyzer.article_intelligence_analyzer import (
    count_entity_mentions, sanitize_forward_event_date, map_gics_to_sector,
    strip_analyst_roster_assets, strip_disclosure_tail,
)


# ── strip_disclosure_tail: cut the footer/disclosure boilerplate ───────────────
def test_strips_motley_fool_footer():
    body = ("Micron stock jumped on strong memory demand and an upbeat outlook. " + "Analysis. " * 30
            + " The Motley Fool has positions in and recommends Nvidia and Tesla.")
    out = strip_disclosure_tail(body)
    assert "Micron stock jumped" in out
    assert "Motley Fool" not in out and "Nvidia and Tesla" not in out

def test_strips_seeking_alpha_disclosure():
    body = ("Schlumberger remains a buy on the pullback. " + "Detail. " * 30
            + " Disclosure: I/we have a beneficial long position in the shares of PM.")
    out = strip_disclosure_tail(body)
    assert "PM" not in out and "Schlumberger remains a buy" in out

def test_keeps_body_without_footer():
    body = "Just an article about Micron and its strong quarter, no boilerplate here."
    assert strip_disclosure_tail(body) == body

def test_does_not_truncate_on_early_word_disclosure():
    body = "The disclosure of quarterly earnings was positive for investors. " + "More. " * 40
    assert len(strip_disclosure_tail(body)) >= len(body) * 0.8
from talisman_ai.analyzer.horizon import reconcile_direction_with_horizons as _recon


# ── reconcile_direction_with_horizons: kill direction/horizon sign contradictions ─
def test_bullish_direction_with_all_bearish_horizons_flips():
    # The URI case: FinABSA said bullish, LLM horizons all bearish/neutral.
    assert _recon("bullish", "slightly_bearish", "bearish", "neutral") == "bearish"

def test_bearish_direction_with_all_bullish_horizons_flips():
    assert _recon("bearish", "bullish", "bullish", "slightly_bullish") == "bullish"

def test_direction_kept_when_a_horizon_agrees():
    assert _recon("bullish", "bullish", "slightly_bullish", "neutral") == "bullish"

def test_direction_kept_when_horizons_only_neutral():
    # No sign opposition (just neutrals) -> leave direction as-is.
    assert _recon("bullish", "neutral", "neutral", "neutral") == "bullish"

def test_neutral_direction_unchanged():
    assert _recon("neutral", "bearish", "bearish", "bearish") == "neutral"


def _a(ticker, name, spans, primary=False):
    return SimpleNamespace(ticker=ticker, asset_name=name, evidence_spans=spans,
                           is_primary_subject=primary)


# ── strip_analyst_roster_assets: drop sell-side banks in earnings transcripts ───
_TRANSCRIPT = (
    "Accenture plc (ACN) Q3 2026 Earnings Call Transcript\n"
    "Conference Call Participants\n"
    "Bryan Keane - Citigroup Inc., Research Division\n"
    "James Schneider - Goldman Sachs Group, Research Division\n"
    "Tien-Tsin Huang - JPMorgan Chase & Co, Research Division\n"
    "Operator\n"
    "Accenture reported strong revenue and raised full-year guidance.")

def test_strips_analyst_roster_banks_in_transcript():
    assets = [
        _a("ACN", "Accenture", ["Accenture"], primary=True),
        _a("C", "Citigroup", ["Citigroup"]),
        _a("GS", "Goldman Sachs Group", ["Goldman Sachs"]),
        _a("JPM", "JPMorgan Chase", ["JPMorgan"]),
    ]
    out = {a.ticker for a in strip_analyst_roster_assets(assets, _TRANSCRIPT)}
    assert out == {"ACN"}  # subject kept, sell-side desks dropped

def test_keeps_primary_subject_even_if_bank():
    # A real Goldman article (not a transcript): GS appears as a subject -> kept.
    text = ("Goldman Sachs reported blowout trading revenue; the bank's shares rose "
            "as Goldman raised its outlook for the year.")
    out = {a.ticker for a in strip_analyst_roster_assets(
        [_a("GS", "Goldman Sachs Group", ["Goldman Sachs", "Goldman"], primary=True)], text)}
    assert out == {"GS"}

def test_non_transcript_assets_unchanged():
    text = "Apple and Microsoft both rose on strong earnings."
    assets = [_a("AAPL", "Apple", ["Apple"], primary=True), _a("MSFT", "Microsoft", ["Microsoft"])]
    assert len(strip_analyst_roster_assets(assets, text)) == 2


# ── map_gics_to_sector: GICS string -> coarse 9-bucket taxonomy ─────────────────
def test_gics_tech_maps_to_tech():
    assert map_gics_to_sector("Information Technology")[1] == "TECH"
    assert map_gics_to_sector("Technology")[1] == "TECH"

def test_gics_energy_maps_to_commodities():
    assert map_gics_to_sector("Energy")[1] == "COMMODITIES"

def test_gics_health_maps_to_science():
    assert map_gics_to_sector("Health Care")[1] == "SCIENCE"

def test_gics_financials_maps_to_equities():
    assert map_gics_to_sector("Financials")[1] == "EQUITIES"
    assert map_gics_to_sector("Industrials")[1] == "EQUITIES"

def test_gics_unknown_maps_to_other():
    assert map_gics_to_sector("")[1] == "OTHER"
    assert map_gics_to_sector("Nonsense Sector")[1] == "OTHER"
    assert map_gics_to_sector(None)[1] == "OTHER"

def test_gics_returns_consistent_id_symbol():
    sid, sym = map_gics_to_sector("Information Technology")
    assert sid == 7 and sym == "TECH"


# ── count_entity_mentions ──────────────────────────────────────────────────────
def test_counts_multiple_mentions():
    text = "Micron rose. Micron Technology guided higher. Analysts like Micron."
    count, first = count_entity_mentions(text, ["Micron", "Micron Technology"])
    assert count == 3
    assert first == 0

def test_single_mention_counts_one():
    count, _ = count_entity_mentions("Only Apple appears here.", ["Apple", "Apple Inc."])
    assert count == 1

def test_no_mention_floors_at_one():
    # An entity the NER resolved but whose surface form isn't found verbatim still
    # gets at least 1 (it was detected) — never 0.
    count, _ = count_entity_mentions("Nothing relevant here.", ["Tesla"])
    assert count == 1

def test_overlapping_forms_not_double_counted():
    # "Apple" inside "Apple Inc." must count once, not twice.
    count, _ = count_entity_mentions("Apple Inc. reported earnings.", ["Apple", "Apple Inc."])
    assert count == 1


# ── sanitize_forward_event_date ────────────────────────────────────────────────
PUB = "2026-06-17T12:00:00+00:00"

def test_past_full_date_dropped():
    assert sanitize_forward_event_date("2025-09-25", PUB) is None

def test_future_date_kept():
    assert sanitize_forward_event_date("2026-09-25", PUB) == "2026-09-25"

def test_nullish_dropped():
    for junk in ("", "null", "none", "N/A", "TBD", "unknown", None):
        assert sanitize_forward_event_date(junk, PUB) is None

def test_past_year_textual_dropped():
    assert sanitize_forward_event_date("Q2 2025", PUB) is None

def test_future_textual_kept():
    assert sanitize_forward_event_date("Q1 2027", PUB) == "Q1 2027"


# ── _reconcile_asset_directions: horizons MUST stay deterministic ──────────────
# The validator hard-gates per-asset outlook determinism (scoring.py Tier-2b,
# tol=0.9). Horizons therefore come ONLY from the off-LLM temporal projector and
# must NEVER be overridden by the LLM (two miner/validator LLM calls are not
# bit-identical even at temperature 0). This pass only reconciles `direction`
# against those already-deterministic horizons.
def _bare_analyzer():
    from talisman_ai.analyzer.article_intelligence_analyzer import ArticleIntelligenceAnalyzer
    return ArticleIntelligenceAnalyzer.__new__(ArticleIntelligenceAnalyzer)

def test_reconcile_leaves_deterministic_horizons_untouched():
    a = _bare_analyzer()
    sents = [{"ticker": "AAPL", "direction": "bullish",
              "short_term": "bullish", "medium_term": "slightly_bullish", "long_term": "neutral"}]
    a._reconcile_asset_directions(sents)
    assert sents[0]["short_term"] == "bullish"
    assert sents[0]["medium_term"] == "slightly_bullish"
    assert sents[0]["long_term"] == "neutral"

def test_reconcile_flips_direction_against_all_opposing_horizons():
    a = _bare_analyzer()
    sents = [{"ticker": "X", "direction": "bullish",
              "short_term": "bearish", "medium_term": "bearish", "long_term": "slightly_bearish"}]
    a._reconcile_asset_directions(sents)
    assert sents[0]["direction"] == "bearish"
    # horizons unchanged by reconciliation
    assert sents[0]["short_term"] == "bearish" and sents[0]["long_term"] == "slightly_bearish"

def test_reconcile_keeps_direction_when_a_horizon_agrees():
    a = _bare_analyzer()
    sents = [{"ticker": "X", "direction": "bullish",
              "short_term": "bullish", "medium_term": "neutral", "long_term": "bearish"}]
    a._reconcile_asset_directions(sents)
    assert sents[0]["direction"] == "bullish"
