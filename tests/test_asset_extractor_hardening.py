"""Hardening tests for AssetExtractor — kills the §3.1 false-positive ticker bugs.

These encode the MINER_DATA_QUALITY_REVIEW §3.1 findings as regression tests:
ambiguous dictionary-word / foreign-word / single-letter collisions must NOT
produce tickers unless corroborated by a cashtag, an exact case-sensitive ticker,
a non-ambiguous name, or an in-context financial cue.
"""

import pytest

from talisman_ai.analyzer.asset_extractor import AssetExtractor


@pytest.fixture(scope="module")
def ax():
    return AssetExtractor()


def _tickers(matches):
    return {m.ticker for m in matches}


# --- Common-word collisions: ambient finance language must NOT rescue an
#     ordinary English word; only strong evidence or the NER org path may. ----

def test_price_target_not_matched_as_target_corp(ax):
    m = ax.extract_assets("Palantir is expensive",
                          "Palantir (PLTR) shares; analysts set a price target given strong revenue.")
    assert "TGT" not in _tickers(m)
    assert "PLTR" in _tickers(m)


def test_optimism_word_not_matched_as_op_token(ax):
    m = ax.extract_assets("Fed holds rates",
                          "Markets showed optimism as the Fed held rates amid stock gains and revenue.")
    assert "OP" not in _tickers(m)


def test_travel_visa_not_matched_as_visa_inc(ax):
    m = ax.extract_assets("Goalkeeper's mother gets visa",
                          "She received a visa to travel; the cost and stock of tickets were limited.")
    assert "V" not in _tickers(m)
    assert "COST" not in _tickers(m)


def test_common_word_with_strong_evidence_is_kept(ax):
    # Cashtag / distinctive full name still corroborates the very-common word.
    assert "V" in _tickers(ax.extract_assets("Visa earnings",
                          "Visa Inc reported revenue; $V shares rose on the nasdaq."))
    assert "COST" in _tickers(ax.extract_assets("Costco",
                          "Costco Wholesale Corporation membership revenue grew."))


def test_asset_only_names_are_not_treated_as_common_words(ax):
    # "bitcoin"/"ethereum" are frequent but not ordinary dictionary words, so
    # nearby context still corroborates them (they have no non-asset meaning).
    m = ax.extract_assets("Bitcoin and Ethereum rally",
                          "Bitcoin surged past $100K while Ethereum hit $4,500. Solana also gained.")
    assert {"BTC", "ETH", "SOL"} <= _tickers(m)


def test_italian_uso_not_matched_as_oil_fund(ax):
    # "uso" = Italian for "use"; must not become USO (US Oil Fund).
    m = ax.extract_assets("Trump guida 16 miliardari",
                          "Un accordo commerciale per uso strategico delle risorse.",
                          language="it")
    assert "USO" not in _tickers(m)


def test_ada_person_name_not_matched_without_context(ax):
    # Soap-opera character "Ada" must not become ADA (Cardano).
    m = ax.extract_assets("Be My Sunshine replay",
                          "The character Ada Masal returns in tonight's episode.")
    assert "ADA" not in _tickers(m)


def test_gold_prize_not_matched_in_nonfinancial(ax):
    # Winning "gold" at a festival must not become XAU.
    m = ax.extract_assets("Jazz festival results",
                          "The young musician won gold at the international competition.")
    assert "XAU" not in _tickers(m)


def test_lone_letter_v_not_matched(ax):
    # A lone capital "V" (TV schedule) must not become Visa.
    m = ax.extract_assets("Diretta Tennis", "Stasera V in onda alle 21 sul canale.")
    assert "V" not in _tickers(m)


# --- True positives that MUST survive (corroboration paths) -------------------

def test_ada_matched_with_cashtag(ax):
    m = ax.extract_assets("Cardano update", "$ADA staking rewards increased this week.")
    assert "ADA" in _tickers(m)


def test_ada_matched_with_proper_name(ax):
    # "cardano" is a non-ambiguous identifier -> corroborates the ADA match.
    m = ax.extract_assets("Cardano rally",
                          "Cardano's ADA token surged 12% as investors piled into the exchange.")
    assert "ADA" in _tickers(m)


def test_gold_matched_with_financial_context(ax):
    m = ax.extract_assets("Gold hits record",
                          "Gold prices rallied as investors sought a safe haven; gold futures jumped 3%.")
    assert "XAU" in _tickers(m)


def test_visa_matched_with_cashtag(ax):
    m = ax.extract_assets("Visa earnings", "$V Visa reported strong quarterly revenue.")
    assert "V" in _tickers(m)


# --- Backward-compat: existing behavior must not regress ----------------------

def test_cashtag_still_works(ax):
    assert "BTC" in _tickers(ax.extract_assets("$BTC rallies", "$BTC is up 5% today"))


def test_proper_names_still_work(ax):
    m = ax.extract_assets("Bitcoin and Ethereum rally",
                          "Bitcoin surged past $100K while Ethereum hit $4,500. Solana also gained.")
    assert {"BTC", "ETH", "SOL"} <= _tickers(m)


def test_language_param_defaults_to_en(ax):
    # Calling without language must still work (backward compatible signature).
    assert "BTC" in _tickers(ax.extract_assets("$BTC", "$BTC up"))
