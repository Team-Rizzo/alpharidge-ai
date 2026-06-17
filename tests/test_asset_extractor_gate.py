"""
Task 2: Ticker uppercase-acronym false positive gate tests.

RED/GREEN verification:
- test_leo_low_earth_orbit_not_a_ticker: LEO case-sensitive identifier was
  auto-getting strong_evidence=True (3-char, above _AMBIGUOUS_CS_MAX_LEN=2),
  bypassing the corroboration gate. After fix, LEO is in _ACRONYM_BLOCKLIST
  so it is ambiguous AND non-corroborating.
- test_real_leo_cashtag_kept: $LEO cashtag sets strong_evidence in Phase 1,
  unaffected by the fix.
- test_common_word_gold_not_ticker_in_prose: "gold" is in common_english_words.txt,
  so _is_very_common_word("gold") is True; after fix _is_noncorroborating also
  covers blocklisted acronyms; the word "gold" alone in a non-financial context
  must not surface the GOLD ticker.
- test_barrick_gold_company_kept: "barrick gold" is a distinctive non-ambiguous
  alias/identifier for Barrick Gold (NYSE: GOLD); it is not a very common word
  and not blocklisted, so it sets strong_evidence=True and the asset is kept.
"""
from talisman_ai.analyzer.asset_extractor import AssetExtractor

ax = AssetExtractor()


def tickers(title, body):
    return {m.ticker for m in ax.extract_assets(title, body)}


def test_leo_low_earth_orbit_not_a_ticker():
    t = tickers(
        "SpaceX launch",
        "The satellite reached low earth orbit (LEO) as shares of the firm rallied 19% on the IPO.",
    )
    assert "LEO" not in t


def test_real_leo_cashtag_kept():
    assert "LEO" in tickers(
        "Crypto update",
        "Bitfinex token $LEO surged 12% amid heavy trading volume.",
    )


def test_common_word_gold_not_ticker_in_prose():
    assert "GOLD" not in tickers(
        "Olympics",
        "She won the gold medal after a record run; the crowd cheered.",
    )


def test_barrick_gold_company_kept():
    # Barrick Gold's distinctive name should corroborate GOLD
    assert "GOLD" in tickers(
        "Earnings",
        "Barrick Gold raised its dividend as gold prices rallied and the miner beat earnings.",
    )
