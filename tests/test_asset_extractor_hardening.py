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


# --- Category-term misattribution: a generic product/sector word must not be a
#     `unique_identifier`, or every article about that category gets pinned to one
#     company. Root cause of the Sandisk(NAND)->Micron(MU) bug. -----------------

def test_nand_category_not_attributed_to_micron(ax):
    # Sandisk-style NAND article that never names Micron. "nand" was a Micron
    # unique_identifier, so the NAND flash category misattributed to MU.
    m = ax.extract_assets(
        "Sandisk: Momentum With A Capital M",
        "Sandisk Corporation has surged, driven by hyperscaler demand and soaring "
        "NAND prices. The data storage provider posted record EPS as memory margins expanded.")
    assert "MU" not in _tickers(m)


def test_dram_hbm_category_not_attributed_to_micron(ax):
    # Generic memory-industry commentary mentioning DRAM/HBM but not Micron.
    m = ax.extract_assets(
        "Memory prices climb",
        "DRAM and HBM contract prices rose this quarter as AI servers absorbed supply; "
        "Samsung and SK Hynix led shipments.")
    assert "MU" not in _tickers(m)


def test_micron_still_resolves_by_name_and_cashtag(ax):
    # Regression guard: removing category terms must NOT break genuine Micron detection.
    assert "MU" in _tickers(ax.extract_assets(
        "Micron earnings", "Micron Technology (MU) reported record revenue; $MU rose 5%."))


# --- Short-ticker acronym collisions: a bare 3-letter UPPERCASE ticker is also a
#     common non-financial acronym (Indian politics/law, "IP"=intellectual
#     property). It must require nearby financial context, like 1-2 letter tickers
#     already do — otherwise the expanded universe pins acronyms to random stocks.

def test_three_letter_caps_acronym_not_matched_in_nonfinancial(ax):
    # DKS (Dick's), BNS (Bank of Nova Scotia), NTR (Nutrien) collide with
    # D.K. Shivakumar, Bharatiya Nyaya Sanhita, and N.T. Rama Rao in Indian news.
    m = ax.extract_assets(
        "Congress consolidates position in Karnataka",
        "DKS strengthened his standing after the party's win; BNS provisions were cited "
        "by the court, and the NTR district administration issued a statement.")
    t = _tickers(m)
    assert "DKS" not in t and "BNS" not in t and "NTR" not in t


def test_ip_not_matched_as_international_paper_in_media(ax):
    # "IP" in entertainment/tech = intellectual property, not International Paper —
    # even amid ambient commercial language (prices, market, revenue). A bare
    # 2-letter ticker is too weak a signal to claim on its own.
    m = ax.extract_assets("Studios chase franchise IP",
                          "As streaming prices rise and the market shifts, studios race to own valuable "
                          "IP, betting franchise revenue will follow the most bankable IP.")
    assert "IP" not in _tickers(m)


def test_three_letter_ticker_still_matched_with_financial_context(ax):
    # Regression: the same-length ticker MUST resolve when financial context is present.
    assert "SLB" in _tickers(ax.extract_assets(
        "SLB: Buy the pullback",
        "SLB shares rallied as investors bought the energy stock; quarterly revenue beat estimates."))


def test_three_letter_ticker_matched_when_only_later_mention_is_financial(ax):
    # Real Seeking-Alpha shape: ticker leads the title (no cue word there), the
    # company's full name never appears, but later body mentions are clearly
    # financial. Context must be checked at EVERY occurrence, not just the first.
    m = ax.extract_assets(
        "SLB: Buy The Pullback On This Energy Technology Compounder",
        "Companies in transition can be attractive. SLB is evolving from oilfield services "
        "into digital workflows. SLB's forward P/E of 18.6 is below its normal valuation. "
        "I maintain a Strong Buy on SLB given a well-covered 2.4% yield and resilient shares.")
    assert "SLB" in _tickers(m)


# --- Moderate-frequency single-word names that ALSO have a common non-company
#     meaning ("costar"=co-star/Spanish "to cost", "intuit"=to intuit) must require
#     financial context, not match bare. Root cause of "costar"->CSGP on a soccer
#     article from the expanded universe. ---------------------------------------

def test_costar_word_not_matched_as_costar_group_in_nonfinancial(ax):
    # Spanish-language soccer text where "costar" means "to cost".
    m = ax.extract_assets("Doblete de Manzambi para Suiza",
                          "El gol llegó a costar caro a Bosnia tras el doblete en el segundo tiempo.")
    assert "CSGP" not in _tickers(m)

def test_intuit_verb_not_matched_in_nonfinancial(ax):
    m = ax.extract_assets("A coach's instinct",
                          "Great managers intuit what their players need before being told.")
    assert "INTU" not in _tickers(m)

def test_ambiguous_single_word_name_kept_with_financial_context(ax):
    # The same names MUST still resolve in a genuine market article (context present).
    assert "CSGP" in _tickers(ax.extract_assets(
        "CoStar earnings", "CoStar reported revenue growth as the real-estate data firm's shares rose."))
    assert "INTU" in _tickers(ax.extract_assets(
        "Intuit earnings", "Intuit stock rallied after quarterly revenue and guidance beat estimates."))


# --- Indices/forex are market CONTEXT, not the subject of a single-stock article.
#     They may only be primary when no equity/crypto/commodity asset is present. --

def test_index_not_primary_when_equity_present(ax):
    # Market-wrap shape: the index is mentioned far more than the stock, so it
    # out-scores it — but an index is never the *subject* when an equity is present.
    m = ax.extract_assets(
        "Stock Market Today: Nasdaq Jumps",
        "The Nasdaq Composite (IXIC) rallied. IXIC gained 2% as the Nasdaq hit a record. "
        "$MQ Marqeta also rose on the day.")
    prim = {x.ticker for x in m if x.is_primary_subject}
    assert "IXIC" not in prim
    assert "MQ" in prim  # the equity is the subject, not the index

def test_index_can_be_primary_when_alone(ax):
    m = ax.extract_assets(
        "Nasdaq hits record", "The Nasdaq Composite rallied to a record high as the index gained 2%.")
    prim = {x.ticker for x in m if x.is_primary_subject}
    assert "IXIC" in prim


# --- Primary subject = the asset named in the TITLE. A name-only subject (no
#     cashtag) must still be primary even though its raw score is low, and a
#     heavily-mentioned customer/peer/index must NOT crowd it out. ----------------

def _primaries(matches):
    return {m.ticker for m in matches if m.is_primary_subject}

def test_title_subject_is_primary_even_with_low_score(ax):
    # "Take-Two Shares Jump..." — TTWO is named once in the body but is the subject;
    # a sell-side bank (BMO) mentioned alongside must not be primary instead.
    m = ax.extract_assets(
        "Take-Two Shares Jump As 'Grand Theft Auto VI' Pre-Orders Open",
        "Take-Two Interactive shares jumped. A BMO analyst raised the price target.")
    assert "TTWO" in _primaries(m)
    assert "BMO" not in _primaries(m)

def test_only_title_asset_is_primary_not_customers(ax):
    # "Why Is ASML Stock Up Again Today?" — ASML is the subject; INTC/NVDA are
    # customers mentioned in the body and must not also be primary.
    m = ax.extract_assets(
        "Why Is ASML Stock Up Again Today?",
        "$ASML rose. Its customers $INTC and $NVDA also gained as chip demand recovered.")
    prim = _primaries(m)
    assert "ASML" in prim
    assert "INTC" not in prim and "NVDA" not in prim


# --- With no title asset, only a DOMINANT body asset becomes primary; a flat
#     list of holdings (an ETF comparison) must not promote a random constituent. -

def test_cashtag_does_not_substring_match_longer_cashtag(ax):
    # "$M" (Macy's) must NOT match inside "$MQ"; "$A" must NOT match inside "$AMD".
    t = _tickers(ax.extract_assets("Movers", "$MQ and $AMD both rallied today."))
    assert "M" not in t and "A" not in t and "AM" not in t
    assert {"MQ", "AMD"} <= t


def test_no_primary_for_flat_holdings_list(ax):
    # "VOO vs SCHD" shape: many holdings at similar relevance, none in the title.
    m = ax.extract_assets(
        "Which Fund Is the Smarter Buy Right Now?",
        "Top holdings include $GS, $UNH, $KO, $MSFT and $NVDA in roughly equal weights.")
    assert not _primaries(m)  # no single subject -> no primary

def test_no_primary_in_many_asset_roundup_even_if_one_leads(ax):
    # A holdings roundup with no title subject: one name leads (cashtag + full name)
    # but it's still just a constituent — many assets => no single subject.
    m = ax.extract_assets(
        "Which Fund Is the Smarter Buy Right Now?",
        "$GS Goldman Sachs is a notable holding. The fund also holds UnitedHealth, "
        "Coca-Cola, Microsoft, Nvidia and Amazon across the portfolio.")
    assert not _primaries(m)


def test_lone_dominant_body_asset_is_primary_without_title(ax):
    # The subject isn't in the vague title but clearly dominates the body.
    m = ax.extract_assets(
        "This Chip Stock Is Soaring",
        "$NVDA jumped 10%. Nvidia's revenue beat as Nvidia raised guidance; Nvidia shares "
        "hit a record high on strong demand.")
    assert "NVDA" in _primaries(m)


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
