"""Anti-cheat: validator-resolved-but-miner-missing asset presence gate.

The Tier-2b asset-sentiment gate is skipped when miner and validator share no
common assets (asset_sentiment_agreement returns (1.0, 0)). A miner that submits
ZERO assets therefore bypasses the hard gate entirely and forfeits only Tier-3
partial credit — the same "omission = free pass" hole that §2.3 closed for
embeddings. Mirror that fix: if the validator resolved assets but the miner sent
none, hard-fail. Asymmetric (validator-missing never penalizes the miner) and
safe against cross-hardware jitter — asset extraction is deterministic gazetteer
matching, so a total miner-absence against a non-empty validator set is a skip
signal, not jitter (jitter only flips an asset's sentiment class).
"""
from talisman_ai.analyzer.scoring import asset_presence_ok, ASSET_PRESENCE_FLOOR


from types import SimpleNamespace


def _assets(*tickers):
    # bare objects -> resolved_via defaults to the deterministic "keyword" path
    return {t: object() for t in tickers}


def _src(**by_ticker):
    return {t: SimpleNamespace(resolved_via=src) for t, src in by_ticker.items()}


def test_both_empty_passes():
    assert asset_presence_ok(_assets(), _assets()) is True


def test_validator_assets_miner_none_fails():
    assert asset_presence_ok(_assets(), _assets("AAPL", "MSFT")) is False


def test_single_validator_asset_miner_none_fails_at_default_floor():
    # default floor is 1 -> a lone validator asset with an empty miner still fails
    assert ASSET_PRESENCE_FLOOR == 1
    assert asset_presence_ok(_assets(), _assets("AAPL")) is False


def test_both_have_assets_passes():
    assert asset_presence_ok(_assets("AAPL", "MSFT"), _assets("AAPL", "MSFT")) is True


def test_partial_miner_omission_passes_here():
    # miner missing SOME assets is handled by the agreement gate (scored on common
    # assets), not this presence check — only TOTAL absence is a bypass.
    assert asset_presence_ok(_assets("AAPL"), _assets("AAPL", "MSFT", "GOOG")) is True


def test_miner_extra_assets_passes():
    # validator found none, miner sent some -> not a gate bypass, allowed here
    assert asset_presence_ok(_assets("AAPL"), _assets()) is True


def test_floor_override_raises_tolerance():
    # with a floor of 2, a single validator asset + empty miner is tolerated
    assert asset_presence_ok(_assets(), _assets("AAPL"), floor=2) is True
    assert asset_presence_ok(_assets(), _assets("AAPL", "MSFT"), floor=2) is False


# ── only DETERMINISTIC (gazetteer/override) assets count toward the floor ──────
# Neural-NER-resolved assets (ReFinED/GLiNER) can diverge across hardware, so a
# validator-only neural asset must NOT hard-fail an honest miner that resolved 0.
def test_neural_only_validator_assets_do_not_trigger_gate():
    assert asset_presence_ok({}, _src(AAPL="refined", MSFT="refined")) is True


def test_keyword_validator_asset_with_miner_none_still_fails():
    assert asset_presence_ok({}, _src(AAPL="keyword")) is False


def test_override_counts_as_deterministic():
    # financial_overrides.json is a static dict lookup -> deterministic -> gated
    assert asset_presence_ok({}, _src(BTC="override")) is False


def test_mixed_sources_count_only_deterministic_toward_floor():
    # 1 keyword (deterministic) + 1 neural; floor 1 -> keyword alone meets it -> fail
    assert asset_presence_ok({}, _src(AAPL="keyword", MSFT="refined")) is False
    # but a lone neural asset with floor 1 does NOT meet it -> pass
    assert asset_presence_ok({}, _src(MSFT="refined")) is True
