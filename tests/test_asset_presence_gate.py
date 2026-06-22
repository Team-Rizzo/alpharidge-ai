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


def _assets(*tickers):
    return {t: object() for t in tickers}


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
