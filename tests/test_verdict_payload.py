from talisman_ai.utils import attestation_crypto as ac
from talisman_ai.validator.verdict_payload import build_verdict_fields


class _A:
    sentiment = "bull"; asset_symbol = "BTC"; content_type = "analysis"
    technical_quality = "high"; market_analysis = "bullish"; impact_potential = "high"


def test_build_verdict_fields_matches_crypto():
    analysis = _A()
    ah = ac.analysis_hash(ac.analysis_to_dict(analysis))
    fields = build_verdict_fields(
        miner_hotkey="minerX", miner_signature="sig", nonce="n1",
        analysis=analysis, validator_verdict="valid", points_awarded=1.0, epoch=7,
    )
    assert fields == {
        "miner_hotkey": "minerX", "miner_signature": "sig", "nonce": "n1",
        "miner_analysis_hash": ah, "validator_verdict": "valid",
        "categorical_key": "bull|BTC|analysis|high|bullish|high",
        "points_awarded": 1.0, "epoch": 7,
    }


def test_invalid_verdict_zero_points():
    fields = build_verdict_fields(
        miner_hotkey="m", miner_signature="s", nonce="n", analysis=_A(),
        validator_verdict="invalid", points_awarded=1.0, epoch=7,
    )
    assert fields["validator_verdict"] == "invalid"
    assert fields["points_awarded"] == 0.0


from talisman_ai.validator.verdict_payload import collect_verdict_meta


class _It:
    def __init__(self, rid):
        self.id = rid


def test_collect_verdict_meta_skips_unsigned():
    items = [_It("1"), _It("2"), _It("3")]
    sigs = {"1": "s1", "2": "s2"}        # item 3 unsigned
    nonces = {"1": "n1", "2": "n2"}
    meta = collect_verdict_meta(items, sigs, nonces, "valid", 7)
    assert set(meta) == {"1", "2"}
    assert meta["1"] == {"miner_signature": "s1", "nonce": "n1",
                         "validator_verdict": "valid", "epoch": 7}
