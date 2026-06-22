from alpharidge_ai.utils import attestation_crypto as ac
from alpharidge_ai.utils.miner_signing import sign_items

from bittensor_wallet import Keypair  # sr25519 (default)


class _Item:
    def __init__(self, rid, analysis):
        self.id = rid
        self.analysis = analysis


class _A:
    sentiment = "bull"; asset_symbol = "BTC"; content_type = "analysis"
    technical_quality = "high"; market_analysis = "bullish"; impact_potential = "high"


def test_sign_items_produces_verifiable_sigs():
    kp = Keypair.create_from_seed("0x" + "77" * 32)
    items = [_Item("100", _A()), _Item("200", _A())]
    sigs, nonces = sign_items(kp, items, id_attr="id")
    assert set(sigs) == {"100", "200"} and set(nonces) == {"100", "200"}
    for rid in ("100", "200"):
        ah = ac.analysis_hash(ac.analysis_to_dict(_A()))
        assert ac.verify_miner_signature(kp.ss58_address, rid, ah, nonces[rid], sigs[rid]) is True
