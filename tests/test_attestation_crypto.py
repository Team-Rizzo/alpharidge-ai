from alpharidge_ai.utils import attestation_crypto as ac

from bittensor_wallet import Keypair  # sr25519 (default)


def test_canonical_json_sorted_compact():
    assert ac.canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_merkle_root_order_independent():
    v1 = {"resource_type": "tweet", "resource_id": "1", "miner_hotkey": "m1",
          "validator_verdict": "valid", "categorical_key": "BTC|bull", "points_awarded": 1.0}
    v2 = {"resource_type": "tweet", "resource_id": "2", "miner_hotkey": "m2",
          "validator_verdict": "valid", "categorical_key": "ETH|bear", "points_awarded": 1.0}
    assert ac.merkle_root([v1, v2]) == ac.merkle_root([v2, v1])


def test_attestation_verify_roundtrip():
    kp = Keypair.create_from_seed("0x" + "11" * 32)
    msg = ac.attestation_message("vali1", 7, {"m1": 3.0}, 3.0, "abcd")
    sig = kp.sign(msg.encode("utf-8")).hex()
    assert ac.verify_attestation(kp.ss58_address, msg, sig) is True
    assert ac.verify_attestation(kp.ss58_address, msg, "00") is False


def test_miner_sign_and_verify():
    kp = Keypair.create_from_seed("0x" + "22" * 32)
    ah = ac.analysis_hash({"sentiment": "bull", "asset_symbol": "BTC"})
    msg = ac.miner_sign_message("9", ah, "n1")
    sig = kp.sign(msg.encode("utf-8")).hex()
    assert ac.verify_miner_signature(kp.ss58_address, "9", ah, "n1", sig) is True


def test_categorical_key_and_analysis_to_dict_consistent():
    class A:
        sentiment = "bull"; asset_symbol = "BTC"; content_type = "analysis"
        technical_quality = "high"; market_analysis = "bullish"; impact_potential = "high"
    d = ac.analysis_to_dict(A())
    assert d["sentiment"] == "bull" and d["asset_symbol"] == "BTC"
    assert ac.categorical_key(A()) == "bull|BTC|analysis|high|bullish|high"
