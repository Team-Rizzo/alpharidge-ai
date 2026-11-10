from alpharidge_ai.ledger import attestation as ac

from bittensor_wallet import Keypair  # sr25519 (default)


def test_canonical_json_sorted_compact():
    assert ac.canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_merkle_root_order_independent():
    v1 = {"resource_type": "ledger", "resource_id": "1", "miner_hotkey": "m1",
          "validator_verdict": "valid", "categorical_key": "finance|event", "points_awarded": 1.0}
    v2 = {"resource_type": "ledger", "resource_id": "2", "miner_hotkey": "m2",
          "validator_verdict": "valid", "categorical_key": "sports|event", "points_awarded": 1.0}
    assert ac.merkle_root([v1, v2]) == ac.merkle_root([v2, v1])


def test_attestation_verify_roundtrip():
    kp = Keypair.create_from_seed("0x" + "11" * 32)
    msg = ac.attestation_message("vali1", 7, {"m1": 3.0}, 3.0, "abcd")
    sig = kp.sign(msg.encode("utf-8")).hex()
    assert ac.verify_attestation(kp.ss58_address, msg, sig) is True
    assert ac.verify_attestation(kp.ss58_address, msg, "00") is False


def test_merkle_root_empty_and_leaf_change():
    assert ac.merkle_root([]) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    v1 = {"resource_type": "ledger", "resource_id": "1", "miner_hotkey": "m1",
          "validator_verdict": "valid", "categorical_key": "finance|event", "points_awarded": 1.0}
    assert ac.merkle_root([v1]) != ac.merkle_root([dict(v1, points_awarded=2.0)])


def test_negative_zero_points_canonicalize_to_zero():
    assert (ac.attestation_message("v", 1, {"m": -0.0}, -0.0, "r")
            == ac.attestation_message("v", 1, {"m": 0.0}, 0.0, "r"))


def test_shared_values_ignore_the_environment(monkeypatch):
    import importlib

    from alpharidge_ai import config
    for name in ("API_ATTESTATION_PUBKEY", "BLOCK_LENGTH", "BURN_UID"):
        monkeypatch.setenv(name, "7")
    importlib.reload(config)
    assert config.API_ATTESTATION_PUBKEY == "5DqYRNaJ9FJ2cuJTFxbU5HDLeeotpgj6Zgkrkw4RgS6SA4nf"
    assert (config.BLOCK_LENGTH, config.BURN_UID) == (100, 189)
