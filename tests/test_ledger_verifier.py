from bittensor_wallet import Keypair

from alpharidge_ai.ledger import attestation as ac
from alpharidge_ai.ledger.verifier import MAX_POINTS_PER_UID, verify_points

API_KP = Keypair.create_from_seed("0x" + "11" * 32)


def _signed(validator, epoch, per_miner, kp=API_KP):
    total = float(sum(v for v in per_miner.values() if v == v and abs(v) != float("inf")))
    root = "ab" * 32
    sig = kp.sign(ac.attestation_message(validator, epoch, per_miner, total, root).encode()).hex()
    return {"validatorHotkey": validator, "epoch": epoch, "perMinerPoints": per_miner,
            "totalPoints": total, "merkleRoot": root}, sig


def _verify(att, sig, validator="valiA", uids=None, **kw):
    return verify_points(attestation=att, signature=sig, validator_hotkey=validator,
                         hotkey_to_uid=uids or {"mhk1": 5, "mhk2": 8},
                         pinned_pubkey=API_KP.ss58_address, **kw)


def test_valid_attestation_maps_points_to_uids():
    out, reason = _verify(*_signed("valiA", 7, {"mhk1": 3.0, "mhk2": 1.0}), epoch=7)
    assert reason == "accepted"
    assert out.points == {5: 3, 8: 1} and out.epoch == 7 and out.merkle_root == "ab" * 32


def test_forged_signature_rejected():
    att, _ = _signed("valiA", 7, {"mhk1": 3.0})
    _, forged = _signed("valiA", 7, {"mhk1": 3.0}, kp=Keypair.create_from_seed("0x" + "99" * 32))
    assert _verify(att, forged) == (None, "bad_signature")


def test_validator_and_epoch_must_match():
    att, sig = _signed("valiA", 7, {"mhk1": 3.0})
    assert _verify(att, sig, validator="other") == (None, "validator_mismatch")
    assert _verify(att, sig, epoch=8) == (None, "epoch_mismatch")


def test_out_of_range_and_nonfinite_points_are_dropped():
    att, sig = _signed("valiA", 7, {"mhk1": float(MAX_POINTS_PER_UID + 1), "mhk2": float("inf")})
    assert _verify(att, sig) == (None, "empty_payload")
    att, sig = _signed("valiA", 7, {"mhk1": float("inf"), "mhk2": 2.0})
    assert _verify(att, sig)[0].points == {8: 2}
