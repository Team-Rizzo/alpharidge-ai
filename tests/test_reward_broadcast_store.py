import pytest
from alpharidge_ai.validator.reward_broadcast_store import RewardBroadcastStore
from alpharidge_ai.utils import attestation_crypto as ac

from bittensor_wallet import Keypair  # sr25519 (default)


def _signed_attestation(api_kp, validator_hotkey, epoch, per_miner):
    total = float(sum(v for v in per_miner.values() if isinstance(v, (int, float)) and v == v))
    root = "ab" * 32  # 64-char hex
    msg = ac.attestation_message(validator_hotkey, epoch, per_miner, total, root)
    sig = api_kp.sign(msg.encode("utf-8")).hex()
    att = {"validatorHotkey": validator_hotkey, "epoch": epoch,
           "perMinerPoints": per_miner, "totalPoints": total, "merkleRoot": root}
    return att, sig


@pytest.fixture
def store(tmp_path):
    return RewardBroadcastStore(path=tmp_path / "b.json")


def test_valid_attestation_ingested_and_mapped_to_uids(store):
    api_kp = Keypair.create_from_seed("0x" + "11" * 32)
    att, sig = _signed_attestation(api_kp, "valiA", 7, {"mhk1": 3.0, "mhk2": 1.0})
    hotkey_to_uid = {"mhk1": 5, "mhk2": 8}
    accepted, reason = store.ingest_attestation(
        attestation=att, signature=sig, sender_hotkey="valiA",
        hotkey_to_uid=hotkey_to_uid, pinned_pubkey=api_kp.ss58_address)
    assert accepted is True, reason
    assert store.aggregate_epoch(7) == {5: 3, 8: 1}


def test_forged_signature_rejected(store):
    api_kp = Keypair.create_from_seed("0x" + "11" * 32)
    wrong_kp = Keypair.create_from_seed("0x" + "99" * 32)
    att, _ = _signed_attestation(api_kp, "valiA", 7, {"mhk1": 3.0})
    forged = wrong_kp.sign(ac.attestation_message("valiA", 7, {"mhk1": 3.0}, 3.0, "deadbeef").encode()).hex()
    accepted, reason = store.ingest_attestation(
        attestation=att, signature=forged, sender_hotkey="valiA",
        hotkey_to_uid={"mhk1": 5}, pinned_pubkey=api_kp.ss58_address)
    assert accepted is False
    assert reason == "bad_signature"


def test_sender_mismatch_rejected(store):
    api_kp = Keypair.create_from_seed("0x" + "11" * 32)
    att, sig = _signed_attestation(api_kp, "valiA", 7, {"mhk1": 3.0})
    accepted, reason = store.ingest_attestation(
        attestation=att, signature=sig, sender_hotkey="someoneElse",
        hotkey_to_uid={"mhk1": 5}, pinned_pubkey=api_kp.ss58_address)
    assert accepted is False
    assert reason == "sender_mismatch"


def test_replay_old_seq_rejected(store):
    api_kp = Keypair.create_from_seed("0x" + "11" * 32)
    att1, sig1 = _signed_attestation(api_kp, "valiA", 7, {"mhk1": 1.0})
    store.ingest_attestation(attestation=att1, signature=sig1, sender_hotkey="valiA",
                             hotkey_to_uid={"mhk1": 5}, pinned_pubkey=api_kp.ss58_address)
    accepted, reason = store.ingest_attestation(
        attestation=att1, signature=sig1, sender_hotkey="valiA",
        hotkey_to_uid={"mhk1": 5}, pinned_pubkey=api_kp.ss58_address)
    assert accepted is False
    assert reason.startswith("duplicate_or_old_seq")


def test_per_uid_points_clamped(store):
    from alpharidge_ai.validator.reward_broadcast_store import MAX_POINTS_PER_UID
    api_kp = Keypair.create_from_seed("0x" + "11" * 32)
    att, sig = _signed_attestation(api_kp, "valiA", 7, {"mhk1": float(MAX_POINTS_PER_UID + 5000)})
    accepted, reason = store.ingest_attestation(
        attestation=att, signature=sig, sender_hotkey="valiA",
        hotkey_to_uid={"mhk1": 5}, pinned_pubkey=api_kp.ss58_address)
    assert accepted is False
    assert store.aggregate_epoch(7) == {}


def test_merkle_root_retained_for_deep_verify(store):
    api_kp = Keypair.create_from_seed("0x" + "11" * 32)
    att, sig = _signed_attestation(api_kp, "valiA", 7, {"mhk1": 1.0})
    store.ingest_attestation(attestation=att, signature=sig, sender_hotkey="valiA",
                             hotkey_to_uid={"mhk1": 5}, pinned_pubkey=api_kp.ss58_address)
    assert store.get_merkle_root(epoch=7, sender="valiA") == "ab" * 32


def test_wire_seq_cannot_silence_future_epochs(store):
    from alpharidge_ai.validator.reward_broadcast_store import MAX_SEQ_EPOCH_SKEW
    api_kp = Keypair.create_from_seed("0x" + "11" * 32)
    att, sig = _signed_attestation(api_kp, "valiA", 10, {"mhk1": 1.0})
    # Replay a valid epoch-10 attestation but with an inflated wire seq.
    store.ingest_attestation(attestation=att, signature=sig, sender_hotkey="valiA",
                             hotkey_to_uid={"mhk1": 5}, pinned_pubkey=api_kp.ss58_address,
                             seq=10 + MAX_SEQ_EPOCH_SKEW)
    # The legitimate epoch-11 attestation must still be accepted (wire seq was ignored).
    att2, sig2 = _signed_attestation(api_kp, "valiA", 11, {"mhk1": 2.0})
    accepted2, reason2 = store.ingest_attestation(
        attestation=att2, signature=sig2, sender_hotkey="valiA",
        hotkey_to_uid={"mhk1": 5}, pinned_pubkey=api_kp.ss58_address, seq=11)
    assert accepted2 is True, f"legit follow-up silenced: {reason2}"
    assert store.aggregate_epoch(11) == {5: 2}


def test_nonfinite_points_skipped_not_crash(store):
    api_kp = Keypair.create_from_seed("0x" + "11" * 32)
    att, sig = _signed_attestation(api_kp, "valiA", 7, {"mhk1": float("inf"), "mhk2": 2.0})
    accepted, reason = store.ingest_attestation(
        attestation=att, signature=sig, sender_hotkey="valiA",
        hotkey_to_uid={"mhk1": 5, "mhk2": 6}, pinned_pubkey=api_kp.ss58_address)
    assert accepted is True, reason
    assert store.aggregate_epoch(7) == {6: 2}   # inf skipped, finite kept


def test_route_prefers_attestation_when_present(store):
    from alpharidge_ai.validator import reward_broadcast_store as rbs
    api_kp = Keypair.create_from_seed("0x" + "11" * 32)
    att, sig = _signed_attestation(api_kp, "valiA", 7, {"mhk1": 2.0})
    accepted, reason = rbs.route_reward_broadcast(
        store=store, sender_hotkey="valiA", epoch=7, seq=7,
        uid_points={}, attestation=att, attestation_sig=sig,
        hotkey_to_uid={"mhk1": 4}, pinned_pubkey=api_kp.ss58_address)
    assert accepted is True, reason
    assert store.aggregate_epoch(7) == {4: 2}


def test_route_falls_back_to_legacy_uid_points(store):
    from alpharidge_ai.validator import reward_broadcast_store as rbs
    accepted, reason = rbs.route_reward_broadcast(
        store=store, sender_hotkey="valiB", epoch=7, seq=7,
        uid_points={3: 2}, attestation=None, attestation_sig=None,
        hotkey_to_uid={}, pinned_pubkey="ignored")
    assert accepted is True, reason
    assert store.aggregate_epoch(7) == {3: 2}


def test_route_skips_blacklisted_sender(store):
    from alpharidge_ai.validator import reward_broadcast_store as rbs
    accepted, reason = rbs.route_reward_broadcast(
        store=store, sender_hotkey="badVali", epoch=7, seq=7,
        uid_points={3: 2}, attestation=None, attestation_sig=None,
        hotkey_to_uid={}, pinned_pubkey="x", blacklisted={"badVali"})
    assert accepted is False
    assert reason == "sender_blacklisted"


def test_route_drops_legacy_when_enforced(store):
    from alpharidge_ai.validator import reward_broadcast_store as rbs
    accepted, reason = rbs.route_reward_broadcast(
        store=store, sender_hotkey="vali", epoch=7, seq=7,
        uid_points={3: 2}, attestation=None, attestation_sig=None,
        hotkey_to_uid={}, pinned_pubkey="x", enforce_signed=True)
    assert accepted is False
    assert reason == "unsigned_broadcast_rejected"
