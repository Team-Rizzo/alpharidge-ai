"""
Opt-in subtensor localnet test: real chain + real axon<->dendrite ValidatorRewards
round-trip carrying an API-signed attestation, verified OFFLINE on the receiver.

Covers the one thing the in-process integration harness cannot: that the new
protocol fields (attestation dict + attestation_sig) serialize across a real
bittensor wire between two chain-registered validators, and the receiver's
route_reward_broadcast offline-verifies + ingests them.

Gated behind E2E_LOCALNET=1 and skipped if no chain is reachable.

Prereq: a running node-subtensor (local chainspec). Start one with:
    cd <subtensor>; ./scripts/localnet.sh          # or a single --dev node
Run:
    E2E_LOCALNET=1 python -m pytest tests/test_e2e_localnet.py -v

Config (env overrides):
    E2E_LOCALNET_WS      default ws://127.0.0.1:9946
    E2E_LOCALNET_NETUID  default 1
    E2E_LOCALNET_PORT    base TCP port for the test axons (default 18091)
"""
import os
import sys

import pytest

if os.environ.get("E2E_LOCALNET") != "1":
    pytest.skip("set E2E_LOCALNET=1 to run the localnet test (needs node-subtensor)",
                allow_module_level=True)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import bittensor as bt  # noqa: E402
from bittensor_wallet import Keypair as WalletKeypair  # noqa: E402

from bittensor_wallet import Keypair  # sr25519 (default)

from talisman_ai.protocol import ValidatorRewards          # noqa: E402
from talisman_ai.utils import attestation_crypto as ac       # noqa: E402
from talisman_ai.validator.reward_broadcast_store import (   # noqa: E402
    RewardBroadcastStore, route_reward_broadcast,
)

WS = os.environ.get("E2E_LOCALNET_WS", "ws://127.0.0.1:9946")
NETUID = int(os.environ.get("E2E_LOCALNET_NETUID", "1"))
BASE_PORT = int(os.environ.get("E2E_LOCALNET_PORT", "18091"))
WALLET_PATH = "/tmp/st_wallets_pytest"

# Skip the whole module if the chain isn't reachable / the subnet doesn't exist.
try:
    _sub = bt.Subtensor(network=WS)
    _sub.get_current_block()
    if not _sub.subnet_exists(NETUID):
        pytest.skip(f"netuid {NETUID} does not exist on {WS}", allow_module_level=True)
except Exception as e:  # pragma: no cover
    pytest.skip(f"localnet not reachable at {WS}: {e}", allow_module_level=True)


def _make_wallet(name, cold_uri, hot_uri):
    w = bt.Wallet(name=name, hotkey=name + "_hk", path=WALLET_PATH)
    w.set_coldkey(WalletKeypair.create_from_uri(cold_uri), encrypt=False, overwrite=True)
    w.set_coldkeypub(WalletKeypair.create_from_uri(cold_uri), encrypt=False, overwrite=True)
    w.set_hotkey(WalletKeypair.create_from_uri(hot_uri), encrypt=False, overwrite=True)
    return w


@pytest.mark.asyncio
async def test_localnet_attestation_roundtrip():
    sub = bt.Subtensor(network=WS)

    # Pinned API attestation key (root of trust).
    att_kp = Keypair.create_from_seed("0x" + "7f" * 32)
    pinned = att_kp.ss58_address

    wA = _make_wallet("e2eA", "//Alice", "//e2eA-hot-v1")
    wB = _make_wallet("e2eB", "//Bob", "//e2eB-hot-v1")
    A_hk, B_hk = wA.hotkey.ss58_address, wB.hotkey.ss58_address

    for w in (wA, wB):
        if not sub.is_hotkey_registered(netuid=NETUID, hotkey_ss58=w.hotkey.ss58_address):
            sub.burned_register(wallet=w, netuid=NETUID,
                                wait_for_inclusion=True, wait_for_finalization=False)

    mg = sub.metagraph(NETUID)
    hotkeys = list(mg.hotkeys)
    assert A_hk in hotkeys and B_hk in hotkeys, hotkeys
    hotkey_to_uid = {hk: i for i, hk in enumerate(hotkeys)}

    # API-signed attestation for validator A; "miner" is a registered hotkey -> maps to a uid.
    miner_hk = hotkeys[0]
    epoch = sub.get_current_block() // 100 + 5000
    per_miner = {miner_hk: 3.0}
    root = "ab" * 32
    msg = ac.attestation_message(A_hk, epoch, per_miner, 3.0, root)
    att_sig = att_kp.sign(msg.encode("utf-8")).hex()
    attestation_obj = {"validatorHotkey": A_hk, "epoch": epoch, "perMinerPoints": per_miner,
                       "totalPoints": 3.0, "merkleRoot": root}
    assert ac.verify_attestation(pinned, msg, att_sig)

    # ---- Happy path: real axon<->dendrite round-trip ----
    recv = RewardBroadcastStore(path="/tmp/localnet_pytest_recv.json")
    recv.last_seen_seq.clear(); recv.by_epoch_by_sender.clear(); recv.merkle_by_epoch_by_sender.clear()
    result = {}

    async def forward_ok(synapse: ValidatorRewards) -> ValidatorRewards:
        sender = synapse.dendrite.hotkey  # authenticated by axon signature check
        accepted, reason = route_reward_broadcast(
            store=recv, sender_hotkey=sender, epoch=synapse.epoch, seq=synapse.seq,
            uid_points=synapse.uid_points, attestation=getattr(synapse, "attestation", None),
            attestation_sig=getattr(synapse, "attestation_sig", None),
            hotkey_to_uid=hotkey_to_uid, pinned_pubkey=pinned, blacklisted=set())
        recv.save()
        result.update(accepted=accepted, reason=reason, sender=sender)
        return synapse

    axon = bt.Axon(wallet=wB, port=BASE_PORT, ip="127.0.0.1", external_ip="127.0.0.1")
    axon.attach(forward_fn=forward_ok)
    axon.start()
    try:
        info = bt.AxonInfo(version=1, ip="127.0.0.1", port=BASE_PORT, ip_type=4,
                           hotkey=B_hk, coldkey=wB.coldkeypub.ss58_address)
        syn = ValidatorRewards(epoch=epoch, uid_points={hotkey_to_uid[miner_hk]: 3},
                               sender_hotkey=A_hk, seq=epoch,
                               attestation=attestation_obj, attestation_sig=att_sig)
        dend = bt.Dendrite(wallet=wA)
        try:
            await dend.forward(axons=[info], synapse=syn, timeout=12, deserialize=False)
        finally:
            await dend.aclose_session()
    finally:
        axon.stop()

    import asyncio
    await asyncio.sleep(0.5)
    assert result.get("sender") == A_hk            # sender authenticated over the wire
    assert result.get("accepted") is True, result.get("reason")
    assert recv.aggregate_epoch(epoch).get(hotkey_to_uid[miner_hk]) == 3
    assert recv.get_merkle_root(epoch=epoch, sender=A_hk) == root  # survived the wire

    # ---- Negative: a forged/altered attestation over the wire is rejected ----
    recv2 = RewardBroadcastStore(path="/tmp/localnet_pytest_recv2.json")
    recv2.last_seen_seq.clear(); recv2.by_epoch_by_sender.clear()
    result.clear()

    async def forward_bad(synapse: ValidatorRewards) -> ValidatorRewards:
        accepted, reason = route_reward_broadcast(
            store=recv2, sender_hotkey=synapse.dendrite.hotkey, epoch=synapse.epoch,
            seq=synapse.seq, uid_points=synapse.uid_points,
            attestation=getattr(synapse, "attestation", None),
            attestation_sig=getattr(synapse, "attestation_sig", None),
            hotkey_to_uid=hotkey_to_uid, pinned_pubkey=pinned, blacklisted=set())
        result.update(accepted=accepted, reason=reason)
        return synapse

    axon2 = bt.Axon(wallet=wB, port=BASE_PORT + 1, ip="127.0.0.1", external_ip="127.0.0.1")
    axon2.attach(forward_fn=forward_bad)
    axon2.start()
    try:
        info2 = bt.AxonInfo(version=1, ip="127.0.0.1", port=BASE_PORT + 1, ip_type=4,
                            hotkey=B_hk, coldkey=wB.coldkeypub.ss58_address)
        syn_bad = ValidatorRewards(epoch=epoch + 1, uid_points={}, sender_hotkey=A_hk,
                                   seq=epoch + 1,
                                   attestation={**attestation_obj, "epoch": epoch + 1},  # stale sig
                                   attestation_sig=att_sig)
        dend2 = bt.Dendrite(wallet=wA)
        try:
            await dend2.forward(axons=[info2], synapse=syn_bad, timeout=12, deserialize=False)
        finally:
            await dend2.aclose_session()
    finally:
        axon2.stop()

    await asyncio.sleep(0.5)
    assert result.get("accepted") is False and result.get("reason") == "bad_signature", result
