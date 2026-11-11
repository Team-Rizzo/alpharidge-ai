"""Claim receipts: the signature check and the miner keeping only receipts for its own claims."""

from datetime import datetime, timezone
from types import SimpleNamespace

from bittensor_wallet import Keypair

from alpharidge_ai import config
from alpharidge_ai.ledger import receipts as rc
from alpharidge_ai.miner.loop import MinerLoop
from alpharidge_ai.miner.store import MinerStore

KEY = Keypair.create_from_seed("0x" + "5d" * 32)
T0 = 1_790_000_000_000


def receipt(seq=1, claim_id=101, hotkey="hk", body="ab" * 32, at=T0, key=KEY):
    return rc.make(seq=seq, prev=rc.GENESIS, claim_id=claim_id, hotkey=hotkey,
                   body_sha256=body, received_ms=at, sign=lambda t: key.sign(t.encode()).hex())


def test_edited_or_foreign_receipts_fail():
    r = receipt()
    assert rc.valid(r, KEY.ss58_address)
    assert not rc.valid({**r, "received_ms": r["received_ms"] - 5}, KEY.ss58_address)
    assert not rc.valid(receipt(key=Keypair.create_from_seed("0x" + "6e" * 32)),
                        KEY.ss58_address)
    assert rc.matches(r, 101, "hk", "ab" * 32, T0)
    assert not rc.matches(r, 101, "other", "ab" * 32, T0)


def test_miner_keeps_only_a_matching_signed_receipt(monkeypatch):
    monkeypatch.setattr(config, "API_ATTESTATION_PUBKEY", KEY.ss58_address)
    store = MinerStore()
    loop = MinerLoop(SimpleNamespace(ss58_address="hk"), store, None, None, None)
    claim = SimpleNamespace(claim_hash=lambda: "ab" * 32)
    at = datetime.fromtimestamp(T0 / 1000, timezone.utc).isoformat()
    loop.keep_receipt(claim, {"id": 102, "received_at": at, "receipt": receipt(claim_id=101)})
    assert store.receipts() == []
    loop.keep_receipt(claim, {"id": 101, "received_at": at, "receipt": receipt()})
    assert [r["claim_id"] for r in store.receipts()] == ["101"]
