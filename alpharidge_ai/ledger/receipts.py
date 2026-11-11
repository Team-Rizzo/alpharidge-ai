"""Claim receipts: the API signs one for every claim it stores, with the pinned ledger key;
miners verify and keep them.

`received_ms` is the claim's receipt time, the one that decides who was first. `seq` and
`prev` (the previous receipt's hash) place it in the API's sequence of receipts.
"""
from __future__ import annotations

from typing import Callable, Mapping

from bittensor_wallet import Keypair

from alpharidge_ai.schema.hashing import canonical_json, sha256_hex

RECEIPT_VERSION = 1
GENESIS = "0" * 64
_INTS = ("v", "seq", "received_ms")
_STRS = ("prev", "claim_id", "hotkey", "body_sha256")


def message(r: Mapping) -> str:
    """The signed text: the receipt's fields as canonical JSON."""
    out = {k: int(r[k]) for k in _INTS}
    out.update({k: str(r[k]) for k in _STRS})
    return canonical_json(out)


def make(*, seq: int, prev: str, claim_id, hotkey: str, body_sha256: str, received_ms: int,
         sign: Callable[[str], str]) -> dict:
    """A receipt with its hash and signature; `sign` signs a message with the API key."""
    r = {"v": RECEIPT_VERSION, "seq": int(seq), "prev": prev, "claim_id": str(claim_id),
         "hotkey": hotkey, "body_sha256": body_sha256, "received_ms": int(received_ms)}
    text = message(r)
    return {**r, "hash": sha256_hex(text), "signature": sign(text)}


def valid(r: Mapping, pubkey: str) -> bool:
    """Well formed, hashed as it reads and signed by `pubkey`."""
    try:
        text = message(r)
        if int(r["v"]) != RECEIPT_VERSION or int(r["seq"]) < 1 or sha256_hex(text) != r["hash"]:
            return False
        return bool(Keypair(ss58_address=pubkey).verify(
            text.encode("utf-8"), bytes.fromhex(str(r["signature"]))))
    except Exception:
        return False


def matches(r: Mapping, claim_id, hotkey: str, body_sha256: str, received_ms) -> bool:
    """The receipt is for this claim, sent by this hotkey with this body at this time."""
    try:
        return (str(r["claim_id"]) == str(claim_id) and r["hotkey"] == hotkey
                and r["body_sha256"] == body_sha256
                and abs(int(r["received_ms"]) - int(received_ms)) < 1)
    except (KeyError, TypeError, ValueError):
        return False
