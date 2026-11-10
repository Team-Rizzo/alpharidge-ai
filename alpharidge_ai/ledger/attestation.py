"""Leaf hashes, the Merkle root and the attestation message of the signed points ledger,
shared by the API (which signs) and validators (which verify)."""
from __future__ import annotations

import hashlib
from typing import Dict, List

# The API attestation key is sr25519, bittensor_wallet's Keypair default.
from bittensor_wallet import Keypair

from alpharidge_ai.schema.hashing import canonical_json

# The ledger's signer name, the epoch length in blocks, and the blocks past an epoch's end
# before the API serves it.
LEDGER_SIGNER = "event-ledger"
BLOCK_LENGTH = 100
CLOSE_GRACE_BLOCKS = 2

def _round(x) -> float:
    result = round(float(x), 6)
    return 0.0 if result == 0.0 else result  # collapse -0.0 for canonical stability


# Leaves with this version also hash the claim id, its receipt time and bucket, the
# coldkey at receipt when known, and the page-judge record's hash on a paid row.
LEAF_V2 = 2


# Leaves hash canonical JSON objects and internal nodes hash two hex digests, so a leaf
# can never pass as a node.
def leaf_hash(verdict: dict) -> str:
    fields = {
        "resource_type": verdict["resource_type"],
        "resource_id": str(verdict["resource_id"]),
        "miner_hotkey": verdict["miner_hotkey"],
        "validator_verdict": verdict["validator_verdict"],
        "categorical_key": verdict["categorical_key"],
        "points_awarded": _round(verdict["points_awarded"]),
    }
    version = verdict.get("leaf_version")
    if version not in (None, LEAF_V2):
        raise ValueError(f"unknown leaf_version {version!r}")
    if version == LEAF_V2:
        fields.update(leaf_version=LEAF_V2, claim_id=str(verdict["claim_id"] or ""),
                      received_ms=int(verdict["received_ms"]), bucket=int(verdict["bucket"]))
        if verdict.get("coldkey"):
            fields["coldkey"] = str(verdict["coldkey"])
        if verdict.get("verdict_hash"):
            fields["verdict_hash"] = str(verdict["verdict_hash"])
    payload = canonical_json(fields)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def merkle_root(verdicts: List[dict]) -> str:
    if not verdicts:
        return hashlib.sha256(b"").hexdigest()
    layer = sorted(leaf_hash(v) for v in verdicts)
    while len(layer) > 1:
        nxt = []
        for i in range(0, len(layer), 2):
            left = layer[i]
            right = layer[i + 1] if i + 1 < len(layer) else layer[i]
            nxt.append(hashlib.sha256((left + right).encode("utf-8")).hexdigest())
        layer = nxt
    return layer[0]


def attestation_message(validator_hotkey: str, epoch: int, per_miner_points: Dict[str, float],
                        total_points: float, merkle_root_hex: str) -> str:
    return canonical_json({
        "validatorHotkey": validator_hotkey,
        "epoch": int(epoch),
        "perMinerPoints": {k: _round(v) for k, v in sorted(per_miner_points.items())},
        "totalPoints": _round(total_points),
        "merkleRoot": merkle_root_hex,
    })


def verify_attestation(pubkey_ss58: str, message: str, signature_hex: str) -> bool:
    try:
        kp = Keypair(ss58_address=pubkey_ss58)  # sr25519 (default)
        return bool(kp.verify(message.encode("utf-8"), bytes.fromhex(signature_hex)))
    except Exception:
        return False
