"""
Shared crypto primitives for verifiable validator points (validator/miner side).

CRITICAL: canonical_json / miner_sign_message / merkle_root / attestation_message
MUST match talisman-ai-api/utils/attestation_crypto.py byte-for-byte.
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, List

try:
    from bittensor_wallet import Keypair, KeypairType
except ImportError:  # pragma: no cover
    from substrateinterface import Keypair, KeypairType

# V3 exact-match categorical fields (order matters for categorical_key + hashing).
ANALYSIS_FIELDS = ("sentiment", "asset_symbol", "content_type",
                   "technical_quality", "market_analysis", "impact_potential")


def canonical_json(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _round(x) -> float:
    result = round(float(x), 6)
    return 0.0 if result == 0.0 else result  # collapse -0.0 for canonical stability (must match API)


def analysis_to_dict(analysis) -> dict:
    return {f: getattr(analysis, f, None) for f in ANALYSIS_FIELDS}


def categorical_key(analysis) -> str:
    return "|".join("" if getattr(analysis, f, None) is None else str(getattr(analysis, f))
                    for f in ANALYSIS_FIELDS)


def analysis_hash(analysis_dict: dict) -> str:
    return hashlib.sha256(canonical_json(analysis_dict).encode("utf-8")).hexdigest()


def miner_sign_message(resource_id: str, analysis_hash_hex: str, nonce: str) -> str:
    return f"talisman-miner-verdict:{resource_id}:{analysis_hash_hex}:{nonce}"


def sign_miner_item(keypair: "Keypair", resource_id: str, analysis_hash_hex: str, nonce: str) -> str:
    return keypair.sign(miner_sign_message(str(resource_id), analysis_hash_hex, nonce).encode("utf-8")).hex()


def verify_miner_signature(miner_hotkey: str, resource_id: str, analysis_hash_hex: str,
                           nonce: str, signature_hex: str) -> bool:
    try:
        msg = miner_sign_message(str(resource_id), analysis_hash_hex, nonce)
        kp = Keypair(ss58_address=miner_hotkey)
        return bool(kp.verify(msg.encode("utf-8"), bytes.fromhex(signature_hex)))
    except Exception:
        return False


# NOTE: leaves and internal nodes share sha256 with no domain separator. Second-preimage
# resistance relies on leaf content being schema-constrained (all six required fields).
# Do not relax leaf validation.
def _leaf_hash(verdict: dict) -> str:
    payload = canonical_json({
        "resource_type": verdict["resource_type"],
        "resource_id": str(verdict["resource_id"]),
        "miner_hotkey": verdict["miner_hotkey"],
        "validator_verdict": verdict["validator_verdict"],
        "categorical_key": verdict["categorical_key"],
        "points_awarded": _round(verdict["points_awarded"]),
    })
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def merkle_root(verdicts: List[dict]) -> str:
    if not verdicts:
        return hashlib.sha256(b"").hexdigest()
    layer = sorted(_leaf_hash(v) for v in verdicts)
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
        kp = Keypair(ss58_address=pubkey_ss58, crypto_type=KeypairType.ED25519)
        return bool(kp.verify(message.encode("utf-8"), bytes.fromhex(signature_hex)))
    except Exception:
        return False
