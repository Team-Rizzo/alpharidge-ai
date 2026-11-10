"""Verify API-signed points attestations offline against the pinned API key."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from alpharidge_ai.ledger import attestation as ac

MAX_POINTS_PER_UID = 10_000_000.0


@dataclass(frozen=True)
class VerifiedPoints:
    validator: str
    epoch: int
    points: Dict[int, float]
    merkle_root: str


def verify_points(*, attestation: dict, signature: str, validator_hotkey: str,
                  hotkey_to_uid: Dict[str, int], pinned_pubkey: str,
                  epoch: Optional[int] = None) -> Tuple[Optional[VerifiedPoints], str]:
    """Return (points, "accepted") or (None, reason). The pinned pubkey is the root of
    trust; the API key is never taken over the wire."""
    try:
        epoch_i = int(attestation["epoch"])
        per_miner = dict(attestation.get("perMinerPoints") or {})
        total = float(attestation.get("totalPoints") or 0.0)
        merkle_root = str(attestation.get("merkleRoot") or "")
        att_validator = str(attestation.get("validatorHotkey") or "")
    except Exception as e:
        return None, f"malformed_attestation({e})"

    if att_validator != str(validator_hotkey):
        return None, "validator_mismatch"
    if epoch is not None and epoch_i != int(epoch):
        return None, "epoch_mismatch"
    if not pinned_pubkey:
        return None, "no_pinned_pubkey"
    msg = ac.attestation_message(att_validator, epoch_i, per_miner, total, merkle_root)
    if not ac.verify_attestation(pinned_pubkey, msg, signature):
        return None, "bad_signature"

    cleaned: Dict[int, float] = {}
    for hk, pts in per_miner.items():
        uid = hotkey_to_uid.get(hk)
        if uid is None:
            continue
        try:
            fv = float(pts)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(fv):
            continue
        p = round(fv, 6)
        if 0 < p <= MAX_POINTS_PER_UID:
            cleaned[int(uid)] = p
    if not cleaned:
        return None, "empty_payload"

    if not (len(merkle_root) == 64 and all(c in "0123456789abcdef" for c in merkle_root)):
        merkle_root = ""
    return VerifiedPoints(att_validator, epoch_i, cleaned, merkle_root), "accepted"
