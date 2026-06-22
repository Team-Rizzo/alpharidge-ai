"""Build the per-item verdict fields the validator sends to the API."""
from __future__ import annotations

from alpharidge_ai.utils import attestation_crypto as ac


def build_verdict_fields(*, miner_hotkey: str, miner_signature: str, nonce: str,
                         analysis, validator_verdict: str, points_awarded: float,
                         epoch: int) -> dict:
    """Compute the verdict fields for one completed item. An 'invalid' verdict
    contributes zero points (and so zero budget at the API)."""
    pts = float(points_awarded) if validator_verdict == "valid" else 0.0
    return {
        "miner_hotkey": miner_hotkey,
        "miner_signature": miner_signature,
        "nonce": nonce,
        "miner_analysis_hash": ac.analysis_hash(ac.analysis_to_dict(analysis)),
        "validator_verdict": validator_verdict,
        "categorical_key": ac.categorical_key(analysis),
        "points_awarded": pts,
        "epoch": int(epoch),
    }


def collect_verdict_meta(items, miner_signatures, nonces, validator_verdict, epoch,
                         id_attr: str = "id") -> dict:
    """Build {str(resource_id): {miner_signature, nonce, validator_verdict, epoch}} for
    items that carry a miner signature+nonce. Items lacking either are skipped (legacy
    miners / unsigned → not creditable)."""
    out = {}
    sigs = miner_signatures or {}
    ncs = nonces or {}
    for item in items:
        rid = str(getattr(item, id_attr))
        sig = sigs.get(rid)
        nonce = ncs.get(rid)
        if sig and nonce:
            out[rid] = {"miner_signature": sig, "nonce": nonce,
                        "validator_verdict": validator_verdict, "epoch": int(epoch)}
    return out
