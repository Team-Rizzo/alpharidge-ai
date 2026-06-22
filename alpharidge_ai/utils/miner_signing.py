"""Miner-side helper: sign each item's analysis with the miner hotkey."""
from __future__ import annotations

import secrets
from typing import Dict, List, Tuple

from alpharidge_ai.utils import attestation_crypto as ac


def sign_items(keypair, items: List, id_attr: str = "id") -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return (signatures, nonces) keyed by str(resource_id). Items must have an
    `.analysis` attribute carrying the V3 categorical fields. Items without analysis
    are skipped (uncreditable)."""
    sigs: Dict[str, str] = {}
    nonces: Dict[str, str] = {}
    for item in items:
        analysis = getattr(item, "analysis", None)
        if analysis is None:
            continue
        rid = str(getattr(item, id_attr))
        nonce = secrets.token_hex(16)
        ah = ac.analysis_hash(ac.analysis_to_dict(analysis))
        sigs[rid] = ac.sign_miner_item(keypair, rid, ah, nonce)
        nonces[rid] = nonce
    return sigs, nonces
