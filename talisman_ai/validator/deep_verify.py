"""Sampled deep-verification of received attestations against raw /verdicts."""
from __future__ import annotations

from typing import List, Optional

from talisman_ai.utils import attestation_crypto as ac


def should_sample(rate: float, rng_value: float) -> bool:
    """rng_value in [0,1); sample when it falls under the rate."""
    return float(rng_value) < float(rate)


def merkle_mismatch(expected_root: str, leaves: List[dict]) -> Optional[str]:
    """Recompute the Merkle root from raw leaves; return a report reason if it
    disagrees with the signed root, else None."""
    if ac.merkle_root(leaves) != expected_root:
        return "attribution_mismatch"
    return None
