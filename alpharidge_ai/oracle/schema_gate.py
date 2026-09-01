"""Schema grace: what an older submission is scored on, and when that ends.

Miners run their own machines, so a schema change takes real calendar time to reach
everyone. Until the cutover block a submission on the previous schema is accepted and
scored on extraction alone, neither rewarded nor punished for a field it cannot send.
After that block the field is required and its absence fails the floor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from alpharidge_ai.models.article_intelligence import (
    SCHEMA_VERSION, SCHEMA_VERSION_PREV)

ACCEPTED_VERSIONS = (SCHEMA_VERSION, SCHEMA_VERSION_PREV)


@dataclass(frozen=True)
class SchemaVerdict:
    accepted: bool
    score_confidence: bool
    reason: str = ""


def evaluate(schema_version: Optional[str], *, block: int,
             cutover_block: int) -> SchemaVerdict:
    """Decide how to treat a submission's schema version at a given block."""
    version = (schema_version or "").strip()

    if version == SCHEMA_VERSION:
        return SchemaVerdict(True, True, "current")

    past_cutover = int(block) >= int(cutover_block)

    if version == SCHEMA_VERSION_PREV:
        if past_cutover:
            return SchemaVerdict(False, False, f"schema_{version}_after_cutover")
        return SchemaVerdict(True, False, "grace")

    return SchemaVerdict(False, False, f"unsupported_schema_{version or 'missing'}")


def confidences(items) -> dict:
    """Per-item stated confidence, keyed by position. Missing values are left out so
    the scorer applies its own neutral value rather than inventing one here."""
    out = {}
    for i, item in enumerate(items or ()):
        value = getattr(item, "confidence", None)
        if value is not None:
            out[i] = float(value)
    return out
