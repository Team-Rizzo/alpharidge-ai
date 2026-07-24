"""Shared triage schema for miner-side article relevance filtering (schema v3).

Miners triage every dispatched article against the relevance rubric and only
deep-analyze articles they label relevant. Validators grade the triage claims
by audit (see alpharidge_ai/validator/triage_grader.py) instead of producing a
reference analysis for every article.

Rubric (an article is RELEVANT iff either branch holds):
  R1 — concerns >=1 resolvable tradeable asset (equity/fx/crypto/commodity/
       rates/sovereign). Keyword/gazetteer-resolved assets make this branch
       deterministic on both sides.
  R2 — concerns a macro/economic event tied to a named economy or bloc:
       monetary policy, inflation/employment/GDP releases, fiscal/trade/
       sanctions policy, or supply shocks with plausible market transmission.

Wire format: additive keys inside analysis_data (backward compatible — v2
miners simply omit them):
  analysis_data["triage"]        = {"label", "reason_code", "confidence"}
  analysis_data["proof_of_read"] = {"content_hash", "word_count"}

proof_of_read is required for EVERY article in the batch, including ones
labeled irrelevant (which carry no other analysis). Both fields are exactly
recomputable by the validator from its own copy of the article, so a triage
claim is auditable even when no analysis accompanies it.
"""
from __future__ import annotations

from typing import Optional, Tuple

from alpharidge_ai.analyzer.text_stats import compute_text_stats
from alpharidge_ai.models.article_intelligence import ArticleIntelligence

TRIAGE_SCHEMA_VERSION = 3

LABEL_RELEVANT = "relevant"
LABEL_IRRELEVANT = "irrelevant"
LABEL_BORDERLINE = "borderline"
LABELS = frozenset({LABEL_RELEVANT, LABEL_IRRELEVANT, LABEL_BORDERLINE})

REASON_CODES = frozenset({
    "non_economic",
    "local_no_market_impact",
    "promo_spam",
    "other",
})

# Deterministic R1 evidence: asset entries resolved by the keyword gazetteer.
DETERMINISTIC_ASSET_SOURCES = frozenset({"keyword"})


def build_proof_of_read(title: str, body: str) -> dict:
    """Compute the proof-of-read record for an article.

    Uses the same primitives Tier-2 validation already exact-matches on, so a
    correct proof is only producible from the actual article text.
    """
    stats = compute_text_stats(title or "", body or "")
    return {
        "content_hash": ArticleIntelligence.compute_content_hash(title or "", body or ""),
        "word_count": int(stats.word_count),
    }


def verify_proof_of_read(proof: Optional[dict], title: str, body: str) -> bool:
    if not isinstance(proof, dict):
        return False
    expected = build_proof_of_read(title, body)
    try:
        return (
            str(proof.get("content_hash")) == expected["content_hash"]
            and int(proof.get("word_count")) == expected["word_count"]
        )
    except (TypeError, ValueError):
        return False


def build_triage_record(label: str, reason_code: Optional[str] = None,
                        confidence: float = 1.0) -> dict:
    if label not in LABELS:
        raise ValueError(f"invalid triage label: {label}")
    if label == LABEL_IRRELEVANT and reason_code not in REASON_CODES:
        raise ValueError(f"irrelevant label requires a reason_code, got: {reason_code}")
    return {
        "label": label,
        "reason_code": reason_code,
        "confidence": max(0.0, min(1.0, float(confidence))),
    }


def extract_triage(analysis_data: Optional[dict]) -> Tuple[Optional[dict], Optional[str]]:
    """Pull and validate the triage record from an analysis_data dict.

    Returns (record, None) when present and well-formed, (None, None) when the
    miner is pre-v3 (no triage key — grace path), and (None, error) when a
    record is present but malformed (graded as an invalid claim, not silently
    forgiven, so malformation is never an escape hatch).
    """
    if not isinstance(analysis_data, dict) or "triage" not in analysis_data:
        return None, None
    rec = analysis_data.get("triage")
    if not isinstance(rec, dict):
        return None, "triage_not_object"
    label = rec.get("label")
    if label not in LABELS:
        return None, "triage_bad_label"
    if label == LABEL_IRRELEVANT and rec.get("reason_code") not in REASON_CODES:
        return None, "triage_missing_reason"
    return rec, None


def deterministic_relevant(assets: Optional[list]) -> bool:
    """R1 deterministic branch: any gazetteer/keyword-resolved asset present.

    `assets` is the extractor output (list of dicts with `resolved_via`).
    """
    for a in assets or []:
        if isinstance(a, dict) and a.get("resolved_via") in DETERMINISTIC_ASSET_SOURCES:
            return True
    return False
