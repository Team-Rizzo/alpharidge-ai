"""Miner-side article triage (schema v3) — the cheap pass that runs on every
dispatched article before any deep analysis.

Reference implementation, deterministic and LLM-free:
  R1: the keyword gazetteer resolves >=1 tradeable asset  -> relevant
  R2: macro/economic-event language tied to a named economy -> relevant
  macro language without a named economy                     -> borderline
  otherwise                                                  -> irrelevant

Deliberately conservative on the irrelevant side: the profitable failure mode
under audit is a false negative, so anything with a plausible market hook is
kept or marked borderline. Miners are free to substitute a smarter classifier;
this one clears the deterministic audits by construction (it can never label a
gazetteer-positive article irrelevant).
"""
from __future__ import annotations

import re
from typing import Optional

from alpharidge_ai.triage import (
    LABEL_BORDERLINE,
    LABEL_IRRELEVANT,
    LABEL_RELEVANT,
    build_proof_of_read,
    build_triage_record,
    gazetteer_assets,
)

# Macro/economic-event cues (R2). Kept compact: these mark *event* language,
# not any mention of money.
_MACRO_TERMS = re.compile(
    r"\b(central bank|interest rate|rate (?:cut|hike|decision)|monetary policy|"
    r"inflation|cpi|ppi|gdp|unemployment|jobless|nonfarm|payrolls|recession|"
    r"stimulus|quantitative easing|bond yield|sovereign debt|fiscal (?:policy|deficit)|"
    r"tariff|trade (?:war|deal|agreement|deficit)|sanction|embargo|export ban|"
    r"currency devaluation|exchange rate|imf|world bank|opec|"
    r"supply chain|oil (?:price|output|production|refinery|depot)|gas pipeline|"
    r"crude|lng|grain export|energy crisis|price cap)\b",
    re.IGNORECASE,
)

# Named economies / blocs / central banks (R2 requires one). Split by case
# sensitivity: short abbreviations must stay case-sensitive or they collide
# with ordinary words ("us", "eu" in Romance languages, "fed up").
_ECONOMY_NAMES = re.compile(
    r"\b(united states|eurozone|euro area|european union|"
    r"china|chinese|japan|japanese|germany|german|france|french|britain|british|"
    r"india|indian|russia|russian|brazil|canada|canadian|australia|mexico|"
    r"south korea|korean|turkey|turkish|saudi|iran|argentina|indonesia|"
    r"federal reserve|bank of england|bank of japan|bundesbank|"
    r"renminbi|pound sterling)\b",
    re.IGNORECASE,
)
_ECONOMY_ABBREVS = re.compile(
    r"\b(U\.?S\.?A?|EU|UK|IMF|ECB|Fed|BOJ|PBOC|OPEC|Treasury)\b|"
    r"\b(dollar|euro|yen|yuan|ruble|rupee)s?\b")


class TriageStage:
    """Triage one article. `asset_extractor` is the shared keyword gazetteer
    (alpharidge_ai.analyzer.asset_extractor.AssetExtractor)."""

    def __init__(self, asset_extractor):
        self._assets = asset_extractor

    def evaluate(self, title: str, content: str) -> tuple[dict, dict, list]:
        """Returns (triage_record, proof_of_read, asset_matches)."""
        title = title or ""
        content = content or ""
        proof = build_proof_of_read(title, content)

        matches = gazetteer_assets(self._assets, title, content)
        if matches:
            return build_triage_record(LABEL_RELEVANT, confidence=0.95), proof, matches

        text = f"{title}\n{content[:4000]}"
        macro = bool(_MACRO_TERMS.search(text))
        economy = bool(_ECONOMY_NAMES.search(text)) or bool(_ECONOMY_ABBREVS.search(text))
        if macro and economy:
            return build_triage_record(LABEL_RELEVANT, confidence=0.7), proof, []
        if macro:
            # Macro-event language with no named economy: genuinely ambiguous.
            return build_triage_record(LABEL_BORDERLINE, confidence=0.5), proof, []
        # A bare country/currency mention with no macro-event language is how
        # most world news reads; it is not market relevance.
        return (build_triage_record(LABEL_IRRELEVANT, "non_economic", confidence=0.8),
                proof, [])
