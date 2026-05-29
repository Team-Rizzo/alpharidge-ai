"""
Multi-asset keyword extraction from article text.

Loads both crypto (assets_expanded.json) and traditional finance
(assets_traditional.json) registries and extracts ALL matching assets.
Returns a list of AssetMatch objects sorted by relevance.

This is the deterministic layer — no LLM calls. The LLM layer adds
per-asset sentiment on top of these matches.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


@dataclass
class AssetMatch:
    """A single asset detected in article text via keyword matching."""
    asset_id: int
    ticker: str
    asset_name: str
    asset_class: str
    coingecko_id: Optional[str] = None
    yahoo_ticker: Optional[str] = None
    relevance_score: float = 0.0
    is_primary_subject: bool = False
    evidence_spans: List[str] = field(default_factory=list)
    disambiguation_method: str = "none"
    disambiguation_confidence: float = 1.0


def _load_json(filename: str) -> list:
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)


class AssetExtractor:
    """Extracts all matching assets from article text using keyword matching.

    Matching priority per asset:
    1. Cashtag ($BTC) — highest confidence, unambiguous
    2. Case-sensitive identifiers (SOL, NEAR) — must match exact case
    3. Unique identifiers — case-insensitive word-boundary match
    4. Aliases — case-insensitive word-boundary match, lower confidence
    """

    def __init__(self):
        crypto = _load_json("assets_expanded.json")
        traditional = _load_json("assets_traditional.json")
        self.assets: Dict[int, dict] = {}
        for asset in crypto + traditional:
            self.assets[asset["id"]] = asset

        self._cashtag_index: Dict[str, int] = {}
        self._case_sensitive_index: Dict[str, int] = {}
        self._identifier_patterns: List[Tuple[re.Pattern, int, str]] = []
        self._alias_patterns: List[Tuple[re.Pattern, int, str]] = []

        for aid, data in self.assets.items():
            for tag in data.get("cashtags", []):
                self._cashtag_index[tag.lower()] = aid

            for cs_id in data.get("case_sensitive_identifiers", []):
                self._case_sensitive_index[cs_id] = aid

            for uid in data.get("unique_identifiers", []):
                uid_lower = uid.lower()
                if len(uid_lower) < 3:
                    continue
                try:
                    pattern = re.compile(rf"\b{re.escape(uid_lower)}\b")
                    self._identifier_patterns.append((pattern, aid, uid))
                except re.error:
                    pass

            for alias in data.get("aliases", []):
                alias_lower = alias.lower()
                if len(alias_lower) < 4:
                    continue
                try:
                    pattern = re.compile(rf"\b{re.escape(alias_lower)}\b")
                    self._alias_patterns.append((pattern, aid, alias))
                except re.error:
                    pass

    def extract_assets(
        self,
        title: str,
        body: str,
        max_assets: int = 20,
    ) -> List[AssetMatch]:
        """Extract all matching assets from article text.

        Args:
            title: Article headline.
            body: Article body text.
            max_assets: Maximum number of assets to return.

        Returns:
            List of AssetMatch objects sorted by relevance_score descending.
        """
        full_text = f"{title}\n{body}"
        text_lower = full_text.lower()
        title_lower = title.lower()

        # Track matches per asset: {asset_id: AssetMatch}
        matches: Dict[int, AssetMatch] = {}

        def _get_or_create(aid: int) -> AssetMatch:
            if aid not in matches:
                data = self.assets[aid]
                matches[aid] = AssetMatch(
                    asset_id=aid,
                    ticker=data["symbol"],
                    asset_name=data["name"],
                    asset_class=data.get("asset_class", "unknown"),
                    coingecko_id=data.get("coingecko_id"),
                    yahoo_ticker=data.get("yahoo_ticker"),
                )
            return matches[aid]

        # Phase 1: Cashtag matching (highest confidence)
        for tag_lower, aid in self._cashtag_index.items():
            if tag_lower in text_lower:
                m = _get_or_create(aid)
                m.evidence_spans.append(tag_lower)
                m.relevance_score += 3.0
                m.disambiguation_method = "cashtag"
                m.disambiguation_confidence = 1.0
                if tag_lower in title_lower:
                    m.relevance_score += 3.0

        # Phase 2: Case-sensitive identifiers
        for cs_id, aid in self._case_sensitive_index.items():
            pattern = re.compile(rf"\b{re.escape(cs_id)}\b")
            found_in_body = pattern.search(full_text)
            if found_in_body:
                m = _get_or_create(aid)
                if cs_id not in m.evidence_spans:
                    m.evidence_spans.append(cs_id)
                m.relevance_score += 2.0
                if m.disambiguation_method == "none":
                    m.disambiguation_method = "keyword_high"
                    m.disambiguation_confidence = 0.95
                if pattern.search(title):
                    m.relevance_score += 2.0

        # Phase 3: Unique identifiers (case-insensitive, word boundary)
        for pattern, aid, raw_id in self._identifier_patterns:
            all_matches = pattern.findall(text_lower)
            if all_matches:
                m = _get_or_create(aid)
                if raw_id not in m.evidence_spans:
                    m.evidence_spans.append(raw_id)
                m.relevance_score += 1.0 + 0.3 * (len(all_matches) - 1)
                if m.disambiguation_method == "none":
                    m.disambiguation_method = "keyword_high"
                    m.disambiguation_confidence = 0.9
                if pattern.search(title_lower):
                    m.relevance_score += 2.0

        # Phase 4: Aliases (lowest confidence)
        for pattern, aid, raw_alias in self._alias_patterns:
            if pattern.search(text_lower):
                m = _get_or_create(aid)
                if raw_alias not in m.evidence_spans:
                    m.evidence_spans.append(raw_alias)
                m.relevance_score += 0.5
                if m.disambiguation_method == "none":
                    m.disambiguation_method = "keyword_contextual"
                    m.disambiguation_confidence = 0.7

        # Determine primary subjects
        if matches:
            max_score = max(m.relevance_score for m in matches.values())
            for m in matches.values():
                if m.relevance_score >= max_score * 0.8 and m.relevance_score >= 3.0:
                    m.is_primary_subject = True

        # Sort by relevance descending, then by ticker for stability
        result = sorted(
            matches.values(),
            key=lambda m: (-m.relevance_score, m.ticker),
        )

        return result[:max_assets]

    def extract_sectors(self, title: str, body: str) -> List[dict]:
        """Extract matching sectors from text (backward-compatible with identify_sector_from_text).

        Returns list of {id, symbol, confidence, evidence} dicts for all matching sectors,
        not just the top one.
        """
        # Load sectors
        sectors_path = os.path.join(DATA_DIR, "sectors.json")
        if not os.path.exists(sectors_path):
            return [{"id": 9, "symbol": "OTHER", "confidence": "low", "evidence": []}]

        with open(sectors_path, "r") as f:
            sectors_list = json.load(f)

        text_lower = f"{title}\n{body}".lower()
        results = []

        for sector in sectors_list:
            sid = sector["id"]
            if sid == 9:
                continue
            evidence = []

            for tag in sector.get("cashtags", []):
                if tag.lower() in text_lower:
                    evidence.append(tag)

            for uid in sector.get("unique_identifiers", []):
                uid_lower = uid.lower()
                if len(uid_lower) < 3:
                    continue
                if re.search(rf"\b{re.escape(uid_lower)}\b", text_lower):
                    evidence.append(uid)

            if evidence:
                confidence = "high" if len(evidence) > 2 else "medium" if len(evidence) > 1 else "low"
                results.append({
                    "id": sid,
                    "symbol": sector["symbol"],
                    "confidence": confidence,
                    "evidence": evidence,
                })

        results.sort(key=lambda x: len(x["evidence"]), reverse=True)

        if not results:
            return [{"id": 9, "symbol": "OTHER", "confidence": "low", "evidence": []}]

        return results
