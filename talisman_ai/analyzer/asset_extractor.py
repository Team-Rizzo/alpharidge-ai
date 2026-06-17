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

from wordfreq import zipf_frequency


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# An identifier/alias keyword is "ambiguous" when it is a single short alpha
# token that collides with ordinary words (English or foreign) or single
# letters — e.g. "ada" (a name), "uso" (Italian "use"), "gold", lone "V".
# A match whose ONLY evidence is an ambiguous keyword is emitted solely when
# corroborated (cashtag / exact ticker / non-ambiguous name / financial cue).
_AMBIGUOUS_MAX_LEN = 4
# A longer alias that is also an ordinary English word (e.g. "block", "target",
# "apple", "gold") is ambiguous too — length alone misses these. We treat any
# alias whose English word-frequency is at/above this Zipf level as ambiguous,
# so it is only emitted when a financial cue corroborates it. Genuine ticker
# tokens ("aapl"=2.4, "nvda"=2.1) sit well below; common words ("block"=4.9,
# "target"=4.8, "apple"=4.8) sit above. wordfreq is a static table -> deterministic.
_AMBIGUOUS_ZIPF = 3.0
# A *very* common word (e.g. "visa", "cost", "target", "optimism", "block",
# "gold") is an ordinary English dictionary word whose keyword match must NOT be
# salvaged by mere ambient financial language — financial articles are saturated
# with "$", "revenue", "stock", so that cue corroborates these on unrelated text
# (a travel "visa", a "price target", "optimism" the mood). Such a word is only
# emitted with STRONG evidence (cashtag / exact ticker / a distinctive
# non-ambiguous name) or via the NER organization path. Membership is the
# `common_english_words.txt` set (NLTK dictionary ∩ Zipf >= 3.0): asset-only
# names that merely happen to be frequent ("bitcoin", "ethereum", "costco",
# "nvidia") are NOT in it and keep context-based corroboration.
# Case-sensitive identifiers this short (e.g. "V", "ON") are also ambiguous.
_AMBIGUOUS_CS_MAX_LEN = 2
# Window (chars) around an ambiguous match in which we look for a financial cue.
_CONTEXT_WINDOW = 100

# Financial-context cues. "$" and "%" are checked separately (not word chars).
_FIN_CONTEXT = re.compile(
    r"(?i)\b(?:etfs?|funds?|stocks?|shares?|ticker|nasdaq|nyse|cboe|futures|"
    r"options?|tokens?|coins?|crypto|cryptocurrency|blockchain|staking|"
    r"trading|traded|traders?|investors?|rally|rallied|rallies|surged?|"
    r"surges|plunged?|plunges|tumbled?|gains?|gained|selloff|sell-off|"
    r"bullish|bearish|yields?|bonds?|index|indices|markets?|exchange|"
    r"earnings|revenue|prices?|spot|bullion|equit(?:y|ies)|valuations?|"
    r"market\s+cap|safe\s+haven)\b"
)


def _has_fin_context(window: str) -> bool:
    return ("$" in window) or ("%" in window) or bool(_FIN_CONTEXT.search(window))


def _is_ambiguous_word(token: str) -> bool:
    """Ambiguous when a single alpha token that collides with ordinary words:
    either short (<= _AMBIGUOUS_MAX_LEN chars) or a common English word
    (Zipf frequency >= _AMBIGUOUS_ZIPF, e.g. "block"/"target"/"apple")."""
    t = token.strip()
    if not t.isalpha():
        return False
    if len(t) <= _AMBIGUOUS_MAX_LEN:
        return True
    return zipf_frequency(t.lower(), "en") >= _AMBIGUOUS_ZIPF


def _load_wordset(filename: str) -> frozenset:
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return frozenset()
    with open(path) as f:
        return frozenset(line.strip() for line in f if line.strip())


# Ordinary common English words that collide with tickers/aliases (see note above).
_COMMON_WORDS = _load_wordset("common_english_words.txt")


def _is_very_common_word(token: str) -> bool:
    """A keyword so common that nearby financial language is NOT evidence it is
    the asset: a single letter, or an ordinary English dictionary word. Asset-only
    names ("bitcoin", "costco") and multi-word/distinctive aliases ("elon musk")
    are not in the common-word set and never count as very common."""
    t = token.strip().lower()
    if not t.isalpha():
        return False
    if len(t) <= 1:
        return True
    return t in _COMMON_WORDS


# Uppercase tokens that are valid tickers but collide with common non-financial
# acronyms; treated as ambiguous AND non-corroborating so they need a cashtag,
# a distinctive name, or another evidence span to be emitted.
_ACRONYM_BLOCKLIST = frozenset({
    "LEO", "SUI", "ICP", "AI", "IT", "ON", "OP", "ATOM", "GAS",
    "USD", "EUR", "GBP", "JPY", "CNY",  # ISO currency codes seen as tickers
    "GOLD", "ALL", "CAR",               # common-word tickers needing corroboration
})


def _is_noncorroborating(token: str) -> bool:
    """Evidence too generic to rescue an asset on its own.

    Returns True when the token is either a very-common English word or a
    blocklisted acronym that collides with non-financial usage. Such a token
    cannot self-rescue an ambiguous asset match even when ambient financial
    language is present — the asset needs a cashtag, an exact ticker with
    non-blocklisted CS identifier, or a distinctive (non-common) name.
    """
    return _is_very_common_word(token) or token.strip() in _ACRONYM_BLOCKLIST


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
    # Corroboration bookkeeping (not part of the public payload).
    strong_evidence: bool = field(default=False, repr=False)
    context_corroborated: bool = field(default=False, repr=False)


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
        self._case_sensitive_index: Dict[str, Tuple[int, bool]] = {}
        # (pattern, asset_id, raw_keyword, ambiguous)
        self._identifier_patterns: List[Tuple[re.Pattern, int, str, bool]] = []
        self._alias_patterns: List[Tuple[re.Pattern, int, str, bool]] = []

        for aid, data in self.assets.items():
            for tag in data.get("cashtags", []):
                self._cashtag_index[tag.lower()] = aid

            for cs_id in data.get("case_sensitive_identifiers", []):
                ambiguous = (len(cs_id) <= _AMBIGUOUS_CS_MAX_LEN
                             or _is_very_common_word(cs_id)
                             or cs_id in _ACRONYM_BLOCKLIST)
                self._case_sensitive_index[cs_id] = (aid, ambiguous)

            for uid in data.get("unique_identifiers", []):
                uid_lower = uid.lower()
                if len(uid_lower) < 3:
                    continue
                try:
                    pattern = re.compile(rf"\b{re.escape(uid_lower)}\b")
                    self._identifier_patterns.append(
                        (pattern, aid, uid, _is_ambiguous_word(uid_lower)))
                except re.error:
                    pass

            for alias in data.get("aliases", []):
                alias_lower = alias.lower()
                if len(alias_lower) < 4:
                    continue
                try:
                    pattern = re.compile(rf"\b{re.escape(alias_lower)}\b")
                    self._alias_patterns.append(
                        (pattern, aid, alias, _is_ambiguous_word(alias_lower)))
                except re.error:
                    pass

    def extract_assets(
        self,
        title: str,
        body: str,
        max_assets: int = 20,
        language: str = "en",
    ) -> List[AssetMatch]:
        """Extract all matching assets from article text.

        Args:
            title: Article headline.
            body: Article body text.
            max_assets: Maximum number of assets to return.
            language: Detected language code (e.g. "en", "it"). Matches whose
                only evidence is an ambiguous dictionary-word/foreign-word/
                single-letter keyword are dropped unless corroborated by a
                cashtag, an exact case-sensitive ticker, a non-ambiguous name,
                or a financial cue near the mention. This is what stops
                "uso"->USO on Italian text and "Ada"->ADA on a soap opera.

        Returns:
            List of AssetMatch objects sorted by relevance_score descending.
        """
        full_text = f"{title}\n{body}"
        text_lower = full_text.lower()
        title_lower = title.lower()

        def _ctx(start: int, end: int) -> bool:
            return _has_fin_context(
                text_lower[max(0, start - _CONTEXT_WINDOW):end + _CONTEXT_WINDOW])

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

        # Phase 1: Cashtag matching (highest confidence, unambiguous by design)
        for tag_lower, aid in self._cashtag_index.items():
            if tag_lower in text_lower:
                m = _get_or_create(aid)
                m.evidence_spans.append(tag_lower)
                m.relevance_score += 3.0
                m.disambiguation_method = "cashtag"
                m.disambiguation_confidence = 1.0
                m.strong_evidence = True
                if tag_lower in title_lower:
                    m.relevance_score += 3.0

        # Phase 2: Case-sensitive identifiers (exact-case tickers)
        for cs_id, (aid, ambiguous) in self._case_sensitive_index.items():
            pattern = re.compile(rf"\b{re.escape(cs_id)}\b")
            hit = pattern.search(full_text)
            if hit:
                m = _get_or_create(aid)
                if cs_id not in m.evidence_spans:
                    m.evidence_spans.append(cs_id)
                m.relevance_score += 2.0
                if m.disambiguation_method == "none":
                    m.disambiguation_method = "keyword_high"
                    m.disambiguation_confidence = 0.95
                if pattern.search(title):
                    m.relevance_score += 2.0
                if not ambiguous:
                    m.strong_evidence = True
                elif _ctx(hit.start(), hit.end()):
                    m.context_corroborated = True

        # Phase 3: Unique identifiers (case-insensitive, word boundary)
        for pattern, aid, raw_id, ambiguous in self._identifier_patterns:
            hits = list(pattern.finditer(text_lower))
            if hits:
                m = _get_or_create(aid)
                if raw_id not in m.evidence_spans:
                    m.evidence_spans.append(raw_id)
                m.relevance_score += 1.0 + 0.3 * (len(hits) - 1)
                if m.disambiguation_method == "none":
                    m.disambiguation_method = "keyword_high"
                    m.disambiguation_confidence = 0.9
                if pattern.search(title_lower):
                    m.relevance_score += 2.0
                if not ambiguous:
                    m.strong_evidence = True
                elif any(_ctx(h.start(), h.end()) for h in hits):
                    m.context_corroborated = True

        # Phase 4: Aliases (lowest confidence)
        for pattern, aid, raw_alias, ambiguous in self._alias_patterns:
            hits = list(pattern.finditer(text_lower))
            if hits:
                m = _get_or_create(aid)
                if raw_alias not in m.evidence_spans:
                    m.evidence_spans.append(raw_alias)
                m.relevance_score += 0.5
                if m.disambiguation_method == "none":
                    m.disambiguation_method = "keyword_contextual"
                    m.disambiguation_confidence = 0.7
                if not ambiguous:
                    m.strong_evidence = True
                elif any(_ctx(h.start(), h.end()) for h in hits):
                    m.context_corroborated = True

        # Corroboration gate. Keep an asset when:
        #   * it has STRONG evidence (cashtag / exact ticker / distinctive
        #     non-ambiguous name), OR
        #   * it was context-corroborated AND at least one piece of its evidence
        #     is NOT a very-common word and NOT a blocklisted acronym.
        #     Ambient financial language alone never rescues an asset whose only
        #     evidence is a very-common word or a blocklisted acronym — this
        #     is what kills "visa"->V on a travel story, "cost"->COST,
        #     "target"->TGT ("price target"), "optimism"->OP, "leo"->LEO on
        #     space news. Real single-word companies still arrive via the NER
        #     organization path.
        def _kept(m: AssetMatch) -> bool:
            if m.strong_evidence:
                return True
            if not m.context_corroborated:
                return False
            return any(not _is_noncorroborating(ev) for ev in m.evidence_spans)

        kept = {aid: m for aid, m in matches.items() if _kept(m)}

        # Determine primary subjects
        if kept:
            max_score = max(m.relevance_score for m in kept.values())
            for m in kept.values():
                if m.relevance_score >= max_score * 0.8 and m.relevance_score >= 3.0:
                    m.is_primary_subject = True

        # Sort by relevance descending, then by ticker for stability
        result = sorted(
            kept.values(),
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
