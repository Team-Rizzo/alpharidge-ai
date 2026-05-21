"""
News Article Classification for Sector Relevance

Analyzes news articles to determine sector relevance and classify content.
Uses keyword-based sector matching (similar to asset matching for tweets).

Key Features:
- Sector-based taxonomy (Crypto, Monetary Policy, Macro, etc.)
- Keyword-based sector identification (cashtags, sector identifiers)
- Article-specific content type enum (breaking_news, analysis, opinion, etc.)
- Atomic LLM tool calls for each classification dimension
"""

import json
import re
import threading
from openai import OpenAI
from typing import Dict, List, Optional
from dataclasses import dataclass
import bittensor as bt

from .classifications import ArticleContentType, Sentiment, TechnicalQuality, MarketAnalysis, ImpactPotential
from .llm_cache import LLMCache

try:
    from talisman_ai import config
except ImportError:
    config = None


@dataclass
class ArticleClassification:
    """Classification result for a news article"""
    sector_id: int
    sector_symbol: str
    content_type: ArticleContentType
    sentiment: Sentiment
    technical_quality: TechnicalQuality
    market_analysis: MarketAnalysis
    impact_potential: ImpactPotential
    relevance_confidence: str
    evidence_spans: List[str]

    def to_canonical_string(self) -> str:
        sorted_evidence = "|".join(sorted([s.lower() for s in self.evidence_spans]))
        return f"{self.sector_id}|{self.content_type.value}|{self.sentiment.value}|{self.technical_quality.value}|{self.market_analysis.value}|{self.impact_potential.value}|{self.relevance_confidence}|{sorted_evidence}"

    def to_dict(self) -> dict:
        return {
            "sector_id": self.sector_id,
            "sector_symbol": self.sector_symbol,
            "content_type": self.content_type.value,
            "sentiment": self.sentiment.value,
            "technical_quality": self.technical_quality.value,
            "market_analysis": self.market_analysis.value,
            "impact_potential": self.impact_potential.value,
            "relevance_confidence": self.relevance_confidence,
            "evidence_spans": self.evidence_spans,
        }


ARTICLE_CONTENT_TYPE_TOOL = {
    "type": "function",
    "function": {
        "name": "classify_article_content_type",
        "description": "Classify the type of news article content",
        "parameters": {
            "type": "object",
            "properties": {
                "content_type": {
                    "type": "string",
                    "enum": [ct.value for ct in ArticleContentType],
                    "description": "Primary article content type"
                }
            },
            "required": ["content_type"]
        }
    }
}

SENTIMENT_TOOL = {
    "type": "function",
    "function": {
        "name": "classify_sentiment",
        "description": "Classify market sentiment",
        "parameters": {
            "type": "object",
            "properties": {
                "sentiment": {
                    "type": "string",
                    "enum": [s.value for s in Sentiment],
                    "description": "Market sentiment"
                }
            },
            "required": ["sentiment"]
        }
    }
}

TECHNICAL_QUALITY_TOOL = {
    "type": "function",
    "function": {
        "name": "assess_technical_quality",
        "description": "Assess technical content quality",
        "parameters": {
            "type": "object",
            "properties": {
                "quality": {
                    "type": "string",
                    "enum": [tq.value for tq in TechnicalQuality],
                    "description": "Technical quality level"
                }
            },
            "required": ["quality"]
        }
    }
}

MARKET_ANALYSIS_TOOL = {
    "type": "function",
    "function": {
        "name": "classify_market_analysis",
        "description": "Classify market analysis type",
        "parameters": {
            "type": "object",
            "properties": {
                "analysis_type": {
                    "type": "string",
                    "enum": [ma.value for ma in MarketAnalysis],
                    "description": "Type of market analysis"
                }
            },
            "required": ["analysis_type"]
        }
    }
}

IMPACT_TOOL = {
    "type": "function",
    "function": {
        "name": "assess_impact",
        "description": "Assess potential impact",
        "parameters": {
            "type": "object",
            "properties": {
                "impact": {
                    "type": "string",
                    "enum": [ip.value for ip in ImpactPotential],
                    "description": "Expected impact level"
                }
            },
            "required": ["impact"]
        }
    }
}

MAX_CONTENT_CHARS = 3000


class NewsRelevanceAnalyzer:
    """
    News article classifier using sector-based taxonomy and atomic tool calls.

    Each classification dimension is decided independently via its own tool call.
    Sector identification uses keyword matching (no LLM).
    """

    def __init__(self, model: str = None, api_key: str = None, llm_base: str = None, sectors: List[Dict] = None):
        self.sector_registry = {}

        if config:
            self.model = model or config.MODEL
            self.api_key = api_key or config.API_KEY
            self.llm_base = llm_base or config.LLM_BASE
        else:
            self.model = model
            self.api_key = api_key
            self.llm_base = llm_base

        if not self.api_key:
            raise ValueError("API_KEY environment variable is required")

        self.client = OpenAI(base_url=self.llm_base, api_key=self.api_key)

        cache_ttl = float(getattr(config, "LLM_CACHE_TTL", 300)) if config else 300.0
        cache_size = int(getattr(config, "LLM_CACHE_MAX_SIZE", 1024)) if config else 1024
        self._cache = LLMCache(max_size=cache_size, ttl_seconds=cache_ttl)
        self._tl = threading.local()
        bt.logging.info(f"[NEWS_ANALYZER] LLM cache enabled: max_size={cache_size}, ttl={cache_ttl}s")

        if sectors:
            self.sectors = {s["id"]: s for s in sectors}
            for s in sectors:
                self.sector_registry[s["id"]] = s
        else:
            self.sectors = {}

        self.sectors[9] = {
            "id": 9,
            "symbol": "OTHER",
            "name": "Other",
            "unique_identifiers": []
        }

        bt.logging.info(f"[NEWS_ANALYZER] Initialized with model: {self.model}")
        if sectors:
            bt.logging.info(f"[NEWS_ANALYZER] Registered {len(self.sectors)} sectors")

    def identify_sector_from_text(self, text: str) -> Dict:
        """
        Identify sector from text using keyword matching.

        Priority:
        1. Cashtag match → sector 1 (Crypto)
        2. Word-boundary match on unique_identifiers
        3. No match → sector 9 (Other)
        """
        text_lower = text.lower()

        matches = []
        for sid, data in self.sectors.items():
            if sid == 9:
                continue

            symbol = data.get("symbol", "")
            evidence = []

            for tag in data.get("cashtags", []):
                if tag.lower() in text_lower:
                    evidence.append(tag)

            if evidence:
                matches.append((sid, symbol, 1.0, evidence))
                continue

            for identifier in data.get("unique_identifiers", []):
                id_lower = identifier.lower()
                if len(id_lower) < 3:
                    continue
                if re.search(rf'\b{re.escape(id_lower)}\b', text_lower):
                    evidence.append(identifier)

            if evidence:
                confidence_score = 0.9 if len(evidence) > 1 else 0.8
                matches.append((sid, symbol, confidence_score, evidence))

        matches.sort(key=lambda x: x[2], reverse=True)

        if matches:
            top = matches[0]
            confidence = "high" if top[2] >= 0.9 else "medium"
            return {
                "id": top[0],
                "symbol": top[1],
                "confidence": confidence,
                "evidence": top[3],
            }

        return {"id": 9, "symbol": "OTHER", "confidence": "low", "evidence": []}

    def _build_article_text(self, title: str, summary: Optional[str], content: Optional[str]) -> str:
        parts = [title]
        if summary:
            parts.append(summary)
        if content:
            parts.append(content[:MAX_CONTENT_CHARS])
        return "\n\n".join(parts)

    def classify_article(self, title: str, summary: Optional[str] = None, content: Optional[str] = None) -> Optional[ArticleClassification]:
        """
        Classify a news article using sector matching + atomic LLM tool calls.

        Args:
            title: Article headline
            summary: RSS description/summary
            content: Full article body (truncated internally)

        Returns:
            ArticleClassification if successful, None if parsing fails
        """
        article_text = self._build_article_text(title, summary, content)

        cached = self._cache.get(article_text)
        if cached is not None:
            return cached

        try:
            self._tl.had_llm_error = False

            sector_result = self.identify_sector_from_text(article_text)

            content_type = self._classify_content_type(article_text)
            sentiment = self._classify_sentiment(article_text)
            technical_quality = self._assess_technical_quality(article_text)
            market_analysis = self._classify_market_analysis(article_text)
            impact = self._assess_impact(article_text)

            result = ArticleClassification(
                sector_id=sector_result["id"],
                sector_symbol=sector_result["symbol"],
                content_type=ArticleContentType(content_type),
                sentiment=Sentiment(sentiment),
                technical_quality=TechnicalQuality(technical_quality),
                market_analysis=MarketAnalysis(market_analysis),
                impact_potential=ImpactPotential(impact),
                relevance_confidence=sector_result["confidence"],
                evidence_spans=sector_result["evidence"],
            )
            if not getattr(self._tl, "had_llm_error", False):
                self._cache.put(article_text, result)
            return result

        except Exception as e:
            bt.logging.error(f"[NEWS_ANALYZER] Classification error: {e}")
            return None

    def _classify_content_type(self, text: str) -> str:
        prompt = f"""Classify the content type of this news article:

"{text}"

Pick the MOST SPECIFIC category:
- breaking_news: time-sensitive, just-happened events, urgent reports
- analysis: in-depth analysis, data-driven commentary, market analysis
- opinion: editorial, op-ed, personal views, columnist takes
- earnings: quarterly results, financial reports, revenue announcements
- regulatory: SEC filings, legislation, policy changes, compliance
- research: studies, white papers, academic findings, scientific reports
- press_release: official company/organization announcements
- interview: Q&A, interviews, profiles, executive quotes
- other: doesn't fit any category above"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                tools=[ARTICLE_CONTENT_TYPE_TOOL],
                tool_choice={"type": "function", "function": {"name": "classify_article_content_type"}},
                temperature=0,
                max_tokens=50,
            )
            args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
            return args.get("content_type", "other")
        except Exception as e:
            bt.logging.warning(f"[NEWS_ANALYZER] _classify_content_type failed: {e}")
            self._tl.had_llm_error = True
            return "other"

    def _classify_sentiment(self, text: str) -> str:
        prompt = f"""Classify the market sentiment of this news article:

"{text}"

- very_bullish: strongly positive for markets, major breakthroughs, record growth
- bullish: positive outlook, growth signals, encouraging developments
- neutral: factual reporting, balanced, no strong market direction
- bearish: concerns raised, negative outlook, risks highlighted
- very_bearish: crisis, crash, severe problems, major failures"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                tools=[SENTIMENT_TOOL],
                tool_choice={"type": "function", "function": {"name": "classify_sentiment"}},
                temperature=0,
                max_tokens=50,
            )
            args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
            return args.get("sentiment", "neutral")
        except Exception as e:
            bt.logging.warning(f"[NEWS_ANALYZER] _classify_sentiment failed: {e}")
            self._tl.had_llm_error = True
            return "neutral"

    def _assess_technical_quality(self, text: str) -> str:
        prompt = f"""Assess the technical quality of this news article:

"{text}"

- high: 2+ specifics (data points, statistics, named sources, detailed analysis)
- medium: 1 specific detail or named source
- low: claims without specifics or sources
- none: no substantive content"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                tools=[TECHNICAL_QUALITY_TOOL],
                tool_choice={"type": "function", "function": {"name": "assess_technical_quality"}},
                temperature=0,
                max_tokens=50,
            )
            args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
            return args.get("quality", "none")
        except Exception as e:
            bt.logging.warning(f"[NEWS_ANALYZER] _assess_technical_quality failed: {e}")
            self._tl.had_llm_error = True
            return "none"

    def _classify_market_analysis(self, text: str) -> str:
        prompt = f"""Classify the market analysis type in this news article:

"{text}"

- technical: market indicators, price patterns, trading data
- economic: fundamentals, GDP, inflation, employment, costs
- political: regulatory, governance, legislation, policy
- social: narrative-driven, public sentiment, trends
- other: none of the above or mixed"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                tools=[MARKET_ANALYSIS_TOOL],
                tool_choice={"type": "function", "function": {"name": "classify_market_analysis"}},
                temperature=0,
                max_tokens=50,
            )
            args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
            return args.get("analysis_type", "other")
        except Exception as e:
            bt.logging.warning(f"[NEWS_ANALYZER] _classify_market_analysis failed: {e}")
            self._tl.had_llm_error = True
            return "other"

    def _assess_impact(self, text: str) -> str:
        prompt = f"""Assess the potential market impact of this news article:

"{text}"

- HIGH: major policy changes, critical events, market-moving news
- MEDIUM: notable developments, significant updates
- LOW: minor information, routine reports
- NONE: no market relevance"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                tools=[IMPACT_TOOL],
                tool_choice={"type": "function", "function": {"name": "assess_impact"}},
                temperature=0,
                max_tokens=50,
            )
            args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
            return args.get("impact", "NONE")
        except Exception as e:
            bt.logging.warning(f"[NEWS_ANALYZER] _assess_impact failed: {e}")
            self._tl.had_llm_error = True
            return "NONE"
