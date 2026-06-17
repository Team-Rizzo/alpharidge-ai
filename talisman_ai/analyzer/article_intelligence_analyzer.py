"""
ArticleIntelligenceAnalyzer — Three-stage pipeline for full article analysis.

Stage 1: Deterministic + NER (~1s)
  - text_stats, keyword assets, spaCy+GLiNER+Flair+ReFinED NER, FinBERT sentiment
  - Override dict + Wikidata QID resolution
Stage 2: LLM Call 1 "Extract & Classify" (~8-12s)
  - Full article text + NER hints → classification enums, economic data, quotes, event fingerprint
Stage 3: LLM Call 2 "Reason & Summarize" (~3-5s)
  - Structured fact sheet (NOT raw article) → chart summaries + narrative keywords only.
  - Per-asset sentiment (FinBERT aspect) and contagion (dependency-graph) are computed
    deterministically off-LLM in assembly — they are no longer asked of the LLM.

Total: ~12-18s per article via OpenRouter (default model: deepseek/deepseek-v4-flash).
"""

from __future__ import annotations

import json
import os
import threading
import time
import traceback
from datetime import datetime, timezone
from typing import Dict, List, Optional

import bittensor as bt
from openai import OpenAI

from talisman_ai.models.article_intelligence import (
    ArticleContentType, ArticleIntelligence, AssetClass, AssetSentiment,
    ChartSummary, ContagionLink, ContagionMechanism, CredibilityFlag,
    DisambiguationMethod, EconomicDataPoint, EconomicEventType,
    EntityRole, EntityType, EventFingerprint, EventType, ExtractedEntity,
    FactualConfidence, ForwardEventType, GeoImpactZone, ImpactPotential,
    InferredImpact, MNPIRiskFlag, MarketAnalysisType, MarketSession,
    NumericClaim, NumericUnit, PositioningSignalType, QuoteExtraction,
    SCHEMA_VERSION, Sentiment, SentimentDirection, SourceAttributionType,
    SourceCategory, SourceMetadata, StalenessFlag, TargetAudience,
    TechnicalQuality, TemporalFocus, TextStatistics, TopicSignature, Urgency,
)
from talisman_ai.analyzer.text_stats import compute_text_stats
from talisman_ai.analyzer.ner_fusion import NERFusionEngine
from talisman_ai.analyzer.llm_cache import LLMCache

try:
    from talisman_ai import config
except ImportError:
    config = None

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MAX_CONTENT_CHARS = 3000

_NARRATIVE_SLUGS: Optional[List[str]] = None


def _load_narrative_slugs() -> List[str]:
    global _NARRATIVE_SLUGS
    if _NARRATIVE_SLUGS is not None:
        return _NARRATIVE_SLUGS
    try:
        path = os.path.join(DATA_DIR, "narratives.json")
        with open(path) as f:
            data = json.load(f)
        _NARRATIVE_SLUGS = [n["slug"] for n in data]
    except Exception:
        _NARRATIVE_SLUGS = []
    return _NARRATIVE_SLUGS


def _safe_enum(enum_cls, value, default=None):
    if value is None:
        return default
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(str(value).lower().strip())
    except (ValueError, KeyError):
        try:
            return enum_cls(str(value).strip())
        except (ValueError, KeyError):
            return default


def _load_json(filename: str):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


# Map dependency_graph relationship strings -> ContagionMechanism enum values.
# Deterministic, used by the off-LLM graph contagion builder.
_RELATION_MECHANISM_MAP = {
    "supply_chain": "supply_chain",
    "competitor": "competitive",
    "regulatory_spillover": "regulatory_spillover",
    "capital_flow": "capital_flow",
    "ecosystem_dependency": "protocol_dependency",
    "protocol_dependency": "protocol_dependency",
    "collateral": "collateral",
    "macro_sensitivity": "macro_sensitivity",
    "correlation": "correlation",
    "narrative": "narrative",
    "liquid_staking_derivative": "protocol_dependency",
    "ecosystem_token": "protocol_dependency",
}


def _relation_to_mechanism(label: str) -> str:
    return _RELATION_MECHANISM_MAP.get(str(label).lower().strip(), "correlation")


# FinBERT 3-class label -> our coarse sentiment direction (lowercase, matches enum values).
_FINBERT_DIRECTION = {"positive": "bullish", "negative": "bearish", "neutral": "neutral"}


# ============================================================================
# LLM Tool Definitions — Two calls only
# ============================================================================

EXTRACT_CLASSIFY_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_and_classify",
        "description": "Extract structured data and classify the article across all dimensions",
        "parameters": {
            "type": "object",
            "properties": {
                "content_type": {"type": "string", "enum": [e.value for e in ArticleContentType]},
                "sentiment": {"type": "string", "enum": [e.value for e in Sentiment]},
                "sentiment_score": {"type": "number", "description": "-1.0 (very bearish) to 1.0 (very bullish)"},
                "sentiment_direction": {"type": "string", "enum": [e.value for e in SentimentDirection]},
                "market_analysis_type": {"type": "string", "enum": [e.value for e in MarketAnalysisType]},
                "impact_potential": {"type": "string", "enum": [e.value for e in ImpactPotential]},
                "technical_quality": {"type": "string", "enum": [e.value for e in TechnicalQuality]},
                "urgency": {"type": "string", "enum": [e.value for e in Urgency]},
                "temporal_focus": {"type": "string", "enum": [e.value for e in TemporalFocus]},
                "factual_confidence": {"type": "string", "enum": [e.value for e in FactualConfidence]},
                "source_attribution": {"type": "string", "enum": [e.value for e in SourceAttributionType]},
                "positioning_signal": {"type": "string", "enum": [e.value for e in PositioningSignalType]},
                "primary_geo": {"type": "string", "enum": [e.value for e in GeoImpactZone]},
                "target_audience": {"type": "string", "enum": [e.value for e in TargetAudience]},
                "credibility_flag": {"type": "string", "enum": [e.value for e in CredibilityFlag]},
                "economic_data": {"type": "array", "items": {"type": "object", "properties": {
                    "event_type": {"type": "string"}, "event_name": {"type": "string"},
                    "actual_value": {"type": "number"}, "expected_value": {"type": "number"},
                    "previous_value": {"type": "number"}, "unit": {"type": "string"}, "period": {"type": "string"},
                    "reporting_country": {"type": "string"},
                }, "required": ["event_name"]}},
                "quotes": {"type": "array", "items": {"type": "object", "properties": {
                    "speaker": {"type": "string"}, "speaker_title": {"type": "string"},
                    "text": {"type": "string"}, "sentiment": {"type": "string"},
                    "is_market_moving": {"type": "boolean"},
                }, "required": ["speaker", "text"]}},
                "numeric_claims": {"type": "array", "items": {"type": "object", "properties": {
                    "metric_name": {"type": "string"}, "value": {"type": "number"},
                    "unit": {"type": "string"}, "context": {"type": "string"},
                    "is_percentage_change": {"type": "boolean"},
                }, "required": ["metric_name", "value", "unit"]}},
                "event_type": {"type": "string", "enum": [e.value for e in EventType]},
                "event_title": {"type": "string"},
                "event_date": {"type": "string", "description": "YYYY-MM-DD or null"},
                "semantic_fingerprint": {"type": "array", "items": {"type": "string"}},
                "staleness_flag": {"type": "string", "enum": [e.value for e in StalenessFlag]},
                "event_timestamp": {"type": "string"},
                "forward_event_type": {"type": "string", "enum": [e.value for e in ForwardEventType]},
                "forward_event_date": {"type": "string"},
                "forward_event_description": {"type": "string"},
                "additional_tickers": {"type": "array", "items": {"type": "string"},
                    "description": "Asset tickers the NER may have missed"},
            },
            "required": ["content_type", "sentiment", "sentiment_score", "sentiment_direction",
                         "market_analysis_type", "impact_potential", "technical_quality",
                         "urgency", "temporal_focus", "factual_confidence", "positioning_signal",
                         "primary_geo", "target_audience", "credibility_flag",
                         "event_type", "event_title", "semantic_fingerprint",
                         "staleness_flag", "forward_event_type"],
        },
    },
}

REASON_SUMMARIZE_TOOL = {
    "type": "function",
    "function": {
        "name": "reason_and_summarize",
        "description": "Generate chart-ready summaries and narrative keywords from extracted facts",
        "parameters": {
            "type": "object",
            "properties": {
                "headline": {"type": "string", "description": "Max 120 chars"},
                "one_liner": {"type": "string", "description": "Max 280 chars"},
                "context_paragraph": {"type": "string", "description": "2-3 sentences, max 1000 chars"},
                "what_changed": {"type": "string", "description": "Regime shift description or null"},
                "narrative_keywords": {"type": "array", "items": {"type": "string"},
                    "description": "0-3 narrative slugs. USE these known narratives when applicable: "
                    + ", ".join(_load_narrative_slugs())
                    + ". Only invent a new slug for genuinely new themes not covered above."},
            },
            "required": ["headline", "one_liner", "context_paragraph", "narrative_keywords"],
        },
    },
}


class ArticleIntelligenceAnalyzer:
    """Three-stage pipeline: NER fusion → LLM extract/classify → LLM reason/summarize."""

    def __init__(self, model: str = None, api_key: str = None, llm_base: str = None,
                 enable_refined: bool = True, enable_flair: bool = True):
        if config:
            self.model = model or config.MODEL
            self.api_key = api_key or config.API_KEY
            self.llm_base = llm_base or config.LLM_BASE
        else:
            self.model = model
            self.api_key = api_key
            self.llm_base = llm_base

        if not self.api_key:
            raise ValueError("API_KEY is required")

        self.client = OpenAI(base_url=self.llm_base, api_key=self.api_key)

        cache_ttl = float(getattr(config, "LLM_CACHE_TTL", 300)) if config else 300.0
        cache_size = int(getattr(config, "LLM_CACHE_MAX_SIZE", 1024)) if config else 1024
        self._cache = LLMCache(max_size=cache_size, ttl_seconds=cache_ttl)

        self.ner_engine = NERFusionEngine(
            enable_refined=enable_refined,
            enable_flair=enable_flair,
        )
        self.source_profiles = _load_json("source_profiles.json").get("profiles", {})
        self.dependency_graph = _load_json("dependency_graph.json").get("dependencies", {})

        bt.logging.info(f"[ARTICLE_INTEL] Ready: model={self.model} endpoint={self.llm_base}")

    def analyze(
        self, article_id: int, url: str, title: str, source: str,
        published: Optional[str] = None, summary: Optional[str] = None,
        content: Optional[str] = None, miner_hotkey: Optional[str] = None,
        raw_html: Optional[str] = None,
    ) -> Optional[ArticleIntelligence]:
        start_ms = int(time.time() * 1000)
        body = content or summary or ""
        body_truncated = body[:MAX_CONTENT_CHARS]
        article_text = f"{title}\n\n{summary or ''}\n\n{body_truncated}".strip()

        try:
            # ── STAGE 1: Deterministic + NER (~1s) ──
            # text_stats/content_hash stay on the plain `content` (Tier-2 gated
            # fields must not move). Only the NER/entity path consumes raw_html,
            # which lets the content extractor run trafilatura on real DOM.
            text_stats = compute_text_stats(title, body)
            ner_result = self.ner_engine.extract_and_resolve(
                title, body_truncated, raw_html=raw_html)
            # Real detected language (was hardcoded "en"). Drives both the Tier-1
            # detected_language gate and text_stats.language deterministically.
            text_stats.language = ner_result.detected_language
            content_hash = ArticleIntelligence.compute_content_hash(title, body)
            source_meta = self._build_source_metadata(source)
            market_sess = self._compute_market_session(published)

            sector_matches = self.ner_engine._asset_extractor.extract_sectors(title, body_truncated)
            primary_sector = sector_matches[0] if sector_matches else {"id": 9, "symbol": "OTHER"}

            # Build NER hints for LLM Call 1
            ner_hints = self._format_ner_hints(ner_result)
            all_tickers = list({e.ticker for e in ner_result.resolved_assets if e.ticker})

            # ── STAGE 2: LLM Call 1 — Extract & Classify (~8-12s) ──
            call1_prompt = (
                f"Analyze this financial news article. Pre-detected entities are provided as hints — "
                f"confirm, correct, or supplement them.\n\n"
                f"Article:\n\"\"\"{article_text}\"\"\"\n\n"
                f"Pre-detected (NER):\n{ner_hints}"
            )
            call1 = self._llm_call(call1_prompt, EXTRACT_CLASSIFY_TOOL, "extract_and_classify")

            # Merge additional tickers from LLM
            for t in call1.get("additional_tickers", []):
                if t and t.upper() not in {x.upper() for x in all_tickers}:
                    all_tickers.append(t.upper())

            # ── STAGE 3: LLM Call 2 — Reason & Summarize (~5-8s) ──
            fact_sheet = self._build_fact_sheet(title, source, published, primary_sector,
                                                call1, ner_result, all_tickers)
            call2_prompt = (
                f"Based on these extracted facts, write chart-ready summaries (headline, one-liner, "
                f"context paragraph) and 0-3 narrative keyword slugs.\n\n"
                f"{fact_sheet}"
            )
            call2 = self._llm_call(call2_prompt, REASON_SUMMARIZE_TOOL, "reason_and_summarize")

            # ── ASSEMBLY ──
            # Contagion + per-asset sentiment are computed off-LLM from the DETERMINISTIC
            # NER-resolved tickers (NOT all_tickers, which includes non-deterministic
            # LLM-suggested additional_tickers) so they match across the consensus boundary.
            ner_tickers = [e.ticker for e in ner_result.resolved_assets if e.ticker]
            asset_sentiments = self._finbert_asset_sentiments(ner_tickers, ner_result)
            assets = self._build_assets(ner_result, asset_sentiments)
            entities = self._build_entities_from_ner(ner_result)
            inferred = self._compute_inferred_impacts(all_tickers)
            elapsed_ms = int(time.time() * 1000) - start_ms

            # ── STAGE 4: Embeddings (~45ms) ──
            headline = (call2.get("headline") or title[:120])[:120]
            one_liner = (call2.get("one_liner") or title[:280])[:280]
            ctx_para = (call2.get("context_paragraph") or "")[:1000]
            narr_kws = (call2.get("narrative_keywords") or [])[:3]

            title_emb = self.ner_engine.encode_text(title)
            body_emb = self.ner_engine.encode_text(f"{one_liner} {ctx_para}")
            narr_emb = self.ner_engine.encode_text(f"{headline} {', '.join(narr_kws)}")

            result = ArticleIntelligence(
                schema_version=SCHEMA_VERSION,
                article_id=article_id, url=url, title=title,
                published_at=published or "",
                analyzed_at=datetime.now(timezone.utc).isoformat(),
                miner_hotkey=miner_hotkey, analysis_model=self.model,
                analysis_latency_ms=elapsed_ms,
                source=source_meta,
                content_type=_safe_enum(ArticleContentType, call1.get("content_type"), ArticleContentType.OTHER),
                market_analysis_type=_safe_enum(MarketAnalysisType, call1.get("market_analysis_type"), MarketAnalysisType.NONE),
                impact_potential=_safe_enum(ImpactPotential, call1.get("impact_potential"), ImpactPotential.LOW),
                technical_quality=_safe_enum(TechnicalQuality, call1.get("technical_quality"), TechnicalQuality.NONE),
                urgency=_safe_enum(Urgency, call1.get("urgency"), Urgency.SAME_DAY),
                temporal_focus=_safe_enum(TemporalFocus, call1.get("temporal_focus"), TemporalFocus.CURRENT),
                overall_sentiment=_safe_enum(Sentiment, call1.get("sentiment"), Sentiment.NEUTRAL),
                overall_sentiment_score=max(-1.0, min(1.0, float(call1.get("sentiment_score", 0.0) or 0.0))),
                sentiment_direction=_safe_enum(SentimentDirection, call1.get("sentiment_direction"), SentimentDirection.NEUTRAL),
                assets=assets,
                entities=entities,
                economic_data=self._build_economic_data(call1.get("economic_data", [])),
                numeric_claims=self._build_numeric_claims(call1.get("numeric_claims", [])),
                quotes=self._build_quotes(call1.get("quotes", [])),
                contagion_links=self._build_contagion_from_graph(ner_tickers),
                chart_summary=ChartSummary(
                    headline=headline,
                    one_liner=one_liner,
                    context_paragraph=ctx_para,
                    what_changed=(call2.get("what_changed") or "")[:200] or None,
                ),
                event_fingerprint=EventFingerprint(
                    event_type=_safe_enum(EventType, call1.get("event_type"), EventType.OTHER),
                    event_title=(call1.get("event_title") or title[:200])[:200],
                    event_date=call1.get("event_date"),
                    content_hash=content_hash,
                    semantic_fingerprint=sorted(call1.get("semantic_fingerprint") or [title.split()[0].lower()])[:10],
                ),
                narrative_keywords=narr_kws,
                title_embedding=title_emb,
                body_embedding=body_emb,
                narrative_embedding=narr_emb,
                topic_signature=TopicSignature(
                    primary_sector_id=primary_sector["id"],
                    primary_sector_symbol=primary_sector["symbol"],
                    secondary_sector_ids=[s["id"] for s in sector_matches[1:4]],
                ),
                text_stats=text_stats,
                factual_confidence=_safe_enum(FactualConfidence, call1.get("factual_confidence"), FactualConfidence.SPECULATIVE),
                source_attribution_type=_safe_enum(SourceAttributionType, call1.get("source_attribution"), SourceAttributionType.NONE),
                positioning_signal=_safe_enum(PositioningSignalType, call1.get("positioning_signal"), PositioningSignalType.NONE),
                target_audience=_safe_enum(TargetAudience, call1.get("target_audience"), TargetAudience.GENERAL),
                credibility_flag=_safe_enum(CredibilityFlag, call1.get("credibility_flag"), CredibilityFlag.UNVERIFIED),
                primary_geo=_safe_enum(GeoImpactZone, call1.get("primary_geo"), GeoImpactZone.GLOBAL),
                market_session=market_sess,
                detected_language=text_stats.language,
                staleness_flag=_safe_enum(StalenessFlag, call1.get("staleness_flag"), StalenessFlag.UNKNOWN),
                event_timestamp=call1.get("event_timestamp"),
                forward_event_type=_safe_enum(ForwardEventType, call1.get("forward_event_type"), ForwardEventType.NONE),
                forward_event_date_approximate=call1.get("forward_event_date"),
                forward_event_description=call1.get("forward_event_description"),
                inferred_impacts=inferred if inferred else None,
            )
            bt.logging.info(f"[ARTICLE_INTEL] Done article {article_id} in {elapsed_ms}ms: "
                           f"{len(assets)} assets, {len(entities)} entities")
            return result

        except Exception as e:
            bt.logging.error(f"[ARTICLE_INTEL] Failed article {article_id}: {e}")
            bt.logging.error(f"[ARTICLE_INTEL] {traceback.format_exc()}")
            return None

    # ========================================================================
    # LLM
    # ========================================================================

    def _llm_call(self, prompt: str, tool: dict, tool_name: str) -> dict:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                tools=[tool],
                tool_choice={"type": "function", "function": {"name": tool_name}},
                temperature=0,
                max_tokens=4000,
            )
            tc = response.choices[0].message.tool_calls
            if not tc:
                bt.logging.warning(f"[ARTICLE_INTEL] {tool_name}: no tool calls returned")
                return {}
            return json.loads(tc[0].function.arguments)
        except Exception as e:
            bt.logging.warning(f"[ARTICLE_INTEL] {tool_name} failed: {e}")
            return {}

    # ========================================================================
    # Fact Sheet Builder (compact input for LLM Call 2)
    # ========================================================================

    def _format_ner_hints(self, ner_result) -> str:
        lines = []
        tickers = [e.ticker for e in ner_result.resolved_assets if e.ticker]
        if tickers:
            lines.append(f"Assets: {', '.join(tickers)}")
        entities = [(e.canonical_name, e.entity_type, e.role or "") for e in ner_result.resolved_entities
                     if e.entity_type in ("person", "regulatory_body", "organization", "economic_indicator")]
        for name, etype, role in entities[:10]:
            role_str = f"/{role}" if role else ""
            lines.append(f"  {name} ({etype}{role_str})")
        money = [m["text"] for m in ner_result.money_values[:5]]
        if money:
            lines.append(f"Money: {', '.join(money)}")
        pcts = [p["text"] for p in ner_result.percentages[:5]]
        if pcts:
            lines.append(f"Percentages: {', '.join(pcts)}")
        sents = [(s["sentiment"], s["text"][:60]) for s in ner_result.sentence_sentiments[:3]]
        if sents:
            lines.append("FinBERT sentiment hints:")
            for direction, txt in sents:
                lines.append(f"  {direction}: {txt}")
        return "\n".join(lines)

    def _build_fact_sheet(self, title, source, published, sector, call1, ner_result, tickers) -> str:
        lines = [
            f"Title: {title}",
            f"Source: {source} | Published: {published or 'unknown'} | Geo: {call1.get('primary_geo', 'global')}",
            f"Event: [{call1.get('event_type', 'other')}] {call1.get('event_title', 'unknown')}",
            f"Classification: {call1.get('content_type', 'other')} | {call1.get('sentiment', 'neutral')} "
            f"({call1.get('sentiment_score', 0)}) | {call1.get('impact_potential', 'low')} impact",
            f"Sector: {sector.get('symbol', 'OTHER')}",
            f"Detected Assets: {', '.join(tickers)}",
        ]
        entities = [(e.canonical_name, e.entity_type) for e in ner_result.resolved_entities
                     if e.entity_type in ("person", "regulatory_body", "organization")]
        if entities:
            lines.append("Entities: " + ", ".join(f"{n} ({t})" for n, t in entities[:8]))
        econ = call1.get("economic_data", [])
        if econ:
            parts = []
            for d in econ[:5]:
                actual = f"{d.get('actual_value', '?')}" if d.get("actual_value") is not None else "?"
                parts.append(f"{d.get('event_name', '?')}: {actual} {d.get('unit', '')}")
            lines.append("Economic Data: " + " | ".join(parts))
        quotes = call1.get("quotes", [])
        if quotes:
            for q in quotes[:2]:
                lines.append(f"Quote: {q.get('speaker', '?')}: \"{q.get('text', '')[:100]}\"")
        sents = ner_result.sentence_sentiments[:3]
        if sents:
            lines.append("FinBERT hints: " + ", ".join(f"{s['sentiment']}({s['score']:.2f})" for s in sents))
        return "\n".join(lines)

    # ========================================================================
    # Assembly Helpers
    # ========================================================================

    def _build_assets(self, ner_result, sentiments_raw: list) -> List[AssetSentiment]:
        sentiment_by_ticker = {}
        for s in sentiments_raw:
            t = s.get("ticker", "").upper()
            if t:
                sentiment_by_ticker[t] = s

        assets = []
        for e in ner_result.resolved_assets:
            if not e.ticker:
                continue
            s = sentiment_by_ticker.get(e.ticker.upper(), {})
            try:
                assets.append(AssetSentiment(
                    ticker=e.ticker,
                    asset_name=e.canonical_name,
                    asset_class=_safe_enum(AssetClass, e.asset_class, AssetClass.UNKNOWN),
                    coingecko_id=None,
                    yahoo_ticker=None,
                    direction=_safe_enum(Sentiment, s.get("direction"), Sentiment.NEUTRAL),
                    magnitude=max(0.0, min(1.0, float(s.get("magnitude", 0.5) or 0.5))),
                    confidence=max(0.0, min(1.0, float(s.get("confidence", 0.5) or 0.5))),
                    short_term_outlook=_safe_enum(Sentiment, s.get("short_term"), Sentiment.NEUTRAL),
                    medium_term_outlook=_safe_enum(Sentiment, s.get("medium_term"), Sentiment.NEUTRAL),
                    long_term_outlook=_safe_enum(Sentiment, s.get("long_term"), Sentiment.NEUTRAL),
                    causal_driver=(s.get("causal_driver") or "No specific driver identified")[:500],
                    relevance_score=min(1.0, e.confidence),
                    is_primary_subject=getattr(e, "is_primary_subject", False),
                    evidence_spans=[e.text],
                ))
            except Exception as ex:
                bt.logging.debug(f"[ARTICLE_INTEL] Skip asset {e.ticker}: {ex}")
        return assets

    def _build_entities_from_ner(self, ner_result) -> List[ExtractedEntity]:
        entities = []
        for e in ner_result.resolved_entities:
            try:
                etype = _safe_enum(EntityType, e.entity_type, EntityType.ORGANIZATION)
                if etype is None:
                    etype = EntityType.ORGANIZATION
                entities.append(ExtractedEntity(
                    name=e.canonical_name,
                    entity_type=etype,
                    role=_safe_enum(EntityRole, e.role, EntityRole.MENTIONED),
                    ticker=e.ticker,
                    sentiment_toward=_safe_enum(Sentiment, e.sentiment_toward, None),
                ))
            except Exception:
                pass
        return entities[:15]

    def _build_economic_data(self, raw: list) -> List[EconomicDataPoint]:
        points = []
        for d in (raw or [])[:5]:
            try:
                actual = d.get("actual_value")
                expected = d.get("expected_value")
                previous = d.get("previous_value")
                points.append(EconomicDataPoint(
                    event_type=_safe_enum(EconomicEventType, d.get("event_type"), EconomicEventType.OTHER),
                    event_name=d.get("event_name", "Unknown"),
                    actual_value=actual, expected_value=expected, previous_value=previous,
                    delta_vs_expected=(actual - expected) if actual is not None and expected is not None else None,
                    delta_vs_previous=(actual - previous) if actual is not None and previous is not None else None,
                    unit=_safe_enum(NumericUnit, d.get("unit"), NumericUnit.OTHER),
                    period=d.get("period"), reporting_country=d.get("reporting_country"),
                ))
            except Exception:
                pass
        return points

    def _build_numeric_claims(self, raw: list) -> List[NumericClaim]:
        claims = []
        for c in (raw or [])[:10]:
            try:
                claims.append(NumericClaim(
                    metric_name=c["metric_name"], value=c["value"],
                    unit=c.get("unit", "other"),
                    context=(c.get("context") or "")[:200],
                    is_percentage_change=c.get("is_percentage_change", False),
                ))
            except Exception:
                pass
        return claims

    def _build_quotes(self, raw: list) -> List[QuoteExtraction]:
        quotes = []
        for q in (raw or [])[:5]:
            try:
                quotes.append(QuoteExtraction(
                    speaker=q["speaker"], speaker_title=q.get("speaker_title"),
                    text=q["text"][:1000],
                    sentiment=_safe_enum(Sentiment, q.get("sentiment"), Sentiment.NEUTRAL),
                    is_market_moving=q.get("is_market_moving", False),
                ))
            except Exception:
                pass
        return quotes

    def _build_contagion_from_graph(self, tickers: List[str]) -> List[ContagionLink]:
        """Deterministic contagion links from the curated dependency_graph.

        For each detected ticker, emit one ContagionLink per known dependent edge.
        Direction is NEUTRAL and strength/confidence a fixed 0.7 prior (the graph
        encodes structure, not event-specific magnitude). Replaces the former
        LLM-reasoned contagion — both miner and validator compute this identically,
        so it is deterministic across the consensus boundary.
        """
        links = []
        seen = set()
        for t in tickers:
            for dep in self.dependency_graph.get(t, {}).get("dependents", []):
                target = dep.get("ticker")
                if not target:
                    continue
                key = (t, target)
                if key in seen:
                    continue
                seen.add(key)
                try:
                    links.append(ContagionLink(
                        source_ticker=t,
                        target_ticker=target,
                        target_asset_class=_safe_enum(AssetClass, dep.get("asset_class"), AssetClass.UNKNOWN),
                        mechanism=_safe_enum(ContagionMechanism,
                                             _relation_to_mechanism(dep.get("relationship", "")),
                                             ContagionMechanism.CORRELATION),
                        direction=SentimentDirection.NEUTRAL,
                        strength=0.7,
                        confidence=0.7,
                        lag_hours=None,
                        reasoning=f"dependency_graph: {t} -> {target} ({dep.get('relationship', 'correlation')})"[:300],
                    ))
                except Exception:
                    pass
                if len(links) >= 8:
                    return links
        return links

    def _finbert_asset_sentiments(self, tickers: List[str], ner_result) -> List[dict]:
        """Per-asset sentiment via FinBERT aspect over each asset's mention sentences.

        Deterministic (CPU FinBERT, no sampling). Reuses the FinBERT pipeline already
        loaded by the NER engine — no second model load. Returns dicts shaped for
        `_build_assets`: direction (and short/medium/long-term outlooks) from the
        FinBERT vote; magnitude/confidence from the mean FinBERT score; causal_driver
        a templated string. Outlooks all mirror `direction` (no horizon signal in FinBERT).
        """
        # Reuse the per-sentence FinBERT labels the NER engine already computed.
        scored = getattr(ner_result, "sentence_sentiments", None) or []

        out = []
        for tk in tickers:
            if not tk:
                continue
            mentions = [s for s in scored if tk.upper() in (s.get("text") or "").upper()]
            if mentions:
                labels = [m["sentiment"] for m in mentions]  # already bullish/bearish/neutral
                direction = max(set(labels), key=labels.count)
                magnitude = sum(float(m.get("score", 0.5)) for m in mentions) / len(mentions)
            else:
                direction, magnitude = "neutral", 0.5
            out.append({
                "ticker": tk,
                "direction": direction,
                "magnitude": max(0.0, min(1.0, magnitude)),
                "confidence": max(0.0, min(1.0, magnitude)),
                "short_term": direction,
                "medium_term": direction,
                "long_term": direction,
                "causal_driver": f"FinBERT aspect sentiment over {len(mentions)} mention sentence(s)",
            })
        return out

    def _build_source_metadata(self, source: str) -> SourceMetadata:
        key = source.lower().replace(" ", "_").replace("-", "_")
        profile = self.source_profiles.get(key, {})
        return SourceMetadata(
            source_id=key, source_name=profile.get("name", source),
            credibility_score=profile.get("credibility", 0.5),
            source_category=_safe_enum(SourceCategory, profile.get("category"), SourceCategory.UNKNOWN),
        )

    def _compute_market_session(self, published: Optional[str]) -> MarketSession:
        if not published:
            return MarketSession.REGULAR_HOURS
        try:
            dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
            if dt.weekday() >= 5:
                return MarketSession.WEEKEND
            h = dt.hour
            if 13 <= h < 14:
                return MarketSession.PRE_MARKET
            if 14 <= h < 21:
                return MarketSession.REGULAR_HOURS
            return MarketSession.AFTER_HOURS
        except (ValueError, TypeError):
            return MarketSession.REGULAR_HOURS

    def _compute_inferred_impacts(self, tickers: List[str]) -> List[InferredImpact]:
        impacts = []
        seen = set(t.upper() for t in tickers)
        for ticker in tickers:
            deps = self.dependency_graph.get(ticker, {}).get("dependents", [])
            for dep in deps:
                if dep["ticker"].upper() not in seen:
                    seen.add(dep["ticker"].upper())
                    impacts.append(InferredImpact(
                        ticker=dep["ticker"],
                        asset_class=_safe_enum(AssetClass, dep.get("asset_class"), AssetClass.UNKNOWN),
                        relationship=dep.get("relationship", "unknown"),
                        confidence=0.7,
                    ))
        return impacts[:20]
