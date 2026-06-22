# ArticleIntelligence Pipeline: Complete Build Changelog

**Date:** 2026-05-28
**Scope:** alpharidge-ai (miner/validator) + alpharidge-ai-api (FastAPI backend)
**Tests:** 56 passing, 0 failures, 2 skipped

---

## What Was Built

The SN45 Alpharidge AI Bittensor subnet's article analysis was expanded from 6 flat classification fields (sentiment, sector, content_type, technical_quality, market_analysis, relevance_confidence) to a comprehensive 28-feature-group intelligence system. The system processes 350K news articles/day, extracting multi-asset sentiment, cross-market contagion chains, chart summaries, entity extraction with Wikidata linking, economic data points, event fingerprints for clustering, narrative tagging with embedding-based semantic matching, and ML features.

Everything runs inside the SN45 subnet: miners produce all per-article features, validators verify them, and the API handles cross-article aggregation (event clustering, narrative matching, sentiment momentum).

---

## Architecture Overview

```
Article arrives at miner
    │
    ├─ STAGE 1: Deterministic + NER (~1s)
    │   ├─ text_stats (24 features: readability, density, structure, language)
    │   ├─ keyword asset extraction → 206 assets across 7 classes
    │   ├─ spaCy trf + GLiNER + Flair OntoNotes NER → raw entities
    │   ├─ Financial override dict (78 entries) → entity resolution
    │   ├─ ReFinED Wikidata entity linking → QID → ticker mapping
    │   ├─ FinBERT sentence-level sentiment (~18ms/sentence)
    │   ├─ sector matching, content_hash, source metadata
    │   └─ inferred impacts from dependency graph
    │
    ├─ STAGE 2: LLM Call 1 "Extract & Classify" (~8-12s)
    │   Model: anthropic/claude-haiku-4.5 via OpenRouter
    │   Input: full article text + NER results as hints
    │   Output: classification enums, economic data points,
    │           quotes, event fingerprint, supplemental entities
    │
    ├─ STAGE 3: LLM Call 2 "Reason & Summarize" (~5-8s)
    │   Model: anthropic/claude-haiku-4.5 via OpenRouter
    │   Input: structured fact sheet (~300 tokens, NOT raw article)
    │   Output: per-asset sentiment with causal drivers,
    │           contagion chain, chart summaries, narrative keywords
    │   Note: narrative_keywords field has full taxonomy of 38 slugs
    │
    ├─ STAGE 4: Embeddings (~45ms)
    │   Model: all-MiniLM-L6-v2 (384d, L2-normalized)
    │   ├─ title_embedding: encode(title)
    │   ├─ body_embedding: encode(LLM summary text)
    │   └─ narrative_embedding: encode(headline + narrative keywords)
    │
    └─ ASSEMBLY → ArticleIntelligence (58 fields, 28 feature groups)

Validator re-runs pipeline, compares via 4-tier validation:
    Tier 1: Exact enum match (20 fields)
    Tier 2: Deterministic match (content_hash, text_stats)
    Tier 2.5: Embedding cosine verification (≥0.90 title, ≥0.80 narrative)
    Tier 3: Near-deterministic composite (≥0.80 weighted score)

API receives articles, runs cross-article intelligence:
    Event Clustering: fingerprint → content_hash → embedding cosine → title Jaccard
    Narrative Matching: slug exact → embedding cosine → keyword Jaccard
    Background Jobs: cache refresh, lifecycle, centroid drift, auto-discovery
```

Total pipeline latency: ~15-22 seconds per article.

---

## Files Created or Significantly Modified

### Core Data Model

**`alpharidge_ai/models/article_intelligence.py`** (NEW, ~800 lines)

The foundational data model for the entire system. Contains:

- **28 enums** covering every classification dimension: ArticleContentType (16 values like breaking_news, analysis, opinion, data_release), Sentiment (7-point scale from very_bearish to very_bullish), EventType (18 types like fed_decision, earnings, hack, tariff_change), AssetClass (crypto, equity, forex, commodity, index, fixed_income, derivative, other), EntityType (10 types), ContagionMechanism (9 mechanisms like risk_appetite, correlation, liquidity, regulatory), EconomicEventType (33 types covering CPI, NFP, GDP, FOMC, etc.), and more.

- **13 Pydantic sub-models**: AssetSentiment (per-asset with temporal outlook and causal driver), ExtractedEntity (with Wikidata QID, role, disambiguation method), EconomicDataPoint (actual/expected/previous values with units), NumericClaim (extracted numbers with context), QuoteExtraction (speaker attribution + sentiment), ContagionLink (source→target with mechanism, strength, confidence, lag, reasoning), ChartSummary (headline, one-liner, context paragraph, regime shift), EventFingerprint (event_type + title + date + content_hash + semantic_fingerprint), TopicSignature (primary/secondary sectors), TextStatistics (24 deterministic features), SourceMetadata (category, audience, reliability), InferredImpact (from dependency graph).

- **ArticleIntelligence class**: 58 top-level fields organized into 28 feature groups. Includes `to_canonical_string()` for Tier 1 validation (deterministic string from all enum fields) and `compute_content_hash()` for dedup (SHA-256 of normalized title + body).

### NER Fusion Engine

**`alpharidge_ai/analyzer/ner_fusion.py`** (NEW, ~500 lines)

Combines 4 NER models + entity resolution + sentiment into a single pipeline:

- **spaCy en_core_web_trf** (always loaded): People, organizations, money values, percentages, dates. Transformer-based, highest recall.
- **GLiNER** (optional): Zero-shot NER for financial-specific labels — cryptocurrency, stock ticker, index, commodity, government body, regulatory body, economic indicator. Threshold 0.4.
- **Flair OntoNotes** (optional): High-confidence ORG/PERSON/MONEY detection. Acts as confirmation signal for entities found by other models.
- **ReFinED** (optional): Amazon's Wikidata entity linking. Resolves ambiguous entities (e.g., "Apple" → Apple Inc. Q312, not the fruit). Required monkey-patch for transformers 5.x compatibility (removes `add_special_tokens` kwarg from tokenizer init).
- **FinBERT** (optional): ProsusAI/finbert sentiment model. 75% accuracy on sentence-level financial sentiment at ~18ms per sentence. Used as pre-pass hint for LLM.
- **SentenceTransformer** (optional): all-MiniLM-L6-v2 for 384-dim embeddings. ~15ms per encode, L2-normalized for efficient cosine similarity via dot product.

Entity resolution pipeline: Override dict (78 curated entries for CPI, Fed, SEC, key CEOs, etc.) → ReFinED Wikidata linking → QID-to-ticker lookup table → keyword asset registry fallback. Total NER time: ~1.1s per article.

`NERResult` dataclass: resolved_assets, resolved_entities, money_values, percentages, dates, sentence_sentiments.

### Article Intelligence Analyzer

**`alpharidge_ai/analyzer/article_intelligence_analyzer.py`** (NEW, ~600 lines)

The main three-stage pipeline orchestrator:

- **Two LLM tool schemas**: `EXTRACT_CLASSIFY_TOOL` (20+ required fields for Call 1) and `REASON_SUMMARIZE_TOOL` (asset sentiments, contagion links, chart summaries, narrative keywords for Call 2). The narrative_keywords field includes the full taxonomy of 38 known narrative slugs, built dynamically from `narratives.json`.
- **`_build_fact_sheet()`**: Constructs a compact ~300-token structured input for Call 2 from Stage 1 NER results + Call 1 outputs. Includes assets, entities, economic data, quotes, FinBERT sentiment hints. This means Call 2 never sees the raw article text — only the distilled facts.
- **`_safe_enum()`**: Robust enum coercion that lowercases, strips whitespace, and falls back to a default value. Handles all the ways LLMs can return unexpected enum values.
- **`_format_ner_hints()`**: Formats NER results as structured text hints injected into Call 1's prompt.
- **Assembly**: After both LLM calls, pre-computes headline/one_liner/ctx_para/narr_kws variables, generates 3 embeddings via the NER engine's SentenceTransformer, and constructs the full ArticleIntelligence object with all 58 fields.
- **Model**: anthropic/claude-haiku-4.5 via OpenRouter, temperature=0, max_tokens=4000.

### Text Statistics

**`alpharidge_ai/analyzer/text_stats.py`** (NEW, ~200 lines)

Purely deterministic text feature extraction (no ML, no LLM). Computes 24 features:
- Readability: Flesch Reading Ease, Flesch-Kincaid Grade Level
- Density: numeric_density, quote_density, entity_density, ticker counts
- Structure: tables, subheadings, code blocks, images, links
- Language: hedging_score, certainty_score, clickbait_score
- Title analysis: word count, has_number, has_question, title_sentiment

Verified byte-identical between miner and validator runs (Tier 2 validation).

### Asset Extractor

**`alpharidge_ai/analyzer/asset_extractor.py`** (NEW, ~200 lines)

Keyword-based multi-asset extraction from article text:
- Loads 206 assets from `assets_expanded.json` (100 crypto) and `assets_traditional.json` (106 equities, indices, forex, commodities, ETFs).
- 4-tier keyword matching: cashtag ($BTC) > case-sensitive IDs > unique identifiers > aliases.
- Returns `AssetMatch` objects with ticker, name, asset_class, relevance score, evidence_spans, disambiguation method/confidence.
- `extract_sectors()` for multi-sector support.
- Tested: correctly finds BTC, ETH, SOL, NVDA, AAPL, SPX with no false positives on common words like "apple" or "gold" in non-financial contexts.

### LLM Cache

**`alpharidge_ai/analyzer/llm_cache.py`** (NEW, ~57 lines)

Thread-safe in-memory TTL cache for deterministic LLM calls. Since the pipeline uses temperature=0, identical inputs produce identical outputs. The cache prevents redundant API calls when the same article is analyzed by both miner and validator within 300 seconds. Max 1024 entries, FIFO eviction.

### Validation / Scoring

**`alpharidge_ai/analyzer/scoring.py`** (MODIFIED, added ~250 lines)

Extended with ArticleIntelligence validation:

- **`validate_article_intelligence(miner_intel, validator_intel)`**: 4-tier validation returning (is_valid, composite_score, details_dict).
  - Tier 1: Exact match on 20 enum fields (content_type, sentiment, impact, urgency, etc.) + primary asset sentiment directions. Any mismatch → immediate reject.
  - Tier 2: Deterministic match on content_hash, word_count, sentence_count, char_count, ticker_mention_count. Any mismatch → immediate reject.
  - Tier 2.5: Embedding verification. Checks title_embedding cosine ≥ 0.90 and narrative_embedding cosine ≥ 0.80. Also validates 384 dimensions, L2 norm near 1.0, non-zero vectors. Mismatch → reject.
  - Tier 3: Weighted composite of near-deterministic fields: asset_extraction (0.20), asset_sentiment (0.20), chart_summary (0.15), entities (0.10), economic_data (0.10), event_fingerprint (0.10), contagion (0.10), narrative_keywords (0.05). Uses Jaccard similarity for set fields, Levenshtein ratio for text fields. Composite must be ≥ 0.80.
- **`validate_miner_article_intelligence_batch()`**: Batch-level validation that samples articles, runs per-article validation, and adds cross-article adversarial detection (flags miners sending identical embeddings across different articles with pairwise cosine > 0.99).

### Data Files

**`alpharidge_ai/analyzer/data/`** (8 NEW JSON files)

- **`assets_expanded.json`** (51KB, 100 entries): Crypto assets with coingecko_id, yahoo_ticker, cashtags, case_sensitive IDs, unique_identifiers, aliases, thematic_tags. Covers BTC, ETH, SOL, ADA, DOT, AVAX, LINK, UNI, AAVE, and 90+ more.
- **`assets_traditional.json`** (60KB, 106 entries): 54 equities (NVDA, AAPL, MSFT, GOOGL, AMZN, META, TSLA, etc.), 15 indices (SPX, DJI, IXIC, VIX, etc.), 15 forex pairs (DXY, EUR/USD, GBP/USD, etc.), 12 commodities (gold, oil, silver, natural gas, etc.), 10 ETFs (SPY, QQQ, IWM, etc.).
- **`contagion_templates.json`** (26KB, 16 entries): Templates for cross-market impact chains — fed_rate_cut/hike, cpi_hot, tariff_increase, exchange_hack, btc_etf_flows, oil_shock, sec_enforcement, earnings_beat/miss, war_escalation, bank_failure, protocol_upgrade, token_unlock, nfp_strong, merger_acquisition. Each defines source assets, target assets, mechanisms, directions, typical lag.
- **`narratives.json`** (15KB, 38 entries): Seed narratives with slug, name, description, keywords, sector_ids, status. Covers Bitcoin ETF Flows, Fed Pivot, AI Infrastructure Buildout, DeFi Revival, Global Trade War, Inflation Persistence, Semiconductor Cycle, Meme Coin Cycle, and 30 more.
- **`dependency_graph.json`** (7.7KB): Maps 16 source assets to their dependents. ETH→stETH/UNI/AAVE/ARB/OP, BTC→WBTC/MSTR/mining stocks, SPX→sector ETFs, oil→airlines/shipping, DXY→EM currencies, etc.
- **`source_profiles.json`** (8.4KB, 37 entries): News source profiles with audience size, category, political_lean, crypto_stance, specialization scores. Covers Reuters, Bloomberg, CoinDesk, The Block, CNBC, BBC, etc.
- **`financial_overrides.json`** (7.4KB, 78 entries): Curated entity mappings for terms that confuse NER models. Economic indicators (CPI, PPI, GDP, NFP, PCE), regulatory bodies (SEC, CFTC, Fed, FDIC, OCC), indices (S&P 500, Dow Jones, Nasdaq, VIX, DXY), key people (Jerome Powell, Gary Gensler, Janet Yellen, Jamie Dimon, etc.), organizations (Goldman Sachs, JPMorgan, BlackRock, etc.). Each entry has canonical name, type, ticker (if applicable), asset_class, wikidata QID, role.
- **`wikidata_ticker_map.json`** (6.9KB, 58 entries): Maps Wikidata QIDs to tickers/types. Q312→AAPL, Q131723→BTC, Q193326→GS, etc. Used by ReFinED entity linking to resolve QIDs to tradeable assets.

### Miner Integration

**`neurons/miner.py`** (MODIFIED)

- Added `setup_article_intelligence_analyzer()` import and initialization with V1 fallback at startup.
- `_process_and_send_articles()`: V2 analyzer produces full ArticleIntelligence, maps all fields to NewsArticleAnalysisBase via model_dump(), serializes analysis_data as JSONB blob. Falls back to V1 (simple classification) if V2 analyzer fails.

### Validator Integration

**`neurons/validator.py`** (MODIFIED)

- Added `setup_article_intelligence_analyzer` and `validate_miner_article_intelligence_batch` imports.
- Validation flow: detects V2 articles (presence of analysis_data field), uses V2 4-tier validation, falls back to V1 format checking for old miners.

### Environment Configuration

**`.miner_env`** and **`.vali_env`** (MODIFIED)

Changed from Qwen3-32B on Chutes to Claude Haiku 4.5 on OpenRouter:
```
# Previous (commented out):
# MODEL=Qwen/Qwen3-32B
# API_KEY=cpk_...
# LLM_BASE=https://llm.chutes.ai/v1

# Current:
MODEL=anthropic/claude-haiku-4.5
API_KEY=sk-or-v1-...
LLM_BASE=https://openrouter.ai/api/v1
```

### Test Configuration

**`conftest.py`** (NEW)

Adds `--live-llm` and `--live-rss` pytest flags for controlling which integration tests run.

### Dependencies

**`requirements.txt`** (MODIFIED)

Added: `sentence-transformers>=2.2.0` (embeddings), plus existing torch, numpy, openai, spacy, etc.

---

## API-Side Files (alpharidge-ai-api/)

### Database Schema

**`prisma/schema.prisma`** (MODIFIED)

Added 7 new models for the event/narrative system:

- **Event**: Canonical events grouping articles about the same real-world occurrence. Fields: fingerprint (unique hash), eventType, canonicalTitle, eventDate, sectorId, articleCount, sentiment, impactPotential, entities (JSONB), titleEmbedding (Float[384]). Indexes on eventType, eventDate, sector, lastArticleAt.
- **EventArticle**: Many-to-many join between articles and events, with role field (reporting, analysis, opinion).
- **Narrative**: Long-running market themes. Fields: slug (unique), name, keywords (JSONB array), sectorIds (JSONB), phase (emerging/active/peak/fading/dormant), articleCount, eventCount, sentimentScore, momentum, embedding (Float[384]), embeddingModel. Indexes on phase, lastArticleAt.
- **NarrativeArticle**: Many-to-many join between articles and narratives, with confidence score and matchedKeyword for audit.
- **EventNarrative**: Many-to-many join between events and narratives.
- **NarrativeIntensity**: Daily time-series for narrative volume and sentiment tracking. Unique on (narrativeId, date).
- **NarrativeCandidate**: Tracks unmatched keywords for auto-discovery. Fields: keyword (unique), articleCount, sourceCount, promoted flag, embedding (Float[384]).

Also added to existing models:
- **NewsArticleAnalysis**: narrativeEmbedding (Float[384]), impactLevel, factualConfidence, eventType, eventDate, contentHash, primaryGeo, overallSentimentScore. New indexes on eventType, eventDate, contentHash, impactLevel.

### Event Clustering Service

**`services/event_clustering.py`** (NEW, ~220 lines)

Groups articles about the same real-world event using multi-phase matching:

1. **Fingerprint match**: Deterministic hash from event_type + date + top 3 entities. Exact match = same event.
2. **Content hash dedup**: SHA-256 of normalized title+body. Catches the same article republished by different sources.
3. **Embedding cosine similarity**: title_embedding dot product against recent events of the same type. Threshold 0.75 (tighter than narrative matching since events are specific occurrences).
4. **Title word-overlap**: Jaccard on word sets, threshold 0.6. Fallback for articles without embeddings.
5. **Create new event**: If no match, creates a new Event record with the article's title embedding.

Background job `merge_fragmented_clusters()` runs every 15 minutes. Finds recent events of the same type within 48 hours, merges by title Jaccard ≥ 0.6 OR embedding cosine ≥ 0.80.

### Narrative Matching Service

**`services/narrative_matcher.py`** (NEW, ~400 lines)

Three-signal hybrid matching with in-memory caching:

- **NarrativeMatchCache**: Holds all active narrative IDs, slugs, names, keyword lists, and a numpy matrix of embeddings (shape N×384). Refreshed every 15 minutes from the database. The `match()` method performs all 3 signals in one pass:
  - Signal 1 (slug match): Normalized keyword exactly equals a narrative slug → confidence 0.95. This catches the LLM's taxonomy-aware outputs.
  - Signal 2 (embedding cosine): Dot product of article's narrative_embedding against the embedding matrix → threshold 0.55. This catches semantic equivalences.
  - Signal 3 (keyword Jaccard): Word-overlap similarity between keywords → threshold 0.50. Backward-compatible fallback.
- **Narrative lifecycle**: emerging → active (20+ articles) → peak → fading (5 days no articles) → dormant (30 days). Reactivation if dormant narrative gets new articles.
- **Auto-discovery**: Tracks unmatched keywords as candidates. Promotes when 10+ articles from 3+ sources in 7 days. Before promotion, checks semantic dedup against existing narratives (cosine > 0.75 → merge instead). Groups similar candidates (cosine > 0.70) before promoting.
- **Centroid drift**: Exponential moving average (alpha=0.05) updates narrative embeddings from recent article embeddings. Runs every 6 hours.
- **Split detection**: If a narrative's recent articles have mean pairwise cosine < 0.50, warns that the narrative may be splitting into sub-themes.
- **Merge detection**: If two narrative centroids have cosine > 0.85, warns they may be converging.

### Background Tasks

**`jobs/background_tasks.py`** (NEW, ~97 lines)

Async background job scheduler:
- **Startup**: Seeds narratives from narratives.json, initializes NarrativeMatchCache.
- **Every 15 min**: Merge fragmented event clusters, refresh narrative cache.
- **Every 6 hours**: Update narrative lifecycle phases, compute centroid drift.
- **Every 24 hours**: Auto-discover new narratives, detect splits/merges.

### Dashboard

**`dashboard_routes.py`** (NEW, ~400 lines)

Public dashboard API endpoints:
- `/dashboard/stats` — Aggregate source counts, sentiment distribution
- `/dashboard/feed` — Unified feed (tweets, telegram, articles) with filtering
- `/dashboard/articles` — Article list with pagination and filtering
- `/dashboard/article_detail/{id}` — Full article with analysis data
- `/dashboard/events` — Clustered events with article counts
- `/dashboard/events/trending` — Trending events by momentum
- `/dashboard/miners` — Miner stats and profiles
- `/dashboard/validators` — Validator stats
- `/dashboard/assets` — Top mentioned assets
- `/dashboard/sentiment` — Sentiment time-series

**`dashboard_models.py`** (NEW, ~200 lines)

Pydantic response models: EventSummary, EventDetailResponse, NarrativeSummary, NarrativesResponse, FeedItem, ArticleWithAnalysis, MinerProfile, ValidatorProfile, PaginatedResponse, etc.

### API Main

**`main.py`** (MODIFIED)

- `/articles/completed` endpoint: Now extracts V2 fields from analysis_data, stores narrative_embedding in NewsArticleAnalysis, triggers event clustering and narrative matching.
- Lifespan: Starts background_tasks.run_periodic_jobs() as asyncio task at startup.

### API Dependencies

**`requirements.txt`** (MODIFIED)

Added: `numpy>=1` for embedding cosine similarity (no sentence-transformers — API stays lean).

### Embedding Script

**`scripts/compute_narrative_embeddings.py`** (NEW, ~80 lines)

CLI utility that pre-computes narrative embeddings using the same all-MiniLM-L6-v2 model as the miner (critical for vector space consistency). Loads all narratives from the database, encodes "{name}. {description}. {keywords}" into 384-dim vectors, stores in Narrative.embedding. Also handles NarrativeCandidate embeddings. Run at deploy time and daily via cron.

---

## Test Suite

**`tests/test_article_intelligence.py`** (NEW, ~1000 lines, 56 tests + 2 skipped)

Organized into 9 test classes:

1. **TestTextStats** (10 tests): Basic stats, determinism, hedging detection, certainty detection, bullish/bearish title sentiment, empty body handling, numeric density, clickbait detection, quote density.
2. **TestAssetExtractor** (11 tests): Asset loading (≥100), cashtag extraction, multi-asset, equity detection, forex detection, primary subject identification, disambiguation, no false positives on common words, sector extraction, multi-sector, evidence spans.
3. **TestContentHash** (4 tests): Deterministic hashing, different content produces different hashes, whitespace normalization, case normalization.
4. **TestEventDeduplication** (2 tests): Same content produces same hash, copycat articles produce different hashes.
5. **TestSchemaValidation** (3 tests): Minimal valid ArticleIntelligence construction, canonical string generation, serialization roundtrip.
6. **TestLLMAnalysisQuality** (14 tests, requires --live-llm): Fed rate cut (classification, assets, per-asset sentiment, economic data, entities, chart summary, contagion), Nvidia earnings (classification, assets, numeric claims), exchange hack (classification, contagion), tariff article (geo impact), all articles have required fields.
7. **TestEventClustering** (6 tests): Copycats share event type/date/similar fingerprint/similar title, different events produce different fingerprints with low overlap.
8. **TestEmbeddings** (4 tests): All 3 embeddings populated (384d), L2-normalized, different articles have different embeddings, narrative keywords are known slugs (100% match rate).
9. **TestValidationAgreement** (1 test, requires --live-llm): Runs analysis twice on same article, validates Tier 1-3 agreement.
10. **TestLiveRSS** (2 tests, skipped without --live-rss): Fetches real RSS articles, analyzes them.
11. **TestBatchQualityReport** (1 test): Prints comprehensive quality report for all analyzed articles including embedding status.

Uses bittensor mock stub for environments with package conflicts. `_get()` helper skips individual tests gracefully if article analysis returns None.

---

## Key Design Decisions

| Decision | Chosen | Alternatives Considered |
|---|---|---|
| LLM model | Claude Haiku 4.5 via OpenRouter | Qwen3-32B (too slow, 4-7min), Gemini 3.5 Flash (no tool calling) |
| Pipeline structure | 2 LLM calls + deterministic pre-pass | 5 LLM calls (original, too slow), 1 LLM call (too much in one prompt) |
| NER approach | 4-engine fusion (spaCy+GLiNER+Flair+ReFinED) | Single model (missed entities), spaCy only (no financial labels) |
| Entity linking | Override dict → ReFinED → keyword registry | spacy-entity-linker (0 results), REL (server down), mGENRE (mislinked CPI) |
| Sentiment | FinBERT pre-pass + LLM reasoning | FinBERT only (fails multi-asset), LLM only (slower) |
| Embedding model | all-MiniLM-L6-v2 (384d) | all-mpnet-base-v2 (768d, marginal improvement, 2x storage) |
| Narrative matching | 3-signal hybrid (slug+embedding+keyword) | Keyword-only (brittle), embedding-only (no exact matches) |
| API ML dependencies | numpy only (lean), script for embedding pre-compute | sentence-transformers in API (heavy), OpenRouter embeddings (model mismatch) |
| Validation | 4-tier + Tier 2.5 embeddings | Format-only (too lenient), full re-analysis (too expensive) |
| Article scope for Call 2 | Compact fact sheet (~300 tokens) | Full article (wasteful, slow, expensive) |

---

## Performance

| Component | Speed | Notes |
|---|---|---|
| Stage 1 (NER + deterministic) | ~1.1s | One-time model load ~30s at startup |
| Stage 2 (LLM Call 1) | ~8-12s | Full article + NER hints |
| Stage 3 (LLM Call 2) | ~5-8s | Fact sheet only (~300 tokens input) |
| Stage 4 (embeddings) | ~45ms | 3x encode at ~15ms each |
| Total per article | ~15-22s | Down from 4-7 minutes with original approach |
| API narrative matching | <1ms | NumPy dot product against 200 narratives |
| API event clustering | ~5ms | DB queries + fingerprint/cosine checks |
| Wire transport overhead | +4.5KB | 3 embeddings at 384×4 bytes each |

---

## Deployment Checklist

1. Install miner/validator dependencies: `pip install sentence-transformers>=2.2.0`
2. Install API dependency: `pip install numpy>=1`
3. Run Prisma migration: `cd alpharidge-ai-api && npx prisma migrate dev`
4. Pre-compute narrative embeddings: `python scripts/compute_narrative_embeddings.py`
5. First startup downloads models: spaCy en_core_web_trf, GLiNER, Flair OntoNotes, ReFinED aida_model, FinBERT, all-MiniLM-L6-v2 (~2GB total, one-time)
6. Set LLM config in `.miner_env` / `.vali_env` (MODEL, API_KEY, LLM_BASE for OpenRouter)

---

## Backward Compatibility

- All V2 fields are Optional with defaults (None or empty lists).
- V1 miners without analysis_data continue to work — API falls back to keyword-only narrative matching.
- V1 validators without embedding verification skip Tier 2.5.
- NewsArticleAnalysis retains all V1 fields (sentiment, sectorId, contentType, etc.).
- Miner and validator detect V2 presence via analysis_data field and adapt accordingly.
