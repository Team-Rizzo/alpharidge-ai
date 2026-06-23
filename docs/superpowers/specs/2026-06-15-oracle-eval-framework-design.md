# Oracle Dataset + Evaluation Framework — Design

**Date:** 2026-06-15
**Status:** Approved (pending spec review)
**Sub-project:** #1 of 3 (Oracle + framework → Candidate analyzers → Validator/API/miner rewrite)

## 1. Context & motivation

SN45 miners analyze news articles into a ~50-field `ArticleIntelligence` object. ~26 of those
fields are produced by **two LLM calls** (`extract_and_classify`, `reason_and_summarize`) in
`alpharidge_ai/analyzer/article_intelligence_analyzer.py`. The rest are deterministic-local
(text_stats, content_hash, embeddings, NER fusion, sector/source lookups).

**The problem with today's notion of "accuracy":** the validator
(`alpharidge_ai/analyzer/scoring.py:924-1066`) does **not** measure real-world correctness. It
re-runs its *own* LLM on the same article and gates the miner on matching that output:
- Tier 1 — 20 enums + primary-asset sentiment must match the validator's LLM **exactly** (any
  miss → 0 points).
- Tier 2 — deterministic text fields exact.
- Tier 2.5 — title/narrative embedding cosine ≥ 0.90 / 0.80.
- Tier 3 — fuzzy weighted composite ≥ 0.80.

So the system rewards **replicating the validator's specific LLM**, not being correct. To move to
real accuracy (and cheaper models), we must change the validator, API, and miner. Before any of
that, we need a **ground truth** and a **measuring stick**. This sub-project builds exactly that —
offline, no subnet changes.

## 2. Goals / non-goals

**Goals**
- A reusable framework that scores *any* candidate analyzer on **per-field accuracy vs ground
  truth**, plus **cost ($/article)** and **latency (p50/p95)**.
- A ground-truth corpus combining an LLM oracle (in-distribution, full schema) with external
  human/real gold (per-field, calibration).
- A per-field leaderboard that drives the replace/keep decisions in sub-project #2.

**Non-goals (this cycle)**
- The candidate ML models themselves (sub-project #2).
- Any validator/API/miner change (sub-project #3).
- Market-grounded impact labels (deferred to a later cycle).

## 3. Ground truth — three layers

Strongest first. Where real gold exists it is preferred *and* used to score the LLM oracle itself.

**Layer 1 — Real gold (human-labeled / market-grounded).** Maximal coverage:

| Our field(s) | Dataset | Maps to |
|---|---|---|
| overall_sentiment, sentiment_direction | Financial PhraseBank, FiQA, SemEval-2017-5 | sentiment (collapse 7→3 class) |
| assets[].direction (per-entity) | FinEntity, EFSA | entity/asset sentiment |
| event_fingerprint.event_type | EDT, EFSA | corporate event types |
| entities[] (NER) | FiNER-ORD, Financial-NER-NLP | PER/ORG/LOC |
| economic_data, numeric_claims | FiNER-139, FNXL | numeric spans/tags |
| contagion (causal) | FinCausal | causality relations |
| content_type / topic | Twitter-Financial-Topic | topic classes |

Each dataset gets an **adapter** mapping its label space → our schema subset.

**Layer 2 — LLM oracle (GLM 5.1 bulk + Opus 4.6 anchor).** Only for bespoke fields with **no**
public analog: urgency, temporal_focus, positioning_signal, credibility_flag, staleness_flag,
target_audience, market_analysis_type, forward_event_*, narrative_keywords (proprietary slugs),
chart_summary (generative), factual_confidence, source_attribution. Their trust is *inferred* from
the oracle's measured accuracy on Layer-1-covered fields.

**Layer 3 — Market-grounded impact** (DEFERRED): articles → post-publication price/vol for a real
`impact_potential` label.

## 4. Oracle pipeline & data

- **Source:** the 5 downloaded CC-NEWS WARCs (`/home/rizzo/alpharidge/ccnews-bench`).
- **Sample:** ~5,000 articles, **English-filtered** (langdetect — CC-NEWS is ~67% non-English).
  **Finance-dense sampling:** keep articles hitting the `assets.json`/sector gazetteer, plus a
  ~20% uniform-random slice for realism (so models also learn to reject non-financial noise).
- **Labeling:** reuse the existing two-call analyzer with the oracle model.
  - GLM 5.1 (OpenRouter, `z-ai/glm-5.1`, ~$0.98/$3.08 per MTok) over all ~5k. Single pass, temp 0.
  - Opus 4.6 over a ~500 anchor subset.
  - Cost ≈ $75 (GLM) + ~$50 (Opus).
- **Storage:** versioned JSONL gold + a manifest (`model`, `prompt_hash`, `schema_version`,
  `date`, `article_ids`). Large blobs gitignored; manifest tracked.
- **Calibration runs:** GLM-vs-Opus agreement on the 500 anchor (per field), and
  GLM/Opus-vs-external agreement on Layer-1 overlaps → a per-field **oracle-trust** score.

## 5. Architecture — `eval/` (top-level, `/home/rizzo/alpharidge/eval`)

Sibling to the repos; imports `alpharidge_ai` for the analyzer and `scoring.py` helpers. Five small,
independently-testable units:

1. **`Analyzer` protocol** — `analyze(article) -> ArticleIntelligence`; metadata `model`,
   `cost_per_article`. The existing `ArticleIntelligenceAnalyzer` already satisfies it; every
   candidate implements the same interface.
2. **`GoldSet`** — iterates `(article, partial_gold_labels)`. `OracleGoldSet` (CC-NEWS, GLM+Opus)
   and `ExternalGoldSet` (HF dataset + adapter).
3. **`FieldMetric` registry** — per-field scorer dispatched by field type (§6). Reuses
   `scoring.py` `_jaccard` / `_levenshtein_ratio` and the `to_canonical_string` field list so
   framework "accuracy" aligns with validator comparison semantics.
4. **`Runner`** — evaluates `Analyzer × GoldSet` → `EvalReport`: per-field accuracy, aggregate,
   cost/article (from token usage), latency p50/p95.
5. **`OracleBuilder`** — builds/serves the oracle gold and computes the calibration/trust scores.
   **`ExternalAdapters`** — one module per HF dataset (load + map labels).

### Directory layout

```
eval/
  analyzers/        # Analyzer protocol + wrappers (existing LLM analyzer, GLM, Opus, candidates later)
  gold/             # GoldSet, OracleGoldSet, ExternalGoldSet
  adapters/         # one per external dataset (phrasebank, finentity, edt, finer_ord, ...)
  metrics/          # FieldMetric registry + per-type metrics
  runner.py         # Runner -> EvalReport
  oracle/           # OracleBuilder, sampling, calibration
  data/             # gold JSONL + manifests (blobs gitignored, manifests tracked)
  reports/          # generated leaderboards
  tests/            # TDD fixtures
```

## 6. Per-field metrics

| Field type | Examples | Metric |
|---|---|---|
| Categorical enum | content_type, impact_potential, event_type | Exact match (1/0) |
| Ordinal (7-class) | overall_sentiment, *_outlook | Exact **and** ordinal-distance (off-by-one partial credit) |
| Float | sentiment_score, magnitude, confidence | MAE / within-tolerance |
| List[struct] | assets, entities, economic_data, quotes, contagion | Set-F1 on keys (ticker/name) + field-agreement on matched items (reuse `_jaccard`) |
| Free text | headline, one_liner, context_paragraph | ROUGE-L + embedding cosine |
| Keywords/fingerprint | semantic_fingerprint, narrative_keywords | Jaccard |
| Embeddings | title_embedding, narrative_embedding | Cosine |

## 7. Outputs

A per-field **leaderboard**: `candidate × field → {accuracy, cost/article, latency}`, plus
**oracle-trust** (GLM-vs-Opus, GLM/Opus-vs-external) per field. This is the artifact that decides,
field by field, "replace with ML" vs "keep the LLM" in sub-project #2.

## 8. Testing

Built TDD with tiny fixtures (no live LLM in unit tests):
- each `FieldMetric` (enums, ordinal, list-F1, text, cosine) with hand-built pred/gold pairs;
- each external adapter's label mapping;
- `Runner` aggregation + report shape;
- oracle manifest integrity.
One gated end-to-end smoke (`E2E=1`) that labels ~20 real articles via GLM and checks the pipeline.

## 9. Risks / open items

- **Domain shift:** external gold sits on different text (sentences, filings, tweets) than CC-NEWS
  → it *calibrates/validates*; the CC-NEWS oracle set is the in-distribution benchmark.
- **Label-space mismatch:** each adapter is bespoke mapping work; start with the highest-value
  datasets and expand.
- **Oracle ceiling:** GLM is the bulk labeler; the Opus anchor + external gold quantify how far we
  can trust it, but bespoke fields with no external analog inherit only *inferred* trust.
- **GLM tool-calling:** the analyzer forces `tool_choice: function`. Verify GLM 5.1 on OpenRouter
  supports forced function-calling with the analyzer's tool schemas; if not, fall back to a
  JSON-schema/structured-output prompt for the oracle path.
- **Dataset licenses:** some external sets are non-commercial (e.g., FiNER-ORD is CC BY-NC 4.0).
  Fine for offline calibration/eval, but flag per-dataset before any are used to *train* shipped
  models in sub-project #2.
- **Top-level `eval/` is not under a git repo** — decide during implementation whether to
  `git init` it or nest it under a tracked repo.

## 10. Branch alignment (prerequisite for implementation)

All repos target `article-intelligence-v2`. `alpharidge-ai` is still on
`feat/verifiable-validator-points`; per decision, merge `feat/verifiable-validator-points` →
`article-intelligence-v2` (mirroring the API) before building, so V2 has both the analyzer and the
verifiable-points work. This spec lands on V2 via that merge.
