# Plan 2 — Call 2 Replacement (shrink to summary-only) — Design

**Date:** 2026-06-16
**Status:** Approved (pending spec review)
**Predecessor:** `2026-06-16-ml-replacement-benchmark-design.md` (Plan 1, single-value classification, done)
**Repo:** `/home/rizzo/alpharidge/eval` (own git); imports `alpharidge-ai` analyzer + data.

## Goal

Shrink the analyzer's second LLM call (`reason_and_summarize`) by moving its
reasoning/extraction fields onto non-LLM models, leaving only free-text
`chart_summary` on a slim, summary-only LLM call. Validate every replacement
against the GLM 5.1 oracle **and**, where it exists, **human gold** (the first
trustworthy yardstick for these fields). This is a benchmark/validation effort in
the `eval` repo; it does **not** edit the production analyzer (that integration is
a later, separately-reviewed step).

## Call 2 today vs. after

`reason_and_summarize` currently emits four field-groups:

| Field-group | Today | After Plan 2 |
|---|---|---|
| `contagion_links` | LLM reasoning | **GLiREL zero-shot RE + dependency_graph prior + FinBERT direction** |
| `narrative_keywords` | LLM | **KeyBERT (MiniLM) / YAKE baseline** |
| per-asset sentiment (`assets[].direction`, outlooks) | LLM | **FinBERT aspect** over asset-mention sentences |
| `chart_summary.*` (headline/one_liner/context) | LLM | **stays on a slim summary-only LLM call** |

Removing the first three shrinks Call 2's prompt (no longer asks for them) and its
output (drops verbose `causal_driver`/contagion `reasoning` strings), so even
though a summary call remains, the per-article cost falls. The exact saving is
**measured**, not assumed (see Token measurement).

## Decision framing (important)

The Plan-1 bar was "fidelity to GLM ≥ τ." That bar is **not** appropriate for these
fields, because GLM is itself unreliable here: GLM-vs-Opus agreement was **0.26**
for contagion and **0.53** for keywords. So Plan 2 separates two things:

- **The action is decided by directive** (the user's call): contagion, keywords,
  and per-asset sentiment are **moved off the LLM**. We are not gating that on a GLM
  threshold.
- **The benchmark characterizes quality** on two axes: (a) divergence from GLM
  (continuity/characterization only), and (b) **agreement with human gold** where a
  dataset maps to the field — the real quality signal. A field with low GLM
  agreement but high human-gold agreement is a *win*, not a regression.

## Components (all in `eval/`, reusing the Plan-1 framework)

Predictors return the full field value (a list or per-asset map) and are scored by
the **existing** metrics (`contagion_links`→ListStruct, `narrative_keywords`→
KeywordJaccard, `assets`→ListStruct on `direction`). Like regressors they carry no
per-item confidence, so they are **replace-or-characterize**, not gated.

### 1. Contagion: learned relation extraction
`eval/predictors/contagion.py` — `GlirelContagion`:
- Detect candidate nodes: spaCy entities + the asset gazetteer (reuse
  `alpharidge_ai` asset extractor) → resolved tickers/entities with char spans.
- **GLiREL** (zero-shot, `pip install glirel`, CPU) scores relations between node
  pairs against a fixed finance relation-label set
  (`supply_chain`, `competitor`, `regulatory_spillover`, `capital_flow`,
  `ecosystem_dependency`, `collateral`, `macro_sensitivity`, `correlation`,
  `protocol_dependency`) → mapped 1:1 to the `ContagionMechanism` enum.
- `direction` from FinBERT sentiment on the target's mention sentence; `strength`/
  `confidence` from the GLiREL score.
- **dependency_graph.json fused in as a high-precision prior** — curated structural
  edges (ETH→stETH …) unioned with the text-derived edges, so we keep the links the
  graph knows and add the event-specific ones the graph can't.
- A `GraphOnlyContagion` baseline (today's deterministic behavior) is kept as the
  ladder's floor rung for comparison.

### 2. Keywords
`eval/predictors/keywords.py` — `KeyBERTKeywords` (MiniLM-backed, top-3 keyphrases)
with a `YakeKeywords` no-torch baseline. Scored vs GLM `narrative_keywords` by the
existing KeywordJaccard metric.

### 3. Per-asset sentiment
`eval/predictors/aspect_sentiment.py` — `FinBERTAspect`: for each detected asset,
run FinBERT (already loaded by the analyzer) over the sentences that mention it,
aggregate to a `direction` on the 7-class `Sentiment` scale (plus `magnitude`/
`confidence` from FinBERT scores). Scored vs GLM `assets[].direction`.

### 4. Human-gold adapters
`eval/hf/` → per-field `GoldSet`s, cached locally, gated behind a network flag:
- `refind.py` — **REFinD** (financial relation extraction, human gold) → validates
  the GLiREL relation component (entity-pair → relation accuracy).
- `finentity.py` — **FinEntity** (entity-level sentiment, human gold) → validates
  `FinBERTAspect` per-asset direction.
- `phrasebank.py`, `fiqa.py` — **Financial PhraseBank / FiQA** (sentence sentiment,
  human gold) → sentiment cross-check (also re-validates Plan-1's `overall_sentiment`).

### 5. Report + cost
- Extend `eval/bench/report.py`: add a **`vs_human_gold`** column (populated where
  an adapter maps to the field) beside the GLM fidelity column.
- `eval/bench/cost.py` — Call-2 cost projection from measured tokens: full Call-2
  cost vs summary-only Call-2 cost → $/article saved.

### 6. Token measurement (makes the cost claim real)
`eval/oracle/measure_tokens.py` + a CLI step: on a ~200-article sample, run the LLM
with (a) the current full Call-2 prompt and (b) a summary-only prompt; record
input/output tokens for each; feed into `cost.py`. Gated (uses the LLM/OpenRouter).
Replaces the earlier illustrative 40–55% with a measured figure.

## Architecture reuse & new deps

Reuses `Article`, `GoldSet`/`GoldItem`, `FIELD_METRICS`, `runner`/`score`/`report`,
and the Plan-1 ladder mechanism (new predictors are just new rungs). New
dependencies: `glirel`, `keybert`, `yake` (FinBERT/MiniLM/spaCy/dependency_graph
already present). REFinD/FinEntity/PhraseBank/FiQA pulled via `datasets`.

## Testing

Same discipline as Plan 1: synthetic-fixture unit tests for every predictor,
adapter, the relation→mechanism mapping, the report column, and the cost projector —
**no network or model loads in unit tests** (inject fakes). Gated integration tests
exercise the real GLiREL/KeyBERT/FinBERT and the HF pulls behind an env flag.

## Out of scope (later steps)

- Editing the production analyzer to actually run the slim Call 2 + non-LLM
  predictors (integration step; touches production; separately reviewed).
- Stock-relation GNNs / price-correlation contagion enrichment (Plan 3 territory).
- Fine-tuning a dedicated RE model on REFinD — only if GLiREL zero-shot
  underperforms the human gold on specific relations (evidence-driven, later).

## Risks

- **GLiREL relation labels ≠ our mechanism enum cleanly.** Mitigm: an explicit,
  tested mapping table; unmapped relations drop rather than mis-tag.
- **REFinD's schema is corporate relations, not market mechanisms.** It validates
  the RE *machinery* (can the model find the right entity-pair relation), not every
  contagion mechanism 1:1. We report it as RE-component accuracy, not as full
  contagion truth, and say so.
- **GLiREL adds a model load + per-pair scoring cost.** It is CPU-runnable but
  heavier than a dict lookup; the `GraphOnlyContagion` baseline bounds the trade.
- **No human gold for `narrative_keywords`.** It is characterized vs GLM only;
  flagged as such.
