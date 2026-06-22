# Article-Intelligence Quality Overhaul — Design

**Date:** 2026-06-17
**Branch:** `article-intelligence-v2` (alpharidge-ai)
**Status:** Draft for approval

## Problem

The miner's article-intelligence analyzer produces four classes of low-quality output. All four
are confirmed in code, not speculation:

| # | Symptom | Confirmed root cause (file:line) |
|---|---------|----------------------------------|
| 1 | `narrative_keywords` hallucinated — every article gets generic crypto/AI slugs (e.g. `institutional-crypto-adoption`) even for non-crypto stories | Call 2 hands the LLM the full 38-slug taxonomy as a "use when applicable" menu with no relevance gate and no abstention pressure; `narrative_keywords` is in the tool's `required` list. `article_intelligence_analyzer.py:184-201, 311` + `data/narratives.json` |
| 2 | Uppercase-acronym ticker false positives — "LEO" (Low Earth Orbit), and latent risk for ALL/CAR/GOLD/IT/ON | `asset_extractor.py:49,161` — `ambiguous = len(cs_id) <= 2`, so any 3+ char case-sensitive identifier auto-gets `strong_evidence=True` (`:260`) and bypasses the 17k-word common-word gate, which `_kept` only applies to the non-strong branch (`:307-312`) |
| 3 | Per-asset `direction` always `neutral` — even SpaceX (+19%) and bullish dividend articles | `_finbert_asset_sentiments` selects an asset's sentences via `tk.upper() in sentence` — matching the **ticker symbol** against prose that says "Tesla"/"SpaceX". Match set is empty → hard-coded `("neutral", 0.5)` fallback. `article_intelligence_analyzer.py:611-644` (bug at `:627,632-633`) |
| 4 | Entity noise — duplicates not deduped (Medtronic ×4); NER slips ("Anthropic principle", "Elon Musk-backed" as person) | `ner_fusion.py:275-282` appends resolved entities with no canonical-name dedup (only assets dedupe by ticker); `_build_entities_from_ner` only truncates to 15 (`article_intelligence_analyzer.py:521`); `entity_filter.py` dedups by character-offset overlap only and its blocklist (`:90-101`) misses adjectival/appositive fragments |

## Goals & constraints

- **Quality first**, with permission to add LLM scope *only where it clearly wins* (user decision).
- **Determinism matters for the validator.** The validator re-runs the analyzer and scores miners by
  consensus. Assets/entities are deliberately deterministic so independent miners agree. Therefore the
  default for P2/P3/P4 is deterministic fixes; an LLM field there would *reduce* reproducibility.
- **2-CPU miner box.** No heavy new model unless it benchmarks as a clear win (see `ccnews-bench`).
- **No-commit constraint.** Implement + benchmark; leave committing to the user (per project memory).

## Ground-truth findings (verified this session, not assumed)

- Gold set `eval/data/gold_z-ai_glm-5.1.jsonl` (5,000 rows): **37% have empty `narrative_keywords`**;
  top slugs are macro/geo (`risk-sentiment-cycle` ×2121, `supply-chain-reshoring`, `middle-east-tensions`),
  the *opposite* of production's crypto/AI spam. → **Abstention is the single highest-value behavior.**
- `KeywordJaccardMetric` scores ∅-vs-∅ as 1.0, so correct abstention is directly rewarded on that 37%.
- Gold uses 67 distinct slugs but **99% of keyword-instances are coverable by the existing 38-slug
  taxonomy** (the 30 extra slugs are all ≤3 occurrences). → No taxonomy expansion needed; recall ceiling ~99%.
- The eval repo already has injectable predictors we extend rather than rebuild:
  `eval/predictors/{keywords,aspect_sentiment,distilled,contagion}.py`, plus characterization reports
  (`eval/reports/{keywords,aspect_sentiment}_characterization.json`). `FinBERTAspect` already scores
  **0.815** on `assets`.

## Design — per feature

For each: **chosen approach** (what we ship) + **benchmarked alternative** (built behind a flag/predictor
so data decides before we commit to it).

### P1 — narrative_keywords: retrieve-then-verify

**Chosen:** Replace the open 38-slug menu with a two-stage select-and-abstain.

1. **Deterministic shortlist (recall).** At init, embed each narrative's `keywords` list with the
   already-resident `all-MiniLM-L6-v2` to form 38 centroids. At runtime embed the article
   (title + one_liner + context_paragraph), cosine vs centroids, take candidates above a low recall
   threshold τ_lo (≤6 slugs).
2. **Precision + abstention.** Because Call 2 already runs for summaries, fold keyword selection into it
   but constrain the tool: present **only the shortlisted slugs**, require a verbatim `evidence_span`
   per emitted slug, and instruct "return `[]` if none is clearly supported — most articles have none."
   Reject any slug whose evidence span is not a substring of the input (anti-confabulation).
   Drop `narrative_keywords` from `required`.

**Benchmarked alternative:** pure deterministic embedding gate at a calibrated high threshold τ_hi
(no LLM at all). If it matches the hybrid within noise on gold, **prefer it** — it's cheaper, fully
deterministic (validator-friendly), and advances the kill-Call-2 direction. Implemented as a
predictor in `eval/predictors/keywords.py` so the choice is data-driven.

**Why not free-text keyphrase extraction:** YAKE/KeyBERT already score 0.0 — gold is a *closed
controlled vocabulary*, so any fix must select from `narratives.json` slugs, never generate phrases.

### P2 — ticker false positives: fix the ambiguity gate (deterministic)

**Chosen:** Make Phase 2 consistent with Phases 3-4. In `AssetExtractor.__init__`:

```
ambiguous = len(cs_id) <= 2 or _is_very_common_word(cs_id) or cs_id in _ACRONYM_BLOCKLIST
```

`_is_very_common_word` (already loaded, 17k words) catches ALL/CAR/GOLD/IT/ON. `_ACRONYM_BLOCKLIST`
(a small curated frozenset) catches uppercase-but-not-dictionary collisions (LEO, SUI, ISO country
codes, common abbreviations). Ambiguous case-sensitive hits then require `_ctx()` corroboration and
pass the common-word check in `_kept`, exactly like the other phases.

No LLM (would break determinism and not clearly win). No new model. Zero added latency.

### P3 — per-asset direction: match by surface form, fall back to article sentiment (deterministic)

**Chosen:** Replace ticker-symbol substring matching with surface-form matching.

1. Build a per-ticker surface-form set: canonical name + aliases + `AssetMatch.evidence_spans`
   (already computed upstream) + ticker. Thread these through `resolved_assets` (widen
   `ResolvedEntity.text`, currently only `evidence_spans[0]`, `ner_fusion.py:248`).
2. Select mention sentences by **word-boundary** match on any surface form (fixes "Tesla"→TSLA),
   then run the resident FinBERT vote over those sentences (reuse `sentence_sentiments`).
3. When still no mention, fall back to the article-level `overall_sentiment` direction — not hard
   `neutral`.

Reuses FinBERT already loaded and already invoked per sentence → near-zero added CPU, deterministic.

**Benchmarked alternative:** a small aspect-based sentiment (ABSA) model
(`yangheng/deberta-v3-base-absa-v1.1`) keyed on the surface form as the aspect term, for correct
multi-asset attribution within one sentence. Adopt **only if** it beats surface-form+FinBERT on gold
by a meaningful margin and its latency is acceptable on the 2-CPU box (it's ~184M params → hundreds of
ms/article; measure before adopting). Wired through `eval/predictors/aspect_sentiment.py`.

**Explicitly rejected:** moving per-asset direction into Call 1. It adds LLM scope and breaks the
deterministic-consensus property the validator relies on for assets.

### P4 — entity noise: canonical dedup + fragment blocklist (deterministic)

Two independent diffs:

1. **Canonical-name dedup** in the `ner_fusion.py:275-282` resolve loop: key non-asset entities by
   `(canonical_name.lower(), entity_type)`; on collision keep the highest-confidence/longest
   representative and merge `role`/`sentiment_toward` (prefer non-null). This aligns output to the
   `entities` metric key (`name.lower()`).
2. **Fragment/adjectival slip removal** in `entity_filter._blocked`: drop or trim hyphenated modifiers
   ("…-backed/-led/-owned/-based"), and appositive tails ("X principle", "X effect"), using the
   already-parsed spaCy POS tags where helpful.

No new models. Negligible CPU.

## Benchmark methodology

Two layers, run before/after for every change:

1. **Regression on existing gold** — `eval` run/characterize over `gold_z-ai_glm-5.1.jsonl` (5,000) and
   `gold_anthropic_claude-opus-4-6.jsonl` (overfit check). Per-field metrics already exist:
   `narrative_keywords` → `KeywordJaccardMetric`; `assets` → `ListStructMetric` (F1 for ticker FPs,
   `field_agreement` for direction); `entities` → `ListStructMetric` by `name.lower()`.
   Gate: target field improves, **no regression on other fields**.
2. **Hand-curated targeted set (~25 articles)** — ground truth for the specific failure modes, since the
   LLM-labeled gold may share blind spots. Includes: the SpaceX IPO (+19%), dividend-stock and VC-interview
   and AI-shopping articles (must yield zero crypto narrative_keywords; SpaceX direction bullish);
   the LEO/GOLD collision cases (must not emit LEO/GOLD as tickers); a real `$LEO`/"Barrick Gold" positive
   (must still emit); the Medtronic-×4 dedup case; "Anthropic principle"/"Elon Musk-backed" slips.
   Stored as `eval/data/handcurated_overhaul.jsonl` + pytest fixtures under `alpharidge-ai/tests/`.

**Reference ceilings:** compute inter-labeler agreement (GLM-5.1 vs Opus) on `narrative_keywords` and
`assets` to contextualize "how good is achievable" rather than chasing 1.0.

**Calibration:** τ_lo/τ_hi for P1 swept on a calibration split; pick the empty-rate matching gold (~37%).

## Determinism summary

| Feature | Deterministic? | Notes |
|---------|----------------|-------|
| P2 tickers | Yes | pure code |
| P3 direction | Yes | FinBERT is CPU, no sampling |
| P4 entities | Yes | pure code |
| P1 keywords (hybrid) | Partial (Call 2 LLM, temp 0) | Tier-3 scoring, 0.05 weight, Jaccard consensus tolerates it |
| P1 keywords (embedding-only alt) | Yes | preferred if it benchmarks comparably |

## Rollout

- All work on branch `article-intelligence-v2`.
- Order: P2 → P4 → P3 → P1 (cheapest/most-contained first; P1 last as it has the LLM/alternative fork).
- Each change is its own diff with its own before/after benchmark numbers.
- **Do not commit.** Hand back with benchmark deltas for user review and commit.

## Risks / open questions

- P1 hybrid keeps Call 2 alive, which is in tension with the strategic kill-Call-2 effort. Mitigation:
  build the deterministic embedding-only path in parallel and prefer it if benchmarks are comparable.
- P2 blocklist needs occasional curation as the asset registry grows.
- P3 surface-form name matching can over-collect (e.g. "Apple" in a non-Apple sentence); mitigated by
  word-boundary matching and benchmarked against the ABSA alternative.
- ABSA / NLI verifier models would add real latency on 2 CPUs; gated behind benchmark evidence.
