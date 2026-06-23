# ML Replacement Benchmark — Design

**Date:** 2026-06-16
**Status:** Approved (pending spec review)
**Predecessor:** `2026-06-15-oracle-eval-framework-design.md` (oracle + eval framework, complete)
**Repo:** `/home/rizzo/alpharidge/eval` (own git repo); imports `alpharidge-ai` analyzer.

## Goal

Replace as many of the analyzer's LLM-determined `ArticleIntelligence` fields as
possible with cheaper non-LLM techniques (ML models, deterministic rules), while
**faithfully reproducing the GLM 5.1 oracle**. We are not trying to beat GLM or
Opus — GLM is the production teacher, and a candidate "passes" when it reproduces
GLM well enough on held-out data. Cost is the priority; accuracy (= GLM fidelity)
is the bar; speed is least important. Deployment target: **2 CPUs, no GPU.**

## What the LLM currently determines

Two monolithic LLM calls (see `article_intelligence_analyzer.py`):

- **Call 1 `extract_and_classify`** — 15 single-value enums + `overall_sentiment`
  (7-way ordinal) + `overall_sentiment_score` (numeric) + `sentiment_direction`;
  plus structured extraction: `economic_data`, `numeric_claims`, `quotes`,
  `event_fingerprint.*`, `staleness_flag`, `forward_event_*`.
- **Call 2 `reason_and_summarize`** — per-asset sentiment (`assets[].direction`,
  `magnitude`, `confidence`, three outlooks, `causal_driver`), `contagion_links`,
  `chart_summary.*`, `narrative_keywords`.

Everything else (entities, text_stats, embeddings, assets ticker/name/class,
topic_signature, market_session, inferred_impacts, source) is already
deterministic and out of scope.

## Decision bar: fidelity to GLM

GLM 5.1 is both the **training teacher** and the **evaluation reference**. We split
the 5,000 GLM-labeled articles **three ways by article id** — train / calib / test
(default 70/15/15 with a fixed seed). The candidate is *fit* on train; any
confidence gate is *selected* on calib; all reported numbers are measured on the
held-out **test** split. This separation matters: selecting the gate threshold on
the same set it is reported on inflates coverage (in-sample tuning). Scoring uses
the existing `eval/metrics` (categorical exact-match, ordinal sentiment ladder,
numeric tolerance, list/struct set-F1, keyword Jaccard, text ROUGE/embedding).

Per field, let **f = candidate-vs-GLM agreement on the TEST split**.

- **calib f ≥ τ_field → REPLACE.** Default `τ` = 0.90 for categorical/boolean,
  metric-appropriate for ordinal/numeric (configurable per field). The candidate
  reproduces GLM closely enough to stand in for it; reported fidelity is the test value.
- **else → confidence-gated hybrid.** Select on **calib** the confidence threshold
  `c` with maximum coverage whose retained-subset mean reaches τ. Then apply that
  fixed `c` to **test**: report `coverage` (fraction auto-handled = projected
  LLM-call reduction) and `gate_acc` (accuracy above `c`). The HYBRID only stands
  if the gate **still clears τ out-of-sample** (`gate_acc ≥ τ`) and `coverage ≥ 20%`.
- **otherwise → KEEP** (the gate did not generalize, or isn't worth the machinery).

Regressor fields emit a constant per-item confidence, so the gate sweep cannot
fire — they are **REPLACE-or-KEEP only**. Classifier confidences are raw
`predict_proba` maxima (uncalibrated); the out-of-sample `gate_acc ≥ τ` check is
what keeps a HYBRID honest despite that. Probability calibration is a Plan-2 lever.

**Secondary diagnostics (informational only, never gate a decision):** alongside
GLM fidelity, the report also shows each candidate's agreement with the **Opus
anchor (500)** and with **HF human gold** where it maps. These flag cases where GLM
itself is weak (e.g. a candidate matches GLM but both disagree with humans), but
they do not change the replace/keep decision — per the goal, GLM is the target.

## Modeling approach: tiered candidate ladder per field

For each replaceable field the framework evaluates a ladder of candidates and
**auto-selects the cheapest one that clears the bar**:

1. **Deterministic / lexical** — rules, gazetteers, regex, existing data files
   (e.g. `contagion_links` from `dependency_graph.json`; keyword heuristics).
2. **Off-the-shelf, no training** — FinBERT (sentiment), KeyBERT/YAKE
   (`narrative_keywords`), zero-shot NLI (enums) as baselines.
3. **Distilled embed + linear head** — embed each article once with the MiniLM +
   FinBERT models already loaded in the pipeline (CPU), then a logistic-regression
   head per field trained on the GLM train split. Near-free inference.
4. **Distilled embed + LightGBM head** — same features, gradient-boosted head for
   the harder multi-class fields.
5. **Fine-tuned small transformer (DeBERTa-v3-small)** — escalation tier, used
   **only** for fields where tiers 1–4 miss the bar. May require GPU to train; the
   trained model still runs on CPU at inference.

Cheapest-passing wins, so easy fields settle at tier 1–2 and only the hard ones
pay for tiers 4–5. Reuses the embeddings already computed in the pipeline, so the
marginal inference cost of the distilled heads is a matrix multiply.

### Field → primary technique

| Group | Fields | Primary technique |
|---|---|---|
| Classification (15 enums) | content_type, market_analysis_type, impact_potential, technical_quality, urgency, temporal_focus, sentiment_direction, factual_confidence, source_attribution_type, positioning_signal, primary_geo, target_audience, credibility_flag, staleness_flag, forward_event_type | distilled embed + linear/LightGBM (tiers 3–4) |
| Sentiment | overall_sentiment, overall_sentiment_score | FinBERT ordinal + regressor (tier 2), distilled fallback |
| Per-asset sentiment | assets[].direction, magnitude, confidence, outlooks, causal_driver | aspect FinBERT / distilled (tiers 2–3) |
| Keywords | narrative_keywords | KeyBERT/YAKE (tier 2) |
| Structured extraction | economic_data, numeric_claims, quotes | FiNER NER + regex (tiers 1–2) |
| Contagion | contagion_links | deterministic `dependency_graph.json` (tier 1; drop LLM) |
| Generative | chart_summary.headline/one_liner/context_paragraph/what_changed | distilled small summarizer (DistilBART/T5), else keep LLM |

## Call-level economics

The two LLM calls are monolithic, so **cost only drops when a whole call is
eliminated or downsized**. The report projects $/article before vs after under
three levers:

- **Eliminate Call 2** — target outcome: contagion → deterministic, per-asset
  sentiment → FinBERT, keywords → KeyBERT, chart_summary → distilled summarizer.
- **Shrink / downsize Call 1** — keep only fields that fail the bar; run them on a
  cheaper or distilled model. Eliminate entirely if all clear.
- **Hybrid fallback rate** — for hybrid fields, the fraction of articles still
  routed to an LLM, aggregated to an expected calls-per-article figure.

## Architecture (all in the existing `eval/` package)

- `eval/hf/` — HuggingFace adapters → per-field `GoldSet`: Financial PhraseBank &
  FiQA (sentiment), FinEntity (entity sentiment), FiNER-ORD (NER), EDT (events).
  Downloaded and cached locally; used for the secondary human-gold diagnostic.
- `eval/predictors/` — `Predictor` protocol (`fit(train)?`, `predict(article)`,
  `confidence(article)`); implementations `deterministic.py`, `offtheshelf.py`
  (FinBERT/KeyBERT/NLI), `distilled.py` (embed-once + linear/LightGBM), optional
  `finetune.py`.
- `eval/distill/` — feature builder (reuse pipeline MiniLM/FinBERT embeddings +
  text_stats), per-field trainer over the GLM **train** split, persists models to
  `eval/models/` (gitignored).
- `eval/bench/` — runner: per field × candidate → score vs **held-out GLM** (and
  secondary Opus/HF), apply the decision policy, compute confidence threshold,
  coverage, and projected cost; emit a per-field decision report.
- `eval/cli.py` — new subcommands: `hf-pull`, `split` (train/test), `distill`,
  `bench`.

### Outputs

- `eval/reports/replacement_decisions.json` + a rendered markdown table:
  per field — best candidate, tier, GLM-fidelity, decision
  (REPLACE / HYBRID@coverage / KEEP), secondary Opus/HF agreement, projected cost.
- Persisted trained models under `eval/models/`.

## Testing

TDD throughout. Unit tests use synthetic fixtures and **never** hit the network or
download models — adapters, predictors, the train/test splitter, the decision
policy, and the cost projector are all tested with injected data. Real model
downloads and HF pulls run only in **gated integration tests** behind an env flag,
mirroring the existing oracle E2E pattern.

## Risks

- **Distilling from GLM caps fidelity at "reproduce GLM," not "be correct."** This
  is intentional per the goal. The secondary human-gold diagnostic exists precisely
  to flag where GLM (and therefore the distilled model) diverges from human labels,
  so a future cycle can revisit those fields — but it does not block replacement now.
- **Thin support for rare classes** (e.g. ~44 "critical" impacts in 5k) weakens
  both training and the held-out estimate. The report flags low-support fields;
  GLM can cheaply label more articles if a field is starved.
- **`chart_summary`** is the hardest to replace; if the distilled summarizer trails,
  it is the one field that may justify keeping a (smaller/cheaper) LLM call.
- **Train/test split leakage** — split by article id with a fixed seed before any
  feature extraction; the held-out split is touched only at evaluation time.
