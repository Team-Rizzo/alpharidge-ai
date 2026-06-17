# Article-Intelligence Quality Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate four quality defects in the miner's article-intelligence output — hallucinated narrative_keywords, uppercase-acronym ticker false positives, always-neutral per-asset direction, and entity noise — and prove each fix with before/after benchmarks.

**Architecture:** Three of four fixes (tickers, per-asset direction, entities) live in the deterministic Stage-1/assembly path (`asset_extractor.py`, `ner_fusion.py`, `article_intelligence_analyzer.py` builders) and need NO LLM — so they are benchmarked with a no-LLM harness over the existing 5k gold set. The fourth (narrative_keywords) is replaced with an embedding-based slug selector benchmarked through the existing `eval characterize` predictor path, with an optional LLM-shortlist precision layer benchmarked on a small sample.

**Tech Stack:** Python, `sentence-transformers` (all-MiniLM-L6-v2, already resident), FinBERT (already resident), spaCy/GLiNER/Flair NER (already resident), the `eval/` benchmark package (metrics, gold sets, predictor registry).

## Global Constraints

- **DO NOT COMMIT.** Per project policy the user reviews and commits. Each task ends by leaving changes unstaged/staged and reporting a diff summary + benchmark numbers. Replace any "commit" step with "report results".
- **Determinism across the validator consensus boundary is mandatory** for `assets`, `entities`, `contagion_links`, and per-asset sentiment. No LLM, no `random`, no wall-clock, no dict-iteration-order dependence in those paths. (`article_intelligence_analyzer.py:296-301` documents this.)
- **2-CPU miner box.** No new heavy model may be added to the production path unless a benchmark shows a clear win AND latency is acceptable (`ccnews-bench/bench_local.py`). New models for P1/P3 alternatives live in `eval/predictors/` only until proven.
- **Branch:** `article-intelligence-v2` in `/home/rizzo/talisman/talisman-ai`. The eval package is at `/home/rizzo/talisman/eval` (separate import root).
- **Existing public behavior must not regress:** a task that improves its target field must not lower any other field's metric on the gold set.

---

### Task 1: Benchmark harness + hand-curated targeted set

**Files:**
- Create: `/home/rizzo/talisman/eval/scripts/overhaul_bench.py`
- Create: `/home/rizzo/talisman/talisman-ai/tests/fixtures/handcurated_overhaul.jsonl`
- Create: `/home/rizzo/talisman/talisman-ai/tests/test_overhaul_regressions.py`

**Interfaces:**
- Produces: `overhaul_bench.py` CLI: `python scripts/overhaul_bench.py --gold <path> --limit N --fields assets entities` — runs ONLY the deterministic path (`NERFusionEngine.extract_and_resolve` + `_build_assets` + `_finbert_asset_sentiments` + `_build_entities_from_ner`) over each gold row's `article`, scores the named fields against `row["labels"]` using `eval.metrics.fields.FIELD_METRICS`, and prints mean F1 / field_agreement. No LLM calls.
- Produces: `handcurated_overhaul.jsonl` — one JSON object per line: `{"article": {"id","title","content","url","source"}, "expect": {...}}` where `expect` encodes the targeted assertions (see Step 3).

- [ ] **Step 1: Write the failing test for the harness**

```python
# talisman-ai/tests/test_overhaul_regressions.py
import json, os, subprocess, sys, pytest
EVAL = "/home/rizzo/talisman/eval"
def test_overhaul_bench_runs_on_smoke(tmp_path):
    # 3-row smoke gold built from the existing sample
    src = "/home/rizzo/talisman/eval/eval/data/gold_z-ai_glm-5.1.jsonl"
    smoke = tmp_path / "smoke.jsonl"
    with open(src) as f, open(smoke, "w") as o:
        for i, line in zip(range(3), f):
            o.write(line)
    r = subprocess.run([sys.executable, "scripts/overhaul_bench.py",
                        "--gold", str(smoke), "--limit", "3", "--fields", "assets", "entities"],
                       cwd=EVAL, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "assets" in r.stdout and "entities" in r.stdout
```

- [ ] **Step 2: Run it, verify it fails**

Run: `cd /home/rizzo/talisman/talisman-ai && python -m pytest tests/test_overhaul_regressions.py::test_overhaul_bench_runs_on_smoke -q`
Expected: FAIL (script does not exist).

- [ ] **Step 3: Implement the harness**

`overhaul_bench.py` must:
1. Add `/home/rizzo/talisman/talisman-ai` to `sys.path`, import `ArticleIntelligenceAnalyzer` and `NERFusionEngine`.
2. Construct the analyzer once (it loads NER models). For each gold row: call `ner = analyzer.ner_engine.extract_and_resolve(title, content)`, then `ner_tickers = [e.ticker for e in ner.resolved_assets if e.ticker]`, `sents = analyzer._finbert_asset_sentiments(ner_tickers, ner)`, `assets = analyzer._build_assets(ner, sents)`, `entities = analyzer._build_entities_from_ner(ner)`.
3. Serialize each to plain dicts matching gold shape: assets → `[{"ticker":a.ticker, "direction":a.direction.value, ...}]`; entities → `[{"name":e.name, ...}]` (use the pydantic `.model_dump()` / `.dict()` then ensure enum values are `.value` strings, matching how gold labels are stored — inspect one gold row to confirm serialization).
4. Score with `from eval.metrics.fields import FIELD_METRICS`; for each requested field compute `metric.score(pred_value, row["labels"].get(field)).value` and also pull `.details` for f1/field_agreement; print mean per field.
5. Support `--limit`, `--fields`, `--gold`.

Confirm gold serialization first: `python -c "import json;r=json.loads(open('eval/eval/data/gold_z-ai_glm-5.1.jsonl').readline());print(r['labels']['assets'][:1], r['labels']['entities'][:1])"` and mirror exactly.

- [ ] **Step 4: Build the hand-curated set (~25 articles)**

Populate `handcurated_overhaul.jsonl` with real article texts covering every failure mode. Draft article bodies from `eval/eval/data/sample_5k.jsonl` where possible (search for matching stories) or write faithful representative texts. Required cases (each line `{"article":{...}, "expect":{...}}`):
- **Non-crypto → zero crypto keywords (×6):** a VC interview, dividend-stock article, SpaceX IPO, AI-shopping, plus 2 macro/geo. `expect.narrative_keywords_excludes = ["institutional-crypto-adoption","bitcoin-etf-flows","defi-revival","ethereum-scaling"]`.
- **SpaceX bullish (×1):** the IPO/+19% article. `expect.asset_direction = {}` only if a real ticker is present; primarily `expect.narrative_keywords_excludes` crypto. (SpaceX has no public ticker, so use this for keyword + entity assertions.)
- **Ticker collisions (×4):** "Low Earth Orbit (LEO)" with financial language nearby; an "ALL"/"CAR"/"GOLD"/"IT" sentence in non-asset context. `expect.assets_excludes = ["LEO","ALL","CAR","GOLD","IT"]`.
- **Real ticker positives (×4):** "$LEO" cashtag; "Barrick Gold raised its dividend … gold prices rallied"; a Tesla earnings story; an Nvidia story. `expect.assets_includes = ["LEO"]`, `["GOLD"]`(Barrick=GOLD), `["TSLA"]`, `["NVDA"]` respectively. For the Tesla one also `expect.asset_direction = {"TSLA": "bullish"}`.
- **Per-asset direction (×3):** a clearly bullish Apple article (`{"AAPL":"bullish"}`), a clearly bearish one, a mixed one.
- **Entity dedup (×2):** an article repeating "Medtronic" 4× (`expect.entity_no_duplicates = true`), one with "Anthropic principle" and "Elon Musk-backed" (`expect.entity_excludes = ["Anthropic principle","Elon Musk-backed"]`, `expect.entity_includes = ["Anthropic","Elon Musk"]`).

- [ ] **Step 5: Make the smoke test pass; report (do not commit)**

Run: `cd /home/rizzo/talisman/talisman-ai && python -m pytest tests/test_overhaul_regressions.py::test_overhaul_bench_runs_on_smoke -q`
Expected: PASS.
Then capture a **baseline**: `cd /home/rizzo/talisman/eval && python scripts/overhaul_bench.py --gold eval/data/gold_z-ai_glm-5.1.jsonl --limit 500 --fields assets entities` and record the assets/entities numbers in the task report. Do NOT commit.

---

### Task 2: P2 — fix ticker uppercase-acronym false positives

**Files:**
- Modify: `/home/rizzo/talisman/talisman-ai/talisman_ai/analyzer/asset_extractor.py:49,93-103,160-162,298-312`
- Test: `/home/rizzo/talisman/talisman-ai/tests/test_asset_extractor_gate.py` (create)

**Interfaces:**
- Consumes: `AssetExtractor.extract_assets(title, body, language)` (unchanged signature).
- Produces: no API change; behavior change only.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_asset_extractor_gate.py
from talisman_ai.analyzer.asset_extractor import AssetExtractor
ax = AssetExtractor()
def tickers(title, body): return {m.ticker for m in ax.extract_assets(title, body)}
def test_leo_low_earth_orbit_not_a_ticker():
    t = tickers("SpaceX launch", "The satellite reached low earth orbit (LEO) as shares of the firm rallied 19% on the IPO.")
    assert "LEO" not in t
def test_real_leo_cashtag_kept():
    assert "LEO" in tickers("Crypto update", "Bitfinex token $LEO surged 12% amid heavy trading volume.")
def test_common_word_gold_not_ticker_in_prose():
    assert "GOLD" not in tickers("Olympics", "She won the gold medal after a record run; the crowd cheered.")
def test_barrick_gold_company_kept():
    # Barrick Gold's distinctive name should corroborate GOLD
    assert "GOLD" in tickers("Earnings", "Barrick Gold raised its dividend as gold prices rallied and the miner beat earnings.")
```

- [ ] **Step 2: Run, verify failure**

Run: `cd talisman-ai && python -m pytest tests/test_asset_extractor_gate.py -q`
Expected: `test_leo_low_earth_orbit_not_a_ticker` and `test_common_word_gold_not_ticker_in_prose` FAIL (these tickers leak).

- [ ] **Step 3: Implement the gate fix**

Add a curated acronym blocklist near `_COMMON_WORDS` (line ~90):

```python
# Uppercase tokens that are valid tickers but collide with common non-financial
# acronyms; treated as ambiguous AND non-corroborating so they need a cashtag,
# a distinctive name, or another evidence span to be emitted.
_ACRONYM_BLOCKLIST = frozenset({
    "LEO", "SUI", "ICP", "AI", "IT", "ON", "OP", "ATOM", "GAS",
    "USD", "EUR", "GBP", "JPY", "CNY",  # ISO currency codes seen as tickers
})

def _is_noncorroborating(token: str) -> bool:
    """Evidence too generic to rescue an asset on its own."""
    return _is_very_common_word(token) or token.strip() in _ACRONYM_BLOCKLIST
```

Change line 161 so case-sensitive identifiers that are common words or blocklisted acronyms are ambiguous (so Phase 2 won't auto-set `strong_evidence`):

```python
ambiguous = (len(cs_id) <= _AMBIGUOUS_CS_MAX_LEN
             or _is_very_common_word(cs_id)
             or cs_id in _ACRONYM_BLOCKLIST)
```

Change `_kept`'s last line (312) so a blocklisted/common evidence span cannot self-rescue:

```python
return any(not _is_noncorroborating(ev) for ev in m.evidence_spans)
```

- [ ] **Step 4: Run tests, verify pass + no recall loss**

Run: `cd talisman-ai && python -m pytest tests/test_asset_extractor_gate.py -q`
Expected: PASS (all four).
Run the existing NER gold guardrails: `python -m tests.ner_benchmark` — confirm `asset_false_positives` did not rise and `asset_recall` did not fall.

- [ ] **Step 5: Benchmark + report (do not commit)**

Run: `cd /home/rizzo/talisman/eval && python scripts/overhaul_bench.py --gold eval/data/gold_z-ai_glm-5.1.jsonl --limit 500 --fields assets`
Expected: `assets` F1 ≥ baseline (precision up from fewer FPs), field_agreement unchanged. Record delta. Do NOT commit.

---

### Task 3: P4a — deduplicate entities by canonical name

**Files:**
- Modify: `/home/rizzo/talisman/talisman-ai/talisman_ai/analyzer/article_intelligence_analyzer.py:505-521` (`_build_entities_from_ner`)
- Test: `/home/rizzo/talisman/talisman-ai/tests/test_entity_dedup.py` (create)

**Interfaces:**
- Consumes: `ner_result.resolved_entities` (each has `.canonical_name`, `.entity_type`, `.role`, `.ticker`, `.sentiment_toward`, `.confidence`).
- Produces: `_build_entities_from_ner` returns at most one `ExtractedEntity` per `(name.lower(), entity_type)`.

- [ ] **Step 1: Write failing test**

```python
# tests/test_entity_dedup.py
from types import SimpleNamespace
from talisman_ai.analyzer.article_intelligence_analyzer import ArticleIntelligenceAnalyzer
def _ent(name, role="mentioned", conf=0.9):
    return SimpleNamespace(canonical_name=name, entity_type="organization",
                           role=role, ticker=None, sentiment_toward=None, confidence=conf)
def test_dedup_merges_repeated_entity():
    a = ArticleIntelligenceAnalyzer.__new__(ArticleIntelligenceAnalyzer)  # no __init__ (no models)
    ner = SimpleNamespace(resolved_entities=[_ent("Medtronic"), _ent("Medtronic"),
                                             _ent("Medtronic"), _ent("Apple Inc")])
    out = a._build_entities_from_ner(ner)
    names = [e.name for e in out]
    assert names.count("Medtronic") == 1
    assert "Apple Inc" in names
```

- [ ] **Step 2: Run, verify failure**

Run: `cd talisman-ai && python -m pytest tests/test_entity_dedup.py -q`
Expected: FAIL (`Medtronic` count == 3).

- [ ] **Step 3: Implement dedup**

Replace the body of `_build_entities_from_ner` so it keys by `(canonical_name.lower(), entity_type)`, keeps the highest-confidence representative, and prefers a non-`MENTIONED` role / non-null sentiment on merge:

```python
def _build_entities_from_ner(self, ner_result) -> List[ExtractedEntity]:
    best = {}  # (name_lower, etype) -> (confidence, ExtractedEntity)
    for e in ner_result.resolved_entities:
        try:
            etype = _safe_enum(EntityType, e.entity_type, EntityType.ORGANIZATION) or EntityType.ORGANIZATION
            role = _safe_enum(EntityRole, e.role, EntityRole.MENTIONED)
            ent = ExtractedEntity(
                name=e.canonical_name, entity_type=etype, role=role,
                ticker=e.ticker, sentiment_toward=_safe_enum(Sentiment, e.sentiment_toward, None))
            key = (e.canonical_name.lower(), etype)
            conf = float(getattr(e, "confidence", 0.0) or 0.0)
            if key not in best:
                best[key] = (conf, ent)
            else:
                prev_conf, prev = best[key]
                # prefer a more specific role / a sentiment signal / higher confidence
                if prev.role == EntityRole.MENTIONED and role != EntityRole.MENTIONED:
                    prev.role = role
                if prev.sentiment_toward is None and ent.sentiment_toward is not None:
                    prev.sentiment_toward = ent.sentiment_toward
                if conf > prev_conf:
                    best[key] = (conf, ent)
        except Exception:
            pass
    ents = [v[1] for v in best.values()]
    return ents[:15]
```

- [ ] **Step 4: Run test, verify pass**

Run: `cd talisman-ai && python -m pytest tests/test_entity_dedup.py -q`
Expected: PASS.

- [ ] **Step 5: Benchmark + report (do not commit)**

Run: `cd /home/rizzo/talisman/eval && python scripts/overhaul_bench.py --gold eval/data/gold_z-ai_glm-5.1.jsonl --limit 500 --fields entities`
Expected: `entities` F1 ≥ baseline (duplicate keys collapse → fewer fp), no fn increase. Record delta. Do NOT commit.

---

### Task 4: P4b — drop adjectival / appositive NER fragments

**Files:**
- Modify: `/home/rizzo/talisman/talisman-ai/talisman_ai/analyzer/entity_filter.py:90-107` (`_blocked` + a normalize step in `filter`)
- Test: `/home/rizzo/talisman/talisman-ai/tests/test_entity_fragments.py` (create)

**Interfaces:**
- Consumes/Produces: `EntityFilter.filter(candidates, language)` — same signature. Candidate text is normalized (hyphenated modifier tails trimmed) before clustering; pure-fragment candidates are dropped.

- [ ] **Step 1: Write failing test**

```python
# tests/test_entity_fragments.py
from talisman_ai.analyzer.entity_filter import EntityFilter, Candidate
ef = EntityFilter()
def _texts(cands): return [c.text for c in ef.filter(cands, "en")]
def test_trims_hyphenated_modifier():
    out = _texts([Candidate("Elon Musk-backed", 0, 16, "PERSON", "spacy", 0.9)])
    assert "Elon Musk" in out and "Elon Musk-backed" not in out
def test_drops_appositive_principle():
    out = _texts([Candidate("Anthropic principle", 0, 19, "ORG", "spacy", 0.9)])
    assert out == ["Anthropic"] or out == []  # trimmed to Anthropic or dropped, never the fragment
```

- [ ] **Step 2: Run, verify failure**

Run: `cd talisman-ai && python -m pytest tests/test_entity_fragments.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement normalization + blocklist extension**

Add module-level rules and a normalizer, then apply it at the top of `filter` before the noise/blocklist pass:

```python
import re
_HYPHEN_MODIFIER = re.compile(
    r"^(.*?)-(?:backed|led|owned|run|based|controlled|style|era|related|linked|"
    r"funded|driven|focused|themed|branded)\b.*$", re.IGNORECASE)
_APPOSITIVE_TAIL = re.compile(
    r"^(.+?)\s+(?:principle|effect|equation|paradox|theorem|conjecture|law)$",
    re.IGNORECASE)

def _normalize_entity_text(text: str) -> str:
    t = text.strip()
    m = _HYPHEN_MODIFIER.match(t)
    if m and m.group(1).strip():
        t = m.group(1).strip()
    m = _APPOSITIVE_TAIL.match(t)
    if m and m.group(1).strip():
        t = m.group(1).strip()
    return t
```

In `filter`, before step 1, rewrite each candidate's text:

```python
candidates = [c._replace(text=_normalize_entity_text(c.text)) if hasattr(c, "_replace")
              else _retext(c, _normalize_entity_text(c.text)) for c in candidates]
```

(If `Candidate` is a dataclass, set `c.text = _normalize_entity_text(c.text)` in a loop instead — check the `Candidate` definition at the top of `entity_filter.py` and use the matching mutation idiom.)

- [ ] **Step 4: Run tests, verify pass**

Run: `cd talisman-ai && python -m pytest tests/test_entity_fragments.py tests/test_entity_dedup.py -q`
Expected: PASS. Also `python -m tests.ner_benchmark` — confirm `entity_leaks` did not rise.

- [ ] **Step 5: Benchmark + report (do not commit)**

Run the entities bench again (`--fields entities`), confirm F1 ≥ Task-3 result with no fn increase. Record delta. Do NOT commit.

---

### Task 5: P3 — real per-asset direction via surface-form matching

**Files:**
- Modify: `/home/rizzo/talisman/talisman-ai/talisman_ai/analyzer/ner_fusion.py:246-256` (carry surface forms on `ResolvedEntity`)
- Modify: `/home/rizzo/talisman/talisman-ai/talisman_ai/analyzer/article_intelligence_analyzer.py:301,611-644` (`_finbert_asset_sentiments` + call site)
- Test: `/home/rizzo/talisman/talisman-ai/tests/test_asset_direction.py` (create)

**Interfaces:**
- Consumes: `ner_result.resolved_assets` (each with `.ticker`, `.canonical_name`); `ner_result.sentence_sentiments` (list of `{"text","sentiment","score"}`); article overall sentiment string.
- Produces: `_finbert_asset_sentiments(self, resolved_assets, ner_result, fallback_direction="neutral")` returning the same dict shape as before, but `direction` derived from sentences matched by **surface form** (canonical name / aliases / evidence span / ticker), and `fallback_direction` (not hard "neutral") when no mention is found.

- [ ] **Step 1: Write failing test**

```python
# tests/test_asset_direction.py
from types import SimpleNamespace
from talisman_ai.analyzer.article_intelligence_analyzer import ArticleIntelligenceAnalyzer
def test_direction_from_company_name_not_ticker():
    a = ArticleIntelligenceAnalyzer.__new__(ArticleIntelligenceAnalyzer)
    ner = SimpleNamespace(
        resolved_assets=[SimpleNamespace(ticker="TSLA", canonical_name="Tesla")],
        sentence_sentiments=[
            {"text": "Tesla shares surged 19% after record deliveries.", "sentiment": "bullish", "score": 0.95},
            {"text": "The market was quiet otherwise.", "sentiment": "neutral", "score": 0.6},
        ])
    out = a._finbert_asset_sentiments(ner.resolved_assets, ner, fallback_direction="neutral")
    tsla = next(o for o in out if o["ticker"] == "TSLA")
    assert tsla["direction"] == "bullish"
def test_fallback_uses_article_sentiment():
    a = ArticleIntelligenceAnalyzer.__new__(ArticleIntelligenceAnalyzer)
    ner = SimpleNamespace(resolved_assets=[SimpleNamespace(ticker="AAPL", canonical_name="Apple")],
                          sentence_sentiments=[{"text":"Unrelated macro note.","sentiment":"bullish","score":0.8}])
    out = a._finbert_asset_sentiments(ner.resolved_assets, ner, fallback_direction="bearish")
    assert out[0]["direction"] == "bearish"
```

- [ ] **Step 2: Run, verify failure**

Run: `cd talisman-ai && python -m pytest tests/test_asset_direction.py -q`
Expected: FAIL (signature is `(self, tickers, ner_result)`; matching is by ticker symbol).

- [ ] **Step 3: Implement surface-form matching**

Rewrite `_finbert_asset_sentiments`:

```python
def _finbert_asset_sentiments(self, resolved_assets, ner_result, fallback_direction="neutral") -> List[dict]:
    """Per-asset sentiment via FinBERT over sentences mentioning the asset by any
    surface form (name / alias / evidence span / ticker), not just the ticker symbol.
    Deterministic. Falls back to the article-level direction when no mention is found."""
    import re as _re
    scored = getattr(ner_result, "sentence_sentiments", None) or []
    out = []
    for e in resolved_assets:
        tk = getattr(e, "ticker", None)
        if not tk:
            continue
        forms = {tk}
        name = getattr(e, "canonical_name", None)
        if name:
            forms.add(name)
        for extra in (getattr(e, "surface_forms", None) or []):
            forms.add(extra)
        # word-boundary match of any surface form (len>=2) against each sentence
        pats = [_re.compile(rf"\b{_re.escape(f)}\b", _re.IGNORECASE) for f in forms if len(f) >= 2]
        mentions = [s for s in scored if any(p.search(s.get("text") or "") for p in pats)]
        if mentions:
            labels = [m["sentiment"] for m in mentions]
            direction = max(set(labels), key=labels.count)
            magnitude = sum(float(m.get("score", 0.5)) for m in mentions) / len(mentions)
            conf = magnitude
        else:
            direction, magnitude, conf = fallback_direction, 0.5, 0.3
        out.append({
            "ticker": tk, "direction": direction,
            "magnitude": max(0.0, min(1.0, magnitude)), "confidence": max(0.0, min(1.0, conf)),
            "short_term": direction, "medium_term": direction, "long_term": direction,
            "causal_driver": f"FinBERT over {len(mentions)} mention sentence(s) matched by surface form",
        })
    return out
```

Update the call site (`:301`) to pass resolved assets and the article sentiment:

```python
overall_dir = (call1.get("sentiment") or "neutral")
asset_sentiments = self._finbert_asset_sentiments(
    [e for e in ner_result.resolved_assets if e.ticker], ner_result, fallback_direction=overall_dir)
```

(Confirm `call1.get("sentiment")` returns a Sentiment string compatible with `_safe_enum(Sentiment, ...)` used in `_build_assets:490`. If it can be None, default to `"neutral"`.)

Optionally enrich surface forms: in `ner_fusion.py:246-256`, set `surface_forms=list(dict.fromkeys(m.evidence_spans + [m.asset_name]))` on the keyword-path `ResolvedEntity` (add the attribute to the `ResolvedEntity` definition with default `None`). This widens matching to aliases without breaking other consumers.

- [ ] **Step 4: Run tests, verify pass**

Run: `cd talisman-ai && python -m pytest tests/test_asset_direction.py -q`
Expected: PASS.

- [ ] **Step 5: Benchmark + report (do not commit)**

Run: `cd /home/rizzo/talisman/eval && python scripts/overhaul_bench.py --gold eval/data/gold_z-ai_glm-5.1.jsonl --limit 500 --fields assets`
Expected: `assets` field_agreement rises from the near-base-rate baseline toward `FinBERTAspect`'s ~0.815; F1 unchanged or better. Record delta. Do NOT commit.

---

### Task 6: P1 (benchmark) — embedding-based narrative slug predictor

**Files:**
- Modify: `/home/rizzo/talisman/eval/eval/predictors/keywords.py` (add `NarrativeEmbedKeywords`)
- Modify: `/home/rizzo/talisman/eval/eval/cli.py:122-126` (register it)
- Create: `/home/rizzo/talisman/eval/scripts/sweep_narrative_tau.py`

**Interfaces:**
- Produces: predictor `NarrativeEmbedKeywords(tau=…, top_k=3)` with `name="narrative_embed"`, `fit(rows)` (precomputes 38 narrative centroids from `narratives.json` keywords using `all-MiniLM-L6-v2`), `predict_one(row)` → `Prediction([slugs over tau][:top_k], conf)`; returns `[]` when none clear tau.

- [ ] **Step 1: Write failing test**

```python
# eval/tests/test_narrative_embed.py
from eval.predictors.keywords import NarrativeEmbedKeywords
def test_abstains_on_offtopic():
    p = NarrativeEmbedKeywords(tau=0.45).fit([])
    pred = p.predict_one({"article": {"title": "Local bakery wins pie contest",
                                      "content": "A small-town bakery took first prize at the county fair."}})
    assert pred.value == []
def test_picks_crypto_slug_on_crypto():
    p = NarrativeEmbedKeywords(tau=0.30).fit([])
    pred = p.predict_one({"article": {"title": "Spot Bitcoin ETF sees record inflows",
                                      "content": "IBIT and FBTC pulled in billions as institutional inflows surged."}})
    assert any("bitcoin" in s or "etf" in s for s in pred.value)
```

- [ ] **Step 2: Run, verify failure**

Run: `cd /home/rizzo/talisman/eval && python -m pytest tests/test_narrative_embed.py -q`
Expected: FAIL (class missing).

- [ ] **Step 3: Implement the predictor**

```python
# append to eval/eval/predictors/keywords.py
import json, os, functools
_NARR_PATH = "/home/rizzo/talisman/talisman-ai/talisman_ai/analyzer/data/narratives.json"

class NarrativeEmbedKeywords:
    name = "narrative_embed"
    def __init__(self, tau=0.40, top_k=3, model="all-MiniLM-L6-v2"):
        self.tau, self.top_k, self._model_name = tau, top_k, model
        self._model = None; self._slugs = None; self._cent = None
    def _ensure(self):
        if self._model is not None: return
        from sentence_transformers import SentenceTransformer
        import numpy as np
        self._model = SentenceTransformer(self._model_name)
        narr = json.load(open(_NARR_PATH))
        self._slugs = [n["slug"] for n in narr]
        texts = [n["name"] + ": " + ", ".join(n.get("keywords") or []) for n in narr]
        self._cent = self._model.encode(texts, normalize_embeddings=True)
    def fit(self, rows): self._ensure(); return self
    def predict_one(self, row):
        import numpy as np
        from eval.predictors.base import Prediction
        self._ensure()
        a = row["article"]
        q = self._model.encode([f"{a.get('title') or ''}. {a.get('content') or ''}"[:2000]],
                               normalize_embeddings=True)[0]
        sims = self._cent @ q
        order = np.argsort(-sims)
        picks = [(self._slugs[i], float(sims[i])) for i in order if sims[i] >= self.tau][:self.top_k]
        return Prediction([s for s, _ in picks], float(picks[0][1]) if picks else 1.0)
```

Register in `cli.py`: add `from eval.predictors.keywords import ... , NarrativeEmbedKeywords` and `"narrative_embed": NarrativeEmbedKeywords` to the registry dict.

- [ ] **Step 4: Run test, verify pass**

Run: `cd /home/rizzo/talisman/eval && python -m pytest tests/test_narrative_embed.py -q`
Expected: PASS.

- [ ] **Step 5: Sweep tau + characterize; report (do not commit)**

Write `sweep_narrative_tau.py` to loop tau in `[0.25,0.30,…,0.55]`, score `NarrativeEmbedKeywords(tau)` on the calibration split via `eval.bench.score.score_candidate("narrative_keywords", pred, rows)`, and print mean Jaccard + empty-rate per tau; pick the tau whose empty-rate ≈ 37% and Jaccard is maximal.
Then: `cd /home/rizzo/talisman/eval && python -m eval characterize --field narrative_keywords --candidates narrative_embed yake keybert --gold eval/data/gold_z-ai_glm-5.1.jsonl --limit 1000 --out eval/reports/keywords_overhaul.json`
Expected: `narrative_embed` glm_fidelity ≫ yake/keybert (which are ~0.0). Also compute the GLM-vs-Opus ceiling for context. Record chosen tau + numbers. Do NOT commit.

---

### Task 7: P1 (production) — wire narrative slug selection into the analyzer

**Files:**
- Modify: `/home/rizzo/talisman/talisman-ai/talisman_ai/analyzer/article_intelligence_analyzer.py:184-204,286-294,311` (shortlist + Call-2 schema + assembly)
- Modify: `/home/rizzo/talisman/talisman-ai/talisman_ai/analyzer/__init__.py` if a setup hook is needed
- Test: `/home/rizzo/talisman/talisman-ai/tests/test_narrative_selection.py` (create)

**Interfaces:**
- Consumes: chosen approach + tau from Task 6.
- Produces: `narrative_keywords` in the final `ArticleIntelligence` are drawn only from the taxonomy, default to `[]`, and exclude crypto slugs on non-crypto articles.

**Decision gate (read Task 6 results before coding):**
- **If the embedding-only predictor matched/beat the hybrid** in Task 6 → ship deterministic: add a `_select_narratives(self, title, one_liner, ctx)` method mirroring `NarrativeEmbedKeywords` (precompute centroids in `__init__` using `self.ner_engine` encoder), compute `narr_kws` in assembly (`:311`), and **drop `narrative_keywords` from `REASON_SUMMARIZE_TOOL` and its `required` list**. This removes LLM scope (advances kill-Call-2).
- **If the hybrid is clearly better** → keep Call 2 but pass only the embedding shortlist into the tool description (replace the full `_load_narrative_slugs()` menu with the per-article shortlist), require a verbatim `evidence_span` per slug, instruct "return [] if none clearly fits; most articles have none", remove `narrative_keywords` from `required`, and reject any slug whose evidence span is not a substring of the fact sheet.

- [ ] **Step 1: Write failing test (deterministic variant shown; adapt if hybrid chosen)**

```python
# tests/test_narrative_selection.py — uses the no-init analyzer + a stub encoder is complex;
# instead test end-to-end intent on the hand-curated non-crypto cases via the bench in Task 8.
# Minimal unit test of the selector:
def test_select_narratives_abstains(monkeypatch):
    from talisman_ai.analyzer.article_intelligence_analyzer import ArticleIntelligenceAnalyzer as A
    a = A.__new__(A)
    a._init_narrative_index()  # builds centroids using a SentenceTransformer
    out = a._select_narratives("Local bakery wins pie contest", "A bakery won a prize.", "")
    assert out == []
```

- [ ] **Step 2: Run, verify failure**

Run: `cd talisman-ai && python -m pytest tests/test_narrative_selection.py -q`
Expected: FAIL (method missing).

- [ ] **Step 3: Implement chosen variant**

Deterministic variant — add to `ArticleIntelligenceAnalyzer`:

```python
def _init_narrative_index(self):
    import json as _json, numpy as _np
    narr = _json.load(open(os.path.join(DATA_DIR, "narratives.json")))
    self._narr_slugs = [n["slug"] for n in narr]
    texts = [n["name"] + ": " + ", ".join(n.get("keywords") or []) for n in narr]
    self._narr_cent = _np.asarray(self.ner_engine.encode_batch(texts))  # L2-normalized
    self._narr_tau = float(getattr(config, "NARRATIVE_TAU", 0.40)) if config else 0.40

def _select_narratives(self, title, one_liner, ctx):
    import numpy as _np
    q = _np.asarray(self.ner_engine.encode_text(f"{title}. {one_liner} {ctx}"[:2000]))
    sims = self._narr_cent @ q
    order = _np.argsort(-sims)
    return [self._narr_slugs[i] for i in order if sims[i] >= self._narr_tau][:3]
```

Call `self._init_narrative_index()` at the end of `__init__`. (Confirm `ner_engine` exposes a batch encode; if only `encode_text` exists, add `encode_batch` or loop.) In assembly replace line 311:

```python
narr_kws = self._select_narratives(headline, one_liner, ctx_para)
```

and remove `narrative_keywords` from `REASON_SUMMARIZE_TOOL.properties` and from `required`.

- [ ] **Step 4: Run unit test, verify pass**

Run: `cd talisman-ai && python -m pytest tests/test_narrative_selection.py -q`
Expected: PASS.

- [ ] **Step 5: Report (do not commit)** — full validation happens in Task 8.

---

### Task 8: Full benchmark + hand-curated regression + report

**Files:**
- Modify: `/home/rizzo/talisman/talisman-ai/tests/test_overhaul_regressions.py` (add the targeted assertions over `handcurated_overhaul.jsonl`)
- Create: `/home/rizzo/talisman/talisman-ai/docs/superpowers/overhaul_benchmark_report.md`

- [ ] **Step 1: Write the hand-curated regression tests**

Add a parametrized test that, for each line in `handcurated_overhaul.jsonl`, runs the deterministic path (NER + builders, plus `_select_narratives`) and asserts the `expect` block: `assets_excludes`/`assets_includes`, `asset_direction`, `entity_no_duplicates`, `entity_excludes`/`entity_includes`, `narrative_keywords_excludes`. (No LLM needed — all four target fields are deterministic after Task 7's deterministic variant; if hybrid was chosen, gate the keyword assertions behind an `LLM_LIVE` env flag.)

- [ ] **Step 2: Run, verify it exercises every case**

Run: `cd talisman-ai && python -m pytest tests/test_overhaul_regressions.py -q`
Expected: All targeted cases PASS (LEO excluded, $LEO kept, Medtronic deduped, no crypto slugs on non-crypto, Tesla bullish, etc.).

- [ ] **Step 3: Run the full gold benchmark, before vs after**

Run on a 1000-row slice for all deterministic fields:
`cd /home/rizzo/talisman/eval && python scripts/overhaul_bench.py --gold eval/data/gold_z-ai_glm-5.1.jsonl --limit 1000 --fields assets entities`
and the narrative characterize from Task 6. Re-run the same against `gold_anthropic_claude-opus-4-6.jsonl` (overfit check).

- [ ] **Step 4: Confirm no regression on other fields**

Spot-check that `sentiment_direction`, `economic_data`, `contagion_links` metrics are unchanged (these paths were not touched). If the hybrid Call-2 variant was chosen, run a small `eval run` sample to confirm summary fields (`headline`/`one_liner`) did not regress.

- [ ] **Step 5: Write the report (do not commit)**

`overhaul_benchmark_report.md` table: field × (baseline, after) × metric, for assets (F1 + field_agreement), entities (F1), narrative_keywords (Jaccard, empty-rate), plus the hand-curated pass/fail summary and the chosen P1 variant + tau. Hand the report to the user for review and commit. **Do NOT commit anything.**

---

## Self-Review

**Spec coverage:** P1 → Tasks 6+7 (+8 bench); P2 → Task 2; P3 → Task 5; P4 dedup → Task 3; P4 fragments → Task 4; benchmark methodology (existing gold + hand-curated) → Tasks 1+8; determinism constraint honored (no LLM added to assets/entities/sentiment; P1 deterministic variant preferred). All spec sections map to tasks.

**Placeholder scan:** No TBD/TODO. Each code step shows real code. The one deferred decision (P1 deterministic vs hybrid) is an explicit, data-gated branch in Task 7 with both variants specified, not a placeholder.

**Type consistency:** `_finbert_asset_sentiments` new signature `(self, resolved_assets, ner_result, fallback_direction)` is used identically at the call site (Task 5 Step 3) and tests (Task 5 Step 1). `_build_assets` still reads `s.get("direction")` etc. — dict shape unchanged. `NarrativeEmbedKeywords` (eval, Task 6) and `_select_narratives` (production, Task 7) are separate by design (different import roots). `_is_noncorroborating` defined in Task 2 used in same task's `_kept`.

**Ordering:** 1 (harness/baseline) → 2 (P2) → 3,4 (P4) → 5 (P3) → 6 (P1 bench) → 7 (P1 prod) → 8 (final). Each task independently testable; later tasks depend only on Task 1's harness.
