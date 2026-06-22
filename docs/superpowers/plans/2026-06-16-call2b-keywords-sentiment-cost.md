# Plan 2b — Keywords, Per-Asset Sentiment, HF Gold, Call-2 Cost

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the remaining off-LLM Call-2 fields — `narrative_keywords` (KeyBERT/YAKE) and per-asset sentiment (`assets[].direction`, FinBERT aspect) — measure them vs GLM and vs human gold (FinEntity, PhraseBank/FiQA), and **measure the real Call-2 token-cost reduction** of shrinking Call 2 to summary-only.

**Architecture:** New article-reading predictors plug into the existing `Predictor` protocol (`fit` no-op; `predict_one(row)` reads `row["article"]`, returns `Prediction(value, 1.0)`) and are scored by existing metrics (`narrative_keywords`→KeywordJaccard, `assets`→ListStruct on `direction`). Reuses Plan 2a's `characterize_field`/`render_characterization` (non-gated characterization) and `re_eval`-style human-gold harness. Heavy models (KeyBERT/MiniLM, FinBERT) and HF datasets are dependency-injected; unit tests use fakes, real backends are gated behind `RUN_MODEL_TESTS=1`. Contagion is settled (graph-only) and out of scope here.

**Tech Stack:** reuses `eval` framework; new deps `keybert`, `yake`; FinBERT/MiniLM present; `datasets` for FinEntity/PhraseBank/FiQA.

**Branch:** continue on `feat/call2a-contagion` (Plan 2b builds on Plan 2a's characterize runner) OR a new branch off it — controller decides at execution.

**Key APIs:** FinBERT `pipeline("sentiment-analysis","ProsusAI/finbert",device=-1)(s)[0] -> {"label","score"}`. KeyBERT `KeyBERT(model).extract_keywords(text, top_n=3) -> [(phrase, score)]`. YAKE `yake.KeywordExtractor(top=3).extract_keywords(text) -> [(phrase, score)]`. `FIELD_METRICS["assets"]` = ListStruct keyed on `ticker`, compares `["direction"]`; `FIELD_METRICS["narrative_keywords"]` = KeywordJaccard.

---

### Task 1: Keyword predictors (YAKE baseline + KeyBERT)

**Files:**
- Create: `eval/predictors/keywords.py`
- Test: `tests/test_keywords.py`

Both take an injected `extract_fn(text) -> list[str]`; defaults lazy-load the real libs (used only in real runs).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_keywords.py
from eval.predictors.keywords import YakeKeywords, KeyBERTKeywords

def test_yake_returns_top_phrases_from_injected_fn():
    pred = YakeKeywords(extract_fn=lambda text: ["rate hike", "inflation", "fed"])
    out = pred.fit([]).predict_one({"article": {"title": "t", "content": "Fed raises rates"}})
    assert out.value == ["rate hike", "inflation", "fed"]
    assert pred.name == "yake"

def test_keybert_uses_injected_fn_and_caps_three():
    pred = KeyBERTKeywords(extract_fn=lambda text: ["a", "b", "c", "d"])
    out = pred.predict_one({"article": {"title": "t", "content": "x"}})
    assert out.value == ["a", "b", "c"] and pred.name == "keybert"

def test_empty_content_yields_empty():
    pred = YakeKeywords(extract_fn=lambda text: [])
    assert pred.predict_one({"article": {"title": "", "content": ""}}).value == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/rizzo/talisman/eval && pytest tests/test_keywords.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# eval/predictors/keywords.py
from eval.predictors.base import Prediction

def _text(art):
    return f"{art.get('title') or ''}. {art.get('content') or ''}".strip()

def _default_yake():
    import yake
    kw = yake.KeywordExtractor(top=3)
    return lambda text: [p for p, _ in kw.extract_keywords(text)]

def _default_keybert():
    from keybert import KeyBERT
    model = KeyBERT(model="all-MiniLM-L6-v2")
    return lambda text: [p for p, _ in model.extract_keywords(text, top_n=3)]

class _BaseKeywords:
    def __init__(self, extract_fn=None):
        self._extract = extract_fn
    def _ensure(self):
        if self._extract is None:
            self._extract = self._default()
    def fit(self, rows):
        return self
    def predict_one(self, row):
        self._ensure()
        kws = self._extract(_text(row["article"]))
        return Prediction(list(kws)[:3], 1.0)

class YakeKeywords(_BaseKeywords):
    name = "yake"
    def _default(self):
        return _default_yake()

class KeyBERTKeywords(_BaseKeywords):
    name = "keybert"
    def _default(self):
        return _default_keybert()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_keywords.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
cd /home/rizzo/talisman/eval
git add eval/predictors/keywords.py tests/test_keywords.py
git commit -m "feat: YAKE + KeyBERT keyword predictors (injected extractor)"
```

---

### Task 2: FinBERT aspect per-asset sentiment predictor

**Files:**
- Create: `eval/predictors/aspect_sentiment.py`
- Test: `tests/test_aspect_sentiment.py`

Injected: `extract_tickers(title, body) -> [ticker]`, `sentiment_fn(sentence) -> "positive"|"negative"|"neutral"`. Output: a list of `{"ticker", "direction"}` matching the `assets` ListStruct (keyed on ticker, compares direction). 3-class FinBERT maps to BULLISH/BEARISH/NEUTRAL (documented coarse mapping).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_aspect_sentiment.py
from eval.predictors.aspect_sentiment import FinBERTAspect

def test_aspect_assigns_direction_per_ticker():
    pred = FinBERTAspect(
        extract_tickers=lambda title, body: ["NVDA", "INTC"],
        sentiment_fn=lambda sent: "positive" if "NVDA" in sent else "negative")
    art = {"title": "Chips", "content": "NVDA soared on earnings. INTC fell on weak guidance."}
    out = pred.fit([]).predict_one({"article": art})
    by = {a["ticker"]: a["direction"] for a in out.value}
    assert by["NVDA"] == "BULLISH" and by["INTC"] == "BEARISH"
    assert pred.name == "finbert_aspect"

def test_aspect_ticker_with_no_sentence_is_neutral():
    pred = FinBERTAspect(
        extract_tickers=lambda title, body: ["BTC"],
        sentiment_fn=lambda sent: "positive")
    out = pred.predict_one({"article": {"title": "t", "content": "no ticker mention here"}})
    assert out.value == [{"ticker": "BTC", "direction": "NEUTRAL"}]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_aspect_sentiment.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# eval/predictors/aspect_sentiment.py
from eval.predictors.base import Prediction

_DIR = {"positive": "BULLISH", "negative": "BEARISH", "neutral": "NEUTRAL"}

def _sentences(art):
    text = f"{art.get('title') or ''}. {art.get('content') or ''}"
    return [s.strip() for s in text.split(".") if s.strip()]

def _default_extract_tickers(title, body):
    from alpharidge_ai.analyzer.asset_extractor import AssetExtractor
    seen, out = set(), []
    for m in AssetExtractor().extract_assets(title or "", body or ""):
        if m.ticker not in seen:
            seen.add(m.ticker); out.append(m.ticker)
    return out

def _default_sentiment_fn():
    from transformers import pipeline
    clf = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=-1)
    return lambda sent: clf(sent[:512], truncation=True, max_length=512)[0]["label"].lower()

class FinBERTAspect:
    """Per-asset sentiment: FinBERT over each asset's mention sentences -> direction.
    3-class FinBERT maps coarsely to BULLISH/BEARISH/NEUTRAL (not the 7-class scale)."""
    name = "finbert_aspect"
    def __init__(self, extract_tickers=None, sentiment_fn=None):
        self._extract = extract_tickers
        self._sentiment = sentiment_fn
    def _ensure(self):
        if self._extract is None:
            self._extract = _default_extract_tickers
        if self._sentiment is None:
            self._sentiment = _default_sentiment_fn()
    def fit(self, rows):
        return self
    def predict_one(self, row):
        self._ensure()
        art = row["article"]
        sents = _sentences(art)
        out = []
        for tk in self._extract(art.get("title"), art.get("content")):
            mention = [s for s in sents if tk in s]
            if not mention:
                out.append({"ticker": tk, "direction": "NEUTRAL"}); continue
            labels = [self._sentiment(s) for s in mention]
            # majority of mention-sentence labels
            top = max(set(labels), key=labels.count)
            out.append({"ticker": tk, "direction": _DIR.get(top, "NEUTRAL")})
        return Prediction(out, 1.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_aspect_sentiment.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
cd /home/rizzo/talisman/eval
git add eval/predictors/aspect_sentiment.py tests/test_aspect_sentiment.py
git commit -m "feat: FinBERT aspect per-asset sentiment predictor"
```

---

### Task 3: HF sentiment-gold adapters (FinEntity, PhraseBank, FiQA)

**Files:**
- Create: `eval/hf/finentity.py`, `eval/hf/phrasebank.py`, `eval/hf/fiqa.py`
- Test: `tests/test_hf_sentiment_adapters.py`

Each module has a PURE `parse_*` function (unit-tested on synthetic records) + a `load_*` function (gated, network). Schemas are confirmed by inspection first (like FinRED in Plan 2a).

- [ ] **Step 1: Inspect real schemas (network)**

Run:
```bash
cd /home/rizzo/talisman/eval
python - <<'PY'
from datasets import load_dataset
for ds_id, cfg in [("financial_phrasebank","sentences_50agree"),
                   ("ChanceFocus/flare-fiqasa", None), ("yixuantt/FinEntity", None)]:
    try:
        d = load_dataset(ds_id, cfg) if cfg else load_dataset(ds_id)
        k = list(d.keys())[0]
        print("OK", ds_id, d[k].column_names, "| sample:", str(d[k][0])[:300])
    except Exception as e:
        print("FAIL", ds_id, type(e).__name__, str(e)[:160])
PY
```
Record the actual columns. PhraseBank: `sentence` + `label` (0=negative,1=neutral,2=positive). FiQA (flare-fiqasa): instruction-format with `text`/`answer` or `query`/`answer`. FinEntity: `content` + `annotations` (list of `{value/label, start, end}` with Positive/Negative/Neutral). **Adapt the parse functions below to the real columns you observe.**

- [ ] **Step 2: Write the failing test (synthetic records matching observed schema)**

```python
# tests/test_hf_sentiment_adapters.py
from eval.hf.phrasebank import parse_phrasebank
from eval.hf.fiqa import parse_fiqa
from eval.hf.finentity import parse_finentity

def test_phrasebank_maps_label_to_sentiment():
    # label 2 = positive
    items = parse_phrasebank({"sentence": "Profit rose sharply.", "label": 2})
    assert items[0]["labels"]["overall_sentiment"] == "BULLISH"
    assert items[0]["article"]["content"] == "Profit rose sharply."

def test_fiqa_maps_answer_to_sentiment():
    items = parse_fiqa({"text": "Shares tumbled.", "answer": "negative"})
    assert items[0]["labels"]["overall_sentiment"] == "BEARISH"

def test_finentity_yields_per_entity_direction():
    rec = {"content": "AAPL up, TSLA down",
           "annotations": [{"value": "AAPL", "label": "Positive"},
                           {"value": "TSLA", "label": "Negative"}]}
    items = parse_finentity(rec)
    assets = items[0]["labels"]["assets"]
    by = {a["ticker"]: a["direction"] for a in assets}
    assert by["AAPL"] == "BULLISH" and by["TSLA"] == "BEARISH"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_hf_sentiment_adapters.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 4: Write the adapters** (adjust field names to the real schema from Step 1)

```python
# eval/hf/phrasebank.py
"""Financial PhraseBank (sentence sentiment, human gold) -> overall_sentiment rows."""
_LABEL = {0: "BEARISH", 1: "NEUTRAL", 2: "BULLISH"}   # 0 neg, 1 neutral, 2 pos

def parse_phrasebank(rec) -> list:
    sent = rec.get("sentence") or rec.get("text") or ""
    lab = rec.get("label")
    if not sent or lab is None or lab not in _LABEL:
        return []
    return [{"article": {"id": None, "title": "", "content": sent},
             "labels": {"overall_sentiment": _LABEL[lab]}}]

def load_phrasebank(config="sentences_50agree", limit=None) -> list:
    from datasets import load_dataset
    ds = load_dataset("financial_phrasebank", config, split="train")
    out = []
    for i, rec in enumerate(ds):
        items = parse_phrasebank(rec)
        for it in items:
            it["article"]["id"] = i
        out.extend(items)
        if limit and len(out) >= limit:
            return out[:limit]
    return out
```

```python
# eval/hf/fiqa.py
"""FiQA-2018 sentiment (human gold) -> overall_sentiment rows."""
def _to_dir(answer) -> str | None:
    a = str(answer).strip().lower()
    if a in ("positive", "bullish"): return "BULLISH"
    if a in ("negative", "bearish"): return "BEARISH"
    if a in ("neutral",): return "NEUTRAL"
    # numeric sentiment score fallback
    try:
        v = float(a)
        return "BULLISH" if v > 0.1 else "BEARISH" if v < -0.1 else "NEUTRAL"
    except ValueError:
        return None

def parse_fiqa(rec) -> list:
    sent = rec.get("text") or rec.get("sentence") or rec.get("query") or ""
    direction = _to_dir(rec.get("answer") or rec.get("label") or rec.get("sentiment") or "")
    if not sent or direction is None:
        return []
    return [{"article": {"id": None, "title": "", "content": sent},
             "labels": {"overall_sentiment": direction}}]

def load_fiqa(dataset_id="ChanceFocus/flare-fiqasa", split="test", limit=None) -> list:
    from datasets import load_dataset
    ds = load_dataset(dataset_id, split=split)
    out = []
    for i, rec in enumerate(ds):
        items = parse_fiqa(rec)
        for it in items:
            it["article"]["id"] = i
        out.extend(items)
        if limit and len(out) >= limit:
            return out[:limit]
    return out
```

```python
# eval/hf/finentity.py
"""FinEntity entity-level sentiment (human gold) -> per-asset direction rows (assets field)."""
_DIR = {"positive": "BULLISH", "negative": "BEARISH", "neutral": "NEUTRAL"}

def parse_finentity(rec) -> list:
    content = rec.get("content") or rec.get("text") or ""
    anns = rec.get("annotations") or rec.get("entities") or []
    assets = []
    for a in anns:
        ticker = a.get("value") or a.get("entity") or a.get("text") or ""
        label = str(a.get("label") or a.get("tag") or a.get("sentiment") or "").lower()
        if ticker and label in _DIR:
            assets.append({"ticker": ticker, "direction": _DIR[label]})
    if not content or not assets:
        return []
    return [{"article": {"id": None, "title": "", "content": content},
             "labels": {"assets": assets}}]

def load_finentity(dataset_id="yixuantt/FinEntity", split="test", limit=None) -> list:
    from datasets import load_dataset
    try:
        ds = load_dataset(dataset_id, split=split)
    except Exception:
        ds = load_dataset(dataset_id, split="train")
    out = []
    for i, rec in enumerate(ds):
        items = parse_finentity(rec)
        for it in items:
            it["article"]["id"] = i
        out.extend(items)
        if limit and len(out) >= limit:
            return out[:limit]
    return out
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_hf_sentiment_adapters.py -v`
Expected: PASS (3 tests). If a real schema differs from the synthetic assumption, fix BOTH the parser and its synthetic test record to match the real columns.

- [ ] **Step 6: Commit**

```bash
cd /home/rizzo/talisman/eval
git add eval/hf/finentity.py eval/hf/phrasebank.py eval/hf/fiqa.py tests/test_hf_sentiment_adapters.py
git commit -m "feat: HF sentiment-gold adapters (PhraseBank, FiQA, FinEntity)"
```

---

### Task 4: Human-gold validation harness

**Files:**
- Create: `eval/bench/human_eval.py`
- Test: `tests/test_human_eval.py`

Generic: run a predictor over human-gold rows, score with the field's metric, return mean agreement + n. Reuses `score_candidate`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_human_eval.py
from eval.bench.human_eval import validate_human
from eval.predictors.base import Prediction

class FakePred:
    name = "f"
    def fit(self, rows): return self
    def predict_one(self, row):
        # echo gold for ticker AAPL only -> 1 of 2 correct direction
        return Prediction([{"ticker": "AAPL", "direction": "BULLISH"},
                           {"ticker": "TSLA", "direction": "BULLISH"}], 1.0)

def test_validate_human_scores_against_gold():
    rows = [{"article": {"id": 0}, "labels": {"assets": [
        {"ticker": "AAPL", "direction": "BULLISH"},
        {"ticker": "TSLA", "direction": "BEARISH"}]}}]
    res = validate_human("assets", FakePred(), rows)
    assert res["n"] == 1 and 0.0 <= res["agreement"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_human_eval.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# eval/bench/human_eval.py
from statistics import mean
from eval.bench.score import score_candidate

def validate_human(field, predictor, human_rows) -> dict:
    """Mean agreement of predictor vs human gold on `field`, using FIELD_METRICS[field]."""
    predictor.fit(human_rows)
    scored = score_candidate(field, predictor, human_rows)
    agreement = mean(s for s, _ in scored) if scored else 0.0
    return {"agreement": round(agreement, 4), "n": len(scored)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_human_eval.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /home/rizzo/talisman/eval
git add eval/bench/human_eval.py tests/test_human_eval.py
git commit -m "feat: human-gold validation harness (predictor vs HF labels)"
```

---

### Task 5: Token measurement + Call-2 cost projector

**Files:**
- Create: `eval/bench/cost.py`
- Create: `eval/oracle/measure_tokens.py`
- Test: `tests/test_cost.py`

`cost.py` is pure (unit-tested with injected token counts). `measure_tokens.py` is the gated LLM measurement.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cost.py
from eval.bench.cost import call_cost, call2_reduction

def test_call_cost_uses_prices():
    # haiku-4-5 = (1.0, 5.0) $/MTok in eval/report.py PRICES
    c = call_cost(1_000_000, 1_000_000, "claude-haiku-4-5")
    assert abs(c - (1.0 + 5.0)) < 1e-6

def test_call2_reduction_reports_savings():
    full = {"input": 1500, "output": 600}
    slim = {"input": 500, "output": 250}
    r = call2_reduction(full, slim, "claude-haiku-4-5")
    assert r["full_cost"] > r["slim_cost"] > 0
    assert 0.0 < r["reduction_fraction"] < 1.0
    assert abs(r["reduction_fraction"] - (1 - r["slim_cost"] / r["full_cost"])) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cost.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# eval/bench/cost.py
from eval.report import PRICES

def call_cost(input_tokens, output_tokens, model_key) -> float:
    pin, pout = PRICES.get(model_key, (0.0, 0.0))
    return input_tokens / 1e6 * pin + output_tokens / 1e6 * pout

def call2_reduction(full_tokens, slim_tokens, model_key) -> dict:
    """full_tokens/slim_tokens: {'input': avg_in, 'output': avg_out} per article."""
    full = call_cost(full_tokens["input"], full_tokens["output"], model_key)
    slim = call_cost(slim_tokens["input"], slim_tokens["output"], model_key)
    frac = (1 - slim / full) if full else 0.0
    return {"model": model_key, "full_cost": full, "slim_cost": slim,
            "reduction_fraction": frac, "saved_per_article": full - slim}
```

```python
# eval/oracle/measure_tokens.py
"""Measure full Call-2 vs summary-only Call-2 token usage on a sample (gated LLM use).
Runs the analyzer's reason_and_summarize prompt vs a summary-only prompt and records
usage. Wire to the same OpenAI-compatible client the oracle labeler uses."""
def measure(sample_rows, client, model, full_prompt_fn, slim_prompt_fn) -> dict:
    """client.chat(...) -> object with .usage.prompt_tokens / .usage.completion_tokens.
    full_prompt_fn(row)/slim_prompt_fn(row) -> messages list. Returns avg in/out per variant."""
    def _avg(prompt_fn):
        ins, outs = [], []
        for row in sample_rows:
            r = client(prompt_fn(row))
            ins.append(r["input_tokens"]); outs.append(r["output_tokens"])
        n = max(1, len(ins))
        return {"input": sum(ins) / n, "output": sum(outs) / n}
    return {"full": _avg(full_prompt_fn), "slim": _avg(slim_prompt_fn)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cost.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
cd /home/rizzo/talisman/eval
git add eval/bench/cost.py eval/oracle/measure_tokens.py tests/test_cost.py
git commit -m "feat: Call-2 cost projector + token-measurement harness"
```

---

### Task 6: CLI + real run (characterize + human-gold + cost)

**Files:**
- Modify: `eval/cli.py` (extend `characterize` registry; add `validate-human`)
- Test: `tests/test_cli_keywords_sentiment.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_keywords_sentiment.py
import json
from pathlib import Path
from eval.cli import main

def _gold(path):
    with open(path, "w") as fh:
        fh.write(json.dumps({"article": {"id": 0, "title": "t", "content": "Fed hikes rates"},
                             "labels": {"narrative_keywords": ["rate hike", "fed"]}}) + "\n")

def test_characterize_keywords_yake(tmp_path, monkeypatch):
    gold = tmp_path / "g.jsonl"; _gold(gold)
    import eval.predictors.keywords as K
    monkeypatch.setattr(K, "_default_yake", lambda: (lambda text: ["rate hike", "fed"]))
    out = tmp_path / "c.json"
    main(["characterize", "--field", "narrative_keywords", "--gold", str(gold),
          "--candidates", "yake", "--out", str(out)])
    data = json.loads(out.read_text())
    assert data["candidates"][0]["name"] == "yake"
    assert data["candidates"][0]["glm_fidelity"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_keywords_sentiment.py -v`
Expected: FAIL (registry has no `yake`)

- [ ] **Step 3: Extend the `characterize` registry in `eval/cli.py`**

In the `elif ns.cmd == "characterize":` block, replace the `registry = {...}` line with:

```python
        from eval.predictors.contagion import GraphOnlyContagion, GlirelContagion
        from eval.predictors.keywords import YakeKeywords, KeyBERTKeywords
        from eval.predictors.aspect_sentiment import FinBERTAspect
        registry = {"graph_only": GraphOnlyContagion, "glirel": GlirelContagion,
                    "yake": YakeKeywords, "keybert": KeyBERTKeywords,
                    "finbert_aspect": FinBERTAspect}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli_keywords_sentiment.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /home/rizzo/talisman/eval
git add eval/cli.py tests/test_cli_keywords_sentiment.py
git commit -m "feat: register keyword/sentiment candidates in characterize CLI"
```

- [ ] **Step 6: Full suite + install + real runs**

```bash
cd /home/rizzo/talisman/eval
pytest -q                       # all green, gated skip
pip install keybert yake
```

Characterize keywords + per-asset sentiment vs GLM on the held-out test split:
```bash
python -m eval.cli characterize --field narrative_keywords \
  --gold eval/data/distill/test.jsonl --candidates yake keybert \
  --out eval/reports/keywords_characterization.json --limit 300 2>/dev/null
python -m eval.cli characterize --field assets \
  --gold eval/data/distill/test.jsonl --candidates finbert_aspect \
  --out eval/reports/aspect_sentiment_characterization.json --limit 300 2>/dev/null
```

Human-gold validation (real models + HF):
```bash
python - <<'PY'
from eval.hf.finentity import load_finentity
from eval.hf.phrasebank import load_phrasebank
from eval.predictors.aspect_sentiment import FinBERTAspect
from eval.bench.human_eval import validate_human
import json, pathlib
fe = load_finentity(limit=200)
res_aspect = validate_human("assets", FinBERTAspect(), fe)
print("FinBERT aspect vs FinEntity:", res_aspect)
pathlib.Path("eval/reports/human_gold_validation.json").write_text(json.dumps(
    {"finbert_aspect_vs_finentity": res_aspect}, indent=2))
PY
```

Token measurement (optional, uses LLM/OpenRouter — only if API_KEY set): instrument full vs summary-only Call-2 on ~100 articles via `eval/oracle/measure_tokens.py`, then `call2_reduction(...)` → write `eval/reports/call2_cost.json`. If no API budget, skip and note it.

Report all results. Commit reports:
```bash
git add eval/reports/keywords_characterization.json eval/reports/aspect_sentiment_characterization.json eval/reports/human_gold_validation.json
git commit -m "report: keyword/aspect-sentiment characterization + FinEntity human-gold"
```
Confirm `git status --porcelain` shows no leaked data files.

- [ ] **Step 7: Report findings** — keyword fidelity vs GLM (yake vs keybert), aspect-sentiment fidelity vs GLM and vs FinEntity human gold, and (if measured) the Call-2 cost reduction.

---

## Self-Review

**Spec coverage (against `2026-06-16-call2-replacement-design.md`):**
- narrative_keywords → KeyBERT/YAKE → Task 1 ✓
- per-asset sentiment → FinBERT aspect → Task 2 ✓
- HF gold (FinEntity/PhraseBank/FiQA) → Tasks 3–4 ✓
- FinBERT aspect validated on FinEntity → Task 6 ✓
- Token measurement + Call-2 cost projection → Task 5–6 ✓
- contagion (graph-only, decided) → out of scope (Plan 2a) ✓
- chart_summary stays on LLM → no work needed (out of scope) ✓

**Placeholder scan:** none in unit-testable code; the only "inspect & adapt" steps are the genuinely external HF schemas (Task 3 Step 1) and the LLM client wiring (Task 5 `measure_tokens`, gated/optional), each flagged with the real-API verification instruction.

**Type consistency:** predictors return `Prediction(value, 1.0)`; keyword value = `list[str]` (KeywordJaccard); aspect value = `list[{"ticker","direction"}]` matching `FIELD_METRICS["assets"]` ListStruct key `ticker` + compare `direction`; HF adapter rows = `{"article": {...}, "labels": {field: ...}}` consumed by `score_candidate`/`validate_human`; `call2_reduction` keys (`full_cost`,`slim_cost`,`reduction_fraction`,`saved_per_article`) consistent with its test; `characterize_field` reused unchanged from Plan 2a.

**Risk for executor:** Tasks 3 & 5's real backends are the only external-API risks; if an HF schema or the LLM client differ, fix the adapter/measure module and its synthetic test, never the characterize/score/cost logic.
