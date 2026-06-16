# Plan 2a — Contagion Engine + Characterization Framework

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the LLM `contagion_links` field with a learned relation-extraction engine (GLiREL zero-shot + `dependency_graph` prior + FinBERT direction), and build the characterization harness that measures Call-2 replacements against the GLM oracle and (for contagion) REFinD human gold.

**Architecture:** New article-reading predictors plug into the existing `eval` framework via the `Predictor` protocol (`fit` is a no-op; `predict_one(row)` reads `row["article"]` and returns `Prediction(value=<list>, confidence=1.0)`), scored by the existing `FIELD_METRICS["contagion_links"]` ListStruct metric. Because GLM is unreliable for contagion (0.26 vs Opus), these fields are **characterized, not τ-gated**: a new `characterize_field` runner reports mean GLM fidelity per candidate, and a separate `re_eval` harness reports relation accuracy vs REFinD human gold. All heavy models/datasets are **dependency-injected**, so unit tests use fakes and never load a model; real backends live behind gated integration tests.

**Tech Stack:** Python, reuses `eval` framework; new deps `glirel` (+ spaCy, present) for the real backend; `datasets` (present) for REFinD. FinBERT/`dependency_graph.json`/`AssetExtractor` already in `talisman-ai`.

**Predictor row shape:** `{"article": {"id","title","content",...}, "labels": {...}, "features": None}` (same as Plan 1; contagion predictors ignore `features` and read `article`).

**Key external APIs (verified):**
- `talisman_ai.analyzer.asset_extractor.AssetExtractor().extract_assets(title, body, max_assets=20) -> List[AssetMatch]`; `AssetMatch.ticker`, `.asset_class`.
- `dependency_graph.json` at `talisman_ai/analyzer/data/dependency_graph.json`, shape `{"dependencies": {TICKER: {"dependents": [{"ticker","asset_class","relationship"}]}}}`.
- FinBERT: `transformers.pipeline("sentiment-analysis", model="ProsusAI/finbert", device=-1)(sent)[0] -> {"label": "positive|negative|neutral", "score": float}`.
- GLiREL: `glirel.GLiREL.from_pretrained(<id>).predict_relations(tokens, labels, threshold, ner)` → list of `{"head_text","tail_text","label","score"}` (exact shape verified in Task 5's gated test).

---

### Task 1: Characterization runner

**Files:**
- Create: `eval/bench/characterize.py`
- Test: `tests/test_characterize.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_characterize.py
from eval.bench.characterize import characterize_field, FieldChar, CandidateChar, render_characterization
from eval.predictors.base import Prediction

class FakePred:
    def __init__(self, name, mapping): self.name = name; self._m = mapping
    def fit(self, rows): return self
    def predict_one(self, row): return Prediction(self._m[row["article"]["id"]], 1.0)

def _rows():
    # contagion_links scored by ListStruct keyed on (source_ticker, target_ticker)
    gold = [{"source_ticker": "ETH", "target_ticker": "stETH"}]
    return [{"article": {"id": 0}, "labels": {"contagion_links": gold}, "features": None}]

def test_characterize_reports_per_candidate_fidelity():
    rows = _rows()
    perfect = FakePred("a", {0: [{"source_ticker": "ETH", "target_ticker": "stETH"}]})
    empty = FakePred("b", {0: []})
    fc = characterize_field("contagion_links", [("a", perfect), ("b", empty)], rows)
    assert isinstance(fc, FieldChar)
    by = {c.name: c for c in fc.candidates}
    assert by["a"].glm_fidelity == 1.0 and by["a"].n_glm == 1
    assert by["b"].glm_fidelity == 0.0

def test_render_characterization_has_rows():
    fc = FieldChar("contagion_links", [CandidateChar("graph_only", 0.31, 100)])
    md = render_characterization([fc])
    assert "contagion_links" in md and "graph_only" in md and "0.31" in md
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/rizzo/talisman/eval && pytest tests/test_characterize.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'eval.bench.characterize'`

- [ ] **Step 3: Write minimal implementation**

```python
# eval/bench/characterize.py
from dataclasses import dataclass
from statistics import mean
from eval.bench.score import score_candidate

@dataclass
class CandidateChar:
    name: str
    glm_fidelity: float          # mean agreement-with-GLM over scored rows
    n_glm: int
    human_gold: float | None = None    # filled by a separate human-gold harness
    human_source: str | None = None
    n_human: int = 0

@dataclass
class FieldChar:
    field: str
    candidates: list             # list[CandidateChar]

def characterize_field(field, named_predictors, glm_rows) -> FieldChar:
    """named_predictors: list of (name, predictor). Reports GLM fidelity per candidate."""
    cands = []
    for name, predictor in named_predictors:
        predictor.fit(glm_rows)
        scored = score_candidate(field, predictor, glm_rows)
        fid = round(mean(s for s, _ in scored), 4) if scored else 0.0
        cands.append(CandidateChar(name, fid, len(scored)))
    return FieldChar(field, cands)

def render_characterization(field_chars) -> str:
    head = ["field", "candidate", "glm_fidelity", "n_glm", "human_gold", "human_source", "n_human"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for fc in field_chars:
        for c in fc.candidates:
            hg = "" if c.human_gold is None else f"{c.human_gold:.3f}"
            lines.append("| " + " | ".join([
                fc.field, c.name, f"{c.glm_fidelity:.3f}", str(c.n_glm),
                hg, c.human_source or "", str(c.n_human)]) + " |")
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_characterize.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
cd /home/rizzo/talisman/eval
git add eval/bench/characterize.py tests/test_characterize.py
git commit -m "feat: characterization runner for non-gated Call-2 field replacements"
```

---

### Task 2: Relation → mechanism mapping (pure logic)

**Files:**
- Create: `eval/predictors/contagion_map.py`
- Test: `tests/test_contagion_map.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_contagion_map.py
from eval.predictors.contagion_map import relation_to_mechanism, make_link

def test_known_relations_map_to_enum():
    assert relation_to_mechanism("supply_chain") == "SUPPLY_CHAIN"
    assert relation_to_mechanism("competitor") == "COMPETITIVE"
    assert relation_to_mechanism("ecosystem_dependency") == "PROTOCOL_DEPENDENCY"
    assert relation_to_mechanism("liquid_staking_derivative") == "PROTOCOL_DEPENDENCY"
    assert relation_to_mechanism("ecosystem_token") == "PROTOCOL_DEPENDENCY"

def test_unknown_relation_defaults_to_correlation():
    assert relation_to_mechanism("totally_unknown") == "CORRELATION"

def test_make_link_shape():
    link = make_link("ETH", "stETH", "crypto", "PROTOCOL_DEPENDENCY", "NEUTRAL", 0.8)
    assert link["source_ticker"] == "ETH" and link["target_ticker"] == "stETH"
    assert link["target_asset_class"] == "CRYPTO"
    assert link["mechanism"] == "PROTOCOL_DEPENDENCY" and link["direction"] == "NEUTRAL"
    assert 0.0 <= link["strength"] <= 1.0 and 0.0 <= link["confidence"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_contagion_map.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# eval/predictors/contagion_map.py
"""Map relation labels (GLiREL output OR dependency_graph relationship strings)
to the ContagionMechanism enum, and build ContagionLink dicts."""

# ContagionMechanism enum values: CORRELATION, SUPPLY_CHAIN, REGULATORY_SPILLOVER,
# CAPITAL_FLOW, NARRATIVE, COLLATERAL, PROTOCOL_DEPENDENCY, COMPETITIVE, MACRO_SENSITIVITY
_RELATION_MAP = {
    "supply_chain": "SUPPLY_CHAIN",
    "competitor": "COMPETITIVE",
    "regulatory_spillover": "REGULATORY_SPILLOVER",
    "capital_flow": "CAPITAL_FLOW",
    "ecosystem_dependency": "PROTOCOL_DEPENDENCY",
    "protocol_dependency": "PROTOCOL_DEPENDENCY",
    "collateral": "COLLATERAL",
    "macro_sensitivity": "MACRO_SENSITIVITY",
    "correlation": "CORRELATION",
    "narrative": "NARRATIVE",
    # dependency_graph.json relationship strings:
    "liquid_staking_derivative": "PROTOCOL_DEPENDENCY",
    "ecosystem_token": "PROTOCOL_DEPENDENCY",
}

def relation_to_mechanism(label: str) -> str:
    return _RELATION_MAP.get(str(label).lower(), "CORRELATION")

def make_link(source, target, asset_class, mechanism, direction, score) -> dict:
    score = max(0.0, min(1.0, float(score)))
    return {
        "source_ticker": source,
        "target_ticker": target,
        "target_asset_class": str(asset_class).upper(),
        "mechanism": mechanism,
        "direction": direction,
        "strength": score,
        "confidence": score,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_contagion_map.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
cd /home/rizzo/talisman/eval
git add eval/predictors/contagion_map.py tests/test_contagion_map.py
git commit -m "feat: relation->mechanism mapping + ContagionLink builder"
```

---

### Task 3: Graph-only contagion predictor (baseline)

**Files:**
- Create: `eval/predictors/contagion.py`
- Test: `tests/test_contagion_graph.py`

This is the floor rung: today's deterministic behavior. Dependencies (`graph` dict and `extract_tickers` callable) are injected so the unit test loads no data and no model.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_contagion_graph.py
from eval.predictors.contagion import GraphOnlyContagion

def test_graph_only_emits_links_for_detected_tickers():
    graph = {"ETH": {"dependents": [
        {"ticker": "stETH", "asset_class": "crypto", "relationship": "liquid_staking_derivative"},
        {"ticker": "UNI", "asset_class": "crypto", "relationship": "ecosystem_token"}]}}
    pred = GraphOnlyContagion(graph=graph, extract_tickers=lambda title, body: ["ETH"])
    out = pred.fit([]).predict_one({"article": {"title": "t", "content": "ETH news"}})
    links = out.value
    assert {l["target_ticker"] for l in links} == {"stETH", "UNI"}
    assert all(l["source_ticker"] == "ETH" for l in links)
    assert all(l["mechanism"] == "PROTOCOL_DEPENDENCY" for l in links)
    assert pred.name == "graph_only"

def test_graph_only_no_tickers_no_links():
    pred = GraphOnlyContagion(graph={}, extract_tickers=lambda title, body: [])
    out = pred.predict_one({"article": {"title": "", "content": "nothing"}})
    assert out.value == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_contagion_graph.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# eval/predictors/contagion.py
from eval.predictors.base import Prediction
from eval.predictors.contagion_map import relation_to_mechanism, make_link

_GRAPH_PATH = "/home/rizzo/talisman/talisman-ai/talisman_ai/analyzer/data/dependency_graph.json"

def _default_graph():
    import json, pathlib
    return json.loads(pathlib.Path(_GRAPH_PATH).read_text())["dependencies"]

def _default_extract_tickers(title, body):
    from talisman_ai.analyzer.asset_extractor import AssetExtractor
    matches = AssetExtractor().extract_assets(title or "", body or "")
    seen, out = set(), []
    for m in matches:
        if m.ticker not in seen:
            seen.add(m.ticker); out.append(m.ticker)
    return out

class GraphOnlyContagion:
    name = "graph_only"
    def __init__(self, graph=None, extract_tickers=None):
        self._graph = graph
        self._extract = extract_tickers
    def _ensure(self):
        if self._graph is None:
            self._graph = _default_graph()
        if self._extract is None:
            self._extract = _default_extract_tickers
    def fit(self, rows):
        return self
    def predict_one(self, row):
        self._ensure()
        art = row["article"]
        tickers = self._extract(art.get("title"), art.get("content"))
        links = []
        for t in tickers:
            for dep in self._graph.get(t, {}).get("dependents", []):
                links.append(make_link(t, dep["ticker"], dep.get("asset_class", "unknown"),
                                       relation_to_mechanism(dep.get("relationship", "")),
                                       "NEUTRAL", 0.7))
        return Prediction(links, 1.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_contagion_graph.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
cd /home/rizzo/talisman/eval
git add eval/predictors/contagion.py tests/test_contagion_graph.py
git commit -m "feat: graph-only contagion baseline predictor"
```

---

### Task 4: GLiREL contagion predictor (injected backend)

**Files:**
- Modify: `eval/predictors/contagion.py` (add `GlirelContagion`)
- Test: `tests/test_contagion_glirel.py`

The relation scorer and direction function are injected. A `relation_scorer(title, body, tickers)` returns a list of `(head_ticker, tail_ticker, relation_label, score)`. A `direction_fn(title, body, target)` returns one of `POSITIVE/NEUTRAL/NEGATIVE/MIXED`. The graph is fused as a precision prior (graph links always included; text links added; dedup on (source, target)).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_contagion_glirel.py
from eval.predictors.contagion import GlirelContagion

def _scorer(title, body, tickers):
    # event-specific link the static graph would miss
    return [("NVDA", "AMD", "competitor", 0.82)] if "NVDA" in tickers else []

def test_glirel_emits_text_links_with_direction():
    pred = GlirelContagion(
        relation_scorer=_scorer,
        direction_fn=lambda title, body, target: "NEGATIVE",
        extract_tickers=lambda title, body: ["NVDA", "AMD"],
        graph={})
    out = pred.fit([]).predict_one({"article": {"title": "t", "content": "NVDA vs AMD"}})
    links = out.value
    assert len(links) == 1
    l = links[0]
    assert l["source_ticker"] == "NVDA" and l["target_ticker"] == "AMD"
    assert l["mechanism"] == "COMPETITIVE" and l["direction"] == "NEGATIVE"
    assert abs(l["strength"] - 0.82) < 1e-9
    assert pred.name == "glirel"

def test_glirel_fuses_graph_prior_and_dedups():
    # graph says ETH->stETH; scorer also emits ETH->stETH (text) -> only one link
    graph = {"ETH": {"dependents": [
        {"ticker": "stETH", "asset_class": "crypto", "relationship": "liquid_staking_derivative"}]}}
    def scorer(title, body, tickers):
        return [("ETH", "stETH", "ecosystem_dependency", 0.6)]
    pred = GlirelContagion(relation_scorer=scorer,
                           direction_fn=lambda *a: "NEUTRAL",
                           extract_tickers=lambda title, body: ["ETH"],
                           graph=graph)
    links = pred.predict_one({"article": {"title": "t", "content": "ETH"}}).value
    keys = [(l["source_ticker"], l["target_ticker"]) for l in links]
    assert keys.count(("ETH", "stETH")) == 1   # deduped
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_contagion_glirel.py -v`
Expected: FAIL `ImportError: cannot import name 'GlirelContagion'`

- [ ] **Step 3: Write minimal implementation (append to `eval/predictors/contagion.py`)**

```python
class GlirelContagion:
    """Text-derived contagion via an injected relation scorer, fused with the
    dependency_graph prior. Graph links are kept (high precision); text links add
    event-specific edges; dedup on (source, target) with graph winning ties."""
    name = "glirel"
    def __init__(self, relation_scorer=None, direction_fn=None,
                 extract_tickers=None, graph=None):
        self._scorer = relation_scorer
        self._direction = direction_fn
        self._extract = extract_tickers
        self._graph = graph
    def _ensure(self):
        if self._graph is None:
            self._graph = _default_graph()
        if self._extract is None:
            self._extract = _default_extract_tickers
        if self._scorer is None:
            from eval.predictors.glirel_backend import glirel_scorer
            self._scorer = glirel_scorer()
        if self._direction is None:
            from eval.predictors.glirel_backend import finbert_direction
            self._direction = finbert_direction()
    def fit(self, rows):
        return self
    def predict_one(self, row):
        self._ensure()
        art = row["article"]
        title, body = art.get("title"), art.get("content")
        tickers = self._extract(title, body)
        by_key = {}
        # graph prior first (precision)
        for t in tickers:
            for dep in self._graph.get(t, {}).get("dependents", []):
                key = (t, dep["ticker"])
                by_key[key] = make_link(t, dep["ticker"], dep.get("asset_class", "unknown"),
                                        relation_to_mechanism(dep.get("relationship", "")),
                                        "NEUTRAL", 0.7)
        # text-derived links (do not overwrite a graph link)
        for head, tail, label, score in self._scorer(title, body, tickers):
            key = (head, tail)
            if key in by_key:
                continue
            direction = self._direction(title, body, tail)
            by_key[key] = make_link(head, tail, "UNKNOWN",
                                    relation_to_mechanism(label), direction, score)
        return Prediction(list(by_key.values()), 1.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_contagion_glirel.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
cd /home/rizzo/talisman/eval
git add eval/predictors/contagion.py tests/test_contagion_glirel.py
git commit -m "feat: GLiREL contagion predictor (injected scorer + graph fusion)"
```

---

### Task 5: Real GLiREL/FinBERT backend + REFinD RE-eval (gated)

**Files:**
- Create: `eval/predictors/glirel_backend.py`
- Create: `eval/hf/__init__.py`, `eval/hf/refind.py`
- Create: `eval/bench/re_eval.py`
- Test: `tests/test_re_eval.py` (unit, injected) + `tests/test_glirel_backend_gated.py` (gated)

`re_eval` is the human-gold harness: given a relation scorer and REFinD-style gold tuples `(text, head, tail, gold_relation)`, it reports relation accuracy. The unit test injects a fake scorer; the real GLiREL/REFinD paths are gated behind `RUN_MODEL_TESTS=1`.

- [ ] **Step 1: Write the failing unit test (no models)**

```python
# tests/test_re_eval.py
from eval.bench.re_eval import relation_accuracy

def test_relation_accuracy_counts_matches():
    gold = [("Apple supplies chips to Foo", "AAPL", "FOO", "supply_chain"),
            ("A competes with B", "A", "B", "competitor")]
    # scorer returns (head, tail, label, score); label compared case-insensitively
    def scorer(text, head, tail):
        return "supply_chain" if head == "AAPL" else "wrong_label"
    acc = relation_accuracy(scorer, gold)
    assert abs(acc["accuracy"] - 0.5) < 1e-9 and acc["n"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_re_eval.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Write `eval/bench/re_eval.py`**

```python
# eval/bench/re_eval.py
def relation_accuracy(pair_scorer, gold_tuples) -> dict:
    """pair_scorer(text, head, tail) -> predicted relation label (str).
    gold_tuples: iterable of (text, head, tail, gold_label)."""
    n = correct = 0
    for text, head, tail, gold in gold_tuples:
        pred = pair_scorer(text, head, tail)
        n += 1
        if pred is not None and str(pred).lower() == str(gold).lower():
            correct += 1
    return {"accuracy": (correct / n) if n else 0.0, "n": n, "correct": correct}
```

- [ ] **Step 4: Run unit test to verify it passes**

Run: `pytest tests/test_re_eval.py -v`
Expected: PASS

- [ ] **Step 5: Write the real backend + REFinD adapter (exercised only by gated test)**

```python
# eval/predictors/glirel_backend.py
"""Real GLiREL + FinBERT backends. Loaded lazily; only used in real runs."""
RELATION_LABELS = ["supply_chain", "competitor", "regulatory_spillover", "capital_flow",
                   "ecosystem_dependency", "collateral", "macro_sensitivity",
                   "correlation", "protocol_dependency"]

def _load_glirel():
    import spacy
    from glirel import GLiREL
    nlp = spacy.load("en_core_web_sm")
    model = GLiREL.from_pretrained("jackboyla/glirel-large-v0")
    return nlp, model

def glirel_scorer(labels=None):
    """Returns scorer(title, body, tickers) -> list[(head, tail, label, score)].
    Verify the exact predict_relations return shape in the gated test and adapt."""
    labels = labels or RELATION_LABELS
    nlp, model = _load_glirel()
    def _scorer(title, body, tickers):
        text = f"{title or ''}. {body or ''}"[:2000]
        doc = nlp(text)
        tokens = [t.text for t in doc]
        # NER spans for the detected tickers (string match over tokens)
        ner = []
        for tk in tickers:
            for i, tok in enumerate(tokens):
                if tok == tk:
                    ner.append([i, i, "ASSET", tk])
        rels = model.predict_relations(tokens, labels, threshold=0.0, ner=ner)
        out = []
        for r in rels:
            h, t = r.get("head_text"), r.get("tail_text")
            if h in tickers and t in tickers and h != t:
                out.append((h, t, r["label"], float(r.get("score", 0.0))))
        return out
    return _scorer

def finbert_direction():
    from transformers import pipeline
    clf = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=-1)
    _map = {"positive": "POSITIVE", "negative": "NEGATIVE", "neutral": "NEUTRAL"}
    def _dir(title, body, target):
        sents = [s for s in f"{title or ''}. {body or ''}".split(".") if target and target in s]
        if not sents:
            return "NEUTRAL"
        res = clf(sents[0][:512], truncation=True, max_length=512)[0]
        return _map.get(res["label"].lower(), "NEUTRAL")
    return _dir
```

```python
# eval/hf/__init__.py
"""HuggingFace human-gold adapters."""
```

```python
# eval/hf/refind.py
"""REFinD financial relation-extraction human gold -> (text, head, tail, relation) tuples.
Dataset id is configurable; verify availability in the gated test and adjust the field
names to the actual REFinD schema if they differ."""
def load_refind(split="test", dataset_id="gtfintechlab/REFinD", limit=None):
    from datasets import load_dataset
    ds = load_dataset(dataset_id, split=split)
    out = []
    for ex in ds:
        text = ex.get("sentence") or ex.get("text") or ""
        head = (ex.get("e1") or ex.get("head") or "")
        tail = (ex.get("e2") or ex.get("tail") or "")
        rel = (ex.get("relation") or ex.get("rel_group") or "")
        if text and head and tail and rel:
            out.append((text, head, tail, rel))
        if limit and len(out) >= limit:
            break
    return out
```

- [ ] **Step 6: Write the gated test**

```python
# tests/test_glirel_backend_gated.py
import os, pytest
pytestmark = pytest.mark.skipif(os.environ.get("RUN_MODEL_TESTS") != "1",
                                reason="set RUN_MODEL_TESTS=1 for GLiREL/REFinD")

def test_glirel_scorer_runs():
    from eval.predictors.glirel_backend import glirel_scorer
    scorer = glirel_scorer()
    links = scorer("NVDA and AMD compete", "NVDA and AMD compete in GPUs.", ["NVDA", "AMD"])
    assert isinstance(links, list)  # shape: list[(head, tail, label, score)]

def test_refind_loads():
    from eval.hf.refind import load_refind
    rows = load_refind(limit=5)
    assert len(rows) >= 1 and len(rows[0]) == 4
```

- [ ] **Step 7: Run gated test (verify real APIs; adapt code if shapes differ)**

Run: `RUN_MODEL_TESTS=1 pytest tests/test_glirel_backend_gated.py -v` (after Task 6 installs deps)
Expected: PASS, or actionable errors that tell you the real `predict_relations` / REFinD field names — fix `glirel_backend.py` / `refind.py` to match, then re-run. If REFinD is unavailable on HF under any id, report it (DONE_WITH_CONCERNS) so the controller can pick an alternative RE gold set (FinRED).

- [ ] **Step 8: Commit**

```bash
cd /home/rizzo/talisman/eval
git add eval/predictors/glirel_backend.py eval/hf/__init__.py eval/hf/refind.py \
        eval/bench/re_eval.py tests/test_re_eval.py tests/test_glirel_backend_gated.py
git commit -m "feat: real GLiREL/FinBERT backend + REFinD RE-eval (gated)"
```

---

### Task 6: CLI + real contagion characterization run

**Files:**
- Modify: `eval/cli.py` (add `characterize` subcommand)
- Test: `tests/test_cli_characterize.py`

- [ ] **Step 1: Write the failing test (uses fake predictors via the graph-only path, no models)**

```python
# tests/test_cli_characterize.py
import json
from pathlib import Path
from eval.cli import main

def _gold(path):
    with open(path, "w") as fh:
        fh.write(json.dumps({"article": {"id": 0, "title": "t", "content": "ETH news"},
                             "labels": {"contagion_links": [
                                 {"source_ticker": "ETH", "target_ticker": "stETH"}]}}) + "\n")

def test_characterize_contagion_graph_only(tmp_path, monkeypatch):
    gold = tmp_path / "test.jsonl"; _gold(gold)
    # force the graph-only predictor to use an injected tiny graph + extractor via env-free monkeypatch
    import eval.predictors.contagion as C
    monkeypatch.setattr(C, "_default_graph", lambda: {"ETH": {"dependents": [
        {"ticker": "stETH", "asset_class": "crypto", "relationship": "liquid_staking_derivative"}]}})
    monkeypatch.setattr(C, "_default_extract_tickers", lambda title, body: ["ETH"])
    out = tmp_path / "char.json"
    main(["characterize", "--field", "contagion_links", "--gold", str(gold),
          "--candidates", "graph_only", "--out", str(out)])
    data = json.loads(out.read_text())
    assert data["field"] == "contagion_links"
    assert data["candidates"][0]["name"] == "graph_only"
    assert data["candidates"][0]["glm_fidelity"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_characterize.py -v`
Expected: FAIL (argparse `invalid choice: 'characterize'`)

- [ ] **Step 3: Implement the `characterize` subcommand in `eval/cli.py`**

Add subparser (before `return p`):

```python
    ch = sub.add_parser("characterize")
    ch.add_argument("--field", required=True)
    ch.add_argument("--gold", required=True)
    ch.add_argument("--candidates", nargs="+", required=True)
    ch.add_argument("--out")
```

Add a candidate registry + handler (in `main`, before the final `if __name__`):

```python
    elif ns.cmd == "characterize":
        from dataclasses import asdict
        from eval.distill.split import load_rows
        from eval.bench.characterize import characterize_field, render_characterization
        from eval.predictors.contagion import GraphOnlyContagion, GlirelContagion
        registry = {"graph_only": GraphOnlyContagion, "glirel": GlirelContagion}
        named = [(c, registry[c]()) for c in ns.candidates]
        rows = load_rows(ns.gold)
        fc = characterize_field(ns.field, named, rows)
        print(render_characterization([fc]))
        if ns.out:
            Path(ns.out).write_text(json.dumps(
                {"field": fc.field, "candidates": [asdict(c) for c in fc.candidates]}, indent=2))
            print(f"\ncharacterization -> {ns.out}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli_characterize.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /home/rizzo/talisman/eval
git add eval/cli.py tests/test_cli_characterize.py
git commit -m "feat: CLI characterize subcommand"
```

- [ ] **Step 6: Full suite green, install deps, real run**

```bash
cd /home/rizzo/talisman/eval
pytest -q                       # all green, gated tests skip
pip install glirel
python -m spacy download en_core_web_sm
```

Reuse the Plan-1 held-out test split (`eval/data/distill/test.jsonl`, 750 GLM rows). Characterize contagion over GLM, both candidates:

```bash
python -m eval.cli characterize --field contagion_links \
  --gold eval/data/distill/test.jsonl --candidates graph_only glirel \
  --out eval/reports/contagion_characterization.json 2>/dev/null
```

Then REFinD validation (gated path, real models):

```bash
RUN_MODEL_TESTS=1 pytest tests/test_glirel_backend_gated.py -v
python - <<'PY'
from eval.predictors.glirel_backend import glirel_scorer
from eval.hf.refind import load_refind
from eval.bench.re_eval import relation_accuracy
scorer = glirel_scorer()
gold = load_refind(split="test", limit=500)
def pair(text, head, tail):
    links = scorer(head + " " + tail, text, [head, tail])
    return links[0][2] if links else None
print(relation_accuracy(pair, gold))
PY
```

Report: the contagion characterization table (graph_only vs glirel GLM-fidelity) and the REFinD relation accuracy. Note that low GLM-fidelity is expected (GLM contagion is 0.26-noise); REFinD accuracy is the trustworthy signal.

- [ ] **Step 7: Commit the report**

```bash
cd /home/rizzo/talisman/eval
git add eval/reports/contagion_characterization.json
git commit -m "report: contagion characterization (graph_only vs GLiREL) + REFinD accuracy"
```

---

## Self-Review

**Spec coverage (against `2026-06-16-call2-replacement-design.md`, contagion portion):**
- GLiREL zero-shot RE + graph prior + FinBERT direction → Tasks 3–5 ✓
- Relation→mechanism mapping → Task 2 ✓
- REFinD human-gold validation (RE accuracy, separate from GLM) → Task 5 ✓
- Characterize-not-gate framing → Task 1 (no τ, reports fidelity) ✓
- Reuse existing ListStruct metric via `score_candidate` → Tasks 1, 6 ✓
- Keywords / per-asset sentiment / HF sentiment adapters / token-cost → **Plan 2b** (not this plan) ✓

**Placeholder scan:** none — all unit-test code is complete; the only "verify and adapt" steps are the genuinely external GLiREL/REFinD API shapes, isolated to the gated Task 5 with explicit fallback instructions.

**Type consistency:** predictors return `Prediction(value=list, confidence=1.0)`; `score_candidate(field, predictor, rows)` → `[(score, conf)]` reused from Plan 1; `make_link` dict keys match the ListStruct key `(source_ticker, target_ticker)` and the `_contagion` compare; `relation_scorer` signature `(title, body, tickers) -> [(head, tail, label, score)]` consistent between Task 4's fake and Task 5's real backend; `re_eval.relation_accuracy(pair_scorer, gold_tuples)` with `pair_scorer(text, head, tail)->label`.

**Risk flagged for executor:** Task 5 is the only real-API-dependent task; if REFinD's HF id/schema or GLiREL's `predict_relations` shape differ, fix those two files and re-run the gated test — do not touch the unit-tested logic in Tasks 1–4.
