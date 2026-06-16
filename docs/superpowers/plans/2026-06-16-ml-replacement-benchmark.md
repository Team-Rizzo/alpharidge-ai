# ML Replacement Benchmark — Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Benchmark distilled CPU-cheap models against the held-out GLM 5.1 oracle for every single-value `ArticleIntelligence` field, via a tiered candidate ladder, and emit a per-field REPLACE / HYBRID / KEEP decision.

**Architecture:** Split the 5,000 GLM-labeled articles into train/test by article id. Embed each article once (MiniLM, CPU) into a feature vector. For each single-value field, fit a ladder of candidates (majority baseline → logistic-regression head → LightGBM head) on the train split, score each on the held-out test split with the existing `eval/metrics`, and apply a fidelity-to-GLM decision policy (τ default 0.90, confidence-gated hybrid fallback below τ, KEEP under 20% coverage). All decisions are driven by GLM agreement; Opus/HF are out of this plan (Plan 2).

**Tech Stack:** Python, numpy, scikit-learn, LightGBM, sentence-transformers (all already installed); reuses `eval/metrics`, `eval/article`, `eval/gold`.

**Scope — fields covered by this plan (single-value classification/regression):**
`content_type, market_analysis_type, impact_potential, technical_quality, urgency, temporal_focus, sentiment_direction, factual_confidence, source_attribution_type, positioning_signal, primary_geo, target_audience, credibility_flag, staleness_flag, forward_event_type` (classifier, scored by CategoricalMetric), `overall_sentiment` (classifier, scored by OrdinalSentimentMetric), `overall_sentiment_score` (regressor, scored by NumericMetric). List/struct/text/generative fields (`assets`, `entities`, `economic_data`, `contagion_links`, `narrative_keywords`, `chart_summary`) are **Plan 2**.

**Row format (used throughout):** a *row* is `{"article": {...}, "labels": {...}, "features": list[float] | None}`. Splitter produces rows with `features=None`; the feature step fills them.

**Data on disk:** GLM gold at `eval/data/gold_z-ai_glm-5.1.jsonl` (5,000 lines, each `{"article":..., "labels":..., "labeler":...}`).

---

### Task 1: Package scaffolding for distill / predictors / bench

**Files:**
- Create: `eval/distill/__init__.py`
- Create: `eval/predictors/__init__.py`
- Create: `eval/bench/__init__.py`

- [ ] **Step 1: Create the three empty package markers**

Each file contains exactly one line:

```python
# eval/distill/__init__.py
"""Distillation: train/test split + feature extraction + per-field heads."""
```

```python
# eval/predictors/__init__.py
"""Candidate predictors for the replacement ladder."""
```

```python
# eval/bench/__init__.py
"""Benchmark runner, decision policy, and reporting."""
```

- [ ] **Step 2: Verify imports work**

Run: `cd /home/rizzo/talisman/eval && python -c "import eval.distill, eval.predictors, eval.bench; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
cd /home/rizzo/talisman/eval
git add eval/distill/__init__.py eval/predictors/__init__.py eval/bench/__init__.py
git commit -m "scaffold: distill/predictors/bench packages"
```

---

### Task 2: Train/test splitter

**Files:**
- Create: `eval/distill/split.py`
- Test: `tests/test_split.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_split.py
import json
from eval.distill.split import split_rows, load_rows

def _write(tmp_path, n):
    p = tmp_path / "gold.jsonl"
    with p.open("w") as fh:
        for i in range(n):
            fh.write(json.dumps({"article": {"id": i, "url": "u", "title": "t",
                                             "source": "ccnews", "content": "c"},
                                 "labels": {"content_type": "analysis"},
                                 "labeler": "glm"}) + "\n")
    return p

def test_split_is_deterministic_and_disjoint(tmp_path):
    rows = load_rows(_write(tmp_path, 100))
    train, test = split_rows(rows, test_frac=0.2, seed=0)
    assert len(train) == 80 and len(test) == 20
    train_ids = {r["article"]["id"] for r in train}
    test_ids = {r["article"]["id"] for r in test}
    assert train_ids.isdisjoint(test_ids)
    # same seed -> identical split
    train2, test2 = split_rows(rows, test_frac=0.2, seed=0)
    assert {r["article"]["id"] for r in test2} == test_ids

def test_rows_have_features_none(tmp_path):
    rows = load_rows(_write(tmp_path, 5))
    assert all(r["features"] is None for r in rows)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_split.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.distill.split'`

- [ ] **Step 3: Write minimal implementation**

```python
# eval/distill/split.py
import json, random
from pathlib import Path

def load_rows(gold_path) -> list:
    rows = []
    for line in Path(gold_path).read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        rows.append({"article": d["article"], "labels": d["labels"], "features": None})
    return rows

def split_rows(rows, test_frac=0.2, seed=0):
    ids = sorted({r["article"]["id"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_test = int(len(ids) * test_frac)
    test_ids = set(ids[:n_test])
    train = [r for r in rows if r["article"]["id"] not in test_ids]
    test = [r for r in rows if r["article"]["id"] in test_ids]
    return train, test
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_split.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
cd /home/rizzo/talisman/eval
git add eval/distill/split.py tests/test_split.py
git commit -m "feat: deterministic train/test splitter for GLM gold"
```

---

### Task 3: Predictor base (Prediction + Protocol)

**Files:**
- Create: `eval/predictors/base.py`
- Test: `tests/test_predictor_base.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_predictor_base.py
from eval.predictors.base import Prediction

def test_prediction_defaults():
    p = Prediction(value="analysis")
    assert p.value == "analysis" and p.confidence == 1.0
    p2 = Prediction(value="x", confidence=0.3)
    assert p2.confidence == 0.3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_predictor_base.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# eval/predictors/base.py
from dataclasses import dataclass
from typing import Any, Protocol

@dataclass
class Prediction:
    value: Any
    confidence: float = 1.0

class Predictor(Protocol):
    name: str
    def fit(self, train_rows: list) -> "Predictor": ...
    def predict_one(self, row: dict) -> Prediction: ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_predictor_base.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /home/rizzo/talisman/eval
git add eval/predictors/base.py tests/test_predictor_base.py
git commit -m "feat: Predictor protocol + Prediction dataclass"
```

---

### Task 4: Baseline predictors (majority class + mean regressor)

**Files:**
- Create: `eval/predictors/baseline.py`
- Test: `tests/test_baseline.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_baseline.py
from eval.predictors.baseline import MajorityPredictor, MeanRegressor

def _rows(field, values):
    return [{"article": {"id": i}, "labels": {field: v}, "features": None}
            for i, v in enumerate(values)]

def test_majority_picks_most_common():
    rows = _rows("content_type", ["analysis", "analysis", "opinion"])
    p = MajorityPredictor("content_type").fit(rows)
    pred = p.predict_one({"labels": {}, "features": None})
    assert pred.value == "analysis" and pred.confidence == 0.0
    assert p.name == "majority"

def test_mean_regressor_returns_mean():
    rows = _rows("overall_sentiment_score", [0.0, 1.0, 0.5])
    p = MeanRegressor("overall_sentiment_score").fit(rows)
    assert abs(p.predict_one({}).value - 0.5) < 1e-9
    assert p.name == "mean"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_baseline.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# eval/predictors/baseline.py
from collections import Counter
from eval.predictors.base import Prediction

class MajorityPredictor:
    name = "majority"
    def __init__(self, field):
        self.field = field
        self.value = None
    def fit(self, train_rows):
        vals = [r["labels"].get(self.field) for r in train_rows
                if r["labels"].get(self.field) is not None]
        self.value = Counter(vals).most_common(1)[0][0] if vals else None
        return self
    def predict_one(self, row):
        return Prediction(self.value, 0.0)

class MeanRegressor:
    name = "mean"
    def __init__(self, field):
        self.field = field
        self.value = 0.0
    def fit(self, train_rows):
        vals = [r["labels"].get(self.field) for r in train_rows
                if isinstance(r["labels"].get(self.field), (int, float))]
        self.value = sum(vals) / len(vals) if vals else 0.0
        return self
    def predict_one(self, row):
        return Prediction(self.value, 0.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_baseline.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
cd /home/rizzo/talisman/eval
git add eval/predictors/baseline.py tests/test_baseline.py
git commit -m "feat: majority-class + mean-regressor baseline predictors"
```

---

### Task 5: Distilled classifier head (logistic regression)

**Files:**
- Create: `eval/predictors/distilled.py`
- Test: `tests/test_distilled_linear.py`

- [ ] **Step 1: Write the failing test**

Synthetic 2-D features that are linearly separable so the head learns perfectly.

```python
# tests/test_distilled_linear.py
from eval.predictors.distilled import LinearHead

def _rows(field):
    rows = []
    for i in range(20):
        # class A near (0,0), class B near (5,5)
        if i % 2 == 0:
            feat, lab = [0.0 + i*0.01, 0.0], "analysis"
        else:
            feat, lab = [5.0, 5.0 + i*0.01], "opinion"
        rows.append({"article": {"id": i}, "labels": {field: lab}, "features": feat})
    return rows

def test_linear_head_learns_and_confidence_in_unit():
    field = "content_type"
    rows = _rows(field)
    p = LinearHead(field).fit(rows)
    assert p.name == "linear"
    pred_a = p.predict_one({"features": [0.0, 0.0]})
    pred_b = p.predict_one({"features": [5.0, 5.0]})
    assert pred_a.value == "analysis"
    assert pred_b.value == "opinion"
    assert 0.0 <= pred_a.confidence <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_distilled_linear.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# eval/predictors/distilled.py
import numpy as np
from eval.predictors.base import Prediction

def _xy(train_rows, field):
    X, y = [], []
    for r in train_rows:
        lab = r["labels"].get(field)
        if lab is None or r["features"] is None:
            continue
        X.append(r["features"]); y.append(lab)
    return np.array(X, dtype=float), y

class LinearHead:
    name = "linear"
    def __init__(self, field):
        self.field = field
        self.clf = None
        self.fallback = None
    def fit(self, train_rows):
        from sklearn.linear_model import LogisticRegression
        X, y = _xy(train_rows, self.field)
        if len(set(y)) < 2:               # degenerate: only one class present
            self.fallback = y[0] if y else None
            return self
        self.clf = LogisticRegression(max_iter=1000, multi_class="auto").fit(X, y)
        return self
    def predict_one(self, row):
        if self.clf is None:
            return Prediction(self.fallback, 0.0)
        x = np.array(row["features"], dtype=float).reshape(1, -1)
        proba = self.clf.predict_proba(x)[0]
        i = int(proba.argmax())
        return Prediction(self.clf.classes_[i], float(proba[i]))

class LGBMHead:
    name = "lgbm"
    def __init__(self, field):
        self.field = field
        self.clf = None
        self.classes_ = None
        self.fallback = None
    def fit(self, train_rows):
        import lightgbm as lgb
        X, y = _xy(train_rows, self.field)
        classes = sorted(set(y))
        if len(classes) < 2:
            self.fallback = classes[0] if classes else None
            return self
        self.classes_ = classes
        idx = {c: i for i, c in enumerate(classes)}
        yi = np.array([idx[v] for v in y])
        self.clf = lgb.LGBMClassifier(n_estimators=200, num_leaves=31,
                                      verbose=-1).fit(X, yi)
        return self
    def predict_one(self, row):
        if self.clf is None:
            return Prediction(self.fallback, 0.0)
        x = np.array(row["features"], dtype=float).reshape(1, -1)
        proba = self.clf.predict_proba(x)[0]
        i = int(proba.argmax())
        return Prediction(self.classes_[i], float(proba[i]))

class RidgeHead:
    name = "ridge"
    def __init__(self, field):
        self.field = field
        self.reg = None
    def fit(self, train_rows):
        from sklearn.linear_model import Ridge
        X, y = _xy(train_rows, self.field)
        if len(X) == 0:
            return self
        self.reg = Ridge(alpha=1.0).fit(X, np.array(y, dtype=float))
        return self
    def predict_one(self, row):
        if self.reg is None:
            return Prediction(0.0, 0.0)
        x = np.array(row["features"], dtype=float).reshape(1, -1)
        return Prediction(float(self.reg.predict(x)[0]), 1.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_distilled_linear.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /home/rizzo/talisman/eval
git add eval/predictors/distilled.py tests/test_distilled_linear.py
git commit -m "feat: distilled LinearHead/LGBMHead classifiers + RidgeHead regressor"
```

---

### Task 6: LightGBM + Ridge head tests

**Files:**
- Test: `tests/test_distilled_lgbm.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_distilled_lgbm.py
from eval.predictors.distilled import LGBMHead, RidgeHead

def _cls_rows(field):
    rows = []
    for i in range(40):
        if i % 2 == 0:
            feat, lab = [0.0, 0.0, float(i)], "analysis"
        else:
            feat, lab = [9.0, 9.0, float(i)], "opinion"
        rows.append({"article": {"id": i}, "labels": {field: lab}, "features": feat})
    return rows

def test_lgbm_head_learns():
    field = "content_type"
    p = LGBMHead(field).fit(_cls_rows(field))
    assert p.name == "lgbm"
    assert p.predict_one({"features": [0.0, 0.0, 2.0]}).value == "analysis"
    assert p.predict_one({"features": [9.0, 9.0, 3.0]}).value == "opinion"

def test_ridge_head_predicts_linear_target():
    field = "overall_sentiment_score"
    rows = [{"article": {"id": i}, "labels": {field: i / 10.0},
             "features": [float(i)]} for i in range(11)]  # y = x/10
    p = RidgeHead(field).fit(rows)
    out = p.predict_one({"features": [5.0]}).value
    assert abs(out - 0.5) < 0.1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_distilled_lgbm.py -v`
Expected: FAIL only if implementation broken; since Task 5 already wrote these classes, this test should PASS. If it fails, fix `distilled.py`.

- [ ] **Step 3: (No new implementation — classes exist from Task 5.)**

If the test fails, correct `eval/predictors/distilled.py` until it passes.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_distilled_lgbm.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
cd /home/rizzo/talisman/eval
git add tests/test_distilled_lgbm.py
git commit -m "test: LGBMHead + RidgeHead behavior"
```

---

### Task 7: Decision policy (fidelity-to-GLM → REPLACE/HYBRID/KEEP)

**Files:**
- Create: `eval/bench/decide.py`
- Test: `tests/test_decide.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_decide.py
from eval.bench.decide import decide, Decision

def test_replace_when_fidelity_meets_tau():
    items = [(1.0, 0.9)] * 9 + [(0.0, 0.1)]   # mean 0.9
    d = decide("content_type", items, tau=0.90, min_coverage=0.2)
    assert d.decision == "REPLACE" and d.coverage == 1.0 and d.confidence_threshold is None

def test_hybrid_gates_on_confidence():
    # high-confidence items are correct, low-confidence ones wrong
    items = [(1.0, 0.95)] * 6 + [(0.0, 0.3)] * 4   # overall mean 0.6
    d = decide("content_type", items, tau=0.90, min_coverage=0.2)
    assert d.decision == "HYBRID"
    assert d.confidence_threshold is not None
    assert d.coverage >= 0.2          # the 6 high-conf items are auto-handled
    assert abs(d.coverage - 0.6) < 1e-9

def test_keep_when_no_threshold_gives_coverage():
    items = [(0.5, 0.5)] * 10          # never reaches tau at any threshold
    d = decide("content_type", items, tau=0.90, min_coverage=0.2)
    assert d.decision == "KEEP" and d.coverage == 0.0

def test_empty_items_keep():
    d = decide("content_type", [], tau=0.90, min_coverage=0.2)
    assert d.decision == "KEEP" and d.fidelity == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_decide.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# eval/bench/decide.py
from dataclasses import dataclass
from statistics import mean

@dataclass
class Decision:
    field: str
    fidelity: float                 # mean agreement-with-GLM on held-out test
    decision: str                   # REPLACE | HYBRID | KEEP
    coverage: float                 # fraction auto-handled (1.0 REPLACE, 0.0 KEEP)
    confidence_threshold: float | None

def decide(field, scored_items, tau=0.90, min_coverage=0.20) -> Decision:
    """scored_items: list of (score in [0,1], confidence in [0,1])."""
    if not scored_items:
        return Decision(field, 0.0, "KEEP", 0.0, None)
    fidelity = mean(s for s, _ in scored_items)
    if fidelity >= tau:
        return Decision(field, fidelity, "REPLACE", 1.0, None)
    # Sweep candidate thresholds; keep those that hit tau on the retained subset.
    best = None  # (coverage, threshold)
    for c in sorted({conf for _, conf in scored_items}):
        kept = [s for s, conf in scored_items if conf >= c]
        if kept and mean(kept) >= tau:
            coverage = len(kept) / len(scored_items)
            if best is None or coverage > best[0]:
                best = (coverage, c)
    if best and best[0] >= min_coverage:
        return Decision(field, fidelity, "HYBRID", best[0], best[1])
    return Decision(field, fidelity, "KEEP", 0.0, None)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_decide.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
cd /home/rizzo/talisman/eval
git add eval/bench/decide.py tests/test_decide.py
git commit -m "feat: fidelity-to-GLM decision policy (replace/hybrid/keep)"
```

---

### Task 8: Field ladders config

**Files:**
- Create: `eval/bench/ladders.py`
- Test: `tests/test_ladders.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ladders.py
from eval.bench.ladders import SINGLE_VALUE_FIELDS, ladder_for

def test_classifier_field_ladder_order():
    names = [factory("content_type").name for factory in ladder_for("content_type")]
    assert names == ["majority", "linear", "lgbm"]

def test_regressor_field_ladder_order():
    names = [factory("overall_sentiment_score").name
             for factory in ladder_for("overall_sentiment_score")]
    assert names == ["mean", "ridge"]

def test_single_value_fields_cover_expected():
    assert "content_type" in SINGLE_VALUE_FIELDS
    assert "overall_sentiment" in SINGLE_VALUE_FIELDS
    assert "overall_sentiment_score" in SINGLE_VALUE_FIELDS
    # list/struct fields are NOT in this plan
    assert "assets" not in SINGLE_VALUE_FIELDS
    assert "contagion_links" not in SINGLE_VALUE_FIELDS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ladders.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# eval/bench/ladders.py
from eval.predictors.baseline import MajorityPredictor, MeanRegressor
from eval.predictors.distilled import LinearHead, LGBMHead, RidgeHead

# Single-value classification fields (scored by Categorical/Ordinal metrics).
CLASSIFIER_FIELDS = [
    "content_type", "market_analysis_type", "impact_potential", "technical_quality",
    "urgency", "temporal_focus", "sentiment_direction", "factual_confidence",
    "source_attribution_type", "positioning_signal", "primary_geo", "target_audience",
    "credibility_flag", "staleness_flag", "forward_event_type", "overall_sentiment",
]
# Single-value numeric fields (scored by NumericMetric).
REGRESSOR_FIELDS = ["overall_sentiment_score"]

SINGLE_VALUE_FIELDS = CLASSIFIER_FIELDS + REGRESSOR_FIELDS

_CLASSIFIER_LADDER = [MajorityPredictor, LinearHead, LGBMHead]
_REGRESSOR_LADDER = [MeanRegressor, RidgeHead]

def ladder_for(field):
    """Return an ordered (cheap -> expensive) list of predictor factories."""
    if field in REGRESSOR_FIELDS:
        return list(_REGRESSOR_LADDER)
    return list(_CLASSIFIER_LADDER)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ladders.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
cd /home/rizzo/talisman/eval
git add eval/bench/ladders.py tests/test_ladders.py
git commit -m "feat: per-field candidate ladders (classifier + regressor)"
```

---

### Task 9: Candidate scoring against held-out GLM

**Files:**
- Create: `eval/bench/score.py`
- Test: `tests/test_score_candidate.py`

This reuses the existing per-field metrics in `eval/metrics/fields.py` (`FIELD_METRICS[field]` is a `(extractor, metric)` tuple; we call `metric.score(pred_value, gold_value)` directly for single-value fields).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_score_candidate.py
from eval.bench.score import score_candidate
from eval.predictors.base import Prediction

class FakePredictor:
    name = "fake"
    def __init__(self, mapping): self.mapping = mapping
    def fit(self, rows): return self
    def predict_one(self, row):
        return self.mapping[row["article"]["id"]]

def test_score_candidate_uses_field_metric():
    field = "content_type"
    test_rows = [
        {"article": {"id": 0}, "labels": {field: "analysis"}, "features": None},
        {"article": {"id": 1}, "labels": {field: "opinion"}, "features": None},
    ]
    pred = FakePredictor({0: Prediction("analysis", 0.9),   # correct
                          1: Prediction("analysis", 0.4)})  # wrong
    scored = score_candidate(field, pred, test_rows)
    assert len(scored) == 2
    assert scored[0] == (1.0, 0.9)   # exact categorical match -> 1.0
    assert scored[1][0] == 0.0 and scored[1][1] == 0.4

def test_score_candidate_skips_missing_gold():
    field = "content_type"
    test_rows = [{"article": {"id": 0}, "labels": {}, "features": None}]
    pred = FakePredictor({0: Prediction("analysis", 1.0)})
    assert score_candidate(field, pred, test_rows) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_score_candidate.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# eval/bench/score.py
from eval.metrics.fields import FIELD_METRICS

def score_candidate(field, predictor, test_rows):
    """Return list of (score in [0,1], confidence) for each test row that has gold."""
    _extractor, metric = FIELD_METRICS[field]
    out = []
    for row in test_rows:
        gold = row["labels"].get(field)
        if gold is None:
            continue
        pred = predictor.predict_one(row)
        score = metric.score(pred.value, gold).value
        out.append((score, pred.confidence))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_score_candidate.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
cd /home/rizzo/talisman/eval
git add eval/bench/score.py tests/test_score_candidate.py
git commit -m "feat: score a candidate against held-out GLM via field metrics"
```

---

### Task 10: Per-field benchmark runner

**Files:**
- Create: `eval/bench/runner.py`
- Test: `tests/test_bench_runner.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bench_runner.py
from eval.bench.runner import run_field, FieldBench

def _rows(field, n, sep):
    """n rows, two classes perfectly separated by feature[0] when sep=True."""
    rows = []
    for i in range(n):
        cls = "analysis" if i % 2 == 0 else "opinion"
        f0 = (0.0 if cls == "analysis" else 9.0) if sep else 1.0
        rows.append({"article": {"id": i}, "labels": {field: cls},
                     "features": [f0, float(i)]})
    return rows

def test_run_field_replaces_when_linear_separates():
    field = "content_type"
    train = _rows(field, 40, sep=True)
    test = _rows(field, 20, sep=True)
    fb = run_field(field, train, test, tau=0.90, min_coverage=0.2)
    assert isinstance(fb, FieldBench)
    assert fb.chosen_decision.decision == "REPLACE"
    # linear is cheaper than lgbm and should win the tie
    assert fb.chosen_candidate in ("linear", "lgbm")
    # all ladder rungs are recorded
    assert {n for n, _ in fb.candidates} == {"majority", "linear", "lgbm"}

def test_run_field_keeps_when_features_useless():
    field = "content_type"
    train = _rows(field, 40, sep=False)
    test = _rows(field, 20, sep=False)
    fb = run_field(field, train, test, tau=0.90, min_coverage=0.2)
    assert fb.chosen_decision.decision == "KEEP"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_bench_runner.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# eval/bench/runner.py
from dataclasses import dataclass
from eval.bench.ladders import ladder_for, SINGLE_VALUE_FIELDS
from eval.bench.score import score_candidate
from eval.bench.decide import decide, Decision

@dataclass
class FieldBench:
    field: str
    chosen_candidate: str
    chosen_decision: Decision
    candidates: list          # list of (name, Decision) for every ladder rung

def run_field(field, train_rows, test_rows, tau=0.90, min_coverage=0.20) -> FieldBench:
    candidates = []
    for factory in ladder_for(field):
        predictor = factory(field).fit(train_rows)
        scored = score_candidate(field, predictor, test_rows)
        candidates.append((predictor.name, decide(field, scored, tau, min_coverage)))
    # Prefer the cheapest REPLACE; else the HYBRID with best coverage; else KEEP (best fidelity).
    replaces = [(n, d) for n, d in candidates if d.decision == "REPLACE"]
    hybrids = [(n, d) for n, d in candidates if d.decision == "HYBRID"]
    if replaces:
        chosen = replaces[0]
    elif hybrids:
        chosen = max(hybrids, key=lambda nd: nd[1].coverage)
    else:
        chosen = max(candidates, key=lambda nd: nd[1].fidelity)
    return FieldBench(field, chosen[0], chosen[1], candidates)

def run_all(train_rows, test_rows, fields=None, tau=0.90, min_coverage=0.20):
    fields = fields if fields is not None else SINGLE_VALUE_FIELDS
    return [run_field(f, train_rows, test_rows, tau, min_coverage) for f in fields]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_bench_runner.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
cd /home/rizzo/talisman/eval
git add eval/bench/runner.py tests/test_bench_runner.py
git commit -m "feat: per-field benchmark runner over the candidate ladder"
```

---

### Task 11: Report + cost/coverage summary

**Files:**
- Create: `eval/bench/report.py`
- Test: `tests/test_bench_report.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bench_report.py
import json
from eval.bench.decide import Decision
from eval.bench.runner import FieldBench
from eval.bench.report import summarize, render_markdown, to_json

def _fb(field, cand, decision, coverage, fidelity):
    d = Decision(field, fidelity, decision, coverage,
                 None if decision != "HYBRID" else 0.8)
    return FieldBench(field, cand, d, [(cand, d)])

def test_summarize_counts_and_expected_calls():
    fbs = [
        _fb("content_type", "linear", "REPLACE", 1.0, 0.93),
        _fb("urgency", "lgbm", "HYBRID", 0.6, 0.7),
        _fb("technical_quality", "majority", "KEEP", 0.0, 0.4),
    ]
    s = summarize(fbs)
    assert s["counts"] == {"REPLACE": 1, "HYBRID": 1, "KEEP": 1}
    # expected LLM calls/field = 1 - coverage, averaged
    assert abs(s["expected_llm_fraction"] - ((0.0 + 0.4 + 1.0) / 3)) < 1e-9

def test_render_markdown_has_row_per_field():
    fbs = [_fb("content_type", "linear", "REPLACE", 1.0, 0.93)]
    md = render_markdown(fbs)
    assert "content_type" in md and "REPLACE" in md and "linear" in md

def test_to_json_roundtrips():
    fbs = [_fb("content_type", "linear", "REPLACE", 1.0, 0.93)]
    data = json.loads(to_json(fbs))
    assert data["fields"][0]["field"] == "content_type"
    assert data["summary"]["counts"]["REPLACE"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_bench_report.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# eval/bench/report.py
import json

def summarize(field_benches) -> dict:
    counts = {"REPLACE": 0, "HYBRID": 0, "KEEP": 0}
    llm_fractions = []
    for fb in field_benches:
        counts[fb.chosen_decision.decision] += 1
        llm_fractions.append(1.0 - fb.chosen_decision.coverage)
    expected = sum(llm_fractions) / len(llm_fractions) if llm_fractions else 1.0
    return {"counts": counts, "expected_llm_fraction": expected,
            "n_fields": len(field_benches)}

def render_markdown(field_benches) -> str:
    head = ["field", "chosen", "decision", "fidelity", "coverage", "conf_threshold"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for fb in field_benches:
        d = fb.chosen_decision
        ct = "" if d.confidence_threshold is None else f"{d.confidence_threshold:.2f}"
        lines.append("| " + " | ".join([
            fb.field, fb.chosen_candidate, d.decision, f"{d.fidelity:.3f}",
            f"{d.coverage:.2f}", ct]) + " |")
    s = summarize(field_benches)
    lines.append("")
    lines.append(f"**Decisions:** {s['counts']}  •  "
                 f"**expected LLM share:** {s['expected_llm_fraction']:.2f} "
                 f"over {s['n_fields']} fields")
    return "\n".join(lines)

def to_json(field_benches) -> str:
    fields = []
    for fb in field_benches:
        d = fb.chosen_decision
        fields.append({
            "field": fb.field, "chosen_candidate": fb.chosen_candidate,
            "decision": d.decision, "fidelity": d.fidelity, "coverage": d.coverage,
            "confidence_threshold": d.confidence_threshold,
            "candidates": [{"name": n, "decision": cd.decision,
                            "fidelity": cd.fidelity, "coverage": cd.coverage}
                           for n, cd in fb.candidates],
        })
    return json.dumps({"fields": fields, "summary": summarize(field_benches)}, indent=2)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_bench_report.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
cd /home/rizzo/talisman/eval
git add eval/bench/report.py tests/test_bench_report.py
git commit -m "feat: benchmark report (markdown + json) with decision summary"
```

---

### Task 12: Feature extractor (MiniLM embeddings) — gated integration

**Files:**
- Create: `eval/distill/features.py`
- Test: `tests/test_features_gated.py`

The extractor loads `all-MiniLM-L6-v2` (already installed). Unit tests elsewhere inject synthetic features, so this is the only place the real model loads — gated behind `RUN_MODEL_TESTS=1` like the existing oracle E2E.

- [ ] **Step 1: Write the failing (gated) test**

```python
# tests/test_features_gated.py
import os, numpy as np, pytest
from eval.distill.features import FeatureExtractor, attach_features

pytestmark = pytest.mark.skipif(os.environ.get("RUN_MODEL_TESTS") != "1",
                                reason="set RUN_MODEL_TESTS=1 to run model-loading test")

def test_extractor_returns_fixed_width_vector():
    fx = FeatureExtractor()
    v = fx.extract({"title": "Apple beats earnings", "content": "Strong quarter."})
    assert isinstance(v, list) and len(v) == fx.dim
    assert all(isinstance(x, float) for x in v)

def test_attach_features_fills_rows():
    fx = FeatureExtractor()
    rows = [{"article": {"id": 0, "title": "t", "content": "c"},
             "labels": {}, "features": None}]
    attach_features(rows, fx)
    assert rows[0]["features"] is not None and len(rows[0]["features"]) == fx.dim
```

- [ ] **Step 2: Run test to verify it fails**

Run: `RUN_MODEL_TESTS=1 pytest tests/test_features_gated.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# eval/distill/features.py
import numpy as np

class FeatureExtractor:
    """Embed an article into a fixed-width vector: [title_emb | body_emb]."""
    MODEL = "all-MiniLM-L6-v2"
    BODY_CHARS = 4000

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.MODEL)
        self.dim = self.model.get_sentence_embedding_dimension() * 2

    def extract(self, article: dict) -> list:
        title = (article.get("title") or "")[:512]
        body = (article.get("content") or "")[:self.BODY_CHARS]
        emb = self.model.encode([title, body], normalize_embeddings=True)
        return np.concatenate([emb[0], emb[1]]).astype(float).tolist()

def attach_features(rows, extractor):
    """Mutate rows in place, filling row['features']."""
    for row in rows:
        row["features"] = extractor.extract(row["article"])
    return rows

def save_features(rows, path):
    """Persist {id: feature-vector} as an .npz keyed by string id."""
    arrs = {str(r["article"]["id"]): np.array(r["features"], dtype=float)
            for r in rows if r["features"] is not None}
    np.savez_compressed(path, **arrs)

def load_features(path) -> dict:
    data = np.load(path)
    return {int(k): data[k].tolist() for k in data.files}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `RUN_MODEL_TESTS=1 pytest tests/test_features_gated.py -v`
Expected: PASS (2 tests). Without the env var, both SKIP.

- [ ] **Step 5: Commit**

```bash
cd /home/rizzo/talisman/eval
git add eval/distill/features.py tests/test_features_gated.py
git commit -m "feat: MiniLM feature extractor + npz cache (gated test)"
```

---

### Task 13: CLI wiring (split / features / bench)

**Files:**
- Modify: `eval/cli.py` (add three subparsers + handlers)
- Test: `tests/test_cli_bench.py`

- [ ] **Step 1: Write the failing test**

Tests the `split` and `bench` commands end-to-end with injected synthetic features (no model load). `features` is exercised only by the gated test in Task 12.

```python
# tests/test_cli_bench.py
import json, numpy as np
from pathlib import Path
from eval.cli import main

def _make_gold(path, n):
    with open(path, "w") as fh:
        for i in range(n):
            cls = "analysis" if i % 2 == 0 else "opinion"
            fh.write(json.dumps({
                "article": {"id": i, "url": "u", "title": "t",
                            "source": "ccnews", "content": "c"},
                "labels": {"content_type": cls}, "labeler": "glm"}) + "\n")

def _make_feats(path, n):
    arrs = {str(i): np.array([0.0 if i % 2 == 0 else 9.0, float(i)]) for i in range(n)}
    np.savez_compressed(path, **arrs)

def test_split_then_bench(tmp_path, capsys):
    gold = tmp_path / "gold.jsonl"
    _make_gold(gold, 60)
    main(["split", "--gold", str(gold), "--out-dir", str(tmp_path),
          "--test-frac", "0.34", "--seed", "0"])
    assert (tmp_path / "train.jsonl").exists() and (tmp_path / "test.jsonl").exists()

    feats = tmp_path / "feats.npz"
    _make_feats(feats, 60)
    out = tmp_path / "decisions.json"
    main(["bench", "--train", str(tmp_path / "train.jsonl"),
          "--test", str(tmp_path / "test.jsonl"),
          "--features", str(feats), "--fields", "content_type",
          "--out", str(out)])
    data = json.loads(out.read_text())
    assert data["fields"][0]["field"] == "content_type"
    assert data["fields"][0]["decision"] == "REPLACE"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_bench.py -v`
Expected: FAIL (argparse `invalid choice: 'split'`)

- [ ] **Step 3: Write minimal implementation**

Add to `build_parser()` in `eval/cli.py` (after the existing subparsers, before `return p`):

```python
    sp = sub.add_parser("split")
    sp.add_argument("--gold", required=True); sp.add_argument("--out-dir", required=True)
    sp.add_argument("--test-frac", type=float, default=0.2)
    sp.add_argument("--seed", type=int, default=0)

    fe = sub.add_parser("features")
    fe.add_argument("--rows", required=True); fe.add_argument("--out", required=True)

    bn = sub.add_parser("bench")
    bn.add_argument("--train", required=True); bn.add_argument("--test", required=True)
    bn.add_argument("--features", required=True)
    bn.add_argument("--fields", nargs="+")
    bn.add_argument("--tau", type=float, default=0.90)
    bn.add_argument("--min-coverage", type=float, default=0.20)
    bn.add_argument("--out")
```

Add these handler branches in `main()` (before the final `if __name__`):

```python
    elif ns.cmd == "split":
        from eval.distill.split import load_rows, split_rows
        train, test = split_rows(load_rows(ns.gold), ns.test_frac, ns.seed)
        out = Path(ns.out_dir); out.mkdir(parents=True, exist_ok=True)
        for name, rows in (("train", train), ("test", test)):
            with (out / f"{name}.jsonl").open("w") as fh:
                for r in rows:
                    fh.write(json.dumps({"article": r["article"],
                                         "labels": r["labels"]}) + "\n")
        print(f"split -> {len(train)} train / {len(test)} test in {ns.out_dir}")
    elif ns.cmd == "features":
        from eval.distill.split import load_rows
        from eval.distill.features import FeatureExtractor, attach_features, save_features
        rows = load_rows(ns.rows)
        attach_features(rows, FeatureExtractor())
        save_features(rows, ns.out)
        print(f"features -> {len(rows)} vectors in {ns.out}")
    elif ns.cmd == "bench":
        from eval.distill.split import load_rows
        from eval.distill.features import load_features
        from eval.bench.runner import run_all
        from eval.bench.report import render_markdown, to_json
        feats = load_features(ns.features)
        def _attach(rows):
            for r in rows:
                r["features"] = feats.get(r["article"]["id"])
            return rows
        train = _attach(load_rows(ns.train))
        test = _attach(load_rows(ns.test))
        fbs = run_all(train, test, fields=ns.fields, tau=ns.tau,
                      min_coverage=ns.min_coverage)
        print(render_markdown(fbs))
        if ns.out:
            Path(ns.out).write_text(to_json(fbs))
            print(f"\ndecisions -> {ns.out}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli_bench.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /home/rizzo/talisman/eval
git add eval/cli.py tests/test_cli_bench.py
git commit -m "feat: CLI split/features/bench subcommands"
```

---

### Task 14: Full suite green + real benchmark dry-run

**Files:** none (verification task)

- [ ] **Step 1: Run the whole unit suite**

Run: `cd /home/rizzo/talisman/eval && pytest -q`
Expected: all tests pass (existing oracle tests + the new ones). Gated tests skip without `RUN_MODEL_TESTS=1`.

- [ ] **Step 2: Produce the real split from the 5k GLM gold**

Run:
```bash
cd /home/rizzo/talisman/eval
python -m eval.cli split --gold eval/data/gold_z-ai_glm-5.1.jsonl \
  --out-dir eval/data/distill --test-frac 0.2 --seed 0
```
Expected: `split -> 4000 train / 1000 test in eval/data/distill`

- [ ] **Step 3: Extract features for both splits (loads MiniLM, CPU, a few minutes)**

Run:
```bash
cd /home/rizzo/talisman/eval
python -m eval.cli features --rows eval/data/distill/train.jsonl --out eval/data/distill/train.npz
python -m eval.cli features --rows eval/data/distill/test.jsonl  --out eval/data/distill/test.npz
```
Expected: `features -> 4000 vectors ...` and `features -> 1000 vectors ...`

Note: `bench` takes a single `--features` map, so merge or pass per-split. Update the `bench` invocation to attach train features to train rows and test features to test rows by running the command twice is unnecessary — instead extend the merge: combine both npz into one map (ids are globally unique across the split).

```bash
cd /home/rizzo/talisman/eval
python - <<'PY'
import numpy as np
a = dict(np.load("eval/data/distill/train.npz"))
b = dict(np.load("eval/data/distill/test.npz"))
a.update(b)
np.savez_compressed("eval/data/distill/all.npz", **a)
print("merged", len(a))
PY
```

- [ ] **Step 4: Run the benchmark over all single-value fields**

Run:
```bash
cd /home/rizzo/talisman/eval
python -m eval.cli bench --train eval/data/distill/train.jsonl \
  --test eval/data/distill/test.jsonl --features eval/data/distill/all.npz \
  --out eval/reports/replacement_decisions.json
```
Expected: a markdown table printed with one row per field, plus the decision summary; `eval/reports/replacement_decisions.json` written.

- [ ] **Step 5: Commit the report (data/models gitignored)**

```bash
cd /home/rizzo/talisman/eval
# ensure the big artifacts are ignored
grep -q "^eval/data/distill/" .gitignore || echo "eval/data/distill/" >> .gitignore
grep -q "^eval/models/" .gitignore || echo "eval/models/" >> .gitignore
git add eval/reports/replacement_decisions.json .gitignore
git commit -m "report: GLM-fidelity replacement decisions for single-value fields"
```

---

## Self-Review

**Spec coverage check (against `2026-06-16-ml-replacement-benchmark-design.md`):**
- Decision bar: fidelity to GLM, τ=0.90, confidence-gated hybrid, <20% coverage → KEEP → Task 7 ✓
- Tiered candidate ladder per field → Tasks 4–6, 8, 10 ✓
- Train on GLM train split, test on held-out GLM → Tasks 2, 9, 14 ✓
- Embed-once CPU features → Task 12 ✓
- Report with per-field decisions + cost/coverage summary → Task 11, 14 ✓
- Opus/HF as secondary diagnostics → **deferred to Plan 2** (noted in scope; not a gap) ✓
- List/struct/text/generative fields (assets, contagion, keywords, chart_summary) → **deferred to Plan 2** (noted in scope) ✓
- Fine-tune tier 5 → **deferred** (ladder is extensible; not needed to ship the core) ✓

**Placeholder scan:** none — every code step is complete and runnable.

**Type consistency:** `Prediction(value, confidence)` used uniformly; `row` shape `{"article","labels","features"}` consistent across split/score/runner/cli; `Decision(field, fidelity, decision, coverage, confidence_threshold)` consistent across decide/runner/report; predictor `.name` attributes match the ladder-order assertions (`majority/linear/lgbm`, `mean/ridge`); `FIELD_METRICS[field]` unpacked as `(extractor, metric)` matching `eval/metrics/fields.py`.

**Note for executor:** Task 6 Step 2 expects PASS (classes already written in Task 5) — this is intentional, not an error.
