"""
Offline cheat-probe harness for article-intelligence validation (RFC quality-yield).

Bounds cheat-acceptance deterministically WITHOUT a live LLM: builds reference
ArticleIntelligence objects, injects a stub analyzer that returns them, and runs
synthetic miner payloads (honest + defective) through the real
`validate_miner_article_intelligence_batch`. Gates every validation loosening
(clone scoping, fractional accept) by asserting cheats stay rejected and honest /
boundary cases pass.

Run as a metric:   python -m tests.test_cheat_probe   (prints cheat/honest rates)
Run as a gate:     pytest tests/test_cheat_probe.py
"""

import numpy as np

from alpharidge_ai.models.article_intelligence import (
    ArticleIntelligence, SourceMetadata, ChartSummary, EventFingerprint,
    TopicSignature, TextStatistics,
    ArticleContentType, ImpactPotential, TechnicalQuality, Sentiment,
    SentimentDirection, FactualConfidence, MarketSession, EventType,
)
from alpharidge_ai.analyzer.scoring import validate_miner_article_intelligence_batch
from alpharidge_ai.utils.api_models import NewsArticleForScoring, NewsArticleAnalysisBase


# ---- builders -------------------------------------------------------------

def _first(enum_cls):
    return list(enum_cls)[0]


def _unit_vec(seed: int, dim: int = 384) -> list:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim)
    return (v / np.linalg.norm(v)).tolist()


def _vec_with_cosine(seed: int, target_cos: float, dim: int = 384) -> list:
    """A unit vector with an EXACT cosine to _unit_vec(seed):
    v = c·base + sqrt(1-c²)·(unit vector orthogonal to base)."""
    base = np.array(_unit_vec(seed, dim))
    rng = np.random.default_rng(seed + 9999)
    r = rng.standard_normal(dim)
    r = r - (r @ base) * base          # orthogonalize against base
    r = r / np.linalg.norm(r)
    v = target_cos * base + np.sqrt(1.0 - target_cos ** 2) * r
    return v.tolist()                  # already unit norm


def make_reference_intel(article_id: int, *, sector_id: int = 10,
                         embedding=None, market_session=None) -> ArticleIntelligence:
    title = f"Article {article_id} headline about markets"
    content = f"body content for article {article_id}. " * 25
    return ArticleIntelligence(
        article_id=article_id,
        url=f"https://example.com/{article_id}",
        title=title,
        published_at="2026-06-26T15:00:00+00:00",
        analyzed_at="2026-06-26T15:05:00+00:00",
        source=SourceMetadata(source_id="src1", source_name="Example Wire", credibility_score=0.8),
        content_type=_first(ArticleContentType),
        impact_potential=_first(ImpactPotential),
        technical_quality=_first(TechnicalQuality),
        overall_sentiment=_first(Sentiment),
        overall_sentiment_score=0.0,
        sentiment_direction=_first(SentimentDirection),
        chart_summary=ChartSummary(headline=title[:120], one_liner="one liner summary",
                                   context_paragraph="context paragraph"),
        event_fingerprint=EventFingerprint(
            event_type=_first(EventType),
            event_title=title[:200],
            content_hash=ArticleIntelligence.compute_content_hash(title, content),
        ),
        topic_signature=TopicSignature(primary_sector_id=sector_id, primary_sector_symbol="TECH"),
        text_stats=TextStatistics(
            char_count=len(content), word_count=len(content.split()),
            sentence_count=5, paragraph_count=1, avg_sentence_length=10.0,
            avg_word_length=5.0, numeric_density=0.0, quote_density=0.0,
        ),
        factual_confidence=_first(FactualConfidence),
        title_embedding=embedding if embedding is not None else _unit_vec(article_id),
        detected_language="en",
        market_session=market_session or MarketSession.REGULAR_HOURS,
    )


def make_article(intel: ArticleIntelligence) -> NewsArticleForScoring:
    """Batch item whose .analysis.analysis_data is the (miner) intel as a dict."""
    return NewsArticleForScoring(
        id=intel.article_id,
        url=intel.url,
        title=intel.title,
        source=intel.source.source_name,
        published=intel.published_at,
        content="body " * 50,
        analysis=NewsArticleAnalysisBase(analysis_data=intel.model_dump(mode="json")),
    )


class _StubNER:
    """Local-embedder shim: returns the validator's title embedding for a known title.
    Mirrors analyzer.ner_engine.encode_text (the cheap MiniLM path Phase 1 uses to
    corroborate clone candidates), without loading the real model."""
    def __init__(self, title_to_embedding: dict):
        self._by_title = title_to_embedding

    def encode_text(self, text):
        return self._by_title.get(text)


class StubAnalyzer:
    """Returns the precomputed validator reference per article-id — fully offline.
    Exposes `ner_engine.encode_text` so the Phase-1 differential clone check can read
    the validator's own title embeddings (here, straight from the references)."""
    def __init__(self, reference_by_id: dict):
        self._ref = reference_by_id
        self.ner_engine = _StubNER({r.title: r.title_embedding for r in reference_by_id.values()})

    def analyze(self, article_id=None, url=None, title=None, source=None,
                published=None, summary=None, content=None, miner_hotkey=None, raw_html=None):
        return self._ref.get(int(article_id))


# ---- scenarios ------------------------------------------------------------
# Each builds (reference_by_id, miner_batch) and an expected batch_valid.

def _base_refs(n=4):
    return {i: make_reference_intel(i, sector_id=10 + i) for i in range(n)}


def scenario_honest():
    refs = _base_refs()
    batch = [make_article(refs[i]) for i in range(4)]   # miner == validator
    return refs, batch, True


def scenario_boundary_market_session():
    """Honest, but miner differs ONLY on market_session — must still PASS (Phase 0)."""
    refs = _base_refs()
    batch = []
    for i in range(4):
        miner = make_reference_intel(i, sector_id=10 + i, embedding=refs[i].title_embedding,
                                     market_session=MarketSession.WEEKEND)  # validator has REGULAR
        batch.append(make_article(miner))
    return refs, batch, True


def scenario_lowEffort():
    """One article ships a zero (invalid) title_embedding — must REJECT."""
    refs = _base_refs()
    batch = [make_article(refs[i]) for i in range(4)]
    bad = refs[1].model_dump(mode="json")
    bad["title_embedding"] = [0.0] * 384
    batch[1] = NewsArticleForScoring(id=1, url=refs[1].url, title=refs[1].title,
                                     source="Example Wire", published=refs[1].published_at,
                                     content="body " * 50,
                                     analysis=NewsArticleAnalysisBase(analysis_data=bad))
    return refs, batch, False


def scenario_recycled():
    """Article 0's slot ships article 2's analysis (stale/replayed) — must REJECT."""
    refs = _base_refs()
    batch = [make_article(refs[i]) for i in range(4)]
    recycled = refs[2].model_dump(mode="json")
    batch[0] = NewsArticleForScoring(id=0, url=refs[0].url, title=refs[0].title,
                                     source="Example Wire", published=refs[0].published_at,
                                     content="body " * 50,
                                     analysis=NewsArticleAnalysisBase(analysis_data=recycled))
    return refs, batch, False


def scenario_cloned():
    """All articles ship the same title_embedding (cloned) — must REJECT (clone gate)."""
    refs = _base_refs()
    shared = refs[0].title_embedding
    batch = []
    for i in range(4):
        miner = make_reference_intel(i, sector_id=10 + i, embedding=shared)
        batch.append(make_article(miner))
    return refs, batch, False


def scenario_honest_syndicated():
    """Genuinely similar articles (cosine ~0.97, below the 0.99 gate), honest analysis —
    must PASS. Guards that real same-event clusters aren't flagged as clones."""
    e0 = _unit_vec(0)
    e1 = _vec_with_cosine(0, 0.97)    # genuinely similar, below the 0.99 gate
    refs = {0: make_reference_intel(0, embedding=e0), 1: make_reference_intel(1, embedding=e1),
            2: make_reference_intel(2), 3: make_reference_intel(3)}
    batch = [make_article(refs[i]) for i in range(4)]
    return refs, batch, True


def scenario_syndicated_fp_PHASE1():
    """Honest near-duplicate (cosine >0.99). Under the current 0.99 gate this REJECTS
    (the false-positive). MEASURED, not asserted — should flip to PASS after Phase 1."""
    e0 = _unit_vec(0)
    e1 = _vec_with_cosine(0, 0.995)   # honest near-duplicate -> trips the 0.99 gate today
    refs = {0: make_reference_intel(0, embedding=e0), 1: make_reference_intel(1, embedding=e1),
            2: make_reference_intel(2), 3: make_reference_intel(3)}
    batch = [make_article(refs[i]) for i in range(4)]
    return refs, batch, True  # target outcome (Phase 1); will be False on current code


CHEATS = ["lowEffort", "recycled", "cloned"]
HONEST = ["honest", "boundary_market_session", "honest_syndicated"]


def _run_one(builder):
    refs, batch, expected = builder()
    analyzer = StubAnalyzer(refs)
    batch_valid, _details = validate_miner_article_intelligence_batch(
        batch, analyzer, sample_size=len(batch), seed=0)
    return bool(batch_valid), expected


class _clone_config:
    """Pin the clone-gate config for a probe run (restores afterward). Mirrors how the
    fleet flips Phase 1 via served `CLONE_DIFFERENTIAL_ENABLED` — scoring reads it live."""
    _KEYS = ("CLONE_DIFFERENTIAL_ENABLED", "CLONE_COSINE_THRESHOLD", "CLONE_DIVERGENCE_MARGIN")

    def __init__(self, differential, threshold=0.99, margin=0.05):
        self._want = dict(CLONE_DIFFERENTIAL_ENABLED=differential,
                          CLONE_COSINE_THRESHOLD=threshold, CLONE_DIVERGENCE_MARGIN=margin)

    def __enter__(self):
        from alpharidge_ai import config as _cfg
        self._cfg = _cfg
        self._saved = {k: getattr(_cfg, k, None) for k in self._KEYS}
        for k, v in self._want.items():
            setattr(_cfg, k, v)
        return self

    def __exit__(self, *exc):
        for k, v in self._saved.items():
            setattr(self._cfg, k, v)


def run_cheat_probe(differential: bool = False) -> dict:
    scenarios = {
        "honest": scenario_honest,
        "boundary_market_session": scenario_boundary_market_session,
        "honest_syndicated": scenario_honest_syndicated,
        "lowEffort": scenario_lowEffort,
        "recycled": scenario_recycled,
        "cloned": scenario_cloned,
    }
    with _clone_config(differential=differential):
        results = {name: _run_one(b) for name, b in scenarios.items()}
        fp_now, _fp_target = _run_one(scenario_syndicated_fp_PHASE1)
    cheats_accepted = sum(1 for c in CHEATS if results[c][0] is True)
    honest_accepted = sum(1 for h in HONEST if results[h][0] is True)
    return {
        "results": results,
        "cheat_acceptance_rate": cheats_accepted / len(CHEATS),
        "honest_accept_rate": honest_accepted / len(HONEST),
        # legacy gate: False (the FP). Differential gate: True (FP fixed, Phase 1).
        "syndicated_fp_passes": fp_now,
    }


# ---- pytest gate ----------------------------------------------------------

def test_cheats_are_rejected():
    """Legacy (default) gate: no cheat is accepted."""
    out = run_cheat_probe(differential=False)
    assert out["cheat_acceptance_rate"] == 0.0, out["results"]


def test_honest_and_boundary_pass():
    """Legacy (default) gate: honest + boundary cases pass; the near-dup FP still rejects."""
    out = run_cheat_probe(differential=False)
    assert out["honest_accept_rate"] == 1.0, out["results"]
    assert out["syndicated_fp_passes"] is False, "legacy 0.99 gate should still false-reject the near-dup"


def test_differential_keeps_cheats_rejected():
    """Phase 1: the scoped clone check must NOT let cloned/low-effort/recycled through."""
    out = run_cheat_probe(differential=True)
    assert out["cheat_acceptance_rate"] == 0.0, out["results"]


def test_differential_fixes_syndicated_fp():
    """Phase 1 acceptance test: the honest near-duplicate (cosine 0.995) now PASSES,
    because the validator corroborates the similarity — while honest stays 1.0."""
    out = run_cheat_probe(differential=True)
    assert out["syndicated_fp_passes"] is True, out["results"]
    assert out["honest_accept_rate"] == 1.0, out["results"]


def _print_mode(label, out):
    print(f"[CHEAT_PROBE] {label}")
    for name, (actual, expected) in out["results"].items():
        flag = "ok" if actual == expected else "XX"
        print(f"  {flag}  {name:28s} accepted={actual}  expected={expected}")
    print(f"  cheat_acceptance_rate = {out['cheat_acceptance_rate']:.2f}  (target 0.00)")
    print(f"  honest_accept_rate    = {out['honest_accept_rate']:.2f}  (target 1.00)")
    print(f"  syndicated_fp_passes  = {out['syndicated_fp_passes']}")


if __name__ == "__main__":
    _print_mode("legacy gate (CLONE_DIFFERENTIAL_ENABLED=false)", run_cheat_probe(differential=False))
    print()
    _print_mode("Phase 1 gate (CLONE_DIFFERENTIAL_ENABLED=true)", run_cheat_probe(differential=True))
