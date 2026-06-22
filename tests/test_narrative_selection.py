"""Production narrative-slug selector (deterministic embedding match)."""
from alpharidge_ai.analyzer.article_intelligence_analyzer import ArticleIntelligenceAnalyzer


class _StubEngine:
    """Wraps a real all-MiniLM-L6-v2 so the selector runs without the full NER stack."""
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self._embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def encode_text(self, text):
        if not text or not text.strip():
            return None
        return self._embedder.encode(text.strip(), normalize_embeddings=True).tolist()


def _analyzer():
    a = ArticleIntelligenceAnalyzer.__new__(ArticleIntelligenceAnalyzer)  # skip __init__
    a.ner_engine = _StubEngine()
    a._init_narrative_index()
    return a


def test_index_built():
    a = _analyzer()
    assert a._narr_slugs and a._narr_cent is not None
    assert a._narr_cent.shape[0] == len(a._narr_slugs)


def test_abstains_on_offtopic():
    a = _analyzer()
    assert a._select_narratives("Local bakery wins pie contest",
                                "A small-town bakery took first prize for its apple pie.", "") == []


def test_picks_crypto_slug_on_crypto():
    a = _analyzer()
    out = a._select_narratives(
        "Spot Bitcoin ETF sees record inflows",
        "IBIT and FBTC pulled in billions as institutional inflows into spot bitcoin ETFs surged.",
        "Demand for regulated crypto exposure continues to climb.")
    assert out, "expected a slug for a clear crypto article"
    assert all(s in a._narr_slugs for s in out)
    assert len(out) <= 3


def test_no_crypto_slug_on_spacex_story():
    a = _analyzer()
    out = a._select_narratives(
        "SpaceX IPO pops 19% in trading debut",
        "Shares of the rocket company surged on their first day as investors piled in.",
        "The listing values the firm at a record for a private space venture.")
    crypto = {"institutional-crypto-adoption", "bitcoin-etf-flows", "defi-revival", "ethereum-scaling"}
    assert not (set(out) & crypto)
