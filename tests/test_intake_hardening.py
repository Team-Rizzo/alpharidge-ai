"""What a miner returns is checked against what it was sent."""

import asyncio
import types

import numpy as np

import neurons.validator as validator_module
from alpharidge_ai import config
from alpharidge_ai.analyzer import scoring
from alpharidge_ai.protocol import ArticleBatch
from alpharidge_ai.utils.api_models import NewsArticleAnalysisBase, NewsArticleForScoring
from alpharidge_ai.validator.reputation_store import MAX_OBS_PER_TARGET, ReputationStore

HK = "5MinerHotkeyForTests"
DIM = 384


def _article(aid, content="short body", analysis=None):
    return NewsArticleForScoring(id=aid, url=f"http://x/{aid}", title=f"Title {aid}",
                                 content=content, source="test", analysis=analysis)


def _analysis(**data):
    return NewsArticleAnalysisBase(analysis_data={"title": "Title", **data})


class _Store:
    def __init__(self, articles):
        self.articles = {str(a.id): a for a in articles}
        self._articles = {k: types.SimpleNamespace(start_time=None) for k in self.articles}

    def get_status(self, aid):
        return types.SimpleNamespace(value="Processing")

    def get_hotkey(self, aid):
        return HK if str(aid) in self.articles else None

    def get_article(self, aid):
        return self.articles[str(aid)]


class _Intake:
    forward_articles = validator_module.Validator.forward_articles

    def __init__(self, leased):
        self._liveness = types.SimpleNamespace(mark_seen=lambda hk: None)
        self._article_store = _Store(leased)
        self._validating_article_ids = set()
        self.tasks, self.handled = [], []

    def _pop_verification(self, hk, aid):
        return None

    def _track_task(self, task):
        self.tasks.append(task)

    async def _handle_article_miner_batch_response(self, batch, hk, sent, *a, **kw):
        self.handled.append((batch, sent))


def _push(intake, returned):
    synapse = ArticleBatch(article_batch=returned)
    synapse.dendrite.hotkey = HK

    async def run():
        await intake.forward_articles(synapse)
        for t in intake.tasks:
            await t
    asyncio.run(run())
    return intake.handled


def test_a_repeated_article_counts_once():
    intake = _Intake([_article(1)])
    handled = _push(intake, [_article(1, analysis=_analysis())] * 50)
    batch, sent = handled[0]
    assert len(batch) == 1 and len(sent) == 1


def test_pay_and_storage_use_our_copy_of_the_article():
    ours = _article(1, content="short body")
    intake = _Intake([ours])
    returned = _article(1, content="x" * 5000, analysis=_analysis(extra="kept"))
    returned.title = "a different title"
    batch, _ = _push(intake, [returned])[0]
    assert batch[0].content == "short body" and batch[0].title == ours.title
    assert batch[0].analysis.analysis_data["extra"] == "kept"


def test_an_oversized_push_is_ignored(monkeypatch):
    monkeypatch.setattr(config, "MINER_BATCH_SIZE", 24, raising=False)
    intake = _Intake([_article(i) for i in range(300)])
    assert _push(intake, [_article(i) for i in range(300)]) == []


def test_a_non_text_title_is_a_mismatch_not_a_crash():
    ref = _article(1)
    for title in (["x"], 123, {"a": 1}):
        record = _article(1, analysis=NewsArticleAnalysisBase(analysis_data={"title": title}))
        assert scoring._titles_match(record, ref) is False


def _batch_with_embedding(te):
    items = []
    for i in range(3):
        a = _article(i, analysis=NewsArticleAnalysisBase(analysis_data={"title": f"Title {i}", "title_embedding": te}))
        a.title = f"Title {i}"
        items.append(a)
    return items


def test_malformed_embeddings_fail_the_batch_without_raising(monkeypatch):
    monkeypatch.setattr(scoring, "SAMPLE_REPLACEMENTS", 0)
    for te in (["a"] * DIM, [float("nan")] * DIM):
        ok, result = scoring.validate_miner_article_intelligence_batch(
            _batch_with_embedding(te), analyzer=None, sample_size=0)
        assert not ok
        assert "malformed_analysis" in {d.get("reason") for d in result["discrepancies"]}
        assert scoring.classify_article_batch_failure(result["discrepancies"]) == "integrity"


def test_a_non_finite_title_embedding_fails_tier_2_5():
    from tests.test_article_intelligence import _make_intel
    unit = list(np.ones(DIM) / np.sqrt(DIM))
    bad = [float("nan")] * DIM
    ok, _, details = scoring.validate_article_intelligence(
        _make_intel(title_embedding=bad, narrative_embedding=unit),
        _make_intel(title_embedding=unit, narrative_embedding=unit))
    assert not ok


class _Peer:
    _trusted_peer = staticmethod(validator_module.Validator._trusted_peer)
    forward_validator_reputation_obs = validator_module.Validator.forward_validator_reputation_obs

    def __init__(self, hotkeys, store):
        self.metagraph = types.SimpleNamespace(hotkeys=hotkeys)
        self._reputation_store = store


def _obs(sender, epoch, observations):
    return types.SimpleNamespace(dendrite=types.SimpleNamespace(hotkey=sender), epoch=epoch,
                                 seq=epoch, observations=observations)


def test_reputation_from_unlisted_senders_and_unregistered_targets_is_dropped(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "REPUTATION_SCORING_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "REPUTATION_PEERS", "PeerA", raising=False)
    store = ReputationStore(path=tmp_path / "rep.json")
    peer = _Peer(["MinerX", "MinerY"], store)
    asyncio.run(peer.forward_validator_reputation_obs(_obs("Stranger", 10, {"MinerX": [[1, 0.0, 1.0]]})))
    assert not store.obs
    asyncio.run(peer.forward_validator_reputation_obs(
        _obs("PeerA", 11, {"MinerX": [[1, 0.5, 1.0]], "NotRegistered": [[2, 0.0, 1.0]]})))
    assert set(store.obs[11]["PeerA"]) == {"MinerX"}


def test_an_empty_peer_list_keeps_todays_behaviour(monkeypatch):
    monkeypatch.setattr(config, "REPUTATION_PEERS", "", raising=False)
    assert validator_module.Validator._trusted_peer("anyone")


def test_reputation_epochs_and_volume_are_bounded(tmp_path):
    store = ReputationStore(path=tmp_path / "rep.json")
    store.finalized = [100]
    assert store.ingest("PeerA", 160, {"MinerX": [[1, 0.0, 1.0]]}, seq=160)[1].startswith("epoch_out_of_range")
    assert store.ingest("PeerA", 100, {"MinerX": [[1, 0.0, 1.0]]}, seq=100)[0] is False
    ok, reason = store.ingest("PeerA", 102, {"MinerX": [[i, 0.0, 1.0] for i in range(500)]}, seq=102)
    assert ok and len(store.obs[102]["PeerA"]["MinerX"]) == MAX_OBS_PER_TARGET == 64


def test_gazetteer_patterns_are_compiled_once_in_the_same_order():
    import re
    from alpharidge_ai.analyzer.asset_extractor import AssetExtractor
    ex = AssetExtractor()
    assert [(t, a) for t, a, _ in ex._cashtag_patterns] == list(ex._cashtag_index.items())
    assert all(p.pattern == re.escape(t) + r"(?![a-z0-9])" for t, _, p in ex._cashtag_patterns)
    assert [(c, (a, amb)) for c, a, amb, _ in ex._case_sensitive_patterns] == list(ex._case_sensitive_index.items())
    assert all(p.pattern == rf"\b{re.escape(c)}\b" for c, _, _, p in ex._case_sensitive_patterns)


def test_an_unknown_ticker_fails_tier_2():
    from tests.test_article_intelligence import _make_intel
    from alpharidge_ai.models.article_intelligence import AssetClass, AssetSentiment, Sentiment

    def asset(ticker):
        n = Sentiment.NEUTRAL
        return AssetSentiment(ticker=ticker, asset_name="Apple", asset_class=AssetClass.EQUITY,
                              direction=n, magnitude=0.5, confidence=0.5, short_term_outlook=n,
                              medium_term_outlook=n, long_term_outlook=n, causal_driver="x",
                              relevance_score=1.0, is_primary_subject=True, evidence_spans=[ticker])
    real, fake = asset("AAPL"), asset("AAPL.")
    ok, _, details = scoring.validate_article_intelligence(_make_intel(assets=[fake]), _make_intel(assets=[real]))
    assert not ok and details["tier2"]["asset_tickers"]["reason"] == "unknown_ticker"
    ok, _, details = scoring.validate_article_intelligence(_make_intel(assets=[real]), _make_intel(assets=[real]))
    assert "asset_tickers" not in details["tier2"]
    ok, _, details = scoring.validate_article_intelligence(_make_intel(assets=[asset("SOX")]), _make_intel())
    assert "asset_tickers" not in details["tier2"]


def test_a_missing_dependency_graph_is_logged_loudly(monkeypatch):
    from alpharidge_ai.analyzer import article_intelligence_analyzer as aia
    errors = []
    monkeypatch.setattr(aia.bt.logging, "error", lambda msg, *a, **k: errors.append(msg))
    real = aia._load_json
    monkeypatch.setattr(aia, "_load_json", lambda name: {} if name == "dependency_graph.json" else real(name))
    an = aia.ArticleIntelligenceAnalyzer(model="m", api_key="k", llm_base="https://invalid")
    assert an.dependency_graph == {}
    assert any("dependency_graph.json" in e for e in errors)


def test_metric_names_match_whatever_the_case_style():
    from alpharidge_ai.oracle import audit
    claim = lambda m, v, u: types.SimpleNamespace(metric_name=m, value=v, unit=u)
    grader = claim("contract value", 460, "EUR millions")
    for name in ("contract_value", "contractValue", "Contract Value"):
        mine = claim(name, 460e6, "EUR")
        assert audit.same_quantity(mine, grader)
        assert audit.names_overlap(mine, grader) and audit.claims_match(mine, grader)
    assert audit.names_overlap(claim("share_price_close", 14.73, "EUR"), claim("Acme share price close", 14.73, "EUR"))
    assert not audit.names_overlap(claim("revenue", 1.2e9, "USD"), claim("operating_margin", 1.2e9, "USD"))
