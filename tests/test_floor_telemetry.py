"""The floor's counts are reported, and reporting them changes no verdict."""

from alpharidge_ai.analyzer import scoring
from alpharidge_ai.models.article_intelligence import ArticleIntelligence
from alpharidge_ai.oracle import floor
from tests.test_floor_gating import TEXT, TITLE, _payload
from tests.test_keyed_audit_pass import _article, _auditor


def _batch(size=3):
    blob = _payload([{"metric_name": "revenue", "value": 1.2e9, "unit": "USD",
                      "confidence": 0.9}])
    batch = [_article(i, blob, TEXT) for i in range(1, size + 1)]
    for a in batch:
        a.title = TITLE
    return batch


def test_stats_are_collected_per_article():
    stats = []
    scoring._floor_sweep(_batch(3), stats=stats)
    assert len(stats) == 3
    for s in stats:
        assert set(s) == set(scoring._FLOOR_STAT_KEYS)
        assert all(isinstance(v, int) and v >= 0 for v in s.values())
        assert s["numbers"] >= s["grounded"]


def test_collecting_stats_changes_no_verdict():
    batch = _batch(4)
    assert scoring._floor_sweep(batch) == scoring._floor_sweep(batch, stats=[])


def test_the_batch_line_is_logged(monkeypatch):
    lines = []
    monkeypatch.setattr(scoring.bt.logging, "info", lambda m: lines.append(m))
    scoring._log_floor_stats("hotkey-abcdefghijk", [
        {k: 1 for k in scoring._FLOOR_STAT_KEYS},
        {k: 2 for k in scoring._FLOOR_STAT_KEYS}])
    line, = [l for l in lines if l.startswith("[FLOORSIG]")]
    assert "n=2" in line and "grounded=3" in line and "numbers=3" in line


def test_nothing_is_logged_for_an_empty_sweep(monkeypatch):
    lines = []
    monkeypatch.setattr(scoring.bt.logging, "info", lambda m: lines.append(m))
    scoring._log_floor_stats("hk", [])
    assert not lines


def test_the_audit_line_says_how_the_schema_was_treated():
    blob = _payload([{"metric_name": "revenue", "value": 1.2e9, "unit": "USD",
                      "confidence": 0.9}])
    intel = ArticleIntelligence(**blob)
    result = floor.evaluate(intel, TEXT)
    auditor = _auditor(pool_rate=1.0)
    observed = [auditor.audit(i, TEXT, intel, intel, result, 0) for i in range(20)]
    observed = [o for o in observed if o is not None]
    assert observed
    assert all("schema=" in o.detail for o in observed)
    assert all("submitted=" in o.detail for o in observed if o.path == "pool")


def _stats(**kw):
    base = {k: 0 for k in scoring._FLOOR_STAT_KEYS}
    base.update(kw)
    return base


def test_quality_is_none_when_nothing_is_asserted():
    assert scoring.floor_quality(_stats(numbers=12)) is None


def test_quality_is_one_when_everything_holds():
    assert scoring.floor_quality(_stats(grounded=4, aligned=2, evidence=5)) == 1.0


def test_invented_claims_lower_quality():
    assert scoring.floor_quality(_stats(grounded=3, ungrounded=1)) == 0.75


def test_inferred_claims_are_not_counted_against_the_submission():
    assert scoring.floor_quality(_stats(grounded=1, inferred=3)) == 1.0


def test_each_rate_counts_equally():
    q = scoring.floor_quality(_stats(grounded=1, ungrounded=1, aligned=1, evidence=2,
                                     span_fail=2))
    assert q == (0.5 + 1.0 + 0.0) / 3


def test_batch_quality_skips_articles_without_one():
    stats = [_stats(grounded=1), _stats(), _stats(grounded=1, ungrounded=1)]
    assert scoring.batch_floor_quality(stats) == 0.75
    assert scoring.batch_floor_quality([_stats()]) is None


def test_the_line_reports_quality(monkeypatch):
    lines = []
    monkeypatch.setattr(scoring.bt.logging, "info", lambda m: lines.append(m))
    scoring._log_floor_stats("hk", [_stats(grounded=3, ungrounded=1)])
    assert lines[0].endswith("quality=0.750")
