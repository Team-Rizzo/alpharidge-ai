"""Marginal content mismatches get one confirming sample; broken articles are set aside."""
import types

from alpharidge_ai.analyzer import scoring

GARBLED = "đŤđťĐÇĐĂ " * 40


class _Refs:
    model = "m"

    def analyze(self, article_id=None, **kwargs):
        return types.SimpleNamespace(id=int(article_id), numeric_claims=[], quotes=[],
                                     assets=[], economic_data=[])


def _run(monkeypatch, agreements, n=4, marginal=True, texts=None, verdicts=None, sample_size=1):
    from tests.test_floor_gating import _payload, TEXT, TITLE
    from tests.test_keyed_audit_pass import _article
    verdicts = verdicts or {}
    monkeypatch.setattr(scoring, "validate_article_intelligence",
                        lambda m, v: verdicts.get(v.id, (True, 1.0, {})))
    monkeypatch.setattr(scoring, "_summary_agreement", lambda m, v: agreements[v.id])
    cfg = scoring._cfg_get
    monkeypatch.setattr(scoring, "_cfg_get",
                        lambda name, default=None: 2.0 if name == "CLONE_COSINE_THRESHOLD" else cfg(name, default))
    monkeypatch.setattr(scoring.random, "sample", lambda xs, k: list(xs)[:k])
    monkeypatch.setattr(scoring.random, "shuffle", lambda xs: xs.reverse())
    blob = _payload([{"metric_name": "revenue", "value": 1.2e9, "unit": "USD", "confidence": 0.9}])
    batch = [_article(i, blob, (texts or {}).get(i, TEXT)) for i in range(1, n + 1)]
    for a in batch:
        a.title = blob.get("title") or TITLE
    refs = {str(a.id): a for a in batch}
    return scoring.validate_miner_article_intelligence_batch(
        batch, _Refs(), sample_size=sample_size, reference_by_id=refs, marginal_allowed=marginal)


def test_a_clear_pass_is_untouched(monkeypatch):
    ok, r = _run(monkeypatch, {1: 0.9, 2: 0.9, 3: 0.9, 4: 0.9})
    assert ok and not r["marginal_used"] and r["total_sampled"] == 1


def test_a_clear_mismatch_fails_with_no_second_chance(monkeypatch):
    ok, r = _run(monkeypatch, {1: 0.10, 2: 0.9, 3: 0.9, 4: 0.9})
    assert not ok and not r["marginal_used"]
    assert [d["reason"] for d in r["discrepancies"]] == ["article_content_mismatch"]


def test_a_marginal_mismatch_cleared_by_a_strong_second_sample(monkeypatch):
    ok, r = _run(monkeypatch, {1: 0.38, 2: 0.9, 3: 0.9, 4: 0.9})
    assert ok and r["marginal_used"] and not r["discrepancies"]
    assert r["total_sampled"] == 1 and r["skipped_samples"] == 1


def test_a_second_sample_below_the_stricter_bar_keeps_the_failure(monkeypatch):
    ok, r = _run(monkeypatch, {1: 0.38, 2: 0.45, 3: 0.9, 4: 0.9})
    assert not ok and r["marginal_used"]
    assert [d["summary_agreement"] for d in r["discrepancies"]] == [0.38]


def test_a_second_sample_that_fails_validation_keeps_both_failures(monkeypatch):
    ok, r = _run(monkeypatch, {1: 0.38, 2: 0.9, 3: 0.9, 4: 0.9}, verdicts={2: (False, 0.2, {})})
    assert not ok and r["marginal_used"]
    assert {d["reason"] for d in r["discrepancies"]} == {"article_content_mismatch", "validation_failed"}


def test_no_allowance_left_means_todays_behaviour(monkeypatch):
    ok, r = _run(monkeypatch, {1: 0.38, 2: 0.9, 3: 0.9, 4: 0.9}, marginal=False)
    assert not ok and not r["marginal_used"] and len(r["discrepancies"]) == 1


def test_no_spare_means_todays_behaviour(monkeypatch):
    ok, r = _run(monkeypatch, {1: 0.38}, n=1)
    assert not ok and not r["marginal_used"] and len(r["discrepancies"]) == 1


def test_only_one_second_sample_per_batch(monkeypatch):
    ok, r = _run(monkeypatch, {1: 0.38, 2: 0.37, 3: 0.9, 4: 0.9}, sample_size=2)
    assert not ok and r["marginal_used"]
    assert sorted(d["summary_agreement"] for d in r["discrepancies"]) == [0.37]


def test_a_garbled_article_is_set_aside_and_replaced(monkeypatch):
    ok, r = _run(monkeypatch, {1: 0.03, 2: 0.9, 3: 0.9, 4: 0.9}, texts={1: GARBLED})
    assert ok and r["defective_ids"] == ["1"] and not r["discrepancies"]
    assert r["total_sampled"] == 1


def test_an_empty_article_is_set_aside(monkeypatch):
    ok, r = _run(monkeypatch, {1: 0.03, 2: 0.9, 3: 0.9, 4: 0.9}, texts={1: "  "})
    assert ok and r["defective_ids"] == ["1"]


def test_all_defective_gives_no_verdict(monkeypatch):
    ok, r = _run(monkeypatch, {1: 0.03}, n=1, texts={1: GARBLED})
    assert not ok and r["no_verdict"] and not r["discrepancies"]


def test_defects_only_come_from_our_own_copy():
    assert scoring.defective_text(GARBLED) == "garbled"
    assert scoring.defective_text("") == "empty"
    assert scoring.defective_text("Ngân hàng Nhà nước điều chỉnh lãi suất " * 5) == ""
    assert scoring.defective_text("Zażółć gęślą jaźń, spółka giełdowa " * 5) == ""


def test_retired_articles_are_not_requeued():
    from alpharidge_ai.utils.article_store import ArticleStore, ArticleStatus
    store = ArticleStore()
    store._articles["7"] = types.SimpleNamespace(status=ArticleStatus.PROCESSING, start_time=1.0)
    store._articles["8"] = types.SimpleNamespace(status=ArticleStatus.PROCESSING, start_time=1.0)
    store.retire("7")
    store.reset_to_unprocessed("8")
    assert store.get_status("8") == ArticleStatus.UNPROCESSED
    store.reset_to_unprocessed("7")
    assert store.get_status("7") == ArticleStatus.PROCESSED


def _validator_stub():
    from neurons.validator import Validator
    v = types.SimpleNamespace(retired=[])
    v._article_store = types.SimpleNamespace(retire=lambda aid: v.retired.append(aid))
    for name in ("_marginal_available", "_spend_marginal", "_note_defective"):
        setattr(v, name, getattr(Validator, name).__get__(v))
    return v


def test_one_second_sample_per_miner_per_day(monkeypatch):
    from alpharidge_ai import config
    monkeypatch.setattr(config, "MARGINAL_PER_DAY", 1, raising=False)
    v = _validator_stub()
    assert v._marginal_available("hk")
    v._spend_marginal("hk")
    assert not v._marginal_available("hk") and v._marginal_available("other")
    monkeypatch.setattr(scoring.time if hasattr(scoring, "time") else __import__("time"),
                        "time", lambda: 10 * 86400.0)
    import neurons.validator as nv
    monkeypatch.setattr(nv.time, "time", lambda: 10 * 86400.0 + 86400)
    assert v._marginal_available("hk")


def test_a_repeatedly_defective_article_is_retired(monkeypatch):
    from alpharidge_ai import config
    monkeypatch.setattr(config, "DEFECTIVE_RETRY_LIMIT", 2, raising=False)
    v = _validator_stub()
    v._note_defective(["5"])
    assert v.retired == []
    v._note_defective(["5", "6"])
    assert v.retired == ["5"]
