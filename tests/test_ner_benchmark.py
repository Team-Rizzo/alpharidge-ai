"""Pytest wrapper for the NER benchmark. Gated behind --live-ner (loads models).

Asserts the entity-overhaul guarantees on the gold set:
  * zero forbidden tickers (GOOGL/META/ADA/USO/XAU/V from boilerplate/noise)
  * 100% language accuracy on labeled cases (was hardcoded "en")
  * no boilerplate / day-of-week entity leaks
  * meaningful asset recall on genuinely-financial articles
"""

import pytest

from tests.ner_benchmark import run, summarize, _print_report


@pytest.fixture(scope="module")
def engine(request):
    if not request.config.getoption("--live-ner"):
        pytest.skip("needs --live-ner (loads the full model stack)")
    from talisman_ai.analyzer.ner_fusion import NERFusionEngine
    return NERFusionEngine()


@pytest.fixture(scope="module")
def results(engine):
    res = run(engine)
    _print_report(res)
    return res


def test_no_forbidden_tickers(results):
    offenders = {r.case.id: sorted(r.forbidden_assets_emitted)
                 for r in results if r.forbidden_assets_emitted}
    assert offenders == {}, f"false-positive tickers: {offenders}"


def test_language_accuracy_perfect(results):
    wrong = {r.case.id: r.detected_language for r in results if not r.lang_ok}
    assert wrong == {}, f"language mis-detections: {wrong}"


def test_no_entity_leaks(results):
    leaks = {r.case.id: r.entity_leaks for r in results if r.entity_leaks}
    assert leaks == {}, f"entity noise leaks: {leaks}"


def test_asset_recall_reasonable(results):
    s = summarize(results)
    assert s["asset_recall"] is None or s["asset_recall"] >= 0.75, s


def test_deterministic_repeat(engine):
    """Same input -> identical assets/entities/language (validator hard-gates
    reproducibility). Catches nondeterministic ops in the model path."""
    from tests.ner_benchmark import GOLD
    c = next(g for g in GOLD if g.id == "fed_cut")
    a = engine.extract_and_resolve(c.title, c.body)
    b = engine.extract_and_resolve(c.title, c.body)
    assert a.detected_language == b.detected_language
    assert {x.ticker for x in a.resolved_assets} == {x.ticker for x in b.resolved_assets}
    assert [e.canonical_name for e in a.resolved_entities] == \
           [e.canonical_name for e in b.resolved_entities]
