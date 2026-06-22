"""NER quality benchmark — gold cases + metrics for the entity overhaul.

Runs the full NER engine (clean -> detect language -> language-routed NER ->
filter -> resolve) against a labeled gold set and measures the things the
MINER_DATA_QUALITY_REVIEW called out:

  * asset false-positive rate  (forbidden tickers like GOOGL/ADA/USO/XAU/V)
  * asset recall               (genuinely-present tickers found)
  * language accuracy          (was hardcoded "en")
  * entity false-positive leaks (boilerplate / day-of-week / foreign noise)

Run as a script:   python -m tests.ner_benchmark
Run via pytest:     pytest tests/test_ner_benchmark.py --live-ner
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Set


@dataclass
class GoldCase:
    id: str
    title: str
    body: str
    language: str                       # expected detected language
    financial: bool
    expect_assets: Set[str] = field(default_factory=set)   # tickers that SHOULD appear
    forbid_assets: Set[str] = field(default_factory=set)   # tickers that MUST NOT appear
    forbid_entity_substrings: List[str] = field(default_factory=list)  # noise that MUST NOT appear


GOLD: List[GoldCase] = [
    # ---- Non-financial / boilerplate: must stay clean (§3.1, §3.3) ----------
    GoldCase(
        id="jazz_widget", language="en", financial=False,
        title="Makaya McCraven Opens Jazz Festival in Correggio",
        body=("The jazz musician opened the festival at Teatro Asioli to a sold-out crowd. "
              "He won gold at an international competition last year. "
              "Follow us on Google and Facebook for more updates."),
        forbid_assets={"GOOGL", "META", "XAU"},
        forbid_entity_substrings=["Getty", "Facebook", "Google"],
    ),
    GoldCase(
        id="soap_ada", language="en", financial=False,
        title="'Be My Sunshine' Episode Replay Streams Tonight",
        body=("The Turkish soap opera returns as the character Ada Masal confronts her past. "
              "The replay airs on the streaming service this evening."),
        forbid_assets={"ADA"},
    ),
    GoldCase(
        id="italian_uso", language="it", financial=False,
        title="Trump guida 16 miliardari statunitensi in Cina",
        body=("Un accordo commerciale per uso strategico delle risorse naturali. "
              "Domenica i mercati italiani restano chiusi per la festività."),
        forbid_assets={"USO"},
        forbid_entity_substrings=["domenica", "Domenica"],
    ),
    GoldCase(
        id="italian_tv_v", language="it", financial=False,
        title="Diretta Tennis e programmazione TV",
        body="Stasera V va in onda alle 21 sul canale principale. Diretta tennis dalle 18.",
        forbid_assets={"V"},
    ),
    # ---- Financial: must detect real signal --------------------------------
    GoldCase(
        id="fed_cut", language="en", financial=True,
        title="Federal Reserve Cuts Interest Rates by 25 Basis Points",
        body=("The Federal Reserve, led by Chair Jerome Powell, cut interest rates on Wednesday. "
              "Bitcoin rallied above $100,000 and Ethereum gained as investors cheered."),
        expect_assets={"BTC", "ETH"},
    ),
    GoldCase(
        id="nvidia_earnings", language="en", financial=True,
        title="Nvidia Reports Record Quarterly Revenue",
        body=("Nvidia posted $42 billion in revenue as demand for AI chips surged. "
              "NVDA shares jumped 8% in after-hours trading on the strong results."),
        expect_assets={"NVDA"},
    ),
    GoldCase(
        id="russian_pentagon", language="ru", financial=True,
        title="Пентагон заключил контракт с General Dynamics",
        body=("Пентагон заключил контракт на два миллиарда долларов с компанией General Dynamics "
              "на строительство атомных подводных лодок для военно-морского флота."),
        # English-keyword asset registry on Russian text — we don't require ticker
        # recall here, but it must NOT mislabel the language or emit junk.
    ),
]


@dataclass
class CaseResult:
    case: GoldCase
    detected_language: str
    assets: Set[str]
    entities: List[str]

    @property
    def lang_ok(self) -> bool:
        return self.detected_language == self.case.language

    @property
    def forbidden_assets_emitted(self) -> Set[str]:
        return self.assets & self.case.forbid_assets

    @property
    def expected_assets_found(self) -> Set[str]:
        return self.assets & self.case.expect_assets

    @property
    def entity_leaks(self) -> List[str]:
        joined = " ".join(self.entities).lower()
        return [s for s in self.case.forbid_entity_substrings if s.lower() in joined]


def run(engine) -> List[CaseResult]:
    results = []
    for c in GOLD:
        ner = engine.extract_and_resolve(c.title, c.body)
        assets = {a.ticker for a in ner.resolved_assets if a.ticker}
        entities = [e.canonical_name for e in ner.resolved_entities]
        results.append(CaseResult(c, ner.detected_language, assets, entities))
    return results


def summarize(results: List[CaseResult]) -> dict:
    lang_total = sum(1 for r in results if r.case.language)
    lang_ok = sum(1 for r in results if r.lang_ok)
    fp = sum(len(r.forbidden_assets_emitted) for r in results)
    exp_total = sum(len(r.case.expect_assets) for r in results)
    exp_found = sum(len(r.expected_assets_found) for r in results)
    leaks = sum(len(r.entity_leaks) for r in results)
    return {
        "cases": len(results),
        "language_accuracy": (lang_ok / lang_total) if lang_total else None,
        "asset_false_positives": fp,
        "asset_recall": (exp_found / exp_total) if exp_total else None,
        "entity_leaks": leaks,
    }


def _print_report(results: List[CaseResult]) -> None:
    print("\n=== NER BENCHMARK ===")
    for r in results:
        flag = "OK " if (r.lang_ok and not r.forbidden_assets_emitted and not r.entity_leaks) else "!! "
        print(f"{flag}{r.case.id:18} lang={r.detected_language}({'ok' if r.lang_ok else 'WRONG, want '+r.case.language})"
              f" assets={sorted(r.assets)}")
        if r.forbidden_assets_emitted:
            print(f"     FORBIDDEN ticker(s): {sorted(r.forbidden_assets_emitted)}")
        if r.case.expect_assets and r.expected_assets_found != r.case.expect_assets:
            print(f"     missing expected: {sorted(r.case.expect_assets - r.assets)}")
        if r.entity_leaks:
            print(f"     entity leak(s): {r.entity_leaks}")
    s = summarize(results)
    print("\n--- SUMMARY ---")
    for k, v in s.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    from alpharidge_ai.analyzer.ner_fusion import NERFusionEngine
    eng = NERFusionEngine()
    _print_report(run(eng))
