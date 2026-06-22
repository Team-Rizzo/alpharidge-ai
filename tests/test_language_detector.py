"""Tests for language_detector — fixes §3.4 (detected_language hardcoded "en").

Must be deterministic (the validator hard-gates detected_language as an exact
match) and must correctly distinguish the languages actually seen in the corpus
(English, Italian, Russian).
"""

from alpharidge_ai.analyzer.language_detector import detect_language


def test_detects_english():
    r = detect_language("The Federal Reserve cut interest rates on Wednesday, sending stocks higher.")
    assert r.code == "en"
    assert 0.0 <= r.confidence <= 1.0


def test_detects_italian():
    # The §3.4 case: was mislabeled "en".
    r = detect_language("Trump guida 16 miliardari statunitensi in Cina per negoziati commerciali. "
                        "Domenica i mercati restano chiusi.")
    assert r.code == "it"


def test_detects_russian():
    r = detect_language("Пентагон заключил контракт на два миллиарда долларов с компанией "
                        "General Dynamics на строительство подводных лодок.")
    assert r.code == "ru"


def test_empty_text_falls_back_to_en():
    r = detect_language("")
    assert r.code == "en"
    assert r.confidence == 0.0


def test_short_text_does_not_crash():
    # RSS titles can be tiny; must never raise.
    r = detect_language("BTC")
    assert isinstance(r.code, str) and len(r.code) >= 2


def test_deterministic():
    src = "Il mercato azionario italiano ha chiuso in rialzo dopo la decisione della banca centrale."
    assert detect_language(src) == detect_language(src)


def test_result_is_indexable_tuple():
    r = detect_language("Hello world, this is a test of the language detector.")
    code, conf = r
    assert code == "en"
