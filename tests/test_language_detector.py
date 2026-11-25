"""language_detector: deterministic, and tells English from other languages."""

from alpharidge_ai.verify.language_detector import detect_language


def test_detects_english():
    r = detect_language("The Federal Reserve cut interest rates on Wednesday, sending stocks higher.")
    assert r.code == "en"
    assert 0.0 <= r.confidence <= 1.0


def test_detects_italian():
    r = detect_language("La società Acme guida sedici aziende in Cina per negoziati commerciali. "
                        "Domenica i mercati restano chiusi.")
    assert r.code == "it"


def test_detects_russian():
    r = detect_language("Министерство заключило контракт на два миллиарда долларов с компанией "
                        "Глобекс на строительство новых заводов.")
    assert r.code == "ru"


def test_empty_text_falls_back_to_en():
    r = detect_language("")
    assert r.code == "en"
    assert r.confidence == 0.0


def test_short_text_does_not_crash():
    # RSS titles can be tiny; must never raise.
    r = detect_language("Fed")
    assert isinstance(r.code, str) and len(r.code) >= 2


def test_deterministic():
    src = "Il mercato azionario italiano ha chiuso in rialzo dopo la decisione della banca centrale."
    assert detect_language(src) == detect_language(src)


def test_result_is_indexable_tuple():
    r = detect_language("Hello world, this is a test of the language detector.")
    code, conf = r
    assert code == "en"


def test_detections_in_threads_at_first_use_agree():
    import importlib
    from concurrent.futures import ThreadPoolExecutor
    from langdetect import detector_factory
    from alpharidge_ai.verify import language_detector as ld
    text = ("The company said on Tuesday it had agreed to buy its rival for $1.2 billion in cash, "
            "a deal its board approved unanimously after months of talks. ") * 5
    for _ in range(5):
        detector_factory._factory = None
        importlib.reload(ld)
        with ThreadPoolExecutor(8) as pool:
            codes = set(pool.map(lambda _: ld.detect_language(text).code, range(16)))
        assert codes == {"en"}
