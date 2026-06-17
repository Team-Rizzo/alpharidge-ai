from talisman_ai.analyzer.entity_filter import (
    EntityFilter, Candidate, _normalize_entity_text,
)

ef = EntityFilter()


def _texts(cands):
    return [c.text for c in ef.filter(cands, "en")]


def test_normalize_trims_hyphen_modifier():
    assert _normalize_entity_text("Elon Musk-backed") == "Elon Musk"
    assert _normalize_entity_text("SoftBank-led round") == "SoftBank"


def test_normalize_trims_appositive_tail():
    assert _normalize_entity_text("Anthropic principle") == "Anthropic"
    assert _normalize_entity_text("Doppler effect") == "Doppler"


def test_normalize_leaves_clean_entities_untouched():
    assert _normalize_entity_text("Goldman Sachs") == "Goldman Sachs"
    assert _normalize_entity_text("Bank of China") == "Bank of China"


def test_filter_trims_hyphenated_modifier():
    out = _texts([Candidate("Elon Musk-backed", 0, 16, "PERSON", "spacy", 0.9)])
    assert "Elon Musk" in out and "Elon Musk-backed" not in out


def test_filter_trims_appositive_principle():
    out = _texts([Candidate("Anthropic principle", 0, 19, "ORG", "spacy", 0.9)])
    assert out == ["Anthropic"]
