"""Tests for entity_filter — confidence gates, blocklists, offset-overlap dedup.

Backstops the §3.3 NER noise (days/stopwords/numeric garbage) and replaces the
broken substring dedup with character-offset clustering that keeps the longest,
highest-confidence span per region.
"""

from talisman_ai.analyzer.entity_filter import (
    Candidate, EntityFilter, in_salient_range, apply_salience,
)


def _names(cands):
    return [c.text for c in cands]


def test_drops_low_confidence_singleton():
    f = EntityFilter()
    out = f.filter([Candidate("Acme Corp", 0, 9, "ORG", "gliner", 0.2)], "en")
    assert out == []


def test_keeps_high_confidence_singleton():
    f = EntityFilter()
    out = f.filter([Candidate("Goldman Sachs", 0, 13, "ORG", "spacy", 0.95)], "en")
    assert _names(out) == ["Goldman Sachs"]


def test_multi_model_agreement_keeps_moderate_score():
    f = EntityFilter()
    # Same span, two models, each individually below the singleton threshold.
    cands = [
        Candidate("Powell", 10, 16, "PERSON", "spacy", 0.45),
        Candidate("Powell", 10, 16, "PERSON", "flair", 0.45),
    ]
    out = f.filter(cands, "en")
    assert _names(out) == ["Powell"]


def test_offset_overlap_dedup_keeps_longest():
    f = EntityFilter()
    cands = [
        Candidate("Bank of China", 0, 13, "ORG", "spacy", 0.9),
        Candidate("China", 8, 13, "GPE", "gliner", 0.85),
    ]
    out = f.filter(cands, "en")
    assert _names(out) == ["Bank of China"]


def test_distinct_nonoverlapping_entities_both_kept():
    f = EntityFilter()
    cands = [
        Candidate("China", 0, 5, "GPE", "spacy", 0.9),
        Candidate("Tesla", 20, 25, "ORG", "spacy", 0.9),
    ]
    out = f.filter(cands, "en")
    assert set(_names(out)) == {"China", "Tesla"}


def test_drops_english_day_of_week():
    f = EntityFilter()
    out = f.filter([Candidate("Sunday", 0, 6, "GPE", "gliner", 0.9)], "en")
    assert out == []


def test_drops_italian_day_domenica():
    f = EntityFilter()
    out = f.filter([Candidate("domenica", 0, 8, "GPE", "gliner", 0.9)], "it")
    assert out == []


def test_drops_numeric_garbage():
    f = EntityFilter()
    out = f.filter([Candidate("2024", 0, 4, "ORG", "gliner", 0.9)], "en")
    assert out == []


def test_drops_single_char():
    f = EntityFilter()
    out = f.filter([Candidate("V", 0, 1, "ORG", "spacy", 0.9)], "en")
    assert out == []


def test_in_salient_range():
    ranges = [(0, 50), (80, 120)]
    assert in_salient_range(10, 20, ranges) is True
    assert in_salient_range(60, 70, ranges) is False   # in a verbless gap
    assert in_salient_range(80, 119, ranges) is True


def test_apply_salience_drops_entities_in_verbless_fragment():
    # "Federal Reserve" sits in a verb-bearing sentence (0-60); "WhatsApp Gmail"
    # sits in a verbless share-bar fragment (60-90).
    cands = [
        Candidate("Federal Reserve", 5, 20, "ORG", "spacy", 0.9),
        Candidate("WhatsApp", 65, 73, "ORG", "gliner", 0.9),
    ]
    out = apply_salience(cands, salient_ranges=[(0, 60)])
    assert [c.text for c in out] == ["Federal Reserve"]


def test_apply_salience_noop_when_no_ranges():
    # No verb-bearing sentence detected -> don't nuke everything.
    cands = [Candidate("Nvidia", 0, 6, "ORG", "spacy", 0.9)]
    assert apply_salience(cands, salient_ranges=[]) == cands


def test_deterministic_and_sorted_by_position():
    f = EntityFilter()
    cands = [
        Candidate("Tesla", 30, 35, "ORG", "spacy", 0.9),
        Candidate("Nvidia", 0, 6, "ORG", "spacy", 0.9),
    ]
    out = f.filter(cands, "en")
    assert _names(out) == ["Nvidia", "Tesla"]
    assert f.filter(cands, "en") == out
