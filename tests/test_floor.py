"""Deterministic floor: normalisation and number grounding."""

import types

import pytest

from alpharidge_ai.verify import floor


def claim(value, unit="none", metric="revenue"):
    return types.SimpleNamespace(field=metric, value=value, unit=unit)


# ---- normalisation ----------------------------------------------------------------

def test_normalisation_folds_quotes_dashes_and_whitespace():
    n = floor.normalize('The  CEO said “we’re up” — a lot.')
    assert n.text == 'the ceo said "we\'re up" - a lot.'


def test_offsets_round_trip_to_the_original_text():
    original = "Alpha   BETA gamma"
    n = floor.normalize(original)
    i = n.text.index("beta")
    start, end = n.to_original(i, i + 4)
    assert original[start:end] == "BETA"


def test_empty_text_is_handled():
    assert floor.normalize(None).text == ""
    assert floor.normalize("   ").text == ""


# ---- number parsing ---------------------------------------------------------------

@pytest.mark.parametrize("text,value,unit", [
    ("revenue of $1.2 billion this year", 1.2e9, "currency:USD"),
    ("shares fell 12.5% today", 12.5, "pct"),
    ("a headcount of 1,203 people", 1203.0, "count"),
    ("profit of 450 million usd", 450e6, "currency:USD"),
    ("guidance cut to 3.4bn", 3.4e9, "count"),
])
def test_parses_magnitude_and_unit(text, value, unit):
    got = floor.parse_numbers(floor.normalize(text).text)
    assert any(n.value == pytest.approx(value) and n.unit == unit for n in got), got


def test_grouped_digits_are_one_number():
    got = floor.parse_numbers(floor.normalize("revenue was 1,203,400,000 dollars").text)
    assert any(n.value == pytest.approx(1_203_400_000.0) for n in got)


# ---- grounding --------------------------------------------------------------------

def test_exact_value_is_grounded():
    numbers = floor.parse_numbers(floor.normalize("revenue of $1.2 billion").text)
    assert floor.visible(claim(1.2e9, "USD"), numbers)


def test_rounded_claim_grounds_against_the_full_number():
    numbers = floor.parse_numbers(floor.normalize("revenue was $1,203,400,000").text)
    assert floor.visible(claim(1.2e9, "USD"), numbers)


def test_a_number_not_in_the_text_is_ungrounded():
    numbers = floor.parse_numbers(floor.normalize("revenue of $1.2 billion").text)
    assert not floor.visible(claim(9.9e9, "USD"), numbers)


def test_tolerance_boundary():
    numbers = floor.parse_numbers(floor.normalize("it rose 100 points").text)
    assert floor.visible(claim(100.4, "count"), numbers)     # within 0.5%
    assert not floor.visible(claim(107.0, "count"), numbers)


# ---- hashing and end to end ------------------------------------------------------

def test_content_hash_is_sha256_of_the_text():
    assert floor.content_hash("abc") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    assert floor.content_hash(None) == floor.content_hash("")


def test_a_text_value_is_not_a_figure():
    numbers = floor.parse_numbers(floor.normalize("The figure was 5.").text)
    assert floor.visible(claim("five", "count"), numbers)


@pytest.mark.parametrize("value,unit,text", [
    (1.2, "usd_billions", "a deal worth $1.2 billion"),
    (1.2, "usd_billions", "a deal worth 1,200 million"),
    (1200, "usd_millions", "(in millions, except per share data) revenue 1,200"),
    (-116, "usd_millions", "(in millions) net loss (116)"),
    (25, "basis_points", "a cut of a quarter point"),
    (5.1, "percent", "California | 5.1"),
    (22, "thousands", "employers added 22,000 jobs"),
])
def test_a_figure_is_visible_in_any_format_unit_aside(value, unit, text):
    numbers = floor.parse_numbers(floor.normalize(text).text)
    assert floor.visible(claim(value, unit), numbers)
    assert not floor.visible(claim(value * 3, unit), numbers)


def test_a_dollar_figure_is_not_a_number_in_another_currency():
    numbers = floor.parse_numbers(floor.normalize("worth C$1.26 billion or €1.26 billion").text)
    assert not floor.visible(claim(1.26, "usd_billions"), numbers)
    assert floor.visible(claim(1.26e9, "count"), numbers)


def test_basis_points_do_not_ground_against_a_percentage():
    numbers = floor.parse_numbers(floor.normalize("Margin expanded 12.5% in the quarter.").text)
    assert not floor.visible(claim(12.5, "bps", "margin"), numbers)


def test_every_spelling_of_a_basis_point_is_treated_alike():
    for unit in ("bps", "bp", "basis points", "basis_points", "BP"):
        assert floor.unit_class(unit) == "pct", unit
        assert floor.unit_magnitude(unit) == floor.BASIS_POINT, unit
