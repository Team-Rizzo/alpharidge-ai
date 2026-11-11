"""Numbers the floor must read: spelled out, and with either locale's marks."""

import types

import pytest

from alpharidge_ai.verify import floor


def values(text):
    return [round(n.value, 4) for n in floor.parse_numbers(floor.normalize(text).text)]


def units(text):
    return {n.unit for n in floor.parse_numbers(floor.normalize(text).text)}


# ---- spelled-out numbers -----------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("Six children were rescued.", 6),
    ("It lasted twenty-five years.", 25),
    ("Three thousand returns were filed.", 3000),
    ("Two hundred and fifty thousand barrels.", 250_000),
    ("Nineteen people attended.", 19),
    ("Forty-two units sold.", 42),
])
def test_english_number_words_are_read(text, expected):
    assert expected in values(text)


def test_digits_still_win_where_both_appear():
    assert 2 in values("Two ambulances arrived and 2 ambulances left.")


# ---- a scale word alone is not a number --------------------------------------------

def test_a_bare_scale_word_emits_nothing():
    """Reading 'billions' as 1e9 would ground that claim against any article using it."""
    assert values("Losses ran into the billions.") == []
    assert values("The company is worth millions.") == []


def test_a_scale_with_a_unit_still_reads():
    assert 1_000_000 in values("One million shares changed hands.")


# ---- locale formats ----------------------------------------------------------------

def test_a_decimal_comma_is_one_number_not_two():
    got = values("Le prix atteint 17,99 euros.")
    assert 17.99 in got
    assert not (17 in got and 99 in got and 17.99 not in got)


def test_dot_grouping_is_read_as_thousands():
    assert 1_203_400 in values("Ventas de 1.203.400 unidades.")


def test_comma_grouping_still_reads_as_thousands():
    assert 1203 in values("Revenue of 1,203 units.")


def test_an_ambiguous_mark_yields_both_readings():
    """1,203 is one-two-oh-three or one-point-two-oh-three depending on the writer."""
    got = values("The figure was 1,203 today.")
    assert 1203 in got and 1.203 in got


def test_mixed_marks_resolve_by_the_rightmost():
    assert 1234.56 in values("Total 1,234.56 dollars.")
    assert 1234.56 in values("Total 1.234,56 euros.")


# ---- currency words ----------------------------------------------------------------

@pytest.mark.parametrize("text,code", [
    ("It cost 17.99 euros.", "currency:EUR"),
    ("Priced at 20 dollars.", "currency:USD"),
    ("Worth 500 reais.", "currency:BRL"),
    ("About 300 pounds.", "currency:GBP"),
])
def test_currency_words_are_recognised(text, code):
    assert code in units(text)


def test_a_percentage_is_still_a_percentage():
    assert "pct" in units("A queda de 1,5% no periodo.")


# ---- grounding ---------------------------------------------------------------------

def visible(text, value, unit="count"):
    return floor.visible(types.SimpleNamespace(value=value, unit=unit),
                         floor.parse_numbers(floor.normalize(text).text))


@pytest.mark.parametrize("text,claim,unit", [
    ("Six children were rescued from the building.", 6.0, "count"),
    ("The figure reached 17,99 million euros.", 17_990_000.0, "EUR"),
    ("Sales of 1.203.400 units this year.", 1_203_400.0, "count"),
    ("Two hundred and fifty thousand barrels shipped.", 250_000.0, "count"),
    ("Revenue of 25,7 million euros.", 25.7, "million euros"),
])
def test_an_honest_claim_is_visible(text, claim, unit):
    assert visible(text, claim, unit)


def test_an_invented_number_is_not_visible():
    assert not visible("Six children were rescued.", 9_400.0)
    for wrong in (12.03, 120.3, 1_203_000.0, 99.0):
        assert not visible("The figure was 1,203 today.", wrong), wrong
    assert not visible("Revenue of 25,7 million euros.", 99.9, "million euros")


def test_unit_magnitude_reads_the_scale():
    assert floor.unit_magnitude("million euros") == 1e6
    assert floor.unit_magnitude("usd_millions") == 1e6
    assert floor.unit_magnitude("thousands") == 1e3
    assert floor.unit_magnitude("billion USD") == 1e9
    assert floor.unit_magnitude("USD") == 1.0
    assert floor.unit_magnitude(None) == 1.0


def test_ordinary_adjacent_numbers_are_not_summed():
    """'100 and 20' contains no 120, in any reading."""
    assert 120.0 not in values("100 and 20 people")
    assert not visible("The vote was 100 to 20.", 120.0)


def test_either_reading_of_an_ambiguous_mark_is_visible():
    assert visible("Cost 1,203 euros.", 1203.0, "EUR")
    assert visible("Cost 1,203 euros.", 1.203, "EUR")
    assert not visible("Cost 1,203 euros.", 9999.0, "EUR")


def test_a_number_ends_at_a_line_break_and_a_time_is_not_a_scale():
    def read(text):
        return [n.value for n in floor.parse_numbers(floor.normalize(text, lines=True).text)]
    assert 53_634 in read("Units: About 53,634\n3M Company") and 53_634_300 not in read(
        "Units: About 53,634\n3M Company")
    assert read("Trading resumes at 8 a.m. and closes at 4:30 p.m.") == [8, 4, 30]
    assert 5_300_000 in read("Revenue of $5.3m.")
