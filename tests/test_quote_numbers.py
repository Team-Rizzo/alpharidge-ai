"""A quote only counts if the numbers it states are the article's."""

import types

import pytest

from alpharidge_ai.oracle import floor

TEXT = ('The company said sales reached $230 billion last year. '
        '"About 97% of the product ships from two plants," the chief executive said. '
        'Guidance was set at $680 million to $700 million.')


def _quote(text):
    return types.SimpleNamespace(text=text, start_offset=None, end_offset=None)


def _intel(*quotes):
    return types.SimpleNamespace(numeric_claims=[], quotes=list(quotes), assets=[],
                                 entities=[])


def _aligned(quote_text):
    result = floor.evaluate(_intel(_quote(quote_text)), TEXT)
    return 0 in result.aligned_quotes


def test_a_faithful_quote_aligns():
    assert _aligned("About 97% of the product ships from two plants")


@pytest.mark.parametrize("altered", [
    "About 47% of the product ships from two plants",
    "sales reached $730 billion last year",
    "Guidance was set at $180 million to $700 million",
])
def test_a_quote_with_an_altered_number_is_rejected(altered):
    result = floor.evaluate(_intel(_quote(altered)), TEXT)
    assert 0 not in result.aligned_quotes
    assert 0 in result.rejected_quotes


def test_a_quote_without_numbers_is_unaffected():
    assert floor.quote_numbers_hold("ships from two plants", TEXT, 0, len(TEXT))


def test_every_stated_number_must_be_present_as_often_as_stated():
    passage = "rose 5% then 5% again"
    assert floor.quote_numbers_hold("5% then 5%", passage, 0, len(passage))
    assert not floor.quote_numbers_hold("5% then 5% then 5%", passage, 0, len(passage))
