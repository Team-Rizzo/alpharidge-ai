"""Fact matching compares field, unit class and magnitude."""

import types

from alpharidge_ai.verify import facts


def fact(field, value, unit="USD"):
    return types.SimpleNamespace(field=field, value=value, unit=unit)


def test_matching_tolerates_small_value_differences():
    assert facts.claims_match(fact("revenue", 1.2e9), fact("revenue", 1.2004e9))
    assert not facts.claims_match(fact("revenue", 1.2e9), fact("revenue", 1.9e9))


def test_matching_requires_the_same_field_and_unit():
    assert not facts.claims_match(fact("revenue", 12.5), fact("margin", 12.5))
    assert not facts.claims_match(fact("revenue", 12.5, "USD"), fact("revenue", 12.5, "%"))


def test_field_case_and_spacing_do_not_matter():
    assert facts.claims_match(fact("Quarterly  Revenue", 1.2e9), fact("quarterly revenue", 1.2e9))
    assert not facts.claims_match(fact("revenue", 1.2e9), fact("quarterly revenue", 1.2e9))


def test_a_scaled_unit_cannot_match_an_unscaled_one():
    """"1 million" and "1" share the unit class `count`."""
    assert not facts.claims_match(fact("n", 1.0, "million"), fact("n", 1.0, "count"))
    assert not facts.claims_match(fact("n", 1.0, "million USD"), fact("n", 1.0, "USD"))


def test_equivalent_facts_written_differently_still_match():
    assert facts.claims_match(fact("n", 1.0, "million"), fact("n", 1e6, "count"))
    assert facts.claims_match(fact("n", 2.5, "billion USD"), fact("n", 2.5e9, "USD"))


def test_basis_points_match_the_percentage_they_equal():
    for unit in ("bps", "bp", "basis points", "basis_points", "BP"):
        assert facts.claims_match(fact("margin", 50, unit), fact("margin", 0.5, "%")), unit
        assert not facts.claims_match(fact("margin", 50, unit), fact("margin", 50, "%")), unit
    assert not facts.claims_match(fact("margin", 12.5, "basis_points"), fact("margin", 12.5, "pct"))


def test_fact_fields_match_whatever_the_case_style():
    theirs = fact("contract value", 460, "EUR millions")
    for name in ("contract_value", "contractValue", "Contract Value"):
        mine = fact(name, 460e6, "EUR")
        assert facts.same_quantity(mine, theirs)
        assert facts.claims_match(mine, theirs)
