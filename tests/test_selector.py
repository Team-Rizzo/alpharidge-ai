"""Keyed selection."""

import pytest

from alpharidge_ai.validator import selector as sel

KEY_A = b"validator-a-secret-key"
KEY_B = b"validator-b-secret-key"


def test_fractions_are_reproducible_and_keyed():
    a, b = sel.Selector(KEY_A), sel.Selector(KEY_B)
    assert [a.fraction(i) for i in range(50)] == [sel.Selector(KEY_A).fraction(i) for i in range(50)]
    assert [a.fraction(i) for i in range(50)] != [b.fraction(i) for i in range(50)]
    assert a.fraction(7, "x") != a.fraction(7, "y")


def test_fractions_are_uniform():
    s = sel.Selector(KEY_A)
    rate = sum(s.fraction(i) < 0.1 for i in range(5000)) / 5000
    assert rate == pytest.approx(0.10, abs=0.02)


def test_key_is_not_exposed_by_repr():
    assert "secret" not in repr(sel.Selector(b"super-secret"))


def test_empty_key_is_refused():
    with pytest.raises(ValueError):
        sel.Selector(b"")
