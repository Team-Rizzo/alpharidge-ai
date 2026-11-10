"""Property tests for the floor, fact matching and settlement modules.

These drive the real modules rather than a model of them, so the properties below hold
of the code that runs rather than of a description of it.
"""

import random
import types

import pytest

from alpharidge_ai.mechanism import settlement
from alpharidge_ai.verify import facts, floor

ARTICLE = ('Revenue rose to $1.2 billion in the quarter, up 12.5% year on year. '
           'The chief executive said "margins should recover in the second half" '
           'on Monday. Headcount reached 1,203 and the vote was 100 to 20. '
           'Sales of 1.203.400 units.')


def claim(value, unit="count", metric="m"):
    return types.SimpleNamespace(field=metric, value=value, unit=unit)


def visible(submitted, text=ARTICLE):
    """Indexes of the claims whose figure the page prints."""
    numbers = floor.parse_numbers(floor.normalize(text).text)
    return {i for i, c in enumerate(submitted) if floor.visible(c, numbers)}


# ---- values a submitter might construct from what is in the text -------------------

def test_no_route_grants_a_value_absent_from_the_article():
    attacks = [
        claim(120.0),                    # 100 + 20, adjacent numbers
        claim(1.0, "million"),           # unscaled unit against a bare 1
        claim(1e9),                      # from the word "billion"
        claim(1220.0),                   # 1203 plus a neighbour
    ]
    assert visible(attacks) == set()


def test_no_scaled_unit_matches_or_grounds_a_different_magnitude():
    # `1.2 billion USD` is deliberately absent: it equals the reference and SHOULD
    # match. The property concerns claims asserting a different magnitude.
    reference = [claim(1.0, "count", "headcount"), claim(1.2e9, "USD", "revenue")]
    attacks = [claim(1.0, "million", "headcount"),
               claim(1.2e9, "million USD", "revenue"),
               claim(1.2e9, "billion USD", "revenue")]
    assert visible(attacks) == set()
    assert not any(facts.claims_match(a, r) for a in attacks for r in reference)


def test_honest_claims_survive_all_of_it():
    """The properties above must not have been bought by rejecting real work."""
    honest = [claim(1.2e9, "USD", "revenue"), claim(12.5, "%", "growth"),
              claim(1203.0, "count", "headcount"), claim(1203400.0, "count", "sales")]
    assert len(visible(honest)) == 4


def test_an_equivalent_claim_written_differently_still_matches():
    # Both sides name a figure the article states — 1,203 — written at different scales.
    assert facts.claims_match(claim(1.203, "thousand", "headcount"),
                              claim(1203, "count", "headcount"))


# ---- settlement --------------------------------------------------------------------

def test_pay_and_burn_always_account_for_the_whole_emission():
    rng = random.Random(3)
    for _ in range(500):
        work = {f"m{i}": rng.choice([0.0, rng.random() * 5000])
                for i in range(rng.randrange(0, 30))}
        result = settlement.settle(work, rng.choice([1.0, 1000.0, 26697.0]))
        assert 0.0 <= result.burn <= 1.0
        assert result.paid + result.burn == pytest.approx(1.0)


def test_no_uid_is_paid_more_by_splitting_its_work():
    rng = random.Random(5)
    for _ in range(200):
        total, capacity = rng.random() * 5000, rng.choice([100.0, 1000.0, 26697.0])
        others = {"other": rng.random() * 2000}
        solo = settlement.settle({"solo": total, **others}, capacity).shares["solo"]
        n = rng.randrange(2, 8)
        # The rest of the field has to be present on both sides, or the denominators
        # differ and the comparison says nothing.
        split = settlement.settle({**{f"s{i}": total / n for i in range(n)}, **others},
                                  capacity)
        assert sum(split.shares[f"s{i}"] for i in range(n)) <= solo + 1e-9


# ---- generated inputs ---------------------------------------------------------------

CORPORA = [
    'Revenue rose to $1.2 billion, up 12.5%. Headcount reached 1,203.',
    'The vote was 100 to 20. Turnout was 45 percent of 3,000 members.',
    'Sales of 1.203.400 units and 17,99 million euros this year.',
    'Six children and twenty-five adults were rescued over 3 hours.',
    'About 20 million barrels and 1.234,56 euros a tonne.',
]


@pytest.mark.parametrize("text", CORPORA)
def test_every_stated_value_is_visible(text):
    parsed = [n for n in floor.parse_numbers(floor.normalize(text).text) if n.literal]
    assert parsed and len(visible([claim(n.value) for n in parsed], text)) == len(parsed)


def test_the_grounding_tolerance_is_bounded_and_deliberate():
    """A rounded restatement grounds; this pins how far it goes.

    The tolerance is fixed; a silent widening fails here.
    """
    assert floor.VALUE_REL_TOLERANCE == 0.005

    text = "Headcount reached 1,203 this year."
    numbers = floor.parse_numbers(floor.normalize(text).text)
    inside = 1203 * (1 + floor.VALUE_REL_TOLERANCE * 0.9)
    outside = 1203 * (1 + floor.VALUE_REL_TOLERANCE * 3)
    assert floor.visible(claim(inside), numbers)
    assert not floor.visible(claim(outside), numbers)


# ---- invariants ------------------------------------------------------------------
#
# The cases above are individual inputs. This states the general property over generated
# ones: two claims match only when they assert the same figure.
#
# The scale table is written out again rather than imported: a check that reads the
# same table as the code cannot catch a wrong entry in it.

SCALES = {
    "": 1.0, "%": 1.0, "pct": 1.0, "percent": 1.0,
    "bps": 0.01, "bp": 0.01, "basis points": 0.01, "basis_points": 0.01,
    "thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12,
    "lakh": 1e5, "crore": 1e7,
    "barrels": 1.0, "users": 1.0, "vehicles": 1.0,
}

_VALUES = (5, 12.5, 50, 250, 3.2, 1000, 1.203)


def _asserts(value, unit):
    """What a claim says, in one scale, computed independently of the floor."""
    return value * SCALES[unit]


@pytest.mark.parametrize("seed", (21, 22, 23))
def test_two_claims_match_only_when_they_assert_the_same_figure(seed):
    """The same property, on the reference comparison."""
    rng = random.Random(seed)
    units = list(SCALES)
    for _ in range(400):
        lv, lu = rng.choice(_VALUES), rng.choice(units)
        rv, ru = rng.choice(_VALUES), rng.choice(units)
        if not facts.claims_match(claim(lv, lu), claim(rv, ru)):
            continue
        left, right = _asserts(lv, lu), _asserts(rv, ru)
        assert abs(left - right) <= abs(right) * 0.01, (
            f"{lv} {lu!r} ({left}) matched {rv} {ru!r} ({right})")
