"""The reference set is held to the same standard as the submission.

A miner's claim must be found in the article before it earns anything. The grader's own
claims were not checked at all — every one became gold on assertion. Two consequences:
recall was measured against claims no submission could find, and a submission matching one
was credited for it twice, once through gold and once through supported.
"""
import types

import pytest

from alpharidge_ai.mechanism import scoring
from alpharidge_ai.oracle import audit, floor

TEXT = "Margin was 12.5% in the quarter and headcount reached 1,203."


def claim(value, unit, metric):
    return types.SimpleNamespace(value=value, unit=unit, metric_name=metric, context="")


def _floor(claims):
    return floor.evaluate(types.SimpleNamespace(numeric_claims=claims), TEXT)


def test_a_grader_claim_the_article_does_not_state_is_not_gold():
    real, invented = claim(12.5, "%", "margin"), claim(99, "bn", "revenue")
    kept = audit.grounded_grader_claims([real, invented], TEXT)
    assert kept == {0}, "only the claim the article states may be gold"


def test_matching_an_invented_grader_claim_earns_nothing():
    """A claim the article does not state is not part of the reference set."""
    grader = [claim(12.5, "%", "margin"), claim(99, "bn", "revenue")]
    miner = [claim(99, "bn", "revenue")]
    decided = audit.adjudicate(miner, grader, _floor(miner).grounded, TEXT, None)
    assert ("g", 1) not in decided.grader_keys
    assert decided.miner_keys == [("m", 0)], "must not be credited as a grader match"
    scored = scoring.article_score(decided.miner_keys, decided.grader_keys,
                                   decided.valid, {}, score_confidence=False)
    assert scored.precision == 0.0 and scored.recall == 0.0


def test_an_honest_match_is_still_credited():
    grader = [claim(12.5, "%", "margin")]
    miner = [claim(12.5, "%", "margin")]
    decided = audit.adjudicate(miner, grader, _floor(miner).grounded, TEXT, None)
    assert decided.miner_keys == [("g", 0)]
    scored = scoring.article_score(decided.miner_keys, decided.grader_keys,
                                   decided.valid, {}, score_confidence=False)
    assert scored.recall == 1.0 and scored.precision == 1.0


def test_recall_is_measured_only_against_findable_claims():
    """Recall is measured against what the article states, not against the raw set."""
    grader = [claim(12.5, "%", "margin"), claim(99, "bn", "x"), claim(77, "bn", "y")]
    miner = [claim(12.5, "%", "margin")]
    decided = audit.adjudicate(miner, grader, _floor(miner).grounded, TEXT, None)
    scored = scoring.article_score(decided.miner_keys, decided.grader_keys,
                                   decided.valid, {}, score_confidence=False)
    assert scored.recall == 1.0, "the miner found everything the article actually states"


def test_a_grader_that_invents_everything_yields_no_observation():
    """Absence of real gold says nothing about the submission, so it scores nothing —
    rather than scoring zero, which would be evidence it is not."""
    grader = [claim(99, "bn", "x"), claim(77, "bn", "y")]
    miner = [claim(12.5, "%", "margin")]
    decided = audit.adjudicate(miner, grader, _floor(miner).grounded, TEXT, None)
    assert decided.grader_keys == set()
    assert scoring.article_score(decided.miner_keys, decided.grader_keys,
                                 decided.valid, {}) is None
