"""A miner's triage score must not depend on how its work was sliced into batches."""

import pytest

from alpharidge_ai.validator.triage_grader import (
    TriageConfig, TriageEvent, TriageGradeResult)

CFG = TriageConfig()


def _result(n, flagged=(), proof=()):
    return TriageGradeResult(
        events=[TriageEvent("soft", "x", aid) for aid in flagged],
        proof_failures=list(proof), batch_size=n)


def _mean_score(batches):
    obs = [o for res, clean in batches for o in res.observations(CFG, clean)]
    total = sum(w for _, _, w in obs)
    return sum(s * w for _, s, w in obs) / total


def test_one_observation_per_batch_whatever_its_size():
    for n in (1, 2, 8, 32):
        assert len(_result(n).observations(CFG, 0)) == 1
        assert len(_result(n, flagged=[0]).observations(CFG, 1 if n > 1 else 0)) == 1


def test_clean_batch_scores_one():
    assert _result(32).observations(CFG, 5) == [(5, 1.0, CFG.clean_weight)]


def test_score_is_the_unflagged_share():
    (aid, score, weight), = _result(32, flagged=[7]).observations(CFG, 0)
    assert score == pytest.approx(31 / 32)
    assert weight == CFG.clean_weight


@pytest.mark.parametrize("n", [1, 2, 4, 8, 16])
def test_same_work_scores_the_same_however_it_is_sliced(n):
    flagged = {3, 17}
    whole = [(_result(32, flagged=flagged), 0)]
    sliced = []
    for start in range(0, 32, n):
        ids = list(range(start, start + n))
        bad = [i for i in ids if i in flagged]
        clean = next((i for i in ids if i not in flagged), ids[0])
        sliced.append((_result(n, flagged=bad), clean))
    assert _mean_score(sliced) == pytest.approx(_mean_score(whole))


def test_no_clean_credit_still_records_findings():
    (aid, score, weight), = _result(32, flagged=[4, 9]).observations(CFG, None)
    assert score == 0.0
    assert weight == pytest.approx(CFG.clean_weight * 2 / 32)


def test_no_clean_credit_and_no_findings_records_nothing():
    assert _result(32).observations(CFG, None) == []


def test_proof_failure_is_per_article_and_hard():
    obs = _result(32, flagged=[1], proof=[2, 3]).observations(CFG, 0)
    assert (2, 0.0, CFG.hard_weight) in obs
    assert (3, 0.0, CFG.hard_weight) in obs
    assert all(score < 1.0 for _, score, _ in obs)


def test_weights_stay_inside_the_store_bound():
    for n in (1, 32, 64):
        for res in (_result(n), _result(n, flagged=range(n)),
                    _result(n, proof=range(n))):
            for clean in (0, None):
                for _, _, w in res.observations(CFG, clean):
                    assert 0.0 < w <= 10.0
