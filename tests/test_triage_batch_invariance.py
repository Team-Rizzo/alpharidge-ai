"""A miner's triage score must not depend on how its work was sliced into batches."""

import pytest

from alpharidge_ai.validator.triage_grader import (
    MAX_OBSERVATION_WEIGHT, TriageConfig, TriageEvent, TriageGradeResult)

CFG = TriageConfig()


def _result(n, flagged=(), proof=(), hard=()):
    return TriageGradeResult(
        events=([TriageEvent("soft", "x", aid) for aid in flagged]
                + [TriageEvent("hard", "y", aid) for aid in hard]),
        proof_failures=list(proof), batch_size=n)


def _mean_score(batches):
    obs = [o for res, clean in batches for o in res.observations(CFG, clean)]
    total = sum(w for _, _, w in obs)
    return sum(s * w for _, s, w in obs) / total


def test_one_soft_observation_per_batch_whatever_its_size():
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
@pytest.mark.parametrize("hard", [set(), {9}, {9, 20, 21}])
def test_same_work_scores_the_same_however_it_is_sliced(n, hard):
    soft = {3, 17}
    whole = [(_result(32, flagged=soft, hard=hard), 0)]
    sliced = []
    for start in range(0, 32, n):
        ids = list(range(start, start + n))
        bad = soft | hard
        clean = next((i for i in ids if i not in bad), ids[0])
        sliced.append((_result(n, flagged=[i for i in ids if i in soft],
                               hard=[i for i in ids if i in hard]), clean))
    assert _mean_score(sliced) == pytest.approx(_mean_score(whole))


@pytest.mark.parametrize("n", [1, 2, 3, 8, 32])
@pytest.mark.parametrize("kind", ["hard", "proof"])
def test_the_penalty_keeps_its_full_weight_at_every_size(n, kind):
    res = _result(n, **{kind: list(range(n))})
    obs = res.observations(CFG, 0)
    total = sum(w for _, _, w in obs)
    assert total == pytest.approx(CFG.clean_weight * (1 + CFG.hard_severity))
    assert all(w <= MAX_OBSERVATION_WEIGHT for _, _, w in obs)


def test_hard_events_pull_harder_than_soft_ones():
    soft = _mean_score([(_result(32, flagged=[5]), 0)])
    hard = _mean_score([(_result(32, hard=[5]), 0)])
    assert hard < soft


def test_the_hard_term_avoids_articles_already_scored():
    obs = _result(32, hard=[4, 6]).observations(CFG, 0, avoid={4})
    keys = [aid for aid, _, _ in obs]
    assert 6 in keys and 4 not in keys


def test_no_clean_credit_still_records_findings():
    (aid, score, weight), = _result(32, flagged=[4, 9]).observations(CFG, None)
    assert score == 0.0
    assert weight == pytest.approx(CFG.clean_weight * 2 / 32)


def test_no_clean_credit_and_no_findings_records_nothing():
    assert _result(32).observations(CFG, None) == []


def test_proof_failures_are_pooled_like_hard_events():
    obs = _result(32, flagged=[1], proof=[2, 3]).observations(CFG, 0)
    assert (0, pytest.approx(29 / 32), CFG.clean_weight) in obs
    assert (2, 0.0, pytest.approx(CFG.clean_weight * CFG.hard_severity * 2 / 32)) in obs


def test_a_proof_failure_does_not_erase_the_rest_of_the_batch():
    (aid, score, _), *_ = _result(32, proof=[5]).observations(CFG, 0)
    assert aid == 0
    assert score == pytest.approx(31 / 32)


@pytest.mark.parametrize("n", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("proof", [{11}, {11, 12, 30}])
def test_proof_failures_score_the_same_however_the_work_is_sliced(n, proof):
    hard = {9}
    whole = [(_result(32, hard=hard, proof=proof), 0)]
    sliced = []
    for start in range(0, 32, n):
        ids = list(range(start, start + n))
        bad = hard | proof
        clean = next((i for i in ids if i not in bad), ids[0])
        sliced.append((_result(n, hard=[i for i in ids if i in hard],
                               proof=[i for i in ids if i in proof]), clean))
    assert _mean_score(sliced) == pytest.approx(_mean_score(whole))


def test_weights_stay_inside_the_store_bound():
    for n in (1, 32, 64):
        for res in (_result(n), _result(n, flagged=range(n)),
                    _result(n, proof=range(n))):
            for clean in (0, None):
                for _, _, w in res.observations(CFG, clean):
                    assert 0.0 < w <= 10.0
