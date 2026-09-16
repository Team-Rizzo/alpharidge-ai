"""A drawn model's audit scores can be brought onto a common level."""

import pytest

from alpharidge_ai.mechanism import profile as mp
from alpharidge_ai.models.article_intelligence import ArticleIntelligence
from alpharidge_ai.oracle import floor
from alpharidge_ai.oracle.runner import Auditor
from tests.test_floor_gating import TEXT, _payload
from tests.test_profile_client import valid

KEEPER_TEXT = "The committee met on the policy outlook and discussed the risks ahead."


class Judge:
    def judge(self, text, model):
        return {"overall_sentiment": "neutral", "impact_potential": "low",
                "urgency": "same_day", "content_type": "other",
                "assets": [], "entities": []}

    def adjudicate(self, text, claims, model):
        return []


def _auditor(scale=None, keeper_scale=None, keeper=False):
    raw = valid()
    raw["oracle"]["keyed_rate_pool"] = 0.0 if keeper else 1.0
    raw["oracle"]["keyed_rate_keeper"] = 1.0 if keeper else 0.0
    entry = {"id": "m", "weight": 1.0}
    if scale is not None:
        entry["scale"] = scale
    if keeper_scale is not None:
        entry["keeper_scale"] = keeper_scale
    raw["oracle"]["grader_models"] = [entry]
    profile = mp.parse(raw)
    return Auditor(b"k", lambda block: profile, grader=Judge())


def _intel():
    return ArticleIntelligence(**_payload([{"metric_name": "revenue", "value": 1.2e9,
                                            "unit": "USD", "confidence": 0.9}]))


def _pool_scores(auditor):
    intel = _intel()
    result = floor.evaluate(intel, TEXT)
    return [o.score for o in (auditor.audit(i, TEXT, intel, intel, result, 0)
                              for i in range(10)) if o is not None]


def _keeper_scores(auditor):
    intel = _intel()
    result = floor.evaluate(intel, KEEPER_TEXT)
    return [o.score for o in (auditor.audit(i, KEEPER_TEXT, intel, intel, result, 0)
                              for i in range(10)) if o is not None]


def test_unscaled_by_default():
    assert _pool_scores(_auditor()) == _pool_scores(_auditor(scale=1.0))


def test_pool_scores_are_scaled():
    plain = _pool_scores(_auditor())
    scaled = _pool_scores(_auditor(scale=0.5))
    assert plain and scaled == pytest.approx([s * 0.5 for s in plain])


def test_keeper_scores_are_scaled_separately():
    plain = _keeper_scores(_auditor(keeper=True))
    scaled = _keeper_scores(_auditor(scale=0.5, keeper_scale=0.8, keeper=True))
    assert plain and scaled == pytest.approx([s * 0.8 for s in plain])


@pytest.mark.parametrize("field", ["scale", "keeper_scale"])
@pytest.mark.parametrize("bad", [0.0, -0.1, 1.01, "x"])
def test_scales_only_ever_lower_a_score(field, bad):
    raw = valid()
    raw["oracle"]["grader_models"][0][field] = bad
    with pytest.raises(mp.ProfileError):
        mp.parse(raw)


def test_scales_are_read_per_model():
    raw = valid()
    raw["oracle"]["grader_models"] = [{"id": "a", "weight": 1.0, "scale": 0.7},
                                      {"id": "b", "weight": 1.0}]
    models = {m.id: m for m in mp.parse(raw).oracle.grader_models}
    assert models["a"].scale == 0.7 and models["b"].scale == 1.0
