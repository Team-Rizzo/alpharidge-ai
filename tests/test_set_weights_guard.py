"""Weights are not set from an all-zero score vector."""
import types
import numpy as np
from alpharidge_ai.base.validator import BaseValidatorNeuron


def _stub(scores):
    calls = []
    v = types.SimpleNamespace(scores=np.asarray(scores, dtype=np.float32))
    v.subtensor = types.SimpleNamespace(set_weights=lambda **kw: calls.append(kw) or (True, ""))
    return v, calls


def test_all_zero_scores_set_nothing():
    v, calls = _stub([0, 0, 0])
    BaseValidatorNeuron.set_weights(v)
    assert calls == []


def test_nan_only_scores_set_nothing():
    v, calls = _stub([np.nan, 0])
    BaseValidatorNeuron.set_weights(v)
    assert calls == []
