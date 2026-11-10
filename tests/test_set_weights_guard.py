"""Weights are not set from an all-zero score vector, nor while the burn UID check fails."""
import types
import numpy as np
from alpharidge_ai.chain.validator import BaseValidatorNeuron


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


def test_no_weights_while_the_burn_uid_check_fails(monkeypatch):
    from neurons.validator import Validator
    calls = []
    monkeypatch.setattr(BaseValidatorNeuron, "set_weights", lambda self: calls.append(self))
    for ledger in (None, types.SimpleNamespace(held=True), types.SimpleNamespace(held=False)):
        v = Validator.__new__(Validator)
        v._ledger = ledger
        v.set_weights()
    assert len(calls) == 1 and calls[0]._ledger.held is False
