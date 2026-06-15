from talisman_ai.validator import deep_verify as dv
from talisman_ai.utils import attestation_crypto as ac


def test_should_sample_is_deterministic_at_bounds():
    assert dv.should_sample(rate=1.0, rng_value=0.99) is True
    assert dv.should_sample(rate=0.0, rng_value=0.0) is False
    assert dv.should_sample(rate=0.5, rng_value=0.4) is True
    assert dv.should_sample(rate=0.5, rng_value=0.6) is False


def test_merkle_matches_returns_none_when_consistent():
    leaves = [{"resource_type": "tweet", "resource_id": "1", "miner_hotkey": "m1",
               "validator_verdict": "valid", "categorical_key": "BTC|bull", "points_awarded": 1.0}]
    expected = ac.merkle_root(leaves)
    assert dv.merkle_mismatch(expected_root=expected, leaves=leaves) is None


def test_merkle_mismatch_returns_reason():
    leaves = [{"resource_type": "tweet", "resource_id": "1", "miner_hotkey": "m1",
               "validator_verdict": "valid", "categorical_key": "BTC|bull", "points_awarded": 1.0}]
    reason = dv.merkle_mismatch(expected_root="not-the-real-root", leaves=leaves)
    assert reason == "attribution_mismatch"
