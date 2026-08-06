"""Weight-window aggregation.

Covers the parts the archive replay cannot reach: the reward store's retention
and range read, the broadcast re-key filter, and the config clamp. The chain
coefficients are pinned so K=1 and K=7 are compared on equal terms.
"""
import numpy as np
import pytest

from alpharidge_ai import config
from alpharidge_ai.models.reward import Reward
from alpharidge_ai.utils import burn
from alpharidge_ai.utils.reward import MinerReward
from alpharidge_ai.validator.reward_broadcast_store import RewardBroadcastStore

POINTS = [3, 7, 11, 2, 5]
ALPHA_PER_POINT = 500.0
MINER_ALPHA_PER_BLOCK = 0.02


@pytest.fixture(autouse=True)
def pinned_coefficients(monkeypatch):
    monkeypatch.setattr(burn, "get_alpha_per_point", lambda: ALPHA_PER_POINT)
    monkeypatch.setattr(burn, "get_miner_alpha_per_block", lambda: MINER_ALPHA_PER_BLOCK)


class FakeMetagraph:
    def __init__(self, n=256, registered_at=None):
        self.n = n
        self.hotkeys = [f"hk{i}" for i in range(n)]
        self.block_at_registration = registered_at if registered_at is not None else [0] * n


def rewards_for(points, multiplier=1):
    return [Reward(hotkey=f"hk{i}", reward=p * multiplier, epoch=0) for i, p in enumerate(points)]


# --- normalisation -------------------------------------------------------

@pytest.mark.parametrize("k", [1, 4, 7, 12, 24])
def test_same_production_rate_gives_same_weights_at_any_k(k):
    """A miner producing at a steady rate must be paid the same whatever K is.

    Points scale with the window, so the divisor has to as well. If either the
    calculated percent or the minimum-percent floor loses its divisor, this
    fails and the burn control drifts with K.
    """
    mg = FakeMetagraph()
    at_k1 = burn.calculate_weights(rewards_for(POINTS), mg, config.BLOCK_LENGTH)
    at_k = burn.calculate_weights(rewards_for(POINTS, k), mg, k * config.BLOCK_LENGTH)
    assert np.allclose(at_k, at_k1, atol=1e-12)


def test_single_epoch_window_reproduces_the_pre_change_arithmetic():
    """Landing the code at K=1 is a no-op, so step 1 of the rollout is safe.

    The old code divided by the EPOCH_LENGTH constant regardless of how long an
    epoch actually was, so parity is stated against that same span. In service
    the caller passes present_epochs * BLOCK_LENGTH; the two agree only while
    BLOCK_LENGTH == EPOCH_LENGTH, which config warns about at import.
    """
    mg = FakeMetagraph()
    got = burn.calculate_weights(rewards_for(POINTS), mg, config.EPOCH_LENGTH)

    expected = np.zeros(mg.n, dtype=np.float64)
    per, tpn = {}, 0.0
    for i, p in enumerate(POINTS):
        pct = (p / ALPHA_PER_POINT) / (MINER_ALPHA_PER_BLOCK * config.EPOCH_LENGTH) * 100
        pct = max(pct, p * config.MIN_PERCENT_PER_POINT)  # pre-change floor: no divisor
        per[i], tpn = pct, tpn + pct
    scale = 100.0 / tpn if tpn > 100 else 1.0
    for i, pct in per.items():
        expected[i] = pct * scale / 100.0
    expected[config.BURN_UID] = 1 - (min(tpn, 100) / 100)

    assert np.allclose(got, expected, atol=1e-15)


def test_window_blocks_is_required_and_must_be_positive():
    """A silent default here would under-divide and drive burn to zero."""
    mg = FakeMetagraph()
    with pytest.raises(TypeError):
        burn.calculate_weights([], mg)
    with pytest.raises(ValueError):
        burn.calculate_weights([], mg, 0)


# --- reward store --------------------------------------------------------

def test_range_read_skips_absent_epochs_rather_than_zeroing_them():
    """A gap must reduce the reported span, not silently count as zero points."""
    store = MinerReward(block_length=config.BLOCK_LENGTH, block=lambda: 1000)
    store.epoch_rewards = {5: {"a": 2}, 6: {"a": 3, "b": 1}, 8: {"b": 4}}
    store.update_current_epoch = lambda: None

    totals, present = store.get_rewards_range(4, 8)

    assert totals == {"a": 5, "b": 5}
    assert present == 3  # 5, 6 and 8 are held; 4 and 7 are not


def test_retention_follows_the_widest_configured_window(monkeypatch):
    """Retention is sized from the raw keys, never the block-gated active K.

    At rollout step 1 both window keys are still 1 while shadow mode runs at 7,
    and the epochs a window needs after the activation block were recorded
    before it. Either case would come up short if the active K were used.
    """
    monkeypatch.setattr(config, "WEIGHT_WINDOW_EPOCHS", 1)
    monkeypatch.setattr(config, "WEIGHT_WINDOW_EPOCHS_PREV", 1)
    monkeypatch.setattr(config, "WEIGHT_WINDOW_SHADOW_EPOCHS", 0)
    assert config.weight_window_retention() == 10  # never below the historical value

    monkeypatch.setattr(config, "WEIGHT_WINDOW_SHADOW_EPOCHS", 7)
    assert config.weight_window_retention() == 12

    monkeypatch.setattr(config, "WEIGHT_WINDOW_SHADOW_EPOCHS", 0)
    monkeypatch.setattr(config, "WEIGHT_WINDOW_EPOCHS", 7)
    assert config.weight_window_retention() == 12


# --- broadcast store -----------------------------------------------------

def test_uid_that_changed_holder_inside_the_window_is_dropped(tmp_path):
    """Broadcast points are keyed by UID, so a re-keyed UID would pay the wrong miner."""
    store = RewardBroadcastStore(path=tmp_path / "b.json", keep_epochs=99)
    store.by_epoch_by_sender = {10: {"v1": {0: 5, 1: 7}}, 11: {"v1": {0: 5, 2: 9}}}
    window_start_block = 10 * config.BLOCK_LENGTH
    mg = FakeMetagraph(registered_at=[0, window_start_block + 50] + [0] * 254)

    agg, rekeyed = store.aggregate_range(10, 11, mg)

    assert agg == {0: 10, 2: 9}
    assert rekeyed == 1


def test_range_aggregate_without_a_metagraph_drops_nothing(tmp_path):
    store = RewardBroadcastStore(path=tmp_path / "b.json", keep_epochs=99)
    store.by_epoch_by_sender = {10: {"v1": {0: 5, 1: 7}}, 11: {"v1": {0: 5, 2: 9}}}

    agg, rekeyed = store.aggregate_range(10, 11)

    assert agg == {0: 10, 1: 7, 2: 9}
    assert rekeyed == 0


# --- config --------------------------------------------------------------

@pytest.mark.parametrize("raw", ["0", "500", "-3", "abc", ""])
def test_out_of_range_window_falls_back_to_the_default(raw):
    assert config._clamp_window(raw, 1) == 1


def test_shadow_window_may_be_zero_to_disable():
    assert config._clamp_window("0", 0, minimum=0) == 0
    assert config._clamp_window("7", 0, minimum=0) == 7


@pytest.mark.parametrize("raw", ["0", "500", "-3"])
def test_served_window_value_out_of_range_raises_so_the_previous_is_kept(raw):
    """refresh_remote_config catches ValueError and leaves the old value in place."""
    with pytest.raises(ValueError):
        config._cast_window_epochs(raw)


def test_active_block_selects_between_current_and_previous_k(monkeypatch):
    monkeypatch.setattr(config, "WEIGHT_WINDOW_EPOCHS", 7)
    monkeypatch.setattr(config, "WEIGHT_WINDOW_EPOCHS_PREV", 1)
    monkeypatch.setattr(config, "WEIGHT_WINDOW_ACTIVE_BLOCK", 1_000_000)

    assert config.weight_window_epochs(999_999) == 1
    assert config.weight_window_epochs(1_000_000) == 7
    assert config.weight_window_epochs(1_000_001) == 7


def test_window_keys_are_marked_as_consensus_keys():
    """A local OVERRIDE_ on these is a weight split no config push can correct."""
    assert config._CONSENSUS_KEYS == {
        "WEIGHT_WINDOW_EPOCHS",
        "WEIGHT_WINDOW_EPOCHS_PREV",
        "WEIGHT_WINDOW_ACTIVE_BLOCK",
    }
    for key in config._CONSENSUS_KEYS:
        assert key in config._REMOTE_CONFIG_KEYS
