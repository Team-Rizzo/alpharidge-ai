"""Per-epoch reads used by the reporting paths.

The Score synapse and the API submit both label their payload with a single
epoch's block range, so they must read that epoch alone. The weight path sums a
K-epoch window and keeps the _range readers. These tests pin the difference: at
K > 1 the window sum is roughly K times the single epoch, so a reporting path
that reused it would overstate every miner while claiming one epoch.
"""
import pytest

from alpharidge_ai import config
from alpharidge_ai.utils.penalty import MinerPenalty
from alpharidge_ai.utils.reward import MinerReward
from alpharidge_ai.validator.validation_client import ValidationClient

BLOCK_LENGTH = config.BLOCK_LENGTH
BASE_EPOCH = config.START_BLOCK // BLOCK_LENGTH
K = 3


class FakeValidator:
    """Just the two stores the reporting reads touch."""

    def __init__(self, block):
        self._miner_reward = MinerReward(BLOCK_LENGTH, block)
        self._miner_penalty = MinerPenalty(BLOCK_LENGTH, block)


@pytest.fixture
def client():
    """A client parked on the epoch after the window, as the run loop is.

    target_epoch is current_epoch - 2 in service, so the epochs being reported
    are always closed and never the one the stores are still filling.
    """
    state = {"epoch": BASE_EPOCH}

    def block():
        return state["epoch"] * BLOCK_LENGTH

    validator = FakeValidator(block)
    c = ValidationClient.__new__(ValidationClient)
    c._validator = validator

    def advance_to(epoch):
        state["epoch"] = epoch
        # Both stores roll their current epoch off the block function.
        validator._miner_reward.update_current_epoch()
        validator._miner_penalty.update_current_epoch()

    c._test_advance_to = advance_to
    return c


def fill_window(client, hotkey="hk0", points=10, penalties=2):
    """Write the same points/penalties into each of K consecutive epochs."""
    validator = client._validator
    for i in range(K):
        client._test_advance_to(BASE_EPOCH + i)
        validator._miner_reward.add_reward(hotkey, points)
        for _ in range(penalties):
            validator._miner_penalty.add_penalty(hotkey, 1)
    client._test_advance_to(BASE_EPOCH + K + 1)


# --- the point of the fix ------------------------------------------------

def test_epoch_read_is_one_epoch_not_the_window(client):
    """The reporting read must not carry the other K-1 epochs with it."""
    fill_window(client, points=10, penalties=2)
    last = BASE_EPOCH + K - 1

    assert client._epoch_rewards(last) == {"hk0": 10}
    assert client._epoch_penalties(last) == {"hk0": 2}


def test_window_read_is_k_times_the_epoch_read(client):
    """States the factor the wrong fix would have introduced, so it stays visible."""
    fill_window(client, points=10, penalties=2)
    start, end = BASE_EPOCH, BASE_EPOCH + K - 1

    window_rewards, present = client._validator._miner_reward.get_rewards_range(start, end)
    window_penalties, _ = client._validator._miner_penalty.get_penalties_range(start, end)

    assert present == K
    assert window_rewards["hk0"] == K * client._epoch_rewards(end)["hk0"]
    assert window_penalties["hk0"] == K * client._epoch_penalties(end)["hk0"]


def test_each_epoch_reads_back_its_own_amount(client):
    """Different amounts per epoch, so a sum cannot pass by coincidence."""
    validator = client._validator
    for i, points in enumerate([4, 9, 16]):
        client._test_advance_to(BASE_EPOCH + i)
        validator._miner_reward.add_reward("hk0", points)
    client._test_advance_to(BASE_EPOCH + K + 1)

    for i, points in enumerate([4, 9, 16]):
        assert client._epoch_rewards(BASE_EPOCH + i) == {"hk0": points}


# --- absent epochs are a normal state, not a failure ---------------------

def test_absent_epoch_is_empty_not_an_error(client):
    """get_rewards raises KeyError for an unknown epoch; reporting must not abort.

    An epoch that fell out of retention, or one nobody earned in, is ordinary.
    Raising here would take the whole Score/submit block down with it, which is
    the failure mode this fix exists to remove.
    """
    fill_window(client)

    assert client._epoch_rewards(BASE_EPOCH - 500) == {}
    assert client._epoch_penalties(BASE_EPOCH - 500) == {}


def test_negative_epoch_is_a_relative_offset_which_is_why_callers_guard(client):
    """-1 means "the newest stored epoch", not "epoch number -1".

    A negative target_epoch would therefore read a real but wrong epoch and
    report it under the wrong block range, silently. Both call sites guard
    target_epoch >= 0 for exactly this; the guard is the fix, not the helper.
    """
    validator = client._validator
    for i, points in enumerate([4, 9, 16]):
        client._test_advance_to(BASE_EPOCH + i)
        validator._miner_reward.add_reward("hk0", points)

    # -1 lands on the newest stored epoch (the one just written), -2 on the one
    # before it. Neither is "epoch -1".
    assert client._epoch_rewards(-1) == {"hk0": 16}
    assert client._epoch_rewards(-2) == {"hk0": 9}


def test_empty_epoch_reads_as_empty(client):
    """An epoch the store holds but nothing was written to."""
    fill_window(client)
    quiet = BASE_EPOCH + K  # created by the roll, never written to

    assert client._epoch_rewards(quiet) == {}
    assert client._epoch_penalties(quiet) == {}
