"""Per-epoch reads used by the reporting paths.

The Score synapse and the API submit both label their payload with a single
epoch's block range, so they must read that epoch alone. The weight path sums a
K-epoch window and keeps the _range readers. These tests pin the difference: at
K > 1 the window sum is roughly K times the single epoch, so a reporting path
that reused it would overstate every miner while claiming one epoch.
"""
from pathlib import Path

import pytest

from alpharidge_ai import config
from alpharidge_ai.utils.penalty import MinerPenalty
from alpharidge_ai.utils.reward import MinerReward
from alpharidge_ai.validator import validation_client as validation_client_module
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


# --- the class of bug, not just this instance ----------------------------

def test_validation_client_references_no_undefined_names():
    """Every name the module reads must be defined somewhere it can see.

    This is the check that would have caught the original fault. A refactor
    moved local_rewards_map and local_penalties into _aggregate_window and left
    three reads behind in run(); Python only notices at runtime, on the one
    branch per epoch that touches them, so it shipped and ran broken for 16
    days. Reading the whole module statically catches the next one at import
    time instead of in production.

    Scope: names that are read but never bound in their own scope, not free
    from an enclosing one, not a module global, and not a builtin.
    """
    import builtins
    import symtable

    source_path = Path(validation_client_module.__file__)
    top = symtable.symtable(source_path.read_text(), str(source_path), "exec")
    module_names = set(top.get_identifiers())
    builtin_names = set(dir(builtins))

    def undefined(table, trail):
        found = []
        for sym in table.get_symbols():
            name = sym.get_name()
            if (
                sym.is_referenced()
                and not sym.is_assigned()
                and not sym.is_parameter()
                and not sym.is_free()
                and not sym.is_imported()
                and sym.is_global()
                and name not in module_names
                and name not in builtin_names
            ):
                found.append(f"{'.'.join(trail)} -> {name}")
        for child in table.get_children():
            found += undefined(child, trail + [child.get_name()])
        return found

    leaks = sorted(set(undefined(top, [source_path.stem])))
    assert not leaks, "undefined names in validation_client:\n  " + "\n  ".join(leaks)


# --- the API reward payload must be one epoch, not the window ------------

def test_api_reward_payload_aggregates_one_epoch_not_the_window():
    """A row stamped with one epoch's block range must carry one epoch's points.

    The weight path's `rewards` list sums the K-epoch window. The API row is
    stamped start_block..stop_block for target_epoch alone, and the dashboard
    derives epoch = start_block / BLOCK_LENGTH from it, so submitting the window
    total records K epochs of points as though they were one. K is served
    remotely and is 7 in production, so that is a 7x overstatement, not a latent
    one. Aggregating a one-epoch window keeps the pooling and the gate the table
    is expected to reflect while restoring the span.

    Read off the real call site: whatever feeds rewards_payload must come from
    an _aggregate_window call whose start and end are both target_epoch.
    """
    import ast

    source_path = Path(validation_client_module.__file__)
    tree = ast.parse(source_path.read_text())

    run = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "run"
    )

    # The comprehension that builds rewards_payload, and the name it iterates.
    payload = next(
        n for n in ast.walk(run)
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == "rewards_payload" for t in n.targets)
    )
    source_name = payload.value.generators[0].iter.id

    # That name must be bound from a single-epoch _aggregate_window call.
    binding = next(
        n for n in ast.walk(run)
        if isinstance(n, ast.Assign)
        and any(
            source_name in [getattr(e, "id", None) for e in ast.walk(t)]
            for t in n.targets
        )
    )
    call = binding.value
    assert isinstance(call, ast.Call) and call.func.attr == "_aggregate_window", (
        f"rewards_payload is built from {source_name!r}, which is not an "
        "_aggregate_window result"
    )
    _, start, end = call.args[:3]
    assert start.id == "target_epoch" and end.id == "target_epoch", (
        "the API reward payload must aggregate target_epoch alone; got window "
        f"[{ast.dump(start)}..{ast.dump(end)}]"
    )
