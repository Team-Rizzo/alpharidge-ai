"""A reputation record belongs to the registration that earned it."""

import types

import pytest

from alpharidge_ai.mechanism import channels as ch
from alpharidge_ai.validator.reputation_store import ReputationStore
from neurons.validator import Validator

HK_OLD, HK_NEW, HK_OTHER = "hk-old", "hk-new", "hk-other"


def store_with(tmp_path, hotkey, uid, reg_block, score=0.2):
    store = ReputationStore(path=tmp_path / "rep.json")
    store.set_channel_weights({ch.AUDIT_V2: 3.0})
    for epoch in range(10, 40):
        store.record_local(epoch, "me", hotkey, epoch, score, 1.0, channel=ch.AUDIT_V2)
        store.finalize(epoch)
    store.reconcile_identities([(uid, hotkey, reg_block)])
    return store


def test_a_record_is_bound_to_its_registration(tmp_path):
    store = store_with(tmp_path, HK_OLD, uid=7, reg_block=1000)
    assert store.state[HK_OLD]["u"] == 7 and store.state[HK_OLD]["b"] == 1000


def test_the_same_registration_under_a_new_hotkey_keeps_it(tmp_path):
    store = store_with(tmp_path, HK_OLD, uid=7, reg_block=1000)
    before = store.reputation(HK_OLD)
    cleared, moved = store.reconcile_identities([(7, HK_NEW, 1000)])
    assert (cleared, moved) == (0, 1)
    assert HK_OLD not in store.state
    assert store.reputation(HK_NEW) == pytest.approx(before)


def test_a_new_registration_starts_empty_even_on_a_known_hotkey(tmp_path):
    store = store_with(tmp_path, HK_OLD, uid=7, reg_block=1000)
    cleared, moved = store.reconcile_identities([(7, HK_OLD, 2000)])
    assert (cleared, moved) == (1, 0)
    assert HK_OLD not in store.state


def test_a_hotkey_returning_on_another_seat_starts_empty(tmp_path):
    store = store_with(tmp_path, HK_OLD, uid=7, reg_block=1000)
    cleared, _ = store.reconcile_identities([(11, HK_OLD, 2000)])
    assert cleared == 1 and HK_OLD not in store.state


def test_records_predating_the_rule_are_adopted_not_cleared(tmp_path):
    store = store_with(tmp_path, HK_OLD, uid=7, reg_block=1000)
    store.state[HK_OLD].pop("u"), store.state[HK_OLD].pop("b")
    before = store.reputation(HK_OLD)
    cleared, moved = store.reconcile_identities([(7, HK_OLD, 1000)])
    assert (cleared, moved) == (0, 0)
    assert store.reputation(HK_OLD) == pytest.approx(before)


def test_an_unchanged_field_is_left_alone(tmp_path):
    store = store_with(tmp_path, HK_OLD, uid=7, reg_block=1000)
    assert store.reconcile_identities([(7, HK_OLD, 1000)]) == (0, 0)


def test_one_hotkeys_record_is_never_given_to_another_seat(tmp_path):
    store = store_with(tmp_path, HK_OLD, uid=7, reg_block=1000)
    cleared, moved = store.reconcile_identities([(8, HK_OTHER, 1000)])
    assert moved == 0 and HK_OLD in store.state


def test_it_survives_a_reload(tmp_path):
    store = store_with(tmp_path, HK_OLD, uid=7, reg_block=1000)
    store.save()
    again = ReputationStore(path=tmp_path / "rep.json")
    again.load()
    assert again.state[HK_OLD]["b"] == 1000
    assert again.reconcile_identities([(7, HK_OLD, 1000)]) == (0, 0)


def test_both_validators_reach_the_same_state(tmp_path):
    rows = [(7, HK_NEW, 1000), (8, HK_OTHER, 2000)]
    (tmp_path / "a").mkdir(), (tmp_path / "b").mkdir()
    one = store_with(tmp_path / "a", HK_OLD, uid=7, reg_block=1000)
    two = store_with(tmp_path / "b", HK_OLD, uid=7, reg_block=1000)
    one.reconcile_identities(rows)
    two.reconcile_identities(list(reversed(rows)))
    assert set(one.state) == set(two.state)
    assert one.reputation(HK_NEW) == pytest.approx(two.reputation(HK_NEW))


def test_rows_the_chain_did_not_give_are_ignored(tmp_path):
    store = store_with(tmp_path, HK_OLD, uid=7, reg_block=1000)
    assert store.reconcile_identities([]) == (0, 0)
    assert HK_OLD in store.state


# ---- the validator side ------------------------------------------------------------

def test_the_rows_come_from_the_metagraph():
    v = types.SimpleNamespace(metagraph=types.SimpleNamespace(
        hotkeys=["a", "b"], block_at_registration=[10, 20]))
    assert Validator._identity_rows(v) == [(0, "a", 10), (1, "b", 20)]


def test_a_metagraph_without_registrations_yields_nothing(monkeypatch):
    import neurons.validator as nv
    monkeypatch.setattr(nv.bt.logging, "debug", lambda m: None)
    v = types.SimpleNamespace(metagraph=types.SimpleNamespace(
        hotkeys=["a", "b"], block_at_registration=[]))
    assert Validator._identity_rows(v) == []


def test_an_unreadable_metagraph_does_not_raise(monkeypatch):
    import neurons.validator as nv
    monkeypatch.setattr(nv.bt.logging, "debug", lambda m: None)
    v = types.SimpleNamespace(metagraph=None)
    assert Validator._identity_rows(v) == []


def test_a_record_nobody_holds_is_cleared_when_someone_claims_it(tmp_path):
    """A hotkey already departed when the rule arrives must not keep its record."""
    store = ReputationStore(path=tmp_path / "rep.json")
    store.set_channel_weights({ch.AUDIT_V2: 3.0})
    for epoch in range(10, 40):
        store.record_local(epoch, "me", HK_OLD, epoch, 0.9, 1.0, channel=ch.AUDIT_V2)
        store.finalize(epoch)
    store.reconcile_identities([(7, HK_OTHER, 1000)])       # HK_OLD is not in the field
    cleared, moved = store.reconcile_identities([(7, HK_OLD, 5000)])   # it comes back
    assert (cleared, moved) == (1, 0)
    assert HK_OLD not in store.state


def test_a_read_that_goes_backwards_refuses(tmp_path, monkeypatch):
    """A chain fact we no longer understand must raise an alarm, not empty the store."""
    import alpharidge_ai.validator.reputation_store as rs
    errors = []
    monkeypatch.setattr(rs.bt.logging, "error", lambda m: errors.append(m))
    store = ReputationStore(path=tmp_path / "rep.json")
    for i in range(60):
        hk = f"hk{i:03d}"
        store.state[hk] = {"r": 0.6, "n": 30, "c": {}, "u": i, "b": 5000 + i}
    rows = [(i, f"hk{i:03d}", (100 + i if i % 2 else 9_000_000 + i)) for i in range(60)]
    assert store.reconcile_identities(rows) == (0, 0)
    assert len(store.state) == 60
    assert any("no record was cleared" in line for line in errors)


def test_a_long_outage_drains_however_large(tmp_path, monkeypatch):
    """Everyone re-registered while we were down: it drains at the pass limit, never locks."""
    import alpharidge_ai.validator.reputation_store as rs
    monkeypatch.setattr(rs.bt.logging, "info", lambda m: None)
    monkeypatch.setattr(rs.bt.logging, "warning", lambda m: None)
    store = ReputationStore(path=tmp_path / "rep.json")
    for i in range(256):
        store.state[f"hk{i:03d}"] = {"r": 0.6, "n": 30, "c": {}, "u": i, "b": 1000 + i}
    rows = [(i, f"hk{i:03d}", (9_000_000 + i if i < 70 else 1000 + i)) for i in range(256)]
    cleared = [store.reconcile_identities(rows)[0] for _ in range(5)]
    assert cleared == [20, 20, 20, 10, 0]


def test_a_normal_days_churn_is_allowed(tmp_path, monkeypatch):
    import alpharidge_ai.validator.reputation_store as rs
    monkeypatch.setattr(rs.bt.logging, "info", lambda m: None)
    store = ReputationStore(path=tmp_path / "rep.json")
    for i in range(300):
        hk = f"hk{i:03d}"
        store.state[hk] = {"r": 0.6, "n": 30, "c": {}, "u": i, "b": 1000 + i}
    rows = [(i, f"hk{i:03d}", (9_000_000 if i < 14 else 1000 + i)) for i in range(300)]
    cleared, _ = store.reconcile_identities(rows)
    assert cleared == 14


def test_the_first_pass_reports_what_it_bound(tmp_path, monkeypatch):
    import alpharidge_ai.validator.reputation_store as rs
    lines = []
    monkeypatch.setattr(rs.bt.logging, "info", lambda m: lines.append(m))
    store = ReputationStore(path=tmp_path / "rep.json")
    for i in range(5):
        store.state[f"hk{i:03d}"] = {"r": 0.6, "n": 30, "c": {}}
    store.reconcile_identities([(i, f"hk{i:03d}", 1000 + i) for i in range(5)])
    assert any("5 bound for the first time" in line for line in lines)
    assert (tmp_path / "rep.json").exists()


def test_carry_over_does_not_depend_on_insertion_order(tmp_path):
    def build(order):
        store = ReputationStore(path=tmp_path / f"rep-{order[0]}.json")
        for hk in order:
            store.state[hk] = {"r": 0.6, "n": 30, "c": {}, "u": 7, "b": 1000}
        store.reconcile_identities([(7, HK_NEW, 1000)])
        return store.state[HK_NEW]["r"]
    assert build(["hk-a", "hk-b"]) == build(["hk-b", "hk-a"])


def test_a_backlog_clears_oldest_first_and_drains(tmp_path, monkeypatch):
    """A validator that was down finds more than one pass may clear; it drains, not latches."""
    import alpharidge_ai.validator.reputation_store as rs
    monkeypatch.setattr(rs.bt.logging, "info", lambda m: None)
    monkeypatch.setattr(rs.bt.logging, "warning", lambda m: None)
    store = ReputationStore(path=tmp_path / "rep.json")
    for i in range(240):
        store.state[f"hk{i:03d}"] = {"r": 0.6, "n": 30, "c": {}, "u": i, "b": 1000 + i}
    rows = [(i, f"hk{i:03d}", (9_000_000 if i < 25 else 1000 + i)) for i in range(240)]
    assert store.reconcile_identities(rows) == (20, 0)
    assert all(f"hk{i:03d}" not in store.state for i in range(20))
    assert all(f"hk{i:03d}" in store.state for i in range(20, 25))
    assert store.reconcile_identities(rows) == (5, 0)
    assert store.reconcile_identities(rows) == (0, 0)


def test_a_refused_pass_still_carries_a_swap(tmp_path, monkeypatch):
    import alpharidge_ai.validator.reputation_store as rs
    monkeypatch.setattr(rs.bt.logging, "error", lambda m: None)
    monkeypatch.setattr(rs.bt.logging, "info", lambda m: None)
    store = ReputationStore(path=tmp_path / "rep.json")
    for i in range(60):
        store.state[f"hk{i:03d}"] = {"r": 0.6, "n": 30, "c": {}, "u": i, "b": 1000 + i}
    store.state[HK_OLD] = {"r": 0.9, "n": 30, "c": {}, "u": 60, "b": 5000}
    rows = [(i, f"hk{i:03d}", 100 + i) for i in range(60)] + [(60, HK_NEW, 5000)]
    assert store.reconcile_identities(rows) == (0, 1)
    assert len(store.state) == 61 and store.state[HK_NEW]["r"] == 0.9


def test_a_swap_keeps_its_record_when_the_new_hotkey_was_scored_first(tmp_path):
    store = store_with(tmp_path, HK_OLD, uid=7, reg_block=1000)
    before = store.reputation(HK_OLD)
    store.state[HK_NEW] = {"r": 0.5, "n": 1, "c": {}}          # opened before this pass
    cleared, moved = store.reconcile_identities([(7, HK_NEW, 1000)])
    assert (cleared, moved) == (0, 1)
    assert HK_OLD not in store.state
    assert store.reputation(HK_NEW) == pytest.approx(before)


def test_rows_leave_out_registrations_after_the_scored_epoch(monkeypatch):
    import neurons.validator as nv
    monkeypatch.setattr(nv.config, "BLOCK_LENGTH", 100, raising=False)
    v = types.SimpleNamespace(metagraph=types.SimpleNamespace(
        hotkeys=["a", "b", "c"], block_at_registration=[950, 1100, 1101]))
    assert Validator._identity_rows(v, 10) == [(0, "a", 950), (1, "b", 1100)]
    assert Validator._identity_rows(v) == [(0, "a", 950), (1, "b", 1100), (2, "c", 1101)]
