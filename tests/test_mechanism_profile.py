"""Mechanism profile: parsing, range validation, and block-height resolution."""

import copy
import json
from pathlib import Path

import pytest

from alpharidge_ai.mechanism import profile as mp


def _valid() -> dict:
    return {
        "version": 2,
        "publish_block": 9_000,
        "activation_block": 10_000,
        "schema_version": "2.0.0",
        "settlement": {"C": 301_743.0},
        "ledger": {"window_epochs": 216},
        "signature": "",
    }


def test_parses_a_valid_profile():
    p = mp.parse(_valid())
    assert p.version == 2
    assert p.settlement.C == pytest.approx(301_743.0)
    assert p.ledger.window_epochs == 216
    extra = _valid()
    extra["ledger"]["note"] = "ignored"
    assert mp.parse(extra).ledger == p.ledger
    assert p.ledger.reverify_rate is None
    raw = _valid()
    raw["ledger"]["reverify_rate"] = 0.5
    raw["ledger"]["settle_delay_h"] = 48
    assert mp.parse(raw).ledger.reverify_rate == 0.5 and mp.parse(raw).ledger.settle_delay_h == 48
    assert mp.parse(raw).ledger.judges is None
    hk = "5" + "F" * 47
    raw["ledger"]["judges"] = [hk, hk]
    assert mp.parse(raw).ledger.judges == (hk,)


def test_the_example_profile_parses():
    path = Path(__file__).resolve().parent.parent / "profiles" / "profile.example.json"
    assert mp.parse(json.loads(path.read_text())).schema_version == mp.SCHEMA_VERSION


@pytest.mark.parametrize("section,key,bad", [
    ("settlement", "C", 0.0),
    ("ledger", "window_epochs", 0),
    ("ledger", "reverify_rate", 0.0),
    ("ledger", "judges", ["not a hotkey"]),
])
def test_out_of_range_values_are_rejected(section, key, bad):
    raw = _valid()
    raw[section][key] = bad
    with pytest.raises(mp.ProfileError) as e:
        mp.parse(raw)
    assert key in str(e.value)


def test_unknown_schema_version_is_rejected():
    raw = _valid()
    raw["schema_version"] = "9.9.9"
    with pytest.raises(mp.ProfileError) as e:
        mp.parse(raw)
    assert "not supported" in str(e.value)


def test_missing_section_is_rejected():
    raw = _valid()
    del raw["ledger"]
    with pytest.raises(mp.ProfileError) as e:
        mp.parse(raw)
    assert "ledger" in str(e.value)


# ---- resolution -------------------------------------------------------------------

def _resolver_with_current():
    cur = mp.parse(_valid())
    return mp.ProfileResolver(current=cur, refresh_seconds=3600)


def test_min_lead_covers_a_full_refresh_interval():
    assert mp.min_lead_blocks(3600) == 300


def test_profile_activates_only_at_its_block():
    r = _resolver_with_current()
    nxt = _valid()
    nxt["version"] = 3
    nxt["activation_block"] = 20_000
    nxt["settlement"]["C"] = 400_000.0

    assert r.offer(nxt)[0]
    assert r.resolve(19_999).settlement.C == pytest.approx(301_743.0)
    assert r.resolve(20_000).settlement.C == pytest.approx(400_000.0)


def test_a_publish_without_enough_lead_is_rejected():
    r = _resolver_with_current()
    nxt = _valid()
    nxt["version"] = 3
    nxt["publish_block"] = 19_800
    nxt["activation_block"] = 20_000
    ok, reason = r.offer(nxt)  # 200 blocks of lead, needs 300
    assert not ok and "insufficient_lead" in reason
    assert r.next is None


def test_a_lower_version_is_rejected():
    r = _resolver_with_current()
    older = _valid()
    older["version"] = 1
    older["activation_block"] = 20_000
    ok, reason = r.offer(older)
    assert not ok and "stale_version" in reason


def test_the_same_version_is_rejected():
    r = _resolver_with_current()
    same = _valid()
    same["activation_block"] = 20_000
    ok, reason = r.offer(same)
    assert not ok and "stale_version" in reason


def test_rollback_is_a_forward_publish():
    r = _resolver_with_current()
    bad = _valid()
    bad["version"] = 3
    bad["activation_block"] = 20_000
    bad["settlement"]["C"] = 999_999.0
    r.offer(bad)
    assert r.resolve(20_000).settlement.C == pytest.approx(999_999.0)

    restore = _valid()
    restore["version"] = 4
    restore["activation_block"] = 30_000
    assert r.offer(restore)[0]
    assert r.resolve(30_000).settlement.C == pytest.approx(301_743.0)


def test_an_invalid_profile_leaves_the_current_one_in_force():
    r = _resolver_with_current()
    broken = _valid()
    broken["version"] = 3
    broken["activation_block"] = 20_000
    broken["settlement"]["C"] = -1.0
    ok, reason = r.offer(broken)
    assert not ok and "invalid" in reason
    assert r.resolve(20_000).settlement.C == pytest.approx(301_743.0)


def test_the_lead_is_measured_from_the_signed_publish_block():
    """An already-active profile must still be adoptable, however late it is fetched."""
    r = mp.ProfileResolver(refresh_seconds=3600)
    old = _valid()
    old["version"] = 9
    old["publish_block"] = 1_000
    old["activation_block"] = 1_400
    assert r.offer(old)[0]
    assert r.resolve(9_000_000).version == 9


def test_resolution_is_deterministic_across_repeated_calls():
    r = _resolver_with_current()
    nxt = _valid()
    nxt["version"] = 3
    nxt["activation_block"] = 20_000
    r.offer(nxt)
    first = r.resolve(20_000)
    for _ in range(5):
        assert r.resolve(20_000) is first
    assert r.resolve(19_000) is first  # promotion does not un-apply


# ---- signature --------------------------------------------------------------------

def test_signature_is_checked_when_a_verifier_is_supplied():
    r = _resolver_with_current()
    nxt = _valid()
    nxt["version"] = 3
    nxt["activation_block"] = 20_000
    nxt["signature"] = "deadbeef"

    ok, reason = r.offer(nxt, verify=lambda m, s: False)
    assert not ok and reason == "bad_signature"

    ok, _ = r.offer(nxt, verify=lambda m, s: True)
    assert ok


def test_unsigned_profile_is_rejected_when_verification_is_on():
    r = _resolver_with_current()
    nxt = _valid()
    nxt["version"] = 3
    nxt["activation_block"] = 20_000
    ok, reason = r.offer(nxt, verify=lambda m, s: True)
    assert not ok and reason == "unsigned"


def test_signed_payload_excludes_the_signature_and_is_order_independent():
    raw = _valid()
    raw["signature"] = "abc123"
    msg = mp.signing_payload(raw)
    assert "abc123" not in msg

    shuffled = {k: copy.deepcopy(v) for k, v in reversed(list(raw.items()))}
    assert mp.signing_payload(shuffled) == msg


def test_canonical_json_matches_the_attestation_encoder():
    """Both sides sign the same bytes; a divergence here is a fleet-wide reject."""
    from alpharidge_ai.ledger import attestation as ac
    sample = {"b": 1, "a": {"z": [1, 2], "y": "ünicode"}, "c": 2.5}
    assert mp.canonical_json(sample) == ac.canonical_json(sample)


# ---- adopting the active profile ----------------------------------------------------

def _raw(version, publish, activation):
    raw = _valid()
    raw["version"] = version
    raw["publish_block"] = publish
    raw["activation_block"] = activation
    return raw


def test_a_fresh_validator_adopts_the_active_profile():
    r = mp.ProfileResolver(refresh_seconds=3600)
    assert r.adopt(_raw(2, 1_000, 1_400), block=5_000)[0]
    assert r.resolve(5_000).version == 2


def test_the_active_profile_survives_a_staged_successor():
    """Offering a successor does not displace the active profile."""
    r = mp.ProfileResolver(refresh_seconds=3600)
    r.adopt(_raw(2, 1_000, 1_400), block=5_000)
    r.offer(_raw(3, 5_000, 9_000))

    assert r.resolve(5_000).version == 2      # still governed
    assert r.resolve(9_000).version == 3      # and switches on time


def test_a_future_profile_is_not_adopted_as_active():
    r = mp.ProfileResolver(refresh_seconds=3600)
    ok, reason = r.adopt(_raw(3, 5_000, 9_000), block=5_000)
    assert not ok and "not_yet_active" in reason


def test_adopting_an_older_version_is_refused():
    r = mp.ProfileResolver(refresh_seconds=3600)
    r.adopt(_raw(5, 1_000, 1_400), block=5_000)
    assert not r.adopt(_raw(4, 1_000, 1_400), block=5_000)[0]
