import asyncio

import pytest
from bittensor_wallet import Keypair

from alpharidge_ai.ledger import attestation as ac
from alpharidge_ai.ledger.pull import LedgerPull
from alpharidge_ai.mechanism.pools import window_points

KP = Keypair.create_from_seed("0x" + "44" * 32)
LEAVES = {7: [{"resource_type": "ledger", "resource_id": "1", "miner_hotkey": "hk1",
               "validator_verdict": "provisional", "categorical_key": "finance|event",
               "points_awarded": 3.6, "claim_id": "9"}],
          6: [{"resource_type": "ledger", "resource_id": "2", "miner_hotkey": "hk2",
               "validator_verdict": "provisional", "categorical_key": "finance|fact",
               "points_awarded": 0.4, "claim_id": "8"}]}


class Api:
    def __init__(self, tamper=None):
        leaves = [x for e in sorted(LEAVES) for x in LEAVES[e]]
        per = window_points(leaves)
        if tamper:
            per = {**per, **tamper}
        total, root = sum(per.values()), ac.merkle_root(leaves)
        sig = KP.sign(ac.attestation_message("event-ledger", 7, per, total, root).encode()).hex()
        self.body = {"attestation": {"validatorHotkey": "event-ledger", "epoch": 7,
                                     "perMinerPoints": per, "totalPoints": total,
                                     "merkleRoot": root},
                     "signature": sig, "capacity": 100.0}
        self.leaf_calls = []

    async def ledger_points(self, epoch):
        return self.body

    async def ledger_leaves(self, epoch, cursor=None, limit=5000):
        self.leaf_calls.append(epoch)
        return {"leaves": LEAVES.get(epoch, []), "next_cursor": None}


def pull(api, epoch=7):
    p = LedgerPull(api, pinned_pubkey=KP.ss58_address, window=3)
    return p, asyncio.run(p.pull(epoch, {"hk1": 1, "hk2": 2}))


def test_signed_points_with_matching_root_are_accepted():
    api = Api()
    p, (got, reason) = pull(api)
    assert reason == "accepted"
    assert got.points == {"hk1": 3.6, "hk2": 0.4}
    assert sorted(api.leaf_calls) == [4, 5, 6, 7]
    asyncio.run(p.pull(7, {"hk1": 1}))
    assert len(api.leaf_calls) == 4


def test_tampered_points_or_key_fail():
    assert pull(Api(tamper={"hk1": 9.0}))[1][1] == "points_mismatch"
    other = LedgerPull(Api(), pinned_pubkey=Keypair.create_from_seed("0x" + "55" * 32).ss58_address,
                       window=3)
    assert asyncio.run(other.pull(7, {"hk1": 1}))[1] == "bad_signature"


def test_leaf_v2_hashes_receipt_and_v1_is_unchanged():
    v1 = dict(LEAVES[7][0])
    v2 = {**v1, "leaf_version": 2, "received_ms": 1000, "bucket": 4}
    assert ac.merkle_root([v1]) == ac.merkle_root([{**v1, "received_ms": 5}])
    assert ac.merkle_root([v2]) != ac.merkle_root([v1])
    assert ac.merkle_root([v2]) != ac.merkle_root([{**v2, "received_ms": 1001}])
    assert ac.merkle_root([v2]) != ac.merkle_root([{**v2, "claim_id": "10"}])
    with pytest.raises(ValueError):
        ac.merkle_root([{**v1, "leaf_version": 3, "coldkey": "ck"}])


def test_only_v2_leaves_give_a_coldkey():
    from alpharidge_ai.mechanism.pools import coldkeys
    v1 = {**LEAVES[7][0], "coldkey": "ck1", "received_ms": 5}
    v2 = {**LEAVES[7][0], "leaf_version": 2, "received_ms": 1000, "bucket": 4, "coldkey": "ck2"}
    assert coldkeys([v1]) == {} and coldkeys([{**v1, "leaf_version": 3}]) == {}
    assert coldkeys([v1, v2]) == {v2["miner_hotkey"]: "ck2"}


def test_a_mismatch_drops_the_cached_window_and_retries():
    api = Api()
    p = LedgerPull(api, pinned_pubkey=KP.ss58_address, window=3)
    p._leaves[6] = []
    got, reason = asyncio.run(p.pull(7, {"hk1": 1, "hk2": 2}))
    assert reason == "accepted" and got.points == {"hk1": 3.6, "hk2": 0.4}
    assert api.leaf_calls.count(6) == 1


def test_the_signed_window_is_used():
    p = LedgerPull(Api(), pinned_pubkey=KP.ss58_address, window=1)
    got, reason = asyncio.run(p.pull(7, {"hk1": 1, "hk2": 2}, window=3))
    assert reason == "accepted" and got.leaf_count == 2


def test_settle_leaves_must_name_the_paid_leaf():
    paid = LEAVES[6][0]
    settle = {**paid, "resource_id": "3", "validator_verdict": "settle", "points_awarded": 0.6}

    class Leaves:
        def __init__(self, signed):
            root = ac.merkle_root(signed)
            msg = ac.attestation_message("event-ledger", 7, {}, 0.0, root)
            self.body = {"attestation": {"validatorHotkey": "event-ledger", "epoch": 7,
                                         "perMinerPoints": {}, "totalPoints": 0.0,
                                         "merkleRoot": root},
                         "signature": KP.sign(msg.encode()).hex()}

        async def ledger_points(self, epoch):
            return self.body

        async def ledger_leaves(self, epoch, cursor=None, limit=5000):
            return {"leaves": {7: [settle], 6: [paid]}.get(epoch, []), "next_cursor": None}

    def check(settles, signed=(paid, settle)):
        p = LedgerPull(Leaves(list(signed)), pinned_pubkey=KP.ss58_address, window=3)
        return asyncio.run(p.unbound_settles(7, settles))

    good = {"ledger_id": "3", "paid_epoch": 6, "paid": paid}
    assert check([good]) is None
    assert check([good], signed=(settle,)) == "unbound_settle"
    for bad in ([], [good, good], [{**good, "paid_epoch": 5}],
                [{**good, "paid": {**paid, "miner_hotkey": "hk1"}}],
                [{**good, "paid": {**paid, "claim_id": "9"}}]):
        assert check(bad) == "unbound_settle"


def _signed_debt():
    from datetime import datetime, timezone

    from alpharidge_ai.ledger import request_auth as ra
    from alpharidge_ai.schema import Claim, canonical_body
    from alpharidge_ai.schema.types import scoop_debt
    from alpharidge_ai.verify.check import verdict_hash

    miner = Keypair.create_from_seed("0x" + "66" * 32)
    claim = Claim(
        kind="event", headline="Acme to acquire Beta Systems", event_type="merger_acquisition",
        category="finance", event_date="2026-10-13",
        actors=[{"id_scheme": "other", "id": "a", "role": "acquirer", "name": "Acme Inc."},
                {"id_scheme": "other", "id": "b", "role": "target", "name": "Beta Systems"}],
        facts=[], source_url="https://news.example.org/acme-beta",
        quote="Acme Inc. agreed on Tuesday to acquire Beta Systems, the companies said.")
    raw = canonical_body(claim)
    h = ra.signed_headers(miner, "POST", "http://api/claims", raw)
    received = datetime(2026, 10, 13, 12, 40, tzinfo=timezone.utc)
    record = {"outcome": "pass", "answers": {"event": "yes"}}
    pts = scoop_debt(claim.event_type, claim.event_date, received)
    leaf = {"resource_type": "ledger", "resource_id": "5", "miner_hotkey": miner.ss58_address,
            "validator_verdict": "debt", "categorical_key": "finance|scoop",
            "points_awarded": pts, "claim_id": "11", "leaf_version": 2,
            "received_ms": int(received.timestamp() * 1000), "bucket": 0,
            "verdict_hash": verdict_hash(record)}
    debt = {"claim_id": "11", "hotkey": miner.ss58_address, "body": raw.decode(),
            "signature": h["X-Auth-Signature"], "nonce": h["X-Auth-Nonce"],
            "signed_at": int(h["X-Auth-Timestamp"]), "received_at": "2026-10-13T12:40:00Z",
            "published_at": None, "reason": "scoop_unconfirmed", "judge": record}

    return miner, raw, record, pts, leaf, debt


def test_a_debt_leaf_is_the_miss_debt_of_a_signed_scoop():
    from alpharidge_ai.validator.reverify import miner_signed
    from alpharidge_ai.verify.check import verdict_hash

    miner, raw, record, pts, leaf, debt = _signed_debt()

    class Leaves:
        def __init__(self, more=()):
            self.more = list(more)

        async def ledger_leaves(self, epoch, cursor=None, limit=5000):
            return {"leaves": [leaf, *self.more] if epoch == 7 else [], "next_cursor": None}

    def check(settles, more=()):
        p = LedgerPull(Leaves(more), pinned_pubkey=KP.ss58_address, window=3)
        return asyncio.run(p.unbound_debts(7, settles, miner_signed))

    assert pts == -24.0
    good = {"ledger_id": "5", "debt": debt}
    assert check([good]) is None
    assert check([{**good, "debt": {**debt, "reason": "unknown"}}]) == "unbound_debt"
    assert check([{**good, "debt": {**debt, "judge": None}}]) == "unbound_debt"
    # Without an intake record the page was not read: age from the event date alone.
    leaf.pop("verdict_hash")
    stale = {**debt, "judge": None, "published_at": "2026-09-20T12:00:00Z"}
    assert check([{**good, "debt": stale}]) is None
    leaf["verdict_hash"] = verdict_hash(record)
    assert check([{**good, "debt": {**debt, "received_at": "2026-10-16T12:40:00Z"}}]) \
        == "unbound_debt"
    assert check([{**good, "debt": {**debt, "body": raw.decode().replace("Beta", "Gamma")}}]) \
        == "unbound_debt"
    assert check([]) == "unbound_debt"
    paid = {**leaf, "resource_id": "6", "validator_verdict": "provisional", "points_awarded": 1.0}
    assert check([good], more=[paid]) == "unbound_debt"
    leaf["points_awarded"] = 2 * pts
    assert check([good]) == "unbound_debt"


def test_a_penalty_may_name_an_unpaid_scoops_debt_leaf():
    from alpharidge_ai.validator.reverify import miner_signed

    *_, debt, ctx = _signed_debt()
    debt = {**debt, "resource_id": "4"}
    pen = {**debt, "resource_id": "5", "validator_verdict": "penalty", "points_awarded": -120.0}
    pen.pop("verdict_hash")

    class Leaves:
        def __init__(self, signed):
            root = ac.merkle_root(signed)
            msg = ac.attestation_message("event-ledger", 7, {}, 0.0, root)
            self.body = {"attestation": {"validatorHotkey": "event-ledger", "epoch": 7,
                                         "perMinerPoints": {}, "totalPoints": 0.0,
                                         "merkleRoot": root},
                         "signature": KP.sign(msg.encode()).hex()}
            self.by = {7: [x for x in signed if x["resource_id"] == "5"],
                       6: [x for x in signed if x["resource_id"] == "4"]}

        async def ledger_points(self, epoch):
            return self.body

        async def ledger_leaves(self, epoch, cursor=None, limit=5000):
            return {"leaves": self.by.get(epoch, []), "next_cursor": None}

    def check(penalty, paid=debt, served=ctx):
        p = LedgerPull(Leaves([paid, penalty]), pinned_pubkey=KP.ss58_address, window=3)
        entry = {"ledger_id": "5", "paid_epoch": 6, "paid": paid}
        return asyncio.run(p.unbound_settles(
            7, [{**entry, "debt": served} if served else entry], signed=miner_signed))

    assert check(pen) is None
    assert check({**pen, "points_awarded": -60.0}) == "unbound_settle"
    assert check({**pen, "validator_verdict": "clawback"}) == "unbound_settle"
    assert check(pen, paid={**debt, "validator_verdict": "carry"}) == "unbound_settle"
    # The debt is checked again against the signed claim served with the penalty.
    assert check(pen, served=None) == "unbound_settle"
    assert check(pen, served={**ctx, "reason": "judge_reject"}) == "unbound_settle"
    assert check(pen, served={**ctx, "body": ctx["body"].replace("Beta", "Gamma")}) \
        == "unbound_settle"

