"""Validator loop: signed ledger points to weights, and the keyed re-verify."""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from bittensor_wallet import Keypair

from alpharidge_ai.judge.client import ScriptedJudge
from alpharidge_ai.judge.combine import combined
from alpharidge_ai.ledger import attestation as ac
from alpharidge_ai.ledger import request_auth as ra
from alpharidge_ai.ledger.api_client import AlpharidgeAPIClient
from alpharidge_ai.ledger.pull import LedgerPull
from alpharidge_ai.mechanism.pools import window_points
from alpharidge_ai.schema import Claim, canonical_body, confirm_claim, judged
from alpharidge_ai.validator.judge import verdict_body
from alpharidge_ai.validator.loop import LedgerLoop
from alpharidge_ai.validator.reverify import (
    AGREE, DISAGREE, UNREACHABLE, Reverifier, keyed, miner_signed, reverify,
)
from alpharidge_ai.validator.scan import (
    Scanner, mentions, page_items, parse_scoop, parse_sitemap,
)
from alpharidge_ai.miner.discover import Item, parse_feed
from alpharidge_ai.validator.selector import Selector
from alpharidge_ai.verify.check import verdict_hash
from alpharidge_ai.verify.fetch import FetchConfig, Fetcher

API_KEY = Keypair.create_from_seed("0x" + "61" * 32)
VALIDATOR = Keypair.create_from_seed("0x" + "62" * 32)
MINER = Keypair.create_from_seed("0x" + "63" * 32)
JUDGE_KEY = Keypair.create_from_seed("0x" + "64" * 32)
JUDGES = {JUDGE_KEY.ss58_address}
HOTKEYS = [VALIDATOR.ss58_address, "hk1", "hk2", "burn"]
LEAVES = {7: [{"resource_type": "ledger", "resource_id": "1", "miner_hotkey": "hk1",
               "validator_verdict": "provisional", "categorical_key": "finance|event",
               "points_awarded": 3.0, "claim_id": "1"}],
          6: [{"resource_type": "ledger", "resource_id": "2", "miner_hotkey": "hk2",
               "validator_verdict": "provisional", "categorical_key": "finance|fact",
               "points_awarded": 1.0, "claim_id": "2"}]}
PAGE = (Path(__file__).parent / "fixtures" / "pages" / "acme_deal.html").read_text()


def signed_points(key=API_KEY, capacity=10.0):
    leaves = [x for e in sorted(LEAVES) for x in LEAVES[e]]
    per = window_points(leaves)
    total, root = sum(per.values()), ac.merkle_root(leaves)
    sig = key.sign(ac.attestation_message("event-ledger", 7, per, total, root).encode()).hex()
    return {"epoch": 7, "attestation": {"validatorHotkey": "event-ledger", "epoch": 7,
                                        "perMinerPoints": per, "totalPoints": total,
                                        "merkleRoot": root},
            "signature": sig, "capacity": capacity}


def api(points, sample=(), posted=None, down=False, quality=None):
    def handler(request: httpx.Request):
        path, q = request.url.path, request.url.params
        if down:
            raise httpx.ConnectError("unreachable")
        if path == "/ledger/points":
            return httpx.Response(200, json=points)
        if path == "/ledger/leaves":
            return httpx.Response(200, json={"leaves": LEAVES.get(int(q["epoch"]), []),
                                             "next_cursor": None})
        if path == "/ledger/sample":
            assert q["validator"] == VALIDATOR.ss58_address
            return httpx.Response(200, json={"items": list(sample), "next_cursor": None})
        if path == "/ledger/quality" and quality is not None:
            return httpx.Response(200, json=quality)
        if path == "/ledger/reverify":
            posted.append(json.loads(request.content))
            return httpx.Response(200, json={"accepted": 1})
        return httpx.Response(404, json={"detail": "not found"})

    client = AlpharidgeAPIClient("http://api", SimpleNamespace(hotkey=VALIDATOR), max_retries=1)
    client._client = httpx.AsyncClient(base_url="http://api",
                                       transport=httpx.MockTransport(handler))
    return client


def loop(client, meta=None, **kw):
    meta = meta or SimpleNamespace(hotkeys=HOTKEYS, n=len(HOTKEYS))
    return LedgerLoop(client, meta, lambda: 850,
                      pull=LedgerPull(client, pinned_pubkey=API_KEY.ss58_address,
                                      window=3), burn_uid=3,
                      **{"capacity": 10.0, "bind": False, **kw})


def test_points_to_weights_with_burn():
    weights = asyncio.run(loop(api(signed_points())).tick())
    assert weights.tolist() == pytest.approx([0.0, 0.3, 0.1, 0.6])


def test_negative_balance_nets_against_its_coldkey(monkeypatch):
    penalty = {**LEAVES[6][0], "resource_id": "3", "validator_verdict": "penalty",
               "points_awarded": -2.0}
    monkeypatch.setitem(LEAVES, 7, LEAVES[7] + [penalty])
    alone = asyncio.run(loop(api(signed_points())).tick())
    assert alone.tolist() == pytest.approx([0.0, 0.3, 0.0, 0.7])
    meta = SimpleNamespace(hotkeys=HOTKEYS, n=len(HOTKEYS), coldkeys=["v", "ck", "ck", "b"])
    shared = asyncio.run(loop(api(signed_points()), meta).tick())
    assert shared.tolist() == pytest.approx([0.0, 0.2, 0.0, 0.8])


def test_quality_scales_the_weights_only_from_its_start(monkeypatch):
    from datetime import datetime, timedelta, timezone
    from alpharidge_ai.mechanism import quality
    served = {"owners": ["hk1"], "outcomes": [[0, 0]] * 60}
    before = asyncio.run(loop(api(signed_points(), quality=served)).tick())
    assert before.tolist() == pytest.approx([0.0, 0.3, 0.1, 0.6])
    monkeypatch.setattr(quality, "QUALITY_FROM", datetime.now(timezone.utc) + timedelta(days=1))
    assert asyncio.run(loop(api(signed_points(), quality=served)).tick()).tolist() == \
        pytest.approx(before.tolist())
    monkeypatch.setattr(quality, "QUALITY_FROM", datetime.now(timezone.utc) - timedelta(days=1))
    after = asyncio.run(loop(api(signed_points(), quality=served)).tick())
    assert after.tolist() == pytest.approx([0.0, 0.0, 0.1, 0.9])
    # Without the outcomes the step keeps the last weights.
    assert asyncio.run(loop(api(signed_points())).tick()) is None


def test_bad_signature_keeps_the_last_weights():
    lp = loop(api(signed_points(key=Keypair.create_from_seed("0x" + "64" * 32))))
    assert asyncio.run(lp.tick()) is None
    tampered = signed_points()
    tampered["attestation"]["perMinerPoints"]["hk1"] = 30.0
    assert asyncio.run(loop(api(tampered)).tick()) is None


def test_weights_only_when_the_burn_uid_is_the_subnet_owner():
    def boom():
        raise RuntimeError("chain down")
    good = loop(api(signed_points()), owner=lambda: "burn")
    assert asyncio.run(good.tick()) is not None and not good.held
    for owner in (lambda: "hk2", lambda: None, boom):
        lp = loop(api(signed_points()), owner=owner)
        assert lp.held and asyncio.run(lp.tick()) is None and lp.held
        stale = loop(api(None, down=True), accepted_epoch=-11, stale_epochs=18, owner=owner)
        assert asyncio.run(stale.tick()) is None and stale.held
        quiet = loop(api(None, down=True), accepted_epoch=6, stale_epochs=18, owner=owner)
        assert asyncio.run(quiet.tick()) is None and quiet.held


def test_capacity_never_comes_from_the_response():
    weights = asyncio.run(loop(api(signed_points()), capacity=20.0).tick())
    assert weights.tolist() == pytest.approx([0.0, 0.15, 0.05, 0.8])


def test_stale_ledger_burns():
    bad = signed_points(key=Keypair.create_from_seed("0x" + "64" * 32))
    for client in (api(bad), api(None, down=True)):
        assert asyncio.run(loop(client, accepted_epoch=6, stale_epochs=18).tick()) is None
        weights = asyncio.run(loop(client, accepted_epoch=-11, stale_epochs=18).tick())
        assert weights.tolist() == [0.0, 0.0, 0.0, 1.0]
    lp = loop(api(signed_points()), accepted_epoch=-11, stale_epochs=18)
    assert asyncio.run(lp.tick())[3] == pytest.approx(0.6) and lp.accepted_epoch == 7


def test_reverify_catches_up_on_unchecked_epochs():
    class Fake:
        def __init__(self):
            self.runs, self.fail = [], {5}

        async def run(self, epoch, candidates=None):
            self.runs.append(epoch)
            if epoch in self.fail:
                raise RuntimeError("down")

    async def ticks(lp, n):
        for _ in range(n):
            await lp.tick()
            await lp._task

    fake = Fake()
    lp = loop(api(signed_points()), reverifier=fake, reverified=[6, -40])
    asyncio.run(ticks(lp, 1))
    assert fake.runs == [7, 5, 4] and lp.reverified == {4, 6, 7}
    fake.fail = set()
    asyncio.run(ticks(lp, 3))
    assert fake.runs[3:] == [5, 3, 2, 1, 0] and lp.reverified == set(range(8))
    # 72 epochs of 100 blocks make the 24 h deadline; older ones are dropped.
    assert lp.reverify_due(78) == [78, 77, 76] and lp.reverified == {7}


def test_the_signed_profile_sets_the_reverify_rate():
    class Fake:
        rate = 1.0

        async def run(self, epoch, candidates=None):
            pass

    signed = SimpleNamespace(ledger=SimpleNamespace(reverify_rate=0.25, judges=("v1", "v2")))
    fake = Fake()
    lp = loop(api(signed_points()), reverifier=fake,
              profile=SimpleNamespace(resolve=lambda block: signed))
    asyncio.run(lp._reverify([7]))
    assert fake.rate == 0.25 and fake.judges == {"v1", "v2"}


def test_keyed_sample_is_deterministic_per_key():
    items = [{"claim_id": str(i)} for i in range(1, 3001)]
    a = [i["claim_id"] for i in keyed(items, Selector(b"key-a"), 0.10)]
    assert a == [i["claim_id"] for i in keyed(items, Selector(b"key-a"), 0.10)]
    assert 0.07 < len(a) / len(items) < 0.13
    b = [i["claim_id"] for i in keyed(items, Selector(b"key-b"), 0.10)]
    assert a != b and len(set(a) & set(b)) < len(a) / 2


def panel(claim, claim_id, kind="claim", key=JUDGE_KEY, **answers):
    """The record the API keeps for a claim one listed validator judged."""
    raw = ScriptedJudge({"subject": "yes", **answers}).judge_page(claim, "")
    body = verdict_body({"key": "k", "claim_id": claim_id, "kind": kind}, claim, "h", raw, True,
                        None)
    h = ra.signed_headers(key, "POST", "http://api/judge/verdicts", body)
    post = {"validator": key.ss58_address, "body": body.decode(),
            "signed_at": int(h["X-Auth-Timestamp"]), "nonce": h["X-Auth-Nonce"],
            "signature": h["X-Auth-Signature"]}
    return {"content_hash": "h", **combined([json.loads(body)], claim), "posts": [post]}


def _item(claim_id, claim, body=None, judge=None):
    judge = judge or panel(claim, claim_id)
    raw = canonical_body(claim)
    h = ra.signed_headers(MINER, "POST", "http://api/claims", raw)
    return {"claim_id": claim_id, "hotkey": MINER.ss58_address, "route": "event",
            "body": body or raw.decode(), "signature": h["X-Auth-Signature"],
            "nonce": h["X-Auth-Nonce"], "signed_at": int(h["X-Auth-Timestamp"]),
            "received_at": "2026-10-13T12:40:00.000Z", "judge": judge,
            "verdict_hash": verdict_hash(judge), "points": 3.0,
            "factors": {"base": 12, "T": 0.625}}


ANNOUNCED = {"retyped": {"kind": "merger_acquisition:announced"}}


def _deal() -> Claim:
    return Claim(
        kind="event", headline="Acme to acquire Beta Systems", event_type="merger_acquisition",
        category="finance", event_date="2026-10-13",
        actors=[{"id_scheme": "wikidata", "id": "Q1", "role": "acquirer", "name": "Acme Inc."},
                {"id_scheme": "other", "id": "beta", "role": "target", "name": "Beta Systems"}],
        facts=[{"field": "deal_value", "value": 1.2, "unit": "usd_billions"}],
        source_url="https://news.example.org/acme-beta",
        quote="Acme Inc. agreed on Tuesday to acquire Beta Systems in a deal valued at $1.2 "
              "billion, the companies said in a joint statement.")


def _site(request):
    if request.url.path == "/robots.txt":
        return httpx.Response(200, text="User-agent: *\nAllow: /\n")
    return httpx.Response(200, text=PAGE, headers={"content-type": "text/html"})


def test_reverify_checks_signature_and_page():
    claim = _deal()
    forged = canonical_body(claim.model_copy(update={"headline": "Acme buys Beta"})).decode()
    moved = claim.model_copy(update={"quote": "Acme Inc. agreed on Tuesday to acquire Gamma."})
    sample = [_item("1", claim), _item("2", claim, body=forged),
              {**_item("3", claim), "category": "sports"}, _item("4", moved),
              {**_item("5", claim), "verdict_hash": verdict_hash({**panel(claim, "5"),
                                                                  "prompt_version": "x"})},
              {**_item("6", claim), "factors": {"base": 3, "T": 0.625}},
              _item("7", claim, judge={**panel(claim, "7"), "fields": {"deal_value": "unclear"}}),
              _item("8", claim, judge={**panel(judged(claim, ANNOUNCED), "8"), **ANNOUNCED}),
              _item("9", claim, judge={**panel(claim, "9"), "retyped": {"kind": "earnings"}}),
              _item("10", claim, judge={**panel(claim, "10"), "scores": {}}),
              _item("11", claim, judge=panel(claim, "11", key=MINER)),
              _item("12", claim, judge=panel(claim, "1")),
              _item("13", claim, judge={**panel(claim, "13"), "posts": []})]

    async def run():
        posted = []
        client = api(signed_points(), sample, posted)
        fetcher = Fetcher(FetchConfig(public_only=False, domain_rps=1000, retry_delay_s=0),
                          transport=httpx.MockTransport(_site))
        lp = loop(client, reverifier=Reverifier(client, Selector(b"k"), fetcher, rate=1.0,
                                                judges=JUDGES), reverified=range(7))
        assert await lp.tick() is not None
        await lp._task
        await fetcher.aclose()
        return posted

    (post,) = asyncio.run(run())
    assert post["epoch"] == 7
    got = {r["claim_id"]: r for r in post["results"]}
    assert got["1"]["verdict"] == AGREE and got["1"]["observed"]["content_hash"]
    assert got["2"]["verdict"] == DISAGREE and got["2"]["observed"]["reason"] == "bad_signature"
    assert got["3"]["verdict"] == DISAGREE and got["3"]["observed"]["reason"] == "category_mismatch"
    reasons = {k: (got[k]["verdict"], got[k]["observed"]["reason"]) for k in map(str, range(4, 14))}
    # A retyped claim is checked as its new type; one that type cannot hold is inconsistent.
    # The verdict must be the one the listed validators' signed posts combine to.
    assert reasons == {"4": (DISAGREE, "unsupported"), "5": (DISAGREE, "no_judge_record"),
                       "6": (DISAGREE, "points_mismatch"), "7": (DISAGREE, "judge_inconsistent"),
                       "8": (AGREE, None), "9": (DISAGREE, "judge_inconsistent"),
                       "10": (DISAGREE, "verdict_mismatch"), "11": (DISAGREE, "not_a_judge"),
                       "12": (DISAGREE, "post_mismatch"), "13": (DISAGREE, "no_posts")}


def test_reverify_fetches_the_page_the_api_ended_at():
    claim = _deal()
    seen = []

    def site(request):
        seen.append(request.url.path)
        if request.url.path == "/acme-beta":
            return httpx.Response(200, text="<p>Something else entirely.</p>" * 20,
                                  headers={"content-type": "text/html"})
        return _site(request)

    async def run(item):
        fetcher = Fetcher(FetchConfig(public_only=False, domain_rps=1000, retry_delay_s=0),
                          transport=httpx.MockTransport(site))
        try:
            return await reverify(item, fetcher, JUDGES)
        finally:
            await fetcher.aclose()

    moved = {**_item("1", claim), "final_url": "https://news.example.org/2026/acme-beta"}
    assert asyncio.run(run(moved))["verdict"] == AGREE and "/acme-beta" not in seen
    assert asyncio.run(run(_item("1", claim)))["verdict"] == DISAGREE


def test_reverify_checks_a_restamped_page_at_the_date_the_api_saw():
    claim = _deal()
    restamped = PAGE.replace("2026-10-13T12:00:00Z", "2026-10-14T09:00:00Z")

    def site(request):
        if request.url.path == "/robots.txt":
            return _site(request)
        return httpx.Response(200, text=restamped, headers={"content-type": "text/html"})

    async def run(item):
        fetcher = Fetcher(FetchConfig(public_only=False, domain_rps=1000, retry_delay_s=0),
                          transport=httpx.MockTransport(site))
        try:
            return await reverify(item, fetcher, JUDGES)
        finally:
            await fetcher.aclose()

    seen = {**_item("1", claim), "published_at": "2026-10-13T12:00:00.000Z"}
    assert asyncio.run(run(seen))["verdict"] == AGREE
    late = {**seen, "published_at": "2026-10-13T12:50:00.000Z"}
    assert asyncio.run(run(late))["observed"]["reason"] == "pre_published"
    assert asyncio.run(run(_item("1", claim)))["observed"]["reason"] == "pre_published"

def test_a_held_back_claim_binds_by_its_body_hash_and_is_checked_after_settle():
    full = _item("1", _deal())
    held = {**{k: v for k, v in full.items() if k not in ("body", "judge")},
            "body_sha256": ra.body_sha256(full["body"].encode()), "withheld": True}
    assert miner_signed(held) and not miner_signed({**held, "body_sha256": "00"})

    async def run():
        posted, sample = [], [held]
        client = api(signed_points(), sample, posted)
        fetcher = Fetcher(FetchConfig(public_only=False, domain_rps=1000, retry_delay_s=0),
                          transport=httpx.MockTransport(_site))
        rv = Reverifier(client, Selector(b"k"), fetcher, rate=1.0)
        signed = SimpleNamespace(ledger=SimpleNamespace(reverify_rate=None, settle_delay_h=24,
                                                        judges=tuple(JUDGES)))
        lp = loop(client, reverifier=rv, profile=SimpleNamespace(resolve=lambda block: signed))
        await rv.run(7)
        assert rv.withheld == {7: {"1"}} and posted[-1]["results"] == []
        assert lp.settled_due(8) == [] and lp.settled_due(7 + 73) == [7]
        # Still held back at settle: kept for the next pass, dropped after the deadline.
        await lp._reverify([], [7])
        assert rv.withheld == {7: {"1"}}
        assert lp.settled_due(7 + 73 + 72) == [] and rv.withheld == {}
        rv.withheld[7] = {"1"}
        sample[0] = full
        await lp._reverify([], [7])
        await fetcher.aclose()
        return posted, rv

    posted, rv = asyncio.run(run())
    assert [r["verdict"] for r in posted[-1]["results"]] == [AGREE] and rv.withheld == {}


def test_paid_leaves_must_be_sampled_candidates(monkeypatch):
    claim = Claim(
        kind="event", headline="Acme to acquire Beta Systems", event_type="merger_acquisition",
        category="finance", event_date="2026-10-13",
        actors=[{"id_scheme": "wikidata", "id": "Q1", "role": "acquirer", "name": "Acme Inc."}],
        source_url="https://news.example.org/acme-beta")
    item = {**_item("1", claim), "category": "finance", "ledger_ids": ["1"]}
    leaf = {**LEAVES[7][0], "miner_hotkey": MINER.ss58_address, "leaf_version": 2,
            "received_ms": 1791895200000, "bucket": 0, "verdict_hash": item["verdict_hash"]}
    monkeypatch.setitem(LEAVES, 7, [leaf])
    hotkeys = [VALIDATOR.ss58_address, MINER.ss58_address, "hk2", "burn"]

    def run(sample):
        lp = LedgerLoop(api(signed_points(), sample), SimpleNamespace(hotkeys=hotkeys, n=4),
                        lambda: 850, pull=LedgerPull(api(signed_points(), sample),
                                                     pinned_pubkey=API_KEY.ss58_address,
                                                     window=3),
                        burn_uid=3, capacity=10.0, bind=True)
        return asyncio.run(lp.tick())

    assert run([item]) is not None
    assert run([]) is None
    assert run([{**item, "route": "fact"}]) is None
    assert run([{**item, "signature": "00" * 64}]) is None
    assert run([{**item, "points": 2.0}]) is None
    assert run([{**item, "verdict_hash": "00"}]) is None
    assert run([item, {**item, "claim_id": "2", "ledger_ids": []}]) is None
    assert run([{**item, "received_at": "2026-10-13T12:40:01.000Z"}]) is None


def test_every_reverify_run_is_bound_to_the_signed_leaves(monkeypatch):
    class Fake:
        def __init__(self):
            self.runs, self.withheld = [], {}

        async def run(self, epoch, candidates=None, only=None):
            self.runs.append((epoch, candidates))

    monkeypatch.setitem(LEAVES, 6, [{**LEAVES[6][0], "miner_hotkey": MINER.ss58_address}])
    good = {**_item("2", _deal()), "category": "finance", "route": "fact", "points": 1.0,
            "ledger_ids": ["2"], "verdict_hash": None}

    def runs(sample):
        fake = Fake()
        lp = loop(api(signed_points(), sample), reverifier=fake, bind=True,
                  reverified=[7, 5, 4, 3, 2, 1, 0])

        async def tick():
            await lp.tick()
            await lp._task
        asyncio.run(tick())
        return fake.runs

    assert runs([good]) == [(6, [good])]
    assert runs([{**good, "points": 2.0}]) == []
    assert runs([good, {**_item("3", _deal()), "ledger_ids": ["9"]}]) == []


WIRE_URL = "https://wire.example.com/2026/10/13/acme-beta-systems"
WIRE_QUOTE = "Acme Inc. said on Tuesday it will acquire Beta Systems for $1.2 billion in cash."
WIRE = """<!doctype html><html><head><title>Acme to buy Beta Systems</title>
<meta property="article:published_time" content="{pub}"></head><body><article>
<h1>Acme to buy Beta Systems</h1>
<p>{quote}</p>
<p>The acquisition gives Acme a larger presence in industrial software, and the boards of both
companies have approved the terms of the transaction.</p>
<p>The deal is expected to close in the first quarter of next year, subject to regulatory
approval and the usual closing conditions for a transaction of this size.</p>
</article></body></html>"""


def _sites(pub="2026-10-13T13:30:00Z", quote=WIRE_QUOTE, extra=None, own=None):
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        if own and request.url.path == "/acme-beta":
            return httpx.Response(own[0], text=own[1], headers={"content-type": "text/html"})
        if request.url.host == "wire.example.com" and request.url.path.endswith("beta-systems"):
            return httpx.Response(200, text=WIRE.format(pub=pub, quote=quote),
                                  headers={"content-type": "text/html"})
        if extra and request.url.path in extra:
            return httpx.Response(200, text=extra[request.url.path])
        return httpx.Response(200, text=PAGE, headers={"content-type": "text/html"})
    return Fetcher(FetchConfig(public_only=False, domain_rps=1000, retry_delay_s=0),
                   transport=httpx.MockTransport(handler))


def _scoop_item(**confirm):
    on_wire = confirm_claim(_deal(), WIRE_URL, WIRE_QUOTE, WIRE_QUOTE)
    found = {"url": WIRE_URL, "final_url": WIRE_URL, "content_hash": "h",
             "published_at": "2026-10-13T13:30:00Z", "quote": WIRE_QUOTE, "trust": 0.625,
             "claim": on_wire.model_dump(mode="json", exclude_none=True),
             "judge": panel(on_wire, "1", "confirm"), "by": VALIDATOR.ss58_address,
             "found_at": "2026-10-13T14:00:00Z", **confirm}
    return {**_item("1", _deal(), judge={**panel(_deal(), "1"), "confirm": found}), "route": "scoop",
            "points": 15.0, "factors": {"base": 60, "T": 0.625}}


def _unread(item):
    record = {"unread": True, "confirm": item["judge"]["confirm"]}
    return {**item, "judge": record, "verdict_hash": verdict_hash(record)}


PLAIN = panel(_deal(), "1")
EARLY = "2026-10-13T12:35:00Z"


@pytest.mark.parametrize("item,site,want", [
    (_scoop_item(), {}, (AGREE, None)),
    (_scoop_item(), {"quote": WIRE_QUOTE.replace("1.2 billion", "900 million")},
     (DISAGREE, "unsupported")),
    # The stored confirm claim is checked as kept, whatever the refetch's text would give.
    (_scoop_item(), {"quote": WIRE_QUOTE + " It did not give a value for the deal."},
     (AGREE, None)),
    (_scoop_item(claim={**_deal().model_dump(mode="json"), "source_url": WIRE_URL,
                        "quote": WIRE_QUOTE, "event_type": "earnings"}),
     {}, (DISAGREE, "confirm_inconsistent")),
    (_scoop_item(claim=None), {}, (DISAGREE, "confirm_inconsistent")),
    # Confirmed by a page published before the scoop's receipt: the tier price, not the premium.
    ({**_scoop_item(published_at=EARLY), "points": 3.0, "factors": {"base": 12, "T": 0.625}},
     {"pub": EARLY}, (AGREE, None)),
    (_scoop_item(published_at=EARLY), {"pub": EARLY}, (DISAGREE, "points_mismatch")),
    (_scoop_item(), {"pub": EARLY}, (DISAGREE, "published_mismatch")),
    (_scoop_item(trust=1.0), {}, (DISAGREE, "points_mismatch")),
    ({**_scoop_item(), "points": 3.0, "factors": {"base": 12, "T": 0.625}}, {},
     (DISAGREE, "points_mismatch")),
    ({**_scoop_item(), "judge": PLAIN, "verdict_hash": verdict_hash(PLAIN), "points": 3.0,
      "factors": {"base": 12, "T": 0.625}}, {},
     (DISAGREE, "no_confirm")),
    (_scoop_item(judge=panel(confirm_claim(_deal(), WIRE_URL, WIRE_QUOTE, WIRE_QUOTE), "1")),
     {}, (DISAGREE, "post_mismatch")),
    # Its own page is not read again once confirmed.
    (_scoop_item(), {"own": (200, PAGE.replace("Beta Systems", "Gamma"))}, (AGREE, None)),
    (_scoop_item(), {"own": (404, "")}, (AGREE, None)),
    # Contested: the tier price, not the premium.
    ({**_scoop_item(contested=["7"]), "points": 3.0, "factors": {"base": 12, "T": 0.625}}, {},
     (AGREE, None)),
    (_scoop_item(contested=["7"]), {}, (DISAGREE, "points_mismatch")),
    # Its page was not read: no page-judge record of its own, age from the event date alone.
    (_unread(_scoop_item()), {}, (AGREE, None)),
    (_unread({**_scoop_item(), "published_at": "2026-09-20T12:00:00Z"}), {}, (AGREE, None)),
    ({**_scoop_item(), "published_at": "2026-09-20T12:00:00Z"}, {}, (DISAGREE, "points_mismatch")),
    ({**_unread(_scoop_item()), "route": "event"}, {}, (DISAGREE, "no_judge_record")),
])
def test_a_scoop_is_checked_on_the_page_that_confirmed_it(item, site, want):
    async def run():
        fetcher = _sites(**site)
        try:
            return await reverify(item, fetcher, JUDGES)
        finally:
            await fetcher.aclose()

    got = asyncio.run(run())
    assert (got["verdict"], got["observed"].get("reason")) == want
    if want[0] == AGREE:
        assert got["observed"]["confirm_content_hash"]


OPEN = {"id": 4, "event_type": "merger_acquisition", "stage": None, "event_date": "2026-10-13",
        "received_at": "2026-10-13T12:40:00Z", "closes_at": "2026-10-16T12:40:00Z",
        "subjects": ["Beta Systems", "Beta Systems Inc."], "actor_ids": [],
        "facts": [{"field": "deal_value", "value": 1.2, "unit": "usd_billions"},
                  {"field": "stake_pct", "value": 80, "unit": "pct"}]}
SITEMAP = """<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
 xmlns:news="http://www.google.com/schemas/sitemap-news/0.9">
<url><loc>{wire}</loc><news:news><news:publication_date>2026-10-13T13:30:00Z
</news:publication_date><news:title>Deal news</news:title></news:news></url>
<url><loc>https://wire.example.com/2026/10/13/gamma-results</loc>
<lastmod>2026-10-13T13:00:00Z</lastmod></url>
<url><loc>https://wire.example.com/2026/10/01/beta-systems-profile</loc>
<lastmod>2026-10-01T13:00:00Z</lastmod></url>
</urlset>""".format(wire=WIRE_URL)


def test_the_scan_keeps_items_naming_a_subject_and_posts_pages_that_pass():
    scoop = parse_scoop(OPEN)
    items = parse_sitemap(SITEMAP)
    assert [i.url for i in items if mentions(scoop, i)] == [
        WIRE_URL, "https://wire.example.com/2026/10/01/beta-systems-profile"]
    assert mentions(scoop, Item(url="https://x.org/a", title="Beta Systems agrees sale"))
    assert not mentions(scoop, Item(url="https://x.org/beta", title="Betamax returns"))
    listing = page_items('<a href="/2026/10/13/deal">Beta Systems sold</a>'
                         '<a href="https://other.org/beta-systems">x</a>'
                         '<a href="/about"><span>About</span></a>', "https://www.wire.example.com/news")
    assert [(i.url, i.title) for i in listing] == [
        ("https://www.wire.example.com/2026/10/13/deal", "Beta Systems sold"),
        ("https://www.wire.example.com/about", "About")]
    assert [mentions(scoop, i) for i in listing] == [True, False]

    posted = []

    def handler(request):
        if request.url.path == "/scoops/open":
            return httpx.Response(200, json={"scoops": [OPEN], "feeds": [
                {"url": "https://wire.example.com/news.xml", "kind": "sitemap"},
                {"url": "https://wire.example.com/newsroom", "kind": "page"}]})
        if request.url.path == "/scoops/confirm":
            posted.append(json.loads(request.content))
            return httpx.Response(202, json={"status": "queued"})
        return httpx.Response(404, json={"detail": "not found"})

    async def run():
        client = AlpharidgeAPIClient("http://api", SimpleNamespace(hotkey=VALIDATOR),
                                     max_retries=1)
        client._client = httpx.AsyncClient(base_url="http://api",
                                           transport=httpx.MockTransport(handler))
        newsroom = f'<a href="{WIRE_URL}">Beta Systems</a><a href="/other">Beta Systems x</a>'
        fetcher = _sites(extra={"/news.xml": SITEMAP, "/newsroom": newsroom})
        found = datetime(2026, 10, 13, 14, tzinfo=timezone.utc)
        scan = Scanner(client, fetcher, interval=0, concurrency=2, clock=lambda: found)
        first, again = await scan.run(), await scan.run()
        await fetcher.aclose()
        return first, again

    first, again = asyncio.run(run())
    # A trusted page from before the scoop's receipt is sent too: the API rejects it not_first.
    assert posted == [{"scoop_id": 4, "url": WIRE_URL},
                      {"scoop_id": 4, "url": "https://wire.example.com/other"}]
    assert first["posted"] == 2 and first["checked"] == 2 and again["checked"] == 0


def test_the_scan_matches_shorter_and_longer_names_tickers_and_summaries():
    def scoop(*subjects, tickers=()):
        return parse_scoop({**OPEN, "subjects": list(subjects), "tickers": list(tickers)})

    fhlb, gbli = scoop("FHLBank Chicago"), scoop("GBLI", "Global Indemnity Group",
                                                    tickers=["GBLI"])
    item = Item(url="https://x.org/n/1", title="Federal Home Loan Bank of Chicago Announces "
                "Leadership Transition")
    assert mentions(fhlb, item) and not mentions(gbli, item)
    assert mentions(scoop("Federal Home Loan Bank of Chicago"),
                    Item(url="https://x.org/fhlbank-chicago-names-ceo"))
    assert mentions(gbli, Item(url="https://x.org/2", title="Global Indemnity names CEO"))
    assert mentions(gbli, Item(url="https://x.org/3", title="Insurer names CEO",
                               summary="The insurer (NYSE: GBLI) named a new chief."))
    assert mentions(scoop("SST VI"), Item(url="https://x.org/4", title="x", summary=(
        "Strategic Storage Trust VI completes merger.")))
    assert not mentions(scoop("Global Indemnity Group", tickers=["GBLI"]), Item(
        url="https://x.org/5", title="Global markets fall", summary="gbli is lower case here"))
    assert [i.summary for i in parse_feed(
        "<rss><channel><item><title>t</title><link>https://x.org/6</link>"
        "<pubDate>Mon, 02 Nov 2026 10:00:00 GMT</pubDate><description>&lt;p&gt;Shares "
        "&amp;amp; bonds&lt;/p&gt;</description></item></channel></rss>")] == ["Shares & bonds"]


def _scan_api(open_, feeds, answer, posted):
    def handler(request):
        if request.url.path == "/scoops/open":
            return httpx.Response(200, json={"scoops": [open_], "feeds": feeds})
        if request.url.path == "/scoops/confirm":
            url = json.loads(request.content)["url"]
            posted.append(url)
            return httpx.Response(200, json={"status": answer(url)})
        return httpx.Response(404, json={"detail": "not found"})

    client = AlpharidgeAPIClient("http://api", SimpleNamespace(hotkey=VALIDATOR), max_retries=1)
    client._client = httpx.AsyncClient(base_url="http://api",
                                       transport=httpx.MockTransport(handler))
    return client


def test_the_scan_counts_only_new_candidates_and_keeps_scanning_an_open_scoop():
    urls = [f"https://wire.example.com/2026/10/13/n{i}-beta-systems" for i in range(6)]
    sitemap = ('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">' + "".join(
        f"<url><loc>{u}</loc><lastmod>2026-10-13T13:30:00Z</lastmod></url>" for u in urls)
        + "</urlset>")
    posted = []
    client = _scan_api(OPEN, [{"url": "https://wire.example.com/news.xml", "kind": "sitemap"}],
                       lambda u: "duplicate" if u in urls[:2] else "queued", posted)

    async def run():
        fetcher = _sites(extra={"/news.xml": sitemap})
        found = datetime(2026, 10, 13, 14, tzinfo=timezone.utc)
        scan = Scanner(client, fetcher, interval=0, concurrency=2, clock=lambda: found)
        got = [await scan.run(), await scan.run()]
        await fetcher.aclose()
        return got

    first, again = asyncio.run(run())
    assert first["posted"] == 3 and posted[:5] == urls[:5]
    assert again["open"] == 1 and posted[5:] == urls[5:] and again["posted"] == 1


def test_the_scan_rereads_fast_feeds_and_keeps_items_that_rolled_off():
    early = "https://wire.example.com/2026/10/12/b1-beta-systems"

    def urlset(rows):
        return ('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">' + "".join(
            f"<url><loc>{u}</loc><lastmod>{t}</lastmod></url>" for u, t in rows) + "</urlset>")

    filler = [(f"https://wire.example.com/x{i}", f"2026-10-12T09:0{i}:00Z") for i in range(6)]
    reads, now = {}, [datetime(2026, 10, 13, 14, tzinfo=timezone.utc)]

    def page(path):
        reads[path] = reads.get(path, 0) + 1
        if path == "/slow.xml":
            return urlset([("https://wire.example.com/old", "2026-10-01T00:00:00Z"),
                           ("https://wire.example.com/new", "2026-10-13T00:00:00Z")])
        return urlset(([(early, "2026-10-12T09:00:00Z")] if reads[path] == 1 else []) + filler)

    other = {**OPEN, "id": 5, "subjects": ["Gamma Labs"]}
    beta = {**OPEN, "since": "2026-10-10T00:00:00Z"}
    calls, posted = [], []
    client = _scan_api(None, [{"url": "https://wire.example.com/fast.xml", "kind": "sitemap"},
                              {"url": "https://wire.example.com/slow.xml", "kind": "sitemap"}],
                       lambda u: "queued", posted)
    opened = client.scoops_open

    async def scoops_open():
        calls.append(1)
        got = await opened()
        return {**got, "scoops": [other] + ([beta] if len(calls) > 1 else [])}

    client.scoops_open = scoops_open

    class Feeds(dict):
        def __getitem__(self, path):
            return page(path)

    async def run():
        fetcher = _sites(extra=Feeds({"/fast.xml": "", "/slow.xml": ""}))
        scan = Scanner(client, fetcher, interval=0, concurrency=2, clock=lambda: now[0])
        await scan.run()
        now[0] += timedelta(seconds=300)
        await scan.run()
        await fetcher.aclose()

    asyncio.run(run())
    assert posted == [early] and reads == {"/fast.xml": 2, "/slow.xml": 1}


def test_without_a_judge_list_the_posts_cannot_be_checked():
    async def run():
        fetcher = Fetcher(FetchConfig(public_only=False, domain_rps=1000, retry_delay_s=0),
                          transport=httpx.MockTransport(_site))
        try:
            return await reverify(_item("1", _deal()), fetcher, ())
        finally:
            await fetcher.aclose()

    got = asyncio.run(run())
    assert (got["verdict"], got["observed"]["reason"]) == ("unreachable", "no_judge_list")


class JudgeAPI:
    def __init__(self, items):
        self.items, self.posted = items, []

    async def judge_pending(self):
        return {"types": ["merger_acquisition", "earnings"], "items": self.items}

    async def post_verdict(self, body):
        self.posted.append(json.loads(body))
        return {"status": "accepted"}


def test_a_listed_validator_fetches_checks_judges_and_posts_once():
    from alpharidge_ai.validator.judge import JudgeWorker
    deal = _deal()
    moved = deal.model_copy(update={"quote": "Acme Inc. agreed on Tuesday to acquire Gamma."})
    item = {"kind": "claim", "claim_id": "1", "body": canonical_body(deal).decode(),
            "url": deal.source_url, "at": "2026-10-13T12:40:00Z"}
    items = [{**item, "key": "k1"}, {**item, "key": "k2", "claim_id": "2",
                                      "body": canonical_body(moved).decode()}]

    async def run(worker):
        try:
            return await worker.run()
        finally:
            await worker.fetcher.aclose()

    client, judge_ = JudgeAPI(items), ScriptedJudge({"subject": "yes"})

    def worker(**kw):
        return JudgeWorker(client, judge_, Fetcher(FetchConfig(
            public_only=False, domain_rps=1000, retry_delay_s=0),
            transport=httpx.MockTransport(_site)), **kw)

    w = worker()
    asyncio.run(run(w))
    ok, off = sorted(client.posted, key=lambda b: b["key"])
    assert judge_.calls == 1 and (ok["claim_id"], ok["as"]) == ("1", "merger_acquisition")
    assert ok["fixed_check"] == {"passed": True, "reason": None} and ok["content_hash"]
    assert combined([ok], deal)["outcome"] == "pass"
    assert off["fixed_check"]["passed"] is False and off["answers"] == {}
    w._fetcher = Fetcher(FetchConfig(public_only=False, domain_rps=1000, retry_delay_s=0),
                         transport=httpx.MockTransport(_site))
    asyncio.run(run(w))
    assert judge_.calls == 1 and len(client.posted) == 2
    client.items = [{**item, "key": "k3", "unread": True}]
    assert asyncio.run(run(worker())) == {"unread": 1} and judge_.calls == 1
    capped = worker(cap_usd=0)
    assert asyncio.run(run(capped)) == {"capped": 1} and len(client.posted) == 2


def test_a_listed_validator_judges_the_page_the_api_ended_at():
    from alpharidge_ai.validator.judge import JudgeWorker
    deal, seen = _deal(), []
    item = {"kind": "claim", "claim_id": "1", "key": "k1", "body": canonical_body(deal).decode(),
            "url": deal.source_url, "final_url": "https://news.example.org/2026/acme-beta",
            "at": "2026-10-13T12:40:00Z"}

    def site(request):
        seen.append(request.url.path)
        return _site(request)

    async def run():
        w = JudgeWorker(JudgeAPI([item]), ScriptedJudge({"subject": "yes"}), Fetcher(
            FetchConfig(public_only=False, domain_rps=1000, retry_delay_s=0),
            transport=httpx.MockTransport(site)))
        try:
            return await w.run()
        finally:
            await w.fetcher.aclose()

    asyncio.run(run())
    assert "/2026/acme-beta" in seen and "/acme-beta" not in seen


def test_private_fetches_only_on_mock_or_local_chains(monkeypatch):
    from alpharidge_ai import config
    config.check_fetch_guard(False, "finney")
    monkeypatch.setattr(config, "REVERIFY_FETCH_PUBLIC_ONLY", False)
    config.check_fetch_guard(True, "finney")
    config.check_fetch_guard(False, "local")
    with pytest.raises(SystemExit):
        config.check_fetch_guard(False, "finney")


def test_a_stuck_judge_item_times_out_and_the_pull_still_finishes(monkeypatch):
    from alpharidge_ai.validator import judge as vj
    items = [{"kind": "claim", "claim_id": str(i), "key": f"k{i}"} for i in range(3)]
    monkeypatch.setattr(vj, "ITEM_TIMEOUT_S", 0.05)

    async def judge_item(self, item, types=None):
        if item["key"] == "k1":
            await asyncio.sleep(3600)
        self.done.add(item["key"])
        return "posted"
    monkeypatch.setattr(vj.JudgeWorker, "judge_item", judge_item)
    w = vj.JudgeWorker(JudgeAPI(items), ScriptedJudge({"subject": "yes"}))
    assert asyncio.run(w.run()) == {"posted": 2, "timeout": 1}
    assert w.tries == {"k1": 1} and "k1" not in w.done


def test_a_slow_judge_item_never_holds_up_the_next_pull(monkeypatch):
    from alpharidge_ai.validator import judge as vj
    items = [{"kind": "claim", "claim_id": str(i), "key": f"k{i}"} for i in range(3)]
    monkeypatch.setattr(vj, "PULL_WAIT_S", 0.05)
    slow = {"k1": 0.3}
    started = []

    async def judge_item(self, item, types=None):
        started.append(item["key"])
        await asyncio.sleep(slow.get(item["key"], 0))
        self.done.add(item["key"])
        return "posted"
    monkeypatch.setattr(vj.JudgeWorker, "judge_item", judge_item)

    async def pulls():
        w = vj.JudgeWorker(JudgeAPI(items), ScriptedJudge({"subject": "yes"}), concurrency=2)
        first = await w.run()
        again = await w.run()
        await asyncio.sleep(0.4)
        return first, again, await w.run()
    first, again, last = asyncio.run(pulls())
    assert first == {"posted": 2, "running": 1}
    # k1 is still running, so it is not served again; the next pull does not wait for it.
    assert again == {"running": 1} and started.count("k1") == 1
    assert last == {"posted": 1}


def test_the_watchdog_names_a_stalled_step_or_judge_loop():
    from alpharidge_ai.validator import watchdog
    assert watchdog.stall(1000, 900, 950) is None
    assert watchdog.stall(1000, 900, None) is None
    assert "validator step" in watchdog.stall(2000, 1600, 2000)
    assert "judge pull" in watchdog.stall(5000, 4990, 1000)
    assert watchdog.stall(5000, 4990, None) is None


def test_a_finished_judge_pull_is_recorded_for_the_watchdog():
    from alpharidge_ai.validator.judge import JudgeWorker
    w = JudgeWorker(JudgeAPI([]), ScriptedJudge({"subject": "yes"}))
    w.last_pull = 0.0
    asyncio.run(w._safe())
    assert w.last_pull > 0


def test_a_chain_call_that_stops_answering_times_out():
    import threading
    from alpharidge_ai.chain.deadline import ChainTimeout, bounded
    assert bounded(lambda: 7, 1) == 7
    with pytest.raises(ValueError):
        bounded(lambda: (_ for _ in ()).throw(ValueError("x")), 1)
    stuck = threading.Event()
    with pytest.raises(ChainTimeout):
        bounded(stuck.wait, 0.05, "block read")
    stuck.set()


def test_served_config_counts_as_loaded_only_after_a_good_read(monkeypatch):
    from alpharidge_ai import config as cfg
    monkeypatch.setattr(cfg, "_remote_config_last_fetch", 0.0)
    monkeypatch.setattr(cfg, "MINER_API_URL", "http://127.0.0.1:9")
    assert not cfg.remote_config_loaded()
    assert cfg.refresh_remote_config(force=True) == {} and not cfg.remote_config_loaded()
    monkeypatch.setattr(cfg, "_remote_config_last_fetch", 1.0)
    assert cfg.remote_config_loaded()


def test_a_judge_item_on_a_host_that_is_being_left_alone_is_served_again_later(monkeypatch):
    from alpharidge_ai.validator import judge as vj
    from alpharidge_ai.verify.fetch import Paced

    async def judge_item(self, item, types=None):
        raise Paced(120)
    monkeypatch.setattr(vj.JudgeWorker, "judge_item", judge_item)
    w = vj.JudgeWorker(JudgeAPI([{"kind": "claim", "claim_id": "1", "key": "k1"}]),
                       ScriptedJudge({"subject": "yes"}))
    assert asyncio.run(w.run()) == {"paced": 1} and "k1" not in w.done and not w.tries


def test_judge_spend_survives_a_restart_within_the_day(tmp_path):
    from datetime import datetime, timezone
    from alpharidge_ai.validator import judge as vj
    path = tmp_path / ".judge_spend_state.json"
    now = [datetime(2026, 11, 12, 9, tzinfo=timezone.utc)]
    spend = vj.Spend(1.0, lambda: now[0], path)
    spend.add(0.9)
    again = vj.Spend(1.0, lambda: now[0], path)
    assert again.usd == 0.9 and not again.room(0.2)
    now[0] = datetime(2026, 11, 13, 0, 1, tzinfo=timezone.utc)
    assert again.room(0.2)
    path.write_text("not json")
    assert vj.Spend(1.0, lambda: now[0], path).room(0.2)


def test_a_ledger_step_that_runs_over_keeps_the_last_weights(monkeypatch):
    from alpharidge_ai.validator import loop as loop_mod
    monkeypatch.setattr(loop_mod, "WEIGHTS_STEP_S", 0.05)
    lp = loop(api(signed_points()))

    async def stuck(epoch):
        await asyncio.sleep(3600)
    monkeypatch.setattr(lp, "weights", stuck)
    assert asyncio.run(lp.tick()) is None and lp._weights_epoch is None


def test_scan_matching_keeps_to_its_budget_and_never_blocks_the_loop(monkeypatch):
    from dataclasses import replace
    from alpharidge_ai.validator import scan as vscan
    base = parse_scoop(OPEN)
    scoops = [replace(base, id=i) for i in range(3)]
    keyed = [(i, vscan.item_keys(i)) for i in parse_sitemap(SITEMAP)]
    assert list(vscan.match_items(scoops, keyed)) == [0, 1, 2]
    assert vscan.match_items(scoops, keyed)[0] == {WIRE_URL: keyed[0][0].published}
    # Past the budget, scoops not reached are left for a later scan; one is always matched.
    assert list(vscan.match_items(scoops, keyed, budget_s=0)) == [0]
    assert vscan.match_items(scoops, keyed, {0: 3}, [1, 2, 3])[0] == {}

    import time
    real = vscan.mentions

    def slow(*a, **kw):
        time.sleep(0.05)
        return real(*a, **kw)
    monkeypatch.setattr(vscan, "mentions", slow)

    async def both():
        ticks = 0

        async def beat():
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0.01)
        task = asyncio.create_task(beat())
        await asyncio.to_thread(vscan.match_items, scoops, keyed)
        task.cancel()
        return ticks
    assert asyncio.run(both()) > 10
