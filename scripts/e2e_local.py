"""Local end-to-end run of the subnet. Driven by scripts/e2e_local.sh.

    python scripts/e2e_local.py keys          # wallets for the mock chain, as shell exports
    python scripts/e2e_local.py sites         # fixture web server (foreground)
    python scripts/e2e_local.py judge-script  # the scripted page judge's answers, as JSON
    python scripts/e2e_local.py phase 2       # publish the fixture pages of phase 2
    python scripts/e2e_local.py miner         # one reference-miner cycle
    python scripts/e2e_local.py submit negated|hedged|prerelease|off|known|dispute|excluded
    python scripts/e2e_local.py flood         # a third hotkey sends past its check budget
    python scripts/e2e_local.py watch --until verified|settled
    python scripts/e2e_local.py judge validator|validator2  # a listed validator judging pages
    python scripts/e2e_local.py reverify      # validator: keyed re-verify of closed epochs
    python scripts/e2e_local.py scan          # validator: scan rounds until no scoop is held
    python scripts/e2e_local.py weights       # validator: signed points -> settle() -> weights
    python scripts/e2e_local.py check         # expected outcomes
    python scripts/e2e_local.py wrong         # operator corrections and retractions

Everything stays on this machine: the fixture sites run on *.localhost, the chain is mocked
and the API runs on 127.0.0.1.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import socket
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional
from xml.sax.saxutils import escape

os.environ.setdefault("BT_NO_PARSE_CLI_ARGS", "0")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from alpharidge_ai.schema.types import SCOOP_PREMIUM  # noqa: E402
from e2e_fixtures import ATTACH, PAGE, PAGES, RAW, SITE_PORT, Page, site  # noqa: E402

E2E_DIR = Path(os.getenv("E2E_DIR", REPO / ".e2e"))
FIXTURE_ENTITIES = REPO / "tests" / "fixtures" / "entities.json"
API_URL = os.getenv("MINER_API_URL", "http://127.0.0.1:8890")
NETUID = int(os.getenv("E2E_NETUID", "45"))
DASHBOARD_LIMIT = 100

# Mock-chain identities, in uid order after the validator (uid 0).
ROLES = ("miner", "second", "owner")
URIS = {"validator": "//e2e/validator", "validator2": "//e2e/validator2",
        "miner": "//e2e/miner", "second": "//e2e/second", "owner": "//e2e/owner",
        "flood": "//e2e/flood"}
LEDGER_SEED = hashlib.sha256(b"e2e-local-ledger").hexdigest()

TERMINAL = {"verified", "provisional", "settled", "rejected", "clawed_back"}


# The official release time of the fixture's scheduled release, ahead of every receipt.
RELEASE_AHEAD_H = 6.0


def render(page: Page, published: datetime) -> bytes:
    body = "\n".join(f"<p>{escape(t)}</p>" for t in page.paragraphs)
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<title>{escape(page.title)}</title>
<link rel="canonical" href="{page.url}">
<meta property="article:published_time" content="{published.isoformat()}">
</head><body>
<nav>Home | News | Markets</nav>
<article>
<h1>{escape(page.title)}</h1>
{body}
</article>
<footer>Local test fixture.</footer>
</body></html>""".encode()


def rss(host: str, items: List[tuple]) -> bytes:
    entries = "".join(
        f"<item><title>{escape(p.title)}</title><link>{p.url}</link>"
        f"<pubDate>{format_datetime(at)}</pubDate></item>" for p, at in items)
    return (f'<?xml version="1.0"?><rss version="2.0"><channel><title>{host}</title>'
            f"<link>{site(host)}/</link>{entries}</channel></rss>").encode()


def release_calendar(opened: datetime) -> bytes:
    from zoneinfo import ZoneInfo
    at = (opened + timedelta(hours=RELEASE_AHEAD_H)).astimezone(ZoneInfo("America/New_York"))
    return ("BEGIN:VCALENDAR\r\nBEGIN:VEVENT\r\nSUMMARY:Consumer Price Index\r\n"
            f"DTSTART;TZID=US-Eastern:{at:%Y%m%dT%H%M%S}\r\nEND:VEVENT\r\n"
            "END:VCALENDAR\r\n").encode()


class Sites:
    """Fixture pages and feeds, with publish times set when each phase opens."""

    def __init__(self):
        self.lock = threading.Lock()
        self.opened: Dict[int, datetime] = {1: datetime.now(timezone.utc)}

    def published(self, page: Page) -> Optional[datetime]:
        opened = self.opened.get(page.phase)
        if opened is None:
            return None
        return (opened - timedelta(hours=page.age_h)).replace(microsecond=0)

    def open_phase(self, phase: int) -> None:
        with self.lock:
            self.opened.setdefault(phase, datetime.now(timezone.utc))

    def get(self, host: str, path: str):
        if path == "/robots.txt":
            return "text/plain", b"User-agent: *\nAllow: /\n"
        if host == "stats" and path == "/calendar.ics":
            return "text/calendar", release_calendar(self.opened[1])
        if host == "stats" and path == "/latest.rss":
            return "application/rss+xml", rss(host, [])
        if path == "/feed.xml":
            items = [(p, self.published(p)) for p in PAGES if p.host == host and p.listed]
            return "application/rss+xml", rss(host, [(p, at) for p, at in items if at])
        for p in PAGES:
            if p.host == host and p.path == path and self.published(p):
                return "text/html; charset=utf-8", render(p, self.published(p))
        return None


def handler(sites: Sites):
    class Handler(BaseHTTPRequestHandler):
        def _host(self) -> str:
            return (self.headers.get("Host") or "").split(":")[0].removesuffix(".localhost")

        def do_GET(self):
            hit = sites.get(self._host(), self.path.split("?")[0])
            self.send_response(200 if hit else 404)
            self.send_header("Content-Type", hit[0] if hit else "text/plain")
            self.end_headers()
            self.wfile.write(hit[1] if hit else b"not found")

        def do_POST(self):
            if self.path.startswith("/_phase/") and self.path[8:].isdigit():
                sites.open_phase(int(self.path[8:]))
                self.send_response(204)
            else:
                self.send_response(404)
            self.end_headers()

        def log_message(self, fmt, *args):
            print(f"[sites] {self._host()} {fmt % args}", flush=True)

    return Handler


class _V6Server(ThreadingHTTPServer):
    address_family = socket.AF_INET6


def feeds_file() -> Path:
    path = E2E_DIR / "feeds.json"
    hosts = sorted({(p.host, p.category) for p in PAGES})
    path.write_text(json.dumps({"sources": [
        {"url": f"{site(h)}/feed.xml", "category": c} for h, c in hosts]}, indent=1))
    return path


def cmd_sites(_args) -> None:
    sites = Sites()
    # *.localhost resolves to ::1 here and to 127.0.0.1 on some hosts, so serve both.
    servers = [_V6Server(("::1", SITE_PORT), handler(sites))]
    try:
        servers.append(ThreadingHTTPServer(("127.0.0.1", SITE_PORT), handler(sites)))
    except OSError:
        pass
    feeds_file()
    for s in servers[1:]:
        threading.Thread(target=s.serve_forever, daemon=True).start()
    print(f"[sites] serving {len(PAGES)} pages on port {SITE_PORT}", flush=True)
    servers[0].serve_forever()


def cmd_scan_feeds(_args) -> None:
    """The trusted list's feeds the validator scan reads, as the API serves them."""
    print(json.dumps({"feeds": [{"url": f"{site(h)}/feed.xml", "kind": "rss"}
                                for h in ("centralbank", "filings", "wire")]}))


def cmd_judge_script(_args) -> None:
    print(json.dumps({p.path: dict(p.judge) for p in PAGES if p.judge}))


def cmd_phase(args) -> None:
    import httpx
    httpx.post(f"http://127.0.0.1:{SITE_PORT}/_phase/{args.phase}", timeout=5)
    print(f"[sites] phase {args.phase} published")


# ---- keys and neurons ------------------------------------------------------------------

def wallet_path() -> Path:
    return E2E_DIR / "wallets"


def keypair(role: str):
    from bittensor_wallet import Keypair
    return Keypair.create_from_uri(URIS[role])


def wallet(role: str):
    import bittensor as bt
    return bt.Wallet(name=f"e2e-{role}", hotkey="default", path=str(wallet_path()))


def cmd_keys(_args) -> None:
    from bittensor_wallet import Keypair
    E2E_DIR.mkdir(parents=True, exist_ok=True)
    hotkeys = {}
    for role in URIS:
        w, kp = wallet(role), keypair(role)
        w.set_hotkey(kp, encrypt=False, overwrite=True)
        w.set_coldkeypub(Keypair.create_from_uri(URIS[role] + "/cold"), overwrite=True)
        hotkeys[role] = kp.ss58_address
    ledger = Keypair.create_from_seed("0x" + LEDGER_SEED)
    exports = {
        "E2E_VALIDATOR_HOTKEY": hotkeys["validator"],
        "E2E_VALIDATOR_HOTKEYS": ",".join(hotkeys[r] for r in ("validator", "validator2")),
        "E2E_MINER_HOTKEYS": ",".join(hotkeys[r] for r in ("miner", "second", "flood")),
        "MOCK_HOTKEYS": ",".join(hotkeys[r] for r in ROLES),
        "BURN_UID": str(1 + ROLES.index("owner")),
        "API_ATTESTATION_PRIVKEY": LEDGER_SEED,
        "API_ATTESTATION_PUBKEY": ledger.ss58_address,
    }
    (E2E_DIR / "keys.json").write_text(json.dumps(hotkeys, indent=1))
    for k, v in exports.items():
        print(f"export {k}={v}")


def hotkeys() -> Dict[str, str]:
    return json.loads((E2E_DIR / "keys.json").read_text())


def neuron_argv(role: str, name: str) -> List[str]:
    return [name, "--mock", "--netuid", str(NETUID), "--wallet.name", f"e2e-{role}",
            "--wallet.hotkey", "default", "--wallet.path", str(wallet_path()),
            "--logging.logging_dir", str(E2E_DIR / "logs"), "--neuron.dont_save_events"]


def remember(sent: Dict[str, str]) -> None:
    path = E2E_DIR / "sent.json"
    known = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps({**known, **sent}, indent=1))


def sent_urls() -> Dict[str, str]:
    path = E2E_DIR / "sent.json"
    return json.loads(path.read_text()) if path.exists() else {}


def client(role: str):
    from alpharidge_ai.ledger.api_client import AlpharidgeAPIClient
    return AlpharidgeAPIClient(base_url=API_URL, wallet=wallet(role), timeout=30)


# ---- miners ----------------------------------------------------------------------------

def cmd_miner(_args) -> None:
    os.environ.setdefault("MINER_FEEDS", str(feeds_file()))
    os.environ.setdefault("MINER_STATE_PATH", str(E2E_DIR / "miner.sqlite"))
    sys.argv = neuron_argv("miner", "miner")
    from alpharidge_ai import config
    from alpharidge_ai.miner import link
    from neurons.miner import Miner
    config.API_ATTESTATION_PUBKEY = os.environ["API_ATTESTATION_PUBKEY"]

    shipped = link.load_entities
    link.load_entities = lambda path=link.ENTITIES_PATH: shipped(path) + shipped(FIXTURE_ENTITIES)

    miner = Miner()
    loop = miner._miner

    async def cycle():
        try:
            return await loop.tick()
        finally:
            await loop.fetcher.aclose()
            await miner._api.close()

    sent = miner.loop.run_until_complete(cycle())
    rows = loop.store._db.execute("SELECT claim_id, source_url FROM claim").fetchall()
    remember({r["claim_id"]: r["source_url"] for r in rows if r["claim_id"]})
    print(f"[miner] one cycle: {sent} claims sent by {miner.wallet.hotkey.ss58_address[:10]}..")
    kept = loop.store.receipts()
    print(f"[miner] {len(kept)} signed receipts kept")
    if len(kept) != sum(1 for r in rows if r["claim_id"]):
        sys.exit("[miner] FAIL a sent claim has no checked receipt")


def event_of(page: str) -> str:
    claims = json.loads((E2E_DIR / "claims_verified.json").read_text())
    found = {i["event_id"] for i in claims.values() if page_of(i) == page and i["event_id"]}
    if len(found) != 1:
        raise SystemExit(f"[second] no single event for the {page} page: {found}")
    return str(found.pop())


def raw_claim(key: str):
    from alpharidge_ai.schema import TYPE_SPEC, Actor, Claim, Fact, headline
    kind, event_type, category, qid, name, facts = RAW[key]
    role = next(iter(TYPE_SPEC[event_type].actor_roles.values()), None)
    actor = Actor(id_scheme="wikidata", id=qid, role=role, name=name)
    return Claim(kind=kind, headline=headline(event_type, [name]), event_type=event_type,
                 category=category, actors=[actor],
                 facts=[Fact(field=f, value=v, unit=u) for f, v, u in facts],
                 event_id=event_of(ATTACH[key]) if key in ATTACH else None,
                 event_date=datetime.now(timezone.utc).date(), source_url=PAGE[key].url,
                 quote=PAGE[key].paragraphs[0] if kind == "event" else None)


def cmd_submit(args) -> None:
    async def go():
        api = client("second")
        try:
            r = await api.submit_claim(raw_claim(args.page))
        finally:
            await api.close()
        remember({str(r["id"]): PAGE[args.page].url})
        kind, event_type = RAW[args.page][:2]
        print(f"[second] sent {kind} {event_type} id={r.get('id')} {PAGE[args.page].url}")
    asyncio.run(go())


def cmd_flood(_args) -> None:
    """Send distinct claims until the API answers 429, then check nothing more was stored."""
    from alpharidge_ai.ledger.api_client import AlpharidgeAPIError
    from alpharidge_ai.schema import Actor, Claim, Fact, headline
    flood = hotkeys()["flood"]

    def claim(i: int):
        name = "Vandelay Industries"
        return Claim(kind="event", headline=headline("layoffs", [name]), event_type="layoffs",
                     category="finance", actors=[Actor(id_scheme="wikidata", id="Q99999008",
                                                       name=name)],
                     facts=[Fact(field="jobs_cut", value=1000.0, unit="count")],
                     event_date=datetime.now(timezone.utc).date(),
                     source_url=f"{site('blog')}/posts/vandelay-{i}",
                     quote="Vandelay Industries announced layoffs that eliminate 1,000 jobs.")

    async def go():
        api, sent, refused = client("flood"), 0, []
        try:
            for i in range(200):
                try:
                    await api.submit_claim(claim(i))
                    sent += 1
                except AlpharidgeAPIError as e:
                    refused.append((e.status_code, e.detail))
                    if len(refused) == 2:
                        break
        finally:
            await api.close()
        stored = sum(1 for i in await items() if i["hotkey"] == flood)
        return sent, refused, stored

    sent, refused, stored = asyncio.run(go())
    (E2E_DIR / "flood.json").write_text(json.dumps(
        {"sent": sent, "refused": refused, "stored": stored}))
    print(f"[flood] {sent} claims accepted, then {refused}; {stored} stored")


# ---- watching claims --------------------------------------------------------------------

async def items() -> List[dict]:
    """Every claim with its source and reason: the operator sees them before settle."""
    api = client("validator")
    try:
        return (await api._request("GET", "/dashboard/items",
                                   params={"limit": DASHBOARD_LIMIT}))["items"]
    finally:
        await api.close()


async def snapshot(api) -> Dict[str, dict]:
    out, urls, flood = {}, sent_urls(), hotkeys().get("flood")
    for item in await items():
        if item["hotkey"] == flood:
            continue
        status = await api.claim_status(item["id"])
        out[item["id"]] = {**item, "novelty": status.get("novelty"),
                           "settle_at": status.get("settle_at"),
                           "source_url": urls.get(item["id"])}
    return out


def describe(item: dict) -> str:
    pts = item["points"]
    return (f"{item['status']:<11} novelty={item.get('novelty') or '-':<7} "
            f"route={item.get('route') or '-':<6} reason={item.get('reason') or '-':<12} "
            f"trust={item.get('trust_band') or '-':<9} "
            f"pts={pts['provisional']:.3f}+{pts['settled']:.3f}")


def settled_in_full(item: dict) -> bool:
    from alpharidge_ai.mechanism.pools import PROVISIONAL_SHARE
    pts = item["points"]
    share = (1 - PROVISIONAL_SHARE) / PROVISIONAL_SHARE
    return (item["status"] != "settled" or pts["provisional"] <= 0
            or abs(pts["settled"] - share * pts["provisional"]) < 1e-3)


def cmd_watch(args) -> None:
    done = {"verified": lambda i: i["status"] in TERMINAL,
            "settled": lambda i: i["status"] in TERMINAL - {"provisional"}
            and settled_in_full(i)}[args.until]

    async def go():
        api = client("validator")
        seen: Dict[str, str] = {}
        start = time.monotonic()
        try:
            while True:
                snap = await snapshot(api)
                for cid, item in sorted(snap.items(), key=lambda kv: int(kv[0])):
                    line = describe(item)
                    if seen.get(cid) != line:
                        seen[cid] = line
                        print(f"[watch +{time.monotonic() - start:5.1f}s] claim {cid:>3} "
                              f"{item['source_domain']:<22} {line}", flush=True)
                waiting = [c for c, i in snap.items() if not done(i)]
                if snap and not waiting:
                    return snap
                if time.monotonic() - start > args.timeout:
                    raise SystemExit(f"[watch] timed out waiting for {args.until}: {waiting}")
                await asyncio.sleep(0.25)
        finally:
            await api.close()

    snap = asyncio.run(go())
    (E2E_DIR / f"claims_{args.until}.json").write_text(json.dumps(snap, indent=1))
    print(f"[watch] path of each claim, at {args.until}:")
    for cid, item in sorted(snap.items(), key=lambda kv: int(kv[0])):
        print(f"    claim {cid:>3} {trace(item)}")


def trace(item: dict) -> str:
    """received -> novelty -> fetch and check -> verified or rejected -> ledger."""
    steps = [f"received {item['received_at'][11:23]}"]
    if item.get("novelty"):
        steps.append(f"novelty {item['novelty']}" + (f" -> {item['route']}" if item["route"]
                                                     else ""))
    check = item.get("check") or {}
    if check:
        flags = [name for k, name in (("date_gate", "date ok"), ("contrastive", "contrastive"))
                 if check.get(k)]
        if check.get("judge"):
            flags.append(f"judge {check['judge']}")
        steps.append(f"fetch+check {check.get('verdict')}"
                     + (f" ({', '.join(flags)})" if flags else "")
                     + (f" {check['reason']}" if check.get("reason") else ""))
    steps.append(item["status"] + (f"({item['reason']})" if item.get("reason") else ""))
    pts = item["points"]
    if pts["provisional"] or pts["settled"]:
        steps.append(f"ledger 40% {pts['provisional']:.3f}"
                     + (f" + 60% {pts['settled']:.3f}" if pts["settled"] else ""))
    page = page_of(item)
    return f"{page:<9} " + " -> ".join(steps)


# ---- validator -------------------------------------------------------------------------

def validator():
    sys.argv = neuron_argv("validator", "validator")
    from alpharidge_ai import config
    from neurons.validator import Validator
    # The mock chain's burn UID and this run's ledger key.
    config.BURN_UID = int(os.environ["BURN_UID"])
    config.API_ATTESTATION_PUBKEY = os.environ["API_ATTESTATION_PUBKEY"]
    return Validator()


def head_epoch(v) -> int:
    from alpharidge_ai.ledger.attestation import BLOCK_LENGTH, CLOSE_GRACE_BLOCKS
    head = int(v.subtensor.get_current_block()) - CLOSE_GRACE_BLOCKS
    return head // BLOCK_LENGTH


def wait_closed(v, epoch: int) -> None:
    while head_epoch(v) <= epoch:
        time.sleep(0.5)


async def leaf_epochs(api, lo: int, hi: int, kind: Optional[str] = None) -> List[int]:
    out = []
    for e in range(lo, hi + 1):
        leaves = (await api.ledger_leaves(e)).get("leaves") or []
        if any(kind is None or leaf["validator_verdict"] == kind for leaf in leaves):
            out.append(e)
    return out


def cmd_judge(args) -> None:
    """A listed validator's page judging, until stopped: pull, fetch, check, judge, post."""
    from alpharidge_ai.judge.client import from_env
    from alpharidge_ai.validator.judge import JudgeWorker
    from alpharidge_ai.validator.reverify import fetch_config
    from alpharidge_ai.verify.fetch import Fetcher

    worker = JudgeWorker(client(args.role), from_env(os.environ), Fetcher(fetch_config()))

    async def go():
        while True:
            try:
                got = await worker.run()
                if got:
                    print(f"[judge {args.role}] {got}", flush=True)
            except Exception as e:
                print(f"[judge {args.role}] {e!r}", flush=True)
            await asyncio.sleep(worker.interval)

    asyncio.run(go())


def cmd_reverify(args) -> None:
    from alpharidge_ai.validator.reverify import keyed
    v = validator()
    ledger = v._ledger
    last = head_epoch(v)
    wait_closed(v, last)

    async def go():
        epochs = await leaf_epochs(v._api, max(0, last - args.lookback), last, "provisional")
        totals: Dict[str, int] = {}
        rv = ledger.reverifier
        for e in epochs:
            kept = keyed(await rv.candidates(e), rv.selector, rv.rate)
            counts = await rv.run(e)
            print(f"[validator] epoch {e}: keyed sample at rate {rv.rate:.2f} kept claims "
                  f"{sorted(int(i['claim_id']) for i in kept)}; re-fetched and checked: {counts}")
            for k, n in counts.items():
                totals[k] = totals.get(k, 0) + n
        # Every paid scoop, sampled or not, checked on the page that confirmed it.
        from alpharidge_ai.validator.reverify import reverify
        cands = [i for e in epochs for i in await rv.candidates(e)]
        scoops = [await reverify(i, rv.fetcher, rv.judges) for i in cands
                  if i.get("route") == "scoop"]
        await v._api.close()
        return epochs, totals, scoops, cands

    epochs, totals, scoops, cands = v.loop.run_until_complete(go())
    print(f"[validator] scoops re-verified on their confirming page: {scoops}")
    (E2E_DIR / "reverify.json").write_text(json.dumps({"epochs": epochs, "totals": totals,
                                                       "scoops": scoops,
                                                       "panel": panel_of(cands)}))


def panel_of(cands: List[dict]) -> Optional[dict]:
    """A paid claim's judge record: who posted, each one's event probability, the mean."""
    for item in cands:
        record = item.get("judge") or {}
        posts = record.get("posts") or []
        if len(posts) < 2:
            continue
        bodies = [json.loads(p["body"]) for p in posts]
        roles = {hk: r for r, hk in hotkeys().items()}
        each = {roles.get(p["validator"], "?"): b["answers"]["event"]["probabilities"]["yes"]
                for p, b in zip(posts, bodies)}
        return {"claim_id": item["claim_id"], "posted": each,
                "score": record["scores"]["event"], "outcome": record["outcome"],
                "answers": record["answers"]}
    return None


def cmd_scan(args) -> None:
    """Validator scan rounds until no scoop is held any more."""
    v = validator()

    async def held() -> List[str]:
        return [i["id"] for i in await items() if i["route"] == "scoop"
                and i["status"] == "verified"]

    async def go():
        start, rounds = time.monotonic(), []
        try:
            while await held():
                if time.monotonic() - start > args.timeout:
                    raise SystemExit(f"[scan] scoops still held: {await held()}")
                rounds.append(await v._scan.run())
                await asyncio.sleep(1.0)
        finally:
            await v._api.close()
        return rounds

    rounds = v.loop.run_until_complete(go())
    print(f"[scan] {len(rounds)} rounds: {rounds}")


def cmd_weights(_args) -> None:
    import numpy as np
    v = validator()
    ledger = v._ledger
    epoch = head_epoch(v)
    wait_closed(v, epoch)

    async def go():
        weights = await ledger.weights(epoch)
        body = await v._api.ledger_points(epoch)
        await v._api.close()
        return weights, body

    weights, body = v.loop.run_until_complete(go())
    if weights is None:
        raise SystemExit(f"[validator] epoch {epoch}: signed points rejected")
    v.update_scores(weights, v.metagraph.uids.tolist())
    v.set_weights()
    state = v.subtensor.chain_state["SubtensorModule"]["Weights"][NETUID][v.uid]
    on_chain = dict(v.subtensor._get_most_recent_storage(state) or [])
    chain = list(v.metagraph.hotkeys)
    roles = {hk: r for r, hk in hotkeys().items()}
    print(f"[validator] epoch {epoch} weight vector (uid, role, weight, set on the mock chain)")
    for uid, w in enumerate(np.asarray(weights).tolist()):
        if w > 0 or chain[uid] in roles:
            print(f"    uid {uid:>2} {roles.get(chain[uid], '?'):<9} {w:.6f}  "
                  f"u16={on_chain.get(uid, 0)}")
    (E2E_DIR / "weights.json").write_text(json.dumps({
        "epoch": epoch, "weights": np.asarray(weights).tolist(), "hotkeys": chain,
        "on_chain": {str(k): w for k, w in on_chain.items()},
        "burn_uid": ledger.burn_uid, "points": body["attestation"]["perMinerPoints"],
        "capacity": body["capacity"]}, indent=1))


# ---- expected outcomes -----------------------------------------------------------------

def cmd_check(_args) -> None:
    claims = json.loads((E2E_DIR / "claims_settled.json").read_text())
    w = json.loads((E2E_DIR / "weights.json").read_text())
    keys = hotkeys()
    failures: List[str] = []

    def expect(ok: bool, what: str) -> None:
        print(f"[check] {'ok  ' if ok else 'FAIL'} {what}")
        if not ok:
            failures.append(what)

    pages = {cid: page_of(item) for cid, item in claims.items()}
    of = lambda key: [claims[c] for c, k in pages.items() if k == key]  # noqa: E731

    def paid(item) -> float:
        return item["points"]["provisional"] + item["points"]["settled"]

    events: Dict[str, List[dict]] = {}
    for item in claims.values():
        if item["route"] in ("event", "scoop") and paid(item) > 0:
            events.setdefault(item["event_id"], []).append(item)
    expect(all(len(v) == 1 for v in events.values()) and len(events) >= 3,
           f"one event prize per event ({len(events)} events, "
           f"{sum(len(v) for v in events.values())} prizes)")

    # The deal's two pages: first by receipt bucket, then by the bucket's tie draw.
    pair = of("deal") + of("duplicate")
    deal = [i for i in pair if paid(i) > 0]
    dup = [i for i in pair if paid(i) == 0]
    expect(len(pair) == 2 and len(deal) == 1 and deal[0]["route"] == "event"
           and deal[0]["status"] == "settled",
           "the first trusted report of the deal takes the event prize and settles")
    expect(len(dup) == 1 and dup[0]["event_id"] == (deal[0]["event_id"] if deal else None)
           and dup[0]["status"] == "rejected",
           f"the same deal reported twice in one cycle: no second prize "
           f"({dup[0]['reason'] if dup else 'missing'})")
    known = of("known")
    expect(len(known) == 1 and known[0]["novelty"] == "known" and known[0]["status"] == "rejected"
           and paid(known[0]) == 0,
           f"a re-report of a known event is known ({known[0]['reason'] if known else 'missing'})")
    expect(all(i["route"] in ("event", "scoop") for i in claims.values() if paid(i) > 0),
           "only first reports of an event are paid")
    neg = of("negated")
    expect(all(i["status"] == "rejected" and paid(i) == 0 for i in neg)
           and any(i["reason"] == "judge_reject" for i in neg),
           f"claims on the denial page are rejected by the page judge "
           f"({[i['reason'] for i in neg] or 'missing'})")
    scoop = of("scoop")
    expect(len(scoop) == 1 and scoop[0]["status"] == "settled" and scoop[0]["route"] == "scoop"
           and paid(scoop[0]) > 0,
           f"the low-trust first report is paid as a scoop once the scan finds a later "
           f"trusted page ({scoop[0]['status'] + '/' + str(scoop[0]['reason']) if scoop else 'missing'})")
    confirm = of("confirm")
    expect(len(confirm) == 1 and confirm[0]["status"] == "rejected"
           and confirm[0]["reason"] == "known_event" and paid(confirm[0]) == 0,
           f"the later trusted report waits behind the scoop and is then known "
           f"({confirm[0]['reason'] if confirm else 'missing'})")
    scoop2, early = of("scoop2"), of("early")
    expect(len(scoop2) == 1 and len(scoop) == 1 and scoop2[0]["status"] == "settled"
           and scoop2[0]["route"] == "scoop" and paid(scoop2[0]) > 0
           and abs(SCOOP_PREMIUM * paid(scoop2[0]) - paid(scoop[0])) < 1e-6,
           f"a scoop confirmed by a trusted page published before its receipt is paid the "
           f"tier price, a fifth of the later-confirmed scoop "
           f"({scoop2[0]['status'] + '/' + str(paid(scoop2[0])) if scoop2 else 'missing'})")
    expect(len(early) == 1 and early[0]["status"] == "rejected"
           and early[0]["reason"] == "known_event" and paid(early[0]) == 0,
           f"the trusted report held behind it is then known "
           f"({early[0]['status'] + '/' + str(early[0]['reason']) if early else 'missing'})")
    expect(all(i["status"] != "provisional" and settled_in_full(i) for i in claims.values()),
           "every paid claim settled its 60% share")

    # Weights follow the signed points: settle() share per hotkey, the rest burned.
    pts = w["points"]
    total = sum(max(0.0, v) for v in pts.values())
    denom = max(total, float(w["capacity"]))
    uid = {hk: i for i, hk in enumerate(w["hotkeys"])}
    ok = all(abs(w["weights"][uid[hk]] - max(0.0, p) / denom) < 1e-5 for hk, p in pts.items())
    burn = w["weights"][w["burn_uid"]]
    ok = ok and abs(burn - (1 - sum(max(0.0, p) for p in pts.values()) / denom)) < 1e-5
    chain = {int(k): v for k, v in w["on_chain"].items()}
    top = max(chain.values(), default=0)
    ok = ok and top > 0 and all(abs(chain.get(i, 0) / top - x / max(w["weights"])) < 1e-3
                                for i, x in enumerate(w["weights"]))
    expect(ok and abs(sum(w["weights"]) - 1.0) < 1e-4,
           f"weights match the signed points (W={total:.3f}, C={w['capacity']:.0f}, "
           f"burn={burn:.4f})")
    miner_pts = sum(p for hk, p in pts.items() if hk == keys["miner"])
    ledger_pts = sum(paid(i) for i in claims.values() if i["hotkey"] == keys["miner"])
    expect(abs(miner_pts - ledger_pts) < 1e-3,
           f"reference miner's signed points equal its ledger rows ({miner_pts:.3f})")
    expect(keys["second"] not in pts or pts[keys["second"]] == 0,
           "the second hotkey's rejected claim earns nothing")
    rv = json.loads((E2E_DIR / "reverify.json").read_text())
    expect(rv["totals"].get("agree", 0) > 0 and rv["totals"].get("disagree", 0) == 0,
           f"validator re-verify agrees with the API ({rv['totals']})")
    expect(len(rv["scoops"]) == 2 and all(r["verdict"] == "agree" for r in rv["scoops"]),
           f"re-verify agrees with both paid scoops on their confirming pages ({rv['scoops']})")
    pn = rv.get("panel") or {}
    posted = pn.get("posted") or {}
    expect(set(posted) == {"validator", "validator2"} and pn.get("outcome") == "pass"
           and abs(pn.get("score", 0) - sum(posted.values()) / 2) < 1e-9,
           f"a paid claim judged by both validators carries the mean of their answers ({pn})")

    deal_event = deal[0]["event_id"] if deal else None
    detail = asyncio.run(event_detail(deal_event)) if deal_event else {}
    stage = of("stage")
    expect(not stage and detail.get("stage") == "announced",
           f"the reference miner sends nothing for a later stage of the known deal "
           f"({len(stage)} claims, deal {detail.get('stage') or 'missing'})")
    dispute = of("dispute")
    expect(len(dispute) == 1 and dispute[0]["status"] == "rejected"
           and dispute[0]["reason"] == "kind_off" and paid(dispute[0]) == 0,
           f"a fact claim on the known deal is turned away unchecked "
           f"({dispute[0]['reason'] if dispute else 'missing'})")
    excluded = of("excluded")
    expect(len(excluded) == 1 and excluded[0]["status"] == "rejected"
           and excluded[0]["reason"] == "page_type" and paid(excluded[0]) == 0,
           f"a claim on an excluded page type is rejected "
           f"({excluded[0]['reason'] if excluded else 'missing'})")

    off = of("off")
    expect(len(off) == 1 and off[0]["status"] == "rejected" and off[0]["reason"] == "type_off"
           and paid(off[0]) == 0,
           f"a claim of a type that is off is rejected "
           f"({off[0]['reason'] if off else 'missing'})")
    hedged = of("hedged")
    expect(len(hedged) == 1 and paid(hedged[0]) == 0 and not (
        hedged[0]["route"] == "event" and hedged[0]["status"] in ("provisional", "settled")),
           f"a hedged claim is not paid as an event "
           f"({hedged[0]['status'] + '/' + str(hedged[0]['reason']) if hedged else 'missing'})")
    pre = of("prerelease")
    calendar = "/calendar.ics" in (E2E_DIR / "sites.log").read_text()
    expect(calendar and len(pre) == 1 and pre[0]["status"] == "rejected"
           and pre[0]["reason"] == "pre_release" and paid(pre[0]) == 0,
           f"a scheduled release sent before its official release time is not paid "
           f"({pre[0]['status'] + '/' + str(pre[0]['reason']) if pre else 'missing'})")
    fl = json.loads((E2E_DIR / "flood.json").read_text())
    limit = int(os.getenv("EL_CHECK_BUDGET_BASE", "200"))
    expect(fl["sent"] == limit and fl["stored"] == limit and len(fl["refused"]) == 2
           and all(c == 429 and "budget_exhausted" in str(d) for c, d in fl["refused"]),
           f"a hotkey over its budget gets 429 and nothing more is stored "
           f"({fl['sent']} accepted, {fl['stored']} stored)")

    if failures:
        raise SystemExit(f"[check] {len(failures)} failed")
    print("[check] all expected outcomes hold")


def cmd_wrong(_args) -> None:
    """A correction of a paid claim claws it; a retraction of a trusted claim
    claws it without a points penalty and marks the event."""
    claims = json.loads((E2E_DIR / "claims_settled.json").read_text())
    by_page = {page_of(i): i for i in claims.values()}
    low = by_page.get("scoop2")
    trusted = next((i for i in claims.values() if page_of(i) in ("deal", "duplicate")
                    and i["status"] == "settled"), None)
    if not (low and trusted):
        raise SystemExit("[wrong] scoop2 or deal claim missing")

    async def go():
        api = client("validator")
        try:
            sent = []
            for item, body in ((low, {"kind": "correction", "slot": "actor",
                                      "reason": "later report names another company"}),
                               (trusted, {"kind": "retraction", "reason": "story withdrawn"})):
                sent.append(await api._request("POST", f"/ops/claims/{item['id']}/retract",
                                               json=body))
            rows = (await api._request("GET", "/ops/wrong"))["rows"]
            marks = [(await api.event_detail(str(i["event_id"]))).get("marked")
                     for i in (low, trusted)]
        finally:
            await api.close()
        return sent, rows, marks

    sent, rows, marks = asyncio.run(go())
    band = {(r["hotkey"], r["band"]): r for r in rows}
    lo = band.get((low["hotkey"], low["trust_band"]), {})
    hi = band.get((trusted["hotkey"], trusted["trust_band"]), {})
    print(f"[wrong] correction {sent[0]}, retraction {sent[1]}; marks {marks}")
    for r in (lo, hi):
        print(f"[wrong] {r.get('band')}: wrong={r.get('wrong')} earned={r.get('earned')} "
              f"clawed={r.get('clawed')} penalty={r.get('penalty')}")
    checks = [
        (sent[0]["applied"] and sent[0]["status"] == "clawed_back" and lo.get("wrong", 0) > 0
         and lo.get("clawed", 0) > 0 and lo.get("penalty") == 0 and marks[0] == "corrected",
         "a paid claim shown wrong is clawed, with no penalty"),
        (trusted["trust_band"] in ("prize-eligible", "primary") and sent[1]["applied"]
         and sent[1]["status"] == "clawed_back" and hi.get("clawed", 0) > 0
         and hi.get("penalty") == 0 and marks[1] == "retracted",
         "a trusted claim later retracted is clawed without a penalty and the event is marked"),
    ]
    for ok, what in checks:
        print(f"[check] {'ok  ' if ok else 'FAIL'} {what}")
    if not all(ok for ok, _ in checks):
        raise SystemExit("[wrong] failed")


async def event_detail(event_id) -> dict:
    api = client("validator")
    try:
        return await api.event_detail(str(event_id))
    finally:
        await api.close()


def page_of(item: dict) -> str:
    host = item["source_domain"].removesuffix(".localhost")
    url = item.get("source_url") or ""
    for p in PAGES:
        if p.host == host and url.endswith(p.path):
            return p.key
    return "?"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("keys")
    sub.add_parser("sites")
    sub.add_parser("judge-script")
    sub.add_parser("scan-feeds")
    sub.add_parser("phase").add_argument("phase", type=int)
    sub.add_parser("miner")
    sub.add_parser("submit").add_argument("page", choices=sorted(RAW))
    w = sub.add_parser("watch")
    w.add_argument("--until", choices=("verified", "settled"), default="verified")
    w.add_argument("--timeout", type=float, default=180)
    r = sub.add_parser("reverify")
    r.add_argument("--lookback", type=int, default=30)
    sub.add_parser("weights")
    sub.add_parser("judge").add_argument("role", choices=("validator", "validator2"))
    sc = sub.add_parser("scan")
    sc.add_argument("--timeout", type=float, default=120)
    sub.add_parser("flood")
    sub.add_parser("check")
    sub.add_parser("wrong")
    args = p.parse_args()
    E2E_DIR.mkdir(parents=True, exist_ok=True)
    globals()[f"cmd_{args.cmd.replace('-', '_')}"](args)


if __name__ == "__main__":
    main()
