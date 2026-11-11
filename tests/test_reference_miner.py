"""Reference miner: feeds, claim building from a saved page, and no resubmission."""

import asyncio
import json
import threading
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from alpharidge_ai.miner.build import ClaimBuilder, sentences
from alpharidge_ai.miner.discover import Discoverer, Source, parse_feed
from alpharidge_ai.miner.link import Linker, load_entities
from alpharidge_ai.miner.loop import MinerLoop
from alpharidge_ai.miner.store import MinerStore
from alpharidge_ai.schema import EventType, Stage, canonical_body
from alpharidge_ai.verify.fetch import FetchConfig, Fetcher, read_page

FIXTURES = Path(__file__).parent / "fixtures" / "miner"
PAGE = (FIXTURES / "boe_rates.html").read_bytes()
FEED = (FIXTURES / "feed.xml").read_bytes()
NOW = datetime(2026, 10, 13, 15, 0, tzinfo=timezone.utc)
URL = "https://news.example.org/boe-rates"

EXPECTED = {
    "actors": [{"id": "Q183231", "id_scheme": "wikidata", "name": "Bank of England",
                "role": "central_bank"}],
    "event_date": "2026-10-13", "event_type": "monetary_policy",
    "facts": [{"field": "change_bps", "unit": "basis_points", "value": -25.0},
              {"field": "new_rate", "unit": "percent", "value": 4.5}],
    "headline": "Bank of England: rate decision", "kind": "event", "place": {"iso2": "GB"},
    "source_url": URL,
    "quote": "The Bank of England cut its benchmark interest rate by 25 basis points to 4.5% on "
             "Tuesday, the central bank said in a statement after its policy meeting.",
}


@pytest.fixture(scope="module")
def linker():
    return Linker(load_entities() + load_entities(FIXTURES.parent / "entities.json"))


def _page(html: bytes = PAGE, url: str = URL):
    return read_page(url, url, 200, html, html.decode(), FetchConfig())


def test_claim_from_saved_page(linker):
    built = ClaimBuilder(linker).build(_page(), NOW)
    assert len(built) == 1
    assert json.loads(canonical_body(built[0].claim)) == EXPECTED


def test_sentences_split_release_layout():
    text = ("Acme Reports Results \u2022 Revenue was $5 billion \u2022 Net income rose 2% "
            "SEATTLE, October 1, 2026\u2014Acme Inc. (NYSE: ACME) today reported results. (Wire) It "
            "rose.")
    assert sentences(text) == ["Acme Reports Results", "Revenue was $5 billion",
                               "Net income rose 2% SEATTLE, October 1, 2026",
                               "Acme Inc. (NYSE: ACME) today reported results.", "(Wire) It rose."]


def test_stale_or_unlinked_pages_give_no_claim(linker):
    builder = ClaimBuilder(linker)
    assert builder.build(_page(), NOW + timedelta(days=31)) == []
    unknown = PAGE.replace(b"Bank of England", b"Example Bank")
    assert builder.build(_page(unknown), NOW) == []
    assert ClaimBuilder(linker, categories=["sports"]).build(_page(), NOW) == []


def _article(title: str, body: str, url: str = "https://news.example.org/a") -> bytes:
    return (f'<!doctype html><html lang="en"><head><title>{title}</title>'
            '<meta property="article:published_time" content="2026-10-13T11:00:00Z">'
            f"</head><body><article><h1>{title}</h1><p>{body}</p></article></body></html>"
            ).encode()


def test_halted_company_is_the_actor(linker):
    page = _article("Nasdaq halts trading in Acme", "Nasdaq halted trading in Acme Inc. shares "
                    "on Tuesday pending news, the exchange said in a notice to members.")
    (built,) = ClaimBuilder(linker).build(_page(page, "https://news.example.org/a"), NOW)
    claim = built.claim
    assert claim.event_type.value == "trading_halt"
    assert [(a.id, a.role) for a in claim.actors] == [("Q99999001", "subject")]
    assert [(f.field, f.value) for f in claim.facts] == [("exchange", "Nasdaq")]


def test_a_listing_exchange_is_not_a_second_company(linker):
    for body in ("Nasdaq-listed Acme Inc. reported third-quarter revenue of $94.9 billion on "
                 "Tuesday, up 6% from a year earlier.",
                 "Acme (Nasdaq: ACME) reported third-quarter revenue of $94.9 billion on "
                 "Tuesday, up 6% from a year earlier."):
        page = _article("Acme results", body)
        (built,) = ClaimBuilder(linker).build(_page(page), NOW)
        assert built.claim.event_type.value == "earnings"
        assert [a.id for a in built.claim.actors] == ["Q99999001"]


def test_misreads_give_no_wrong_claim(linker):
    def build(title, body):
        return [([(f.field, f.value) for f in b.claim.facts], b.claim.event_type.value)
                for b in ClaimBuilder(linker).build(_page(_article(title, body)), NOW)]

    assert build("Bank of Canada raises rates", "The Bank of Canada raised its policy rate by 25 "
                 "basis points to 2.5 per cent on Tuesday.") == \
        [([("change_bps", 25.0), ("new_rate", 2.5)], "monetary_policy")]
    assert build("Cyberdyne Systems shares rise", "Shares of Nasdaq-listed Cyberdyne Systems rose 3% on "
                 "Tuesday after the company reported results.") == []
    assert build("Tyrell Corp. moves listing", "Tyrell Corp. said on Tuesday it will "
                 "transfer its stock listing from the NYSE to Nasdaq.") == []
    assert build("Stark Motors earnings preview", "Stark Motors is set to announce third-quarter earnings "
                 "after the markets close Tuesday. Stark Motors' 2025 third-quarter results included "
                 "$47.19 billion in automotive revenue and net income of $2.4 billion.") == []
    assert build("Initech offers buyouts", "Initech is offering voluntary buyouts to about "
                 "60,000 employees, roughly a quarter of its workforce, the company said on "
                 "Tuesday.") == []
    assert build("Initech cuts jobs", "Initech said on Tuesday it will cut 6,000 jobs "
                 "across its workforce.") == [([("jobs_cut", 6000.0)], "layoffs")]


def test_place_only_when_one_country_is_named(linker):
    builder = ClaimBuilder(linker)
    assert builder.place(EventType.MONETARY_POLICY, ["Rates rose in Japan."]).iso2 == "JP"
    assert builder.place(EventType.MONETARY_POLICY, ["Japan and Canada both moved."]) is None
    assert builder.place(EventType.EARNINGS, ["Japan sales rose."]) is None


def test_completion_page_reads_its_stage(linker):
    url = "https://news.example.org/deal"
    page = _page(_article("Initech completes deal", "Initech completed its "
                          "acquisition of Acme on Tuesday after the deal closed."), url)
    (built,) = ClaimBuilder(linker).build(page, NOW)
    assert built.claim.stage is Stage.COMPLETED


def test_stage_reads_base_form():
    from alpharidge_ai.miner.build import stage_of
    assert stage_of("Regulators approve the merger", EventType.MERGER_ACQUISITION) is Stage.APPROVED


def test_parse_rss_and_atom():
    rss = parse_feed(FEED.decode(), "https://news.example.org/feed", "finance")
    assert [i.url for i in rss] == [URL, "https://news.example.org/old"]
    assert rss[0].published == datetime(2026, 10, 13, 11, 0, tzinfo=timezone.utc)
    atom = ('<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>T</title>'
            '<link rel="alternate" href="https://a.example.org/x"/>'
            '<updated>2026-10-13T10:00:00Z</updated></entry></feed>')
    assert [i.url for i in parse_feed(atom)] == ["https://a.example.org/x"]
    assert parse_feed("<rss><broken") == []


class _Handler(BaseHTTPRequestHandler):
    pages = {}

    def do_GET(self):
        ctype, body = self.pages.get(self.path, ("text/plain", None))
        self.send_response(200 if body is not None else 404)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body or b"not found")

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def site():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    base = f"http://127.0.0.1:{server.server_port}"
    page = PAGE.replace(b"https://news.example.org/boe-rates", f"{base}/boe-rates".encode())
    _Handler.pages = {"/robots.txt": ("text/plain", b"User-agent: *\nAllow: /\n"),
                      "/feed.xml": ("application/rss+xml", FEED),
                      "/boe-rates": ("text/html; charset=utf-8", page)}
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield base
    server.shutdown()


class FakeAPI:
    def __init__(self, known=None):
        self.sent, self.known = [], known

    async def submit_claim(self, claim):
        self.sent.append(claim)
        return {"id": str(len(self.sent)), "status": "received"}

    async def claim_status(self, claim_id):
        return {"id": claim_id, "status": "verified", "points": {"provisional": 4.0}}

    async def list_events(self, **query):
        return {"items": [self.known] if self.known else [], "next_cursor": None}

    async def event_detail(self, event_id):
        return {"id": event_id, "facts": []}


def _run(site, api, store):
    async def go():
        fetcher = Fetcher(FetchConfig(public_only=False, domain_rps=1000, retry_delay_s=0))
        loop = MinerLoop(api, store, Discoverer([Source(f"{site}/feed.xml", "finance")], fetcher),
                         ClaimBuilder(Linker()), fetcher)
        try:
            return [await loop.tick(NOW), await loop.tick(NOW)]
        finally:
            await fetcher.aclose()
    return asyncio.run(go())


def test_loop_submits_once_and_tracks_outcome(site, tmp_path):
    api, store = FakeAPI(), MinerStore(str(tmp_path / "miner.sqlite"))
    assert _run(site, api, store) == [1, 0]
    assert [c.source_url for c in api.sent] == [f"{site}/boe-rates"]
    assert store.get("1")["status"] == "verified"
    assert store.sent(api.sent[0])

    # A fresh run on the same state doesn't resend the page.
    again = FakeAPI()
    assert _run(site, again, store) == [0, 0] and again.sent == []


def test_known_event_is_not_sent_again(site, tmp_path):
    api = FakeAPI(known={"id": "77", "type": "monetary_policy"})
    assert _run(site, api, MinerStore(str(tmp_path / "miner.sqlite")))[0] == 0
    assert api.sent == []


class BusyAPI(FakeAPI):
    calls = 0

    async def submit_claim(self, claim):
        from alpharidge_ai.ledger.api_client import AlpharidgeAPIError
        self.calls += 1
        raise AlpharidgeAPIError("busy", status_code=429, detail="budget_exhausted",
                                 retry_after=600)


def test_rate_limited_page_is_kept_for_later(site, tmp_path):
    store, busy = MinerStore(str(tmp_path / "miner.sqlite")), BusyAPI()
    assert _run(site, busy, store) == [0, 0] and busy.calls == 1
    assert not store.seen(f"{site}/boe-rates")
    api = FakeAPI()
    assert _run(site, api, store)[0] == 1
