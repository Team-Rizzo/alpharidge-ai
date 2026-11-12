"""fetch(): robots.txt, error pages, access walls and the publish date, on a local server."""

import asyncio
import ssl
import threading
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from alpharidge_ai.verify import official
from alpharidge_ai.verify.dates import parse_datetime, publish_date
from alpharidge_ai.verify.fetch import FetchConfig, Fetcher, Paced, authors, tls_context

ARTICLE = (Path(__file__).parent / "fixtures" / "pages" / "acme_deal.html").read_bytes()
WALL = (b'<html><head><script type="application/ld+json">{"@type":"NewsArticle",'
        b'"isAccessibleForFree":"False"}</script></head><body><p>Subscribe to read.</p>'
        b'</body></html>')
PAGES = {"/robots.txt": (200, b"User-agent: *\nDisallow: /private/\n"), "/a": (200, ARTICLE),
         "/private/a": (200, ARTICLE), "/wall": (200, WALL)}
REDIRECTS = {"/go/a": "/a", "/go/private": "/private/a"}


class _Handler(BaseHTTPRequestHandler):
    seen = []

    def do_GET(self):
        self.seen.append((self.path, self.headers.get("User-Agent"), self.headers.get("Cookie")))
        if self.path in REDIRECTS:
            self.send_response(302)
            self.send_header("Location", REDIRECTS[self.path])
            self.end_headers()
            return
        if self.path == "/slow":
            time.sleep(1.5)
            self.path = "/a"
        if self.path == "/drip":
            self.send_response(200)
            self.send_header("Content-Type", "application/rss+xml")
            self.end_headers()
            for _ in range(30):
                self.wfile.write(b"<item></item>")
                self.wfile.flush()
                time.sleep(0.1)
            return
        if self.path == "/flaky":
            self.send_response(503)
            self.end_headers()
            return
        status, body = PAGES.get(self.path, (404, b"not found"))
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Set-Cookie", "session=1")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def base():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def _fetch(urls, **cfg):
    async def run():
        f = Fetcher(FetchConfig(public_only=False, domain_rps=1000, retry_delay_s=0, **cfg))
        try:
            return [await f.fetch(u) for u in urls]
        finally:
            await f.aclose()
    return asyncio.run(run())


def test_fetch_reads_article_and_respects_robots(base):
    _Handler.seen.clear()
    ok, private, wall, missing = _fetch([f"{base}/a", f"{base}/private/a", f"{base}/wall",
                                         f"{base}/nope"])
    assert ok.ok and "Beta Systems" in ok.text and ok.status == 200
    assert ok.published_at == datetime(2026, 10, 13, 12, 0, tzinfo=timezone.utc)
    assert (private.reason, wall.reason, missing.reason) == ("blocked", "blocked", "fetch_failed")
    assert (ok.unread, private.unread, wall.unread, missing.unread) == (False, True, True, False)
    assert all(ua.startswith("AlphaRidgeBot/") and cookie is None for _, ua, cookie in _Handler.seen)
    assert [p for p, _, _ in _Handler.seen].count("/robots.txt") == 1


def test_robots_is_checked_after_redirects(base):
    _Handler.seen.clear()
    ok, private = _fetch([f"{base}/go/a", f"{base}/go/private"])
    assert ok.ok and ok.final_url == f"{base}/a"
    assert private.reason == "blocked"
    assert "/private/a" not in [p for p, _, _ in _Handler.seen]


def test_one_retry_for_server_errors(base):
    _Handler.seen.clear()
    (flaky,) = _fetch([f"{base}/flaky"])
    assert (flaky.reason, flaky.status, flaky.unread) == ("fetch_failed", 503, True)
    assert [p for p, _, _ in _Handler.seen].count("/flaky") == 2


def test_a_slow_answer_is_read_within_the_total_time_and_a_drip_is_cut_off(base):
    (slow,) = _fetch([f"{base}/slow"], timeout_s=3, connect_timeout_s=0.5)
    assert slow.ok and slow.text

    async def run():
        f = Fetcher(FetchConfig(public_only=False, domain_rps=1000, timeout_s=1))
        try:
            t = time.monotonic()
            return await f.get_text(f"{base}/drip"), time.monotonic() - t
        finally:
            await f.aclose()
    text, took = asyncio.run(run())
    assert text is None and took < 2


def test_tls_settings_match_urllib3():
    ctx = tls_context()
    assert ctx.options & ssl.OP_NO_TICKET and ctx.post_handshake_auth
    assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname


def test_busy_domain_raises_paced_instead_of_waiting(base):
    async def run():
        f = Fetcher(FetchConfig(public_only=False, domain_rps=1, pace_max_wait_s=1.2))
        try:
            await f.fetch(f"{base}/a")
            return await asyncio.gather(f.fetch(f"{base}/a"), f.fetch(f"{base}/a"),
                                        return_exceptions=True)
        finally:
            await f.aclose()
    ok, busy = asyncio.run(run())
    assert ok.ok and isinstance(busy, Paced) and 1.2 < busy.wait_s <= 2


def test_private_hosts_are_refused(base):
    async def run():
        f = Fetcher(FetchConfig())
        try:
            return await f.fetch(f"{base}/a")
        finally:
            await f.aclose()
    got = asyncio.run(run())
    assert got.reason == "blocked" and not got.unread


def test_parse_datetime_forms():
    want = datetime(2026, 10, 13, 12, 0, tzinfo=timezone.utc)
    assert parse_datetime("2026-10-13T12:00:00Z") == want
    assert parse_datetime("2026-10-13T14:00:00+0200") == want
    assert parse_datetime("2026-10-13T12:00:00.5Z") == want.replace(microsecond=500000)
    assert parse_datetime("Tue, 13 Oct 2026 12:00:00 GMT") == want
    assert parse_datetime("2026-10-13") == datetime(2026, 10, 13, tzinfo=timezone.utc)
    assert parse_datetime("soon") is None


def test_authors_from_ld_meta_and_links():
    html = ('<script type="application/ld+json">{"@type": "NewsArticle", "author": '
            '[{"@type": "Person", "name": "Ann Lee", "url": "https://x.example/people/ann"}, '
            '"Bo Chan"]}</script><meta name="author" content="Ann Lee">'
            '<a rel="author" href="/contributors/cy">Cy</a>')
    assert authors(html) == [{"name": "Ann Lee", "url": "https://x.example/people/ann"},
                             {"name": "Bo Chan"}, {"name": "Ann Lee"},
                             {"url": "/contributors/cy"}]
    assert authors("<p>no byline</p>") == []


def test_a_filing_is_published_at_its_acceptance_time():
    folder = "https://www.sec.gov/Archives/edgar/data/1023128/000102312826000057/"
    index = folder + "0001023128-26-000057-index.htm"
    accepted = '<div class="infoHead">Accepted</div>\n<div class="info">2026-10-15 17:33:31</div>'
    seen = []

    def site(request):
        seen.append(str(request.url))
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="")
        if str(request.url) == index:
            return httpx.Response(200 if indexed else 404, text=accepted)
        return httpx.Response(200, content=ARTICLE, headers={"Content-Type": "text/html"})

    async def run(*urls):
        f = Fetcher(FetchConfig(public_only=False, domain_rps=1000, retry_delay_s=0),
                    transport=httpx.MockTransport(site))
        try:
            return [await f.fetch(u) for u in urls]
        finally:
            await f.aclose()

    indexed = True
    pages = asyncio.run(run(folder + "ex99.htm", folder + "ex99-2.htm"))
    assert [p.published_at for p in pages] == [datetime(2026, 10, 15, 21, 33, 31,
                                                        tzinfo=timezone.utc)] * 2
    assert not pages[0].date_inferred and seen.count(index) == 1
    indexed = False
    assert asyncio.run(run(folder + "ex99.htm"))[0].reason == "fetch_failed"


def test_a_filing_index_without_an_acceptance_time_is_read_again():
    folder = "https://www.sec.gov/Archives/edgar/data/1023128/000102312826000057/"
    index = folder + "0001023128-26-000057-index.htm"
    accepted = '<div class="infoHead">Accepted</div>\n<div class="info">2026-10-15 17:33:31</div>'
    bodies = ["<html>Request Rate Threshold Exceeded</html>", accepted]

    def site(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="")
        if str(request.url) == index:
            return httpx.Response(200, text=bodies.pop(0))
        return httpx.Response(200, content=ARTICLE, headers={"Content-Type": "text/html"})

    async def run():
        f = Fetcher(FetchConfig(public_only=False, domain_rps=1000, retry_delay_s=0),
                    transport=httpx.MockTransport(site))
        try:
            return [await f.fetch(folder + "ex99.htm") for _ in range(2)]
        finally:
            await f.aclose()

    first, second = asyncio.run(run())
    assert first.reason == "fetch_failed"
    assert second.published_at == datetime(2026, 10, 15, 21, 33, 31, tzinfo=timezone.utc)


def test_a_busy_domain_never_leaves_a_filing_unread():
    import time
    folder = "https://www.sec.gov/Archives/edgar/data/1023128/000102312826000057/"
    accepted = '<div class="infoHead">Accepted</div>\n<div class="info">2026-10-15 17:33:31</div>'
    holder = []

    def site(request):
        if request.url.path.endswith("-index.htm"):
            return httpx.Response(200, text=accepted)
        if request.url.path != "/robots.txt":
            # Other fetches take every slot on the domain while the page is read.
            holder[0]._next["www.sec.gov"] = time.monotonic() + 60
        return httpx.Response(200, content=ARTICLE, headers={"Content-Type": "text/html"})

    async def run():
        f = Fetcher(FetchConfig(public_only=False, domain_rps=1000, pace_max_wait_s=1),
                    transport=httpx.MockTransport(site))
        holder.append(f)
        try:
            page = await f.fetch(folder + "ex99.htm")
            busy = await asyncio.gather(f.fetch(folder.replace("57/", "58/") + "ex99.htm"),
                                        return_exceptions=True)
            return page, busy[0]
        finally:
            await f.aclose()
    page, busy = asyncio.run(run())
    assert page.ok and page.published_at == datetime(2026, 10, 15, 21, 33, 31, tzinfo=timezone.utc)
    assert isinstance(busy, Paced)


def test_a_page_dated_ahead_of_the_host_clock_keeps_its_date(monkeypatch):
    import time
    monkeypatch.setenv("TZ", "Etc/GMT+12")
    time.tzset()
    try:
        day = (datetime.now(timezone.utc) + timedelta(hours=14)).date()
        html = (f"<html><body><article><h1>Prices rise</h1><p class='date'>{day:%B %d, %Y}</p>"
                "<p>Prices rose 0.4% in the month, the statistics office said.</p>"
                "<p>Last updated January 2, 2020</p></article></body></html>")
        assert publish_date(html)[0].date() == day
    finally:
        monkeypatch.delenv("TZ")
        time.tzset()


def test_an_official_release_is_dated_by_its_written_release_date():
    html = ("<html><body><div>EMBARGOED UNTIL RELEASE AT 8:30 a.m. EDT, Tuesday, October 6, "
            "2026</div><p>The trade deficit narrowed in August.</p>"
            "<p>Next release: November 4, 2026</p>"
            "<div class='date-info'>Page last modified on 10/6/26</div></body></html>")
    url = "https://www.bea.gov/news/2026/trade"
    assert publish_date(html, url) == (datetime(2026, 10, 6, tzinfo=timezone.utc), True)


def test_an_official_site_with_its_own_date_rule_is_dated_only_there():
    # Month-first after the label; the related item's date further down is not read.
    html = ("<html><body><h1>Designations</h1><div>Release Date</div><div>11/06/2026</div>"
            "<p>OFAC designated eight businesses.</p><p>Release Date 10/29/2026</p></body></html>")
    url = "https://ofac.treasury.gov/recent-actions/20261106"
    assert publish_date(html, url) == (datetime(2026, 11, 6, tzinfo=timezone.utc), False)
    html = ('<html><body><p class="news-article-header__date">Friday 6th November 2026</p>'
            "<p>Celtic beat Hearts 2-1 on 1 November 2026.</p></body></html>")
    assert publish_date(html, "https://spfl.co.uk/news/round-up")[0] == datetime(
        2026, 11, 6, tzinfo=timezone.utc)
    # Without its stated date the page has only a guess, which does not count there.
    html = "<html><body><p>OFAC designated eight businesses on 6 November 2026.</p></body></html>"
    dt, inferred = publish_date(html, url)
    assert (dt is None or inferred) and not official.body_dated(url)


def test_markup_scans_stay_fast_on_unclosed_tags():
    import time
    from alpharidge_ai.validator.scan import page_items
    from alpharidge_ai.verify.dates import ld_nodes, meta_tags, publish_date
    from alpharidge_ai.verify.fetch import access_wall, canonical_url, page_title
    for unit in ("<input ", "<a ", '<a rel="author" ', "<meta ", '<link rel="canonical" ',
                 "<script ", '<time itemprop="datePublished" ', "<title>", "<", '<a href="x">'):
        html = unit * (200_000 // len(unit))
        start = time.perf_counter()
        authors(html), page_title(html), canonical_url(html, "https://a.example/")
        access_wall(html, "https://a.example/x", "", FetchConfig())
        meta_tags(html), list(ld_nodes(html)), page_items(html, "https://a.example/")
        assert time.perf_counter() - start < 3, unit


def test_markup_scans_read_tags_like_the_plain_regex():
    from alpharidge_ai.verify.dates import _structured_date, meta_tags
    from alpharidge_ai.verify.fetch import access_wall, canonical_url, page_title
    html = ('<a <a class="x" rel="author" href="/by/jo">Jo</a><meta <meta name="author" '
            'content="Jo"><link rel="canonical" href="/c"><title>A  <b>t</b></title>'
            '<time class="d" itemprop="datePublished" datetime="2026-10-05T10:00:00Z">'
            '<input id=p type="password">')
    assert authors(html)[-1] == {"url": "/by/jo"}
    assert meta_tags(html) == {"author": "Jo"}
    assert canonical_url(html, "https://a.example/x") == "https://a.example/c"
    assert page_title(html) == "A <b>t</b>"
    assert _structured_date(html) == datetime(2026, 10, 5, 10, tzinfo=timezone.utc)
    assert access_wall(html, "https://a.example/x", "", FetchConfig())


def _mock_fetch(url, handler, **cfg):
    async def run():
        f = Fetcher(FetchConfig(public_only=False, domain_rps=1000, retries=0, **cfg),
                    transport=httpx.MockTransport(handler))
        try:
            return await f.fetch(url)
        finally:
            await f.aclose()
    return asyncio.run(run())


def _encoded(body, encodings, robots=b"User-agent: *\nAllow: /\n", robots_enc=()):
    import gzip
    import zlib

    def enc(data, codings):
        for c in codings:
            data = gzip.compress(data) if c == "gzip" else zlib.compress(data)
        return data

    def handler(request):
        if request.url.path == "/robots.txt":
            headers = {"Content-Encoding": ", ".join(robots_enc)} if robots_enc else {}
            return httpx.Response(200, stream=httpx.ByteStream(enc(robots, robots_enc)),
                                  headers=headers)
        headers = {"Content-Type": "text/html"}
        if encodings:
            headers["Content-Encoding"] = ", ".join(encodings)
        return httpx.Response(200, stream=httpx.ByteStream(enc(body, encodings)), headers=headers)
    return handler


def test_compressed_pages_are_capped_after_decoding():
    ok = _mock_fetch("https://a.example/a", _encoded(ARTICLE, ["gzip"]))
    assert ok.ok and ok.text
    assert _mock_fetch("https://a.example/a", _encoded(ARTICLE, ["deflate"])).ok
    bomb = b"<p>" + b"a" * 20_000_000
    assert _mock_fetch("https://a.example/a", _encoded(bomb, ["gzip"])).reason == "fetch_failed"
    assert _mock_fetch("https://a.example/a",
                       _encoded(ARTICLE, ["gzip", "gzip"])).reason == "fetch_failed"


def test_robots_txt_is_read_with_the_same_cap():
    big = b"User-agent: *\nAllow: /\n" + b"#" * 50_000_000
    handler = _encoded(ARTICLE, [], robots=big, robots_enc=("gzip",))
    assert _mock_fetch("https://a.example/a", handler).reason == "blocked"


@pytest.mark.parametrize("status,unread", [(401, True), (402, True), (403, True), (407, True),
                                           (451, True), (404, False),
                                           (410, False), (400, True), (418, True)])
def test_a_page_held_back_is_unread_and_a_missing_one_is_not(status, unread):
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        return httpx.Response(status, text="no", headers={"content-type": "text/html"})
    got = _mock_fetch("https://news.example.org/a", handler)
    assert not got.ok and got.unread is unread


@pytest.mark.parametrize("ctype,body", [("application/pdf", b"%PDF-1.4"),
                                        ("text/html", b"<p>x</p>" * 400_000),
                                        ("text/html", b"<html><body></body></html>")])
def test_a_body_with_no_text_or_too_big_is_unread(ctype, body):
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        return httpx.Response(200, content=body, headers={"content-type": ctype})
    got = _mock_fetch("https://news.example.org/a", handler)
    assert (got.ok, got.reason, got.unread) == (False, "fetch_failed", True)


def test_a_page_the_parser_refuses_is_a_failed_fetch():
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        return httpx.Response(200, content=b'<html><body><p>Beta<span style="display:none">x</span>\x01 Systems</p>'
                              b"</body></html>",
                              headers={"content-type": "text/html"})
    got = _mock_fetch("https://news.example.org/a", handler)
    assert (got.ok, got.reason, got.unread) == (False, "fetch_failed", False)


def test_an_unreadable_robots_txt_blocks_only_for_a_short_while():
    calls = []

    def handler(request):
        if request.url.path == "/robots.txt":
            calls.append(1)
            if len(calls) == 1:
                return httpx.Response(503, text="busy")
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        return httpx.Response(200, content=ARTICLE, headers={"content-type": "text/html"})

    async def go(retry_s):
        f = Fetcher(FetchConfig(public_only=False, retry_delay_s=0, domain_rps=1000,
                                robots_retry_s=retry_s), transport=httpx.MockTransport(handler))
        try:
            first = await f.fetch("https://news.example.org/a")
            second = await f.fetch("https://news.example.org/a")
        finally:
            await f.aclose()
        return first, second

    first, second = asyncio.run(go(0))
    assert (first.reason, first.blocked_by) == ("blocked", "robots") and second.ok
    calls.clear()
    first, second = asyncio.run(go(60))
    assert not first.ok and not second.ok and len(calls) == 1


def test_a_parse_that_never_finishes_counts_as_unread_and_restarts_the_workers(monkeypatch):
    from concurrent.futures import Executor, Future
    from alpharidge_ai.verify import fetch as fetch_mod

    class Stuck(Executor):
        restarts = 0

        def submit(self, fn, /, *args, **kwargs):
            return Future()

        def restart(self):
            Stuck.restarts += 1

    monkeypatch.setattr(fetch_mod, "_PARSE", Stuck())

    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        return httpx.Response(200, content=ARTICLE, headers={"content-type": "text/html"})
    got = _mock_fetch("https://news.example.org/a", handler, parse_timeout_s=0.05)
    assert (got.reason, got.unread, got.status) == ("fetch_failed", True, 200)
    assert Stuck.restarts == 1


def test_pages_parse_the_same_in_worker_processes():
    from dataclasses import asdict
    from alpharidge_ai.verify import parse_pool

    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        return httpx.Response(200, content=ARTICLE, headers={"content-type": "text/html"})
    here = asdict(_mock_fetch("https://news.example.org/a", handler))
    parse_pool.install(1)
    try:
        there = asdict(_mock_fetch("https://news.example.org/a", handler, parse_timeout_s=60))
        parse_pool._installed.restart()
        again = asdict(_mock_fetch("https://news.example.org/a", handler))
    finally:
        parse_pool.uninstall()
    assert there == here == again and here["text"]


def test_a_host_that_answers_429_is_left_alone_for_its_retry_after():
    from alpharidge_ai.verify.fetch import Paced

    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        return httpx.Response(429, text="slow down", headers={"retry-after": "30"})

    async def go():
        f = Fetcher(FetchConfig(public_only=False, retries=0, pace_max_wait_s=1,
                                host_rps=(("news.example.org", 1000),)),
                    transport=httpx.MockTransport(handler))
        try:
            with pytest.raises(Paced) as first:
                await f.fetch("https://news.example.org/a")
            with pytest.raises(Paced) as second:
                await f.fetch("https://news.example.org/b")
        finally:
            await f.aclose()
        return first.value.wait_s, second.value.wait_s

    first, second = asyncio.run(go())
    assert first == 30 and 28 < second <= 30


def test_a_host_rate_spaces_that_hosts_requests_only():
    from alpharidge_ai.verify.fetch import Paced

    async def go():
        f = Fetcher(FetchConfig(public_only=False, pace_max_wait_s=1, domain_rps=1000,
                                host_rps=(("slow.example", 0.2),)))
        await f._pace("slow.example")
        await f._pace("fast.example")
        await f._pace("fast.example")
        with pytest.raises(Paced):
            await f._pace("slow.example")
        await f.aclose()
    asyncio.run(go())
