"""Fetch a claim's source page: honest user agent, robots.txt, size and time limits.

A page that can't be read (HTTP error, robots.txt, paywall, login) comes back with a
`reason` and no text.
"""
from __future__ import annotations

import asyncio
import ipaddress
import os
import re
import socket
import ssl
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime
from html import unescape
from http.cookiejar import CookieJar, DefaultCookiePolicy
from typing import Dict, Iterable, List, Optional, Tuple, Union
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import httpcore
import httpx

from alpharidge_ai import __version__
from alpharidge_ai.verify import floor
from alpharidge_ai.verify.content_extractor import ContentExtractor, tidy
from alpharidge_ai.verify.dates import accepted_at, ld_nodes, meta_tags, publish_date
from alpharidge_ai.verify.markup import last_end, search_upto, tag_matches

FETCH_FAILED, BLOCKED = "fetch_failed", "blocked"
# The crawler's product token.
BOT_NAME = re.sub(r"[^A-Za-z0-9_-]", "", os.getenv("EL_BOT_NAME", "")) or "AlphaRidgeBot"

_TEXT_TYPES = ("text/html", "application/xhtml+xml", "text/plain")
_LOGIN_PATH = re.compile(r"/(?:login|log-in|signin|sign-in|subscribe|paywall|register)(?:[/?.]|$)")
_PASSWORD = re.compile(r"<input[^>]+type\s*=\s*[\"']?password", re.I)
_INPUT = re.compile(r"<input[^>]*", re.I)
_A = re.compile(r"<a\s[^>]*", re.I)
_REL_AUTHOR = re.compile(r"<a\s[^>]*rel\s*=\s*[\"'][^\"']*\bauthor\b[^>]*>", re.I)
_HREF = re.compile(r"href\s*=\s*[\"']([^\"']+)", re.I)
_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)
_TITLE_END = re.compile(r"</title>", re.I)
_LINK = re.compile(r"<link[^>]*", re.I)
_CANONICAL = re.compile(r"<link[^>]+rel\s*=\s*[\"']canonical[\"'][^>]*>", re.I)
_FILING = re.compile(r"/Archives/edgar/data/(\d+)/(\d{18}|\d{10}-\d{2}-\d{6})/")
FILING_INDEX_CACHE_MAX = 2000
ROBOTS_MAX_BYTES = 500_000
_ENCODINGS = {"gzip": zlib.MAX_WBITS | 16, "x-gzip": zlib.MAX_WBITS | 16,
              "deflate": zlib.MAX_WBITS}
# lxml and trafilatura corrupt the heap when pages are parsed in several threads at once, so
# every page is parsed on this one thread.
_PARSE = ThreadPoolExecutor(max_workers=1, thread_name_prefix="read_page")


def user_agent(contact: str = "") -> str:
    """The crawler's user agent, with the operator's contact URL or email when set."""
    contact = (contact or "").strip()
    return f"{BOT_NAME}/{__version__} (+{contact})" if contact else f"{BOT_NAME}/{__version__}"


def tls_context() -> ssl.SSLContext:
    """The TLS settings urllib3 (and so requests) uses; some CDNs refuse Python's bare
    defaults."""
    ctx = httpx.create_ssl_context()
    ctx.options |= ssl.OP_NO_TICKET
    ctx.post_handshake_auth = True
    return ctx


@dataclass(frozen=True)
class FetchConfig:
    user_agent: str = user_agent()
    # Whole request, body and redirects included: some official sites take 8-10 s to answer.
    timeout_s: float = 15.0
    # Connecting, TLS included; a host that does not answer fails fast.
    connect_timeout_s: float = 5.0
    max_bytes: int = 2_000_000
    max_redirects: int = 3
    # Extra attempts, for a 5xx or a timeout only.
    retries: int = 1
    retry_delay_s: float = 2.0
    domain_rps: float = 1.0
    # Longest wait for a domain slot; a longer wait raises Paced.
    pace_max_wait_s: float = float("inf")
    robots_ttl_s: float = 3600.0
    login_min_words: int = 40
    public_only: bool = True
    headers: Tuple[Tuple[str, str], ...] = ()


@dataclass(frozen=True)
class FetchedPage:
    url: str
    final_url: str = ""
    status: Optional[int] = None
    reason: Optional[str] = None
    html: str = ""
    text: str = ""
    published_at: Optional[datetime] = None
    # True when the date was read from the page body rather than its metadata.
    date_inferred: bool = False
    canonical_url: str = ""
    content_hash: str = ""
    bytes: int = 0
    title: str = ""
    # Not read, though the page may be there: an access wall, a 4xx other than 404/410,
    # robots.txt, a timeout, a connection error, a 5xx, a body that is not text, too big or
    # with no text found. A scoop may still cite it.
    unread: bool = False

    @property
    def ok(self) -> bool:
        return self.reason is None


class _Refused(Exception):
    pass


class Paced(Exception):
    """The domain has no free slot within `pace_max_wait_s`; retry after `wait_s`."""

    def __init__(self, wait_s: float):
        super().__init__(f"paced {wait_s:.1f}s")
        self.wait_s = wait_s


class _Retry(Exception):
    def __init__(self, status: Optional[int]):
        self.status = status


# ---- page metadata -------------------------------------------------------------------

def _ld_people(value) -> Iterable[dict]:
    for node in value if isinstance(value, list) else [value]:
        if isinstance(node, str):
            yield {"name": node}
        elif isinstance(node, dict):
            yield {k: str(node[k]) for k in ("name", "url", "jobTitle", "description")
                   if isinstance(node.get(k), str)}


def authors(html: str) -> List[Dict[str, str]]:
    """Bylines the page declares: JSON-LD author or creator, author meta tags, rel=author."""
    out: List[Dict[str, str]] = []
    for node in ld_nodes(html):
        for key in ("author", "creator"):
            out.extend(p for p in _ld_people(node.get(key)) if p)
    meta = meta_tags(html)
    for key in ("author", "article:author", "dc.creator", "parsely-author", "byl"):
        if meta.get(key, "").strip():
            value = meta[key].strip()
            out.append({"url": value} if value.startswith("http") else {"name": value})
    for tag in tag_matches(html, _A, _REL_AUTHOR, last_end(html, ">")):
        href = _HREF.search(tag.group(0))
        if href:
            out.append({"url": href.group(1)})
    return out


def page_title(html: str) -> str:
    """The headline the page declares: JSON-LD, og:title, then <title>."""
    for node in ld_nodes(html):
        if isinstance(node.get("headline"), str) and node["headline"].strip():
            return unescape(node["headline"]).strip()
    meta = meta_tags(html)
    if meta.get("og:title", "").strip():
        return unescape(meta["og:title"]).strip()
    m = search_upto(_TITLE, html, _TITLE_END)
    return re.sub(r"\s+", " ", unescape(m.group(1))).strip() if m else ""


def canonical_url(html: str, base: str) -> str:
    m = next(tag_matches(html, _LINK, _CANONICAL, last_end(html, ">")), None)
    href = _HREF.search(m.group(0)) if m else None
    if not href:
        og = meta_tags(html).get("og:url")
        return urljoin(base, og) if og else base
    return urljoin(base, href.group(1))


def access_wall(html: str, final_url: str, text: str, cfg: FetchConfig) -> bool:
    """A paywall or login page rather than the article."""
    for node in ld_nodes(html):
        free = node.get("isAccessibleForFree")
        if free is False or (isinstance(free, str) and free.strip().lower() == "false"):
            return True
    if _LOGIN_PATH.search(urlsplit(final_url).path.lower()):
        return True
    return (len(text.split()) < cfg.login_min_words
            and next(tag_matches(html, _INPUT, _PASSWORD), None) is not None)


def clean_url(url: str) -> str:
    """The URL with repeated slashes in its path collapsed ("https://a.org//press/x")."""
    parts = urlsplit(url)
    return parts._replace(path=re.sub(r"/{2,}", "/", parts.path)).geturl()


def filing_index_url(url: str) -> Optional[str]:
    """The index page of the SEC filing a page belongs to."""
    parts = urlsplit(clean_url(url))
    if (parts.hostname or "").lower() not in ("sec.gov", "www.sec.gov"):
        return None
    m = _FILING.match(parts.path)
    if not m:
        return None
    acc = m.group(2).replace("-", "")
    return (f"{parts.scheme}://{parts.netloc}/Archives/edgar/data/{int(m.group(1))}/{acc}/"
            f"{acc[:10]}-{acc[10:12]}-{acc[12:]}-index.htm")


def read_page(url: str, final_url: str, status: int, raw: bytes, html: str,
              cfg: FetchConfig) -> FetchedPage:
    base = dict(url=url, final_url=final_url, status=status, bytes=len(raw))
    try:
        text = tidy(ContentExtractor().extract(html)) if html.strip() else ""
    except ValueError:
        # The parser refuses some control characters.
        return FetchedPage(reason=FETCH_FAILED, **base)
    walled = access_wall(html, final_url, text, cfg)
    if not text and not walled:
        return FetchedPage(reason=FETCH_FAILED, unread=True, **base)
    if walled:
        return FetchedPage(reason=BLOCKED, unread=True, **base)
    published, inferred = publish_date(html, final_url or url)
    return FetchedPage(html=html, text=text, published_at=published, date_inferred=inferred,
                       canonical_url=canonical_url(html, final_url),
                       content_hash=floor.content_hash(text), title=page_title(html), **base)


# ---- fetching ------------------------------------------------------------------------

async def read_body(r: httpx.Response, max_bytes: int) -> Optional[bytes]:
    """The decoded body; None when it is over max_bytes as sent or decoded, or is not plain,
    gzip or deflate (one encoding at most)."""
    codings = [c.strip().lower() for c in r.headers.get("content-encoding", "").split(",")]
    codings = [c for c in codings if c and c != "identity"]
    if len(codings) > 1 or (codings and codings[0] not in _ENCODINGS):
        return None
    if r.is_stream_consumed:
        # Built in memory, already read.
        return r.content if len(r.content) <= max_bytes else None
    d = zlib.decompressobj(_ENCODINGS[codings[0]]) if codings else None
    # Some servers send raw deflate without the zlib header.
    retry_raw = codings == ["deflate"]
    sent, out = 0, bytearray()
    try:
        async for chunk in r.aiter_raw():
            sent += len(chunk)
            if sent > max_bytes:
                return None
            if d is None:
                out += chunk
                continue
            try:
                out += d.decompress(chunk, max_bytes + 1 - len(out))
            except zlib.error:
                if not retry_raw:
                    raise
                d = zlib.decompressobj(-zlib.MAX_WBITS)
                out += d.decompress(chunk, max_bytes + 1 - len(out))
            retry_raw = False
            while d.unconsumed_tail and len(out) <= max_bytes:
                out += d.decompress(d.unconsumed_tail, max_bytes + 1 - len(out))
            if len(out) > max_bytes:
                return None
        if d is not None:
            out += d.flush()
    except zlib.error:
        return None
    return bytes(out) if len(out) <= max_bytes else None


async def public_address(host: str, port: Optional[int] = None) -> Optional[str]:
    """The address to connect to when every address the host resolves to is public."""
    try:
        return host if ipaddress.ip_address(host).is_global else None
    except ValueError:
        pass
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, port,
                                                              type=socket.SOCK_STREAM)
    except socket.gaierror:
        return None
    addrs = [i[4][0] for i in infos]
    if not addrs or not all(ipaddress.ip_address(a).is_global for a in addrs):
        return None
    return addrs[0]


async def _public(host: str) -> bool:
    return await public_address(host) is not None


class PublicBackend(httpcore.AsyncNetworkBackend):
    """Connects only to the public address it checked, so the name is resolved once."""

    def __init__(self, inner: Optional[httpcore.AsyncNetworkBackend] = None):
        self.inner = inner or httpcore.AnyIOBackend()

    async def connect_tcp(self, host, port, timeout=None, local_address=None,
                          socket_options=None):
        addr = await public_address(host, port)
        if addr is None:
            raise httpcore.ConnectError(f"non-public host {host}")
        return await self.inner.connect_tcp(addr, port, timeout=timeout,
                                            local_address=local_address,
                                            socket_options=socket_options)

    async def sleep(self, seconds: float) -> None:
        await self.inner.sleep(seconds)


def public_transport(**kw) -> httpx.AsyncHTTPTransport:
    transport = httpx.AsyncHTTPTransport(**kw)
    # Relies on httpx internals; the [verify] extra pins the version.
    if not hasattr(getattr(transport, "_pool", None), "_network_backend"):
        raise RuntimeError("unsupported httpx version: no connection pool backend")
    transport._pool._network_backend = PublicBackend()
    return transport


class Fetcher:
    """One client per process: robots.txt cache, per-domain pacing, no cookies."""

    def __init__(self, cfg: FetchConfig = FetchConfig(), transport=None):
        self.cfg = cfg
        self._robots: Dict[Tuple[str, str], Tuple[float, Optional[RobotFileParser]]] = {}
        self._next: Dict[str, float] = {}
        self._indexes: Dict[str, str] = {}
        hooks = {"request": [self._guard]} if cfg.public_only else {}
        tls = tls_context()
        if transport is None and cfg.public_only:
            transport = public_transport(verify=tls)
        self.client = httpx.AsyncClient(
            headers={"Accept": "text/html,*/*;q=0.5", "Accept-Encoding": "gzip, deflate",
                     **dict(cfg.headers),
                     "User-Agent": cfg.user_agent},
            timeout=httpx.Timeout(cfg.timeout_s, connect=min(cfg.connect_timeout_s,
                                                             cfg.timeout_s)),
            verify=tls, follow_redirects=True, max_redirects=cfg.max_redirects,
            cookies=CookieJar(policy=DefaultCookiePolicy(allowed_domains=[])),
            event_hooks=hooks, transport=transport)

    async def aclose(self) -> None:
        await self.client.aclose()

    async def _guard(self, request: httpx.Request) -> None:
        if request.url.scheme not in ("http", "https") or not await _public(request.url.host):
            raise _Refused(request.url.host)

    async def _pace(self, host: str) -> None:
        now = time.monotonic()
        start = max(self._next.get(host, 0.0), now)
        if start - now > self.cfg.pace_max_wait_s:
            raise Paced(start - now)
        self._next[host] = start + 1.0 / max(self.cfg.domain_rps, 1e-6)
        if start > now:
            await asyncio.sleep(start - now)

    async def allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        key = (parts.scheme, parts.netloc)
        hit = self._robots.get(key)
        if hit is None or hit[0] < time.monotonic():
            rp = await self._load_robots(f"{parts.scheme}://{parts.netloc}/robots.txt")
            hit = (time.monotonic() + self.cfg.robots_ttl_s, rp)
            self._robots[key] = hit
        return hit[1] is not None and hit[1].can_fetch(self.cfg.user_agent, url)

    async def _load_robots(self, robots_url: str) -> Optional[RobotFileParser]:
        """None means disallow everything (robots.txt unreachable)."""
        rp = RobotFileParser(robots_url)
        try:
            await self._pace(urlsplit(robots_url).netloc)
            status, text = await asyncio.wait_for(self._robots_txt(robots_url),
                                                  self.cfg.timeout_s)
        except _Refused:
            return None
        except (httpx.HTTPError, asyncio.TimeoutError):
            return None
        if status >= 500 or text is None:
            return None
        rp.parse(text.splitlines())
        return rp

    async def _robots_txt(self, url: str) -> Tuple[int, Optional[str]]:
        async with self.client.stream("GET", url) as r:
            if r.status_code >= 400:
                return r.status_code, ""
            body = await read_body(r, ROBOTS_MAX_BYTES)
            return r.status_code, (None if body is None
                                   else body.decode(r.encoding or "utf-8", errors="replace"))

    async def fetch(self, url: str) -> FetchedPage:
        """The page; an SEC filing is published at its acceptance time, and is not read
        without it. A busy domain raises Paced for the page or its index alike."""
        index = await self.filing_index(url)
        page = await self._fetch(url)
        final = page.final_url or url
        if not page.ok or filing_index_url(final) is None:
            return page
        if filing_index_url(final) != filing_index_url(url):
            index = await self.filing_index(final)
        accepted = accepted_at(index or "")
        if accepted is None:
            return FetchedPage(url=url, final_url=page.final_url, status=page.status,
                               reason=FETCH_FAILED)
        return replace(page, published_at=accepted, date_inferred=False)

    async def filing_index(self, url: str) -> Optional[str]:
        """The index page of the SEC filing a page belongs to, kept per filing."""
        index = filing_index_url(url)
        if index is None:
            return None
        if index not in self._indexes:
            try:
                text = await asyncio.wait_for(self.get_text(index), self.cfg.timeout_s)
            except asyncio.TimeoutError:
                text = None
            if text is None:
                return None
            while len(self._indexes) >= FILING_INDEX_CACHE_MAX:
                del self._indexes[next(iter(self._indexes))]
            self._indexes[index] = text
        return self._indexes[index]

    async def get_text(self, url: str) -> Optional[str]:
        """A raw body (a feed, a filing index), or None."""
        try:
            if not await self.allowed(url):
                return None
            await self._pace(urlsplit(url).netloc)
            return await asyncio.wait_for(self._text(url), self.cfg.timeout_s)
        except Paced:
            raise
        except Exception:
            return None

    async def _text(self, url: str) -> Optional[str]:
        async with self.client.stream("GET", url) as r:
            if r.status_code >= 400:
                return None
            raw = await read_body(r, self.cfg.max_bytes)
            return None if raw is None else raw.decode(r.encoding or "utf-8", errors="replace")

    async def _fetch(self, url: str) -> FetchedPage:
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname or (
                self.cfg.public_only and not await _public(parts.hostname)):
            return FetchedPage(url=url, reason=BLOCKED)
        try:
            if not await self.allowed(url):
                return FetchedPage(url=url, reason=BLOCKED, unread=True)
        except _Refused:
            return FetchedPage(url=url, reason=BLOCKED)
        for attempt in range(self.cfg.retries + 1):
            if attempt:
                await asyncio.sleep(self.cfg.retry_delay_s)
            await self._pace(urlsplit(url).netloc)
            try:
                got = await asyncio.wait_for(self._get(url), self.cfg.timeout_s)
                if isinstance(got, FetchedPage):
                    return got
                # Parsing is outside the time limit: pages wait their turn on one thread.
                return await asyncio.get_running_loop().run_in_executor(
                    _PARSE, read_page, url, *got, self.cfg)
            except _Refused:
                return FetchedPage(url=url, reason=BLOCKED)
            except _Retry as e:
                status = e.status
            except (asyncio.TimeoutError, httpx.TimeoutException):
                status = None
            except httpx.TransportError as e:
                return FetchedPage(url=url, reason=FETCH_FAILED,
                                   unread=not isinstance(e, httpx.UnsupportedProtocol))
            except httpx.HTTPError:
                return FetchedPage(url=url, reason=FETCH_FAILED)
        return FetchedPage(url=url, status=status, reason=FETCH_FAILED, unread=True)

    async def _get(self, url: str) -> Union[FetchedPage, Tuple[str, int, bytes, str]]:
        """Redirects are followed by hand so robots.txt is checked for every hop."""
        request = self.client.build_request("GET", url)
        for _ in range(self.cfg.max_redirects + 1):
            r = await self.client.send(request, stream=True, follow_redirects=False)
            try:
                if not r.is_redirect:
                    return await self._response(url, r)
                request = r.next_request
            finally:
                await r.aclose()
            if request is None:
                return FetchedPage(url=url, final_url=str(r.url), status=r.status_code,
                                   reason=FETCH_FAILED)
            if not await self.allowed(str(request.url)):
                return FetchedPage(url=url, final_url=str(request.url), reason=BLOCKED,
                                   unread=True)
        return FetchedPage(url=url, final_url=str(request.url), reason=FETCH_FAILED)

    async def _response(self, url: str,
                        r: httpx.Response) -> Union[FetchedPage, Tuple[str, int, bytes, str]]:
        final, status = str(r.url), r.status_code
        if status >= 500:
            raise _Retry(status)
        if status in (401, 402, 403, 407, 451):
            return FetchedPage(url=url, final_url=final, status=status, reason=BLOCKED,
                               unread=True)
        if status >= 400:
            return FetchedPage(url=url, final_url=final, status=status, reason=FETCH_FAILED,
                               unread=status not in (404, 410))
        ctype = r.headers.get("content-type", "text/html").split(";")[0].strip().lower()
        if ctype not in _TEXT_TYPES:
            return FetchedPage(url=url, final_url=final, status=status, reason=FETCH_FAILED,
                               unread=True)
        raw = await read_body(r, self.cfg.max_bytes)
        if raw is None:
            return FetchedPage(url=url, final_url=final, status=status, reason=FETCH_FAILED,
                               unread=True)
        return final, status, raw, raw.decode(r.encoding or "utf-8", errors="replace")
