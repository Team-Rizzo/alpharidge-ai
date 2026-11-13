"""Listed validators read the trusted list's feeds, sitemaps and listing pages and the SEC
filings of open scoops' subjects, and send the API pages that pass the shared check for a held
scoop. The API queues them for the listed validators' page judges.

`GET /scoops/open` gives `feeds` as [{"url": str, "kind": "rss" | "sitemap" | "page"}]; a page
is a listing whose same-site links are read with their anchor text.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from datetime import time as dtime
from typing import Callable, Dict, List, Optional, Set, Tuple
from html import unescape
from urllib.parse import unquote, urljoin, urlsplit

import bittensor as bt

from alpharidge_ai import config
from alpharidge_ai.ledger.api_client import AlpharidgeAPIError, AuthorizationError, NotFoundError
from alpharidge_ai.miner.discover import Item, parse_feed
from alpharidge_ai.schema import TYPE_SPEC, Claim, EventType
from alpharidge_ai.validator.reverify import fetch_config
from alpharidge_ai.verify import floor
from alpharidge_ai.verify.check import (
    PIVOT_SLOTS, check_claim, find, name_forms, pick_quote, ticker_on, unstated_figures,
)
from alpharidge_ai.verify.dates import parse_datetime
from alpharidge_ai.verify.fetch import Fetcher
from alpharidge_ai.verify.filing import FORMS, submissions_url
from alpharidge_ai.verify.markup import last_end, sub_upto, tag_matches
from alpharidge_ai.verify.names import name_words, named_in

POSTS_PER_PASS = 3
CHECKS_PER_RUN = 10
OFF_S = 6 * 3600
FEED_MIN_S, FEED_MAX_S = 120, 1800
ITEMS_KEEP_S = 5 * 86400
ITEMS_MAX = 100_000
# Seconds of matching per scan; scoops not reached are matched on later scans.
MATCH_BUDGET_S = 60
FILING_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"


@dataclass(frozen=True)
class Scoop:
    id: int
    event_type: EventType
    stage: Optional[str]
    event_date: date
    received_at: datetime
    subjects: Tuple[str, ...]
    ciks: Tuple[str, ...]
    facts: Tuple[dict, ...]
    look_from: Optional[datetime] = None
    tickers: Tuple[str, ...] = ()

    @property
    def since(self) -> datetime:
        """Pages and filings from this time on are looked at."""
        return self.look_from or datetime.combine(self.event_date, dtime(), timezone.utc)


def parse_scoop(raw: dict) -> Optional[Scoop]:
    try:
        received = parse_datetime(raw["received_at"])
        if received is None:
            return None
        return Scoop(id=int(raw["id"]), event_type=EventType(raw["event_type"]),
                     stage=raw.get("stage"), event_date=date.fromisoformat(raw["event_date"]),
                     received_at=received,
                     subjects=tuple(str(s) for s in raw.get("subjects") or () if s),
                     ciks=tuple(str(a["id"]) for a in raw.get("actor_ids") or ()
                                if a.get("id_scheme") == "cik"),
                     facts=tuple(dict(f) for f in raw.get("facts") or ()),
                     look_from=parse_datetime(raw.get("since")),
                     tickers=tuple(str(t) for t in raw.get("tickers") or () if t))
    except (KeyError, TypeError, ValueError):
        return None


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def parse_sitemap(body: str) -> List[Item]:
    """Pages of a `<urlset>` sitemap, with their news title and publication date when given."""
    if "<!ENTITY" in body:
        return []
    try:
        root = ET.fromstring(body.encode("utf-8"))
    except ET.ParseError:
        return []
    out = []
    for node in root.iter():
        if _local(node.tag) != "url":
            continue
        got: Dict[str, str] = {}
        for c in node.iter():
            name = _local(c.tag)
            if name in ("loc", "title", "publication_date", "lastmod") and name not in got:
                got[name] = (c.text or "").strip()
        url = got.get("loc", "")
        if url.startswith(("http://", "https://")):
            when = got.get("publication_date") or got.get("lastmod")
            out.append(Item(url=url, title=got.get("title", ""), published=parse_datetime(when)))
    return out


_ANCHOR = re.compile(r"<a\s[^>]*?href\s*=\s*[\"']([^\"'#]+)[\"'][^>]*>(.*?)</a>", re.I | re.S)
_A, _A_END = re.compile(r"<a\s[^>]*", re.I), re.compile(r"</a>", re.I)
_TAG = re.compile(r"<[^>]+>")


def _host(url: str) -> str:
    return urlsplit(url).netloc.lower().removeprefix("www.")


def page_items(html: str, base: str) -> List[Item]:
    """Same-site links on a listing page, with their anchor text as the title."""
    host, out = _host(base), {}
    html = html or ""
    for m in tag_matches(html, _A, _ANCHOR, last_end(html, _A_END)):
        href, inner = m.groups()
        url = urljoin(base, unescape(href.strip()))
        if url.startswith(("http://", "https://")) and _host(url) == host and url not in out:
            title = unescape(sub_upto(_TAG, " ", inner, ">"))
            out[url] = Item(url=url, title=" ".join(title.split()))
    return list(out.values())


def filings(submissions: dict, cik: str, after: datetime) -> List[Item]:
    """The filer's current reports accepted after `after`."""
    try:
        recent = submissions["filings"]["recent"]
        rows = zip(recent["accessionNumber"], recent["form"], recent["acceptanceDateTime"],
                   recent["primaryDocument"])
    except (KeyError, TypeError):
        return []
    out = []
    for acc, form, accepted, doc in rows:
        when = parse_datetime(str(accepted))
        if form in FORMS and when and when > after and doc:
            out.append(Item(url=FILING_URL.format(cik=int(cik), acc=str(acc).replace("-", ""),
                                                  doc=doc), published=when))
    return out


def _slug(url: str) -> str:
    path = unquote(urlsplit(url).path)
    for c in "-_/.+":
        path = path.replace(c, " ")
    return floor.normalize(path).text


def item_words(item: Item) -> Tuple[Tuple[str, ...], ...]:
    return (name_words(item.title), name_words(unquote(urlsplit(item.url).path)),
            name_words(item.summary))


def item_keys(item: Item) -> Tuple[str, str, Tuple]:
    """What an item is matched on, worked out once per item: its normalised title, its URL
    slug and its words."""
    return floor.normalize(item.title).text, _slug(item.url), item_words(item)


def mentions(scoop: Scoop, item: Item, keys: Optional[Tuple] = None) -> bool:
    """The item's title, URL slug or summary names one of the scoop's subjects, as written,
    shorter or longer, or gives one of its tickers."""
    title, slug, words = keys or item_keys(item)
    if any(find(title, f) or find(slug, f.replace(".", "").replace(",", ""))
           for s in scoop.subjects for f in name_forms(s)):
        return True
    if any(ticker_on(t, item.title) or ticker_on(t, item.summary) for t in scoop.tickers):
        return True
    return any(named_in(w, s) for s in scoop.subjects for w in words if w)


def match_items(scoops: List[Scoop], items: List[Tuple[Item, Tuple]],
                after: Optional[Dict[int, int]] = None, numbers: Optional[List[int]] = None,
                budget_s: Optional[float] = None) -> Dict[int, Dict[str, Optional[datetime]]]:
    """Per scoop matched, the items that name it, published from its `since` on; with `after`
    and `numbers`, only items numbered after the scoop's last look. Scoops are taken in turn
    until `budget_s` is spent; those not reached are left out. CPU only, for a worker
    thread."""
    stop = None if budget_s is None else time.monotonic() + budget_s
    out: Dict[int, Dict[str, Optional[datetime]]] = {}
    for s in scoops:
        if stop is not None and out and time.monotonic() >= stop:
            break
        named = out.setdefault(s.id, {})
        upto = (after or {}).get(s.id, 0)
        for k, (item, keys) in enumerate(items):
            if numbers is not None and numbers[k] <= upto:
                continue
            if (item.published is None or item.published >= s.since) and mentions(s, item, keys):
                named[item.url] = item.published
    return out


def held_claim(scoop: Scoop, url: str, text: str) -> Optional[Claim]:
    """The scoop as checked on a page: the subject name the page shows, its other facts less the
    figures the page does not show; None when a figure its type needs is missing."""
    norm = floor.normalize(text).text
    name = next((s for s in scoop.subjects if any(find(norm, f) for f in name_forms(s))), None)
    if name is None:
        return None
    spec = TYPE_SPEC[scoop.event_type]
    pivot = PIVOT_SLOTS.get(scoop.event_type.value)
    facts = [f for f in scoop.facts if f.get("field") != pivot]
    body = dict(kind="event", headline="scan", event_type=scoop.event_type.value,
                stage=scoop.stage, event_date=scoop.event_date.isoformat(),
                actors=[{"id_scheme": "other", "id": name[:64], "name": name[:200]}],
                facts=facts, source_url=url)
    try:
        claim = Claim.model_validate(body)
        drop = set(unstated_figures(claim, text))
        had = {f["field"] for f in facts}
        if any((set(g) & had) and (set(g) & had) <= drop for g in spec.required):
            return None
        body["facts"] = [f for f in facts if f["field"] not in drop]
        claim = Claim.model_validate(body)
        return claim.model_copy(update={"quote": pick_quote(claim, text)})
    except ValueError:
        return None


@dataclass
class _Seen:
    urls: Set[str] = field(default_factory=set)
    done: bool = False
    # Listed items that name the scoop, by URL, and the last item looked at.
    named: Dict[str, Optional[datetime]] = field(default_factory=dict)
    upto: int = 0


@dataclass
class _Feed:
    due: float = 0.0
    read: float = 0.0
    urls: frozenset = frozenset()
    # New items per second, smoothed.
    rate: Optional[float] = None


def next_read(state: _Feed, items: List[Item], now: float) -> float:
    """When to read a feed again, from how fast its items turn over."""
    urls = frozenset(i.url for i in items)
    stay = None
    if state.read and now > state.read:
        got = len(urls - state.urls) / (now - state.read)
        state.rate = got if state.rate is None else (state.rate + got) / 2
        stay = len(urls) / state.rate if state.rate else None
    else:
        dated = sorted(i.published for i in items if i.published)
        if len(dated) > 1:
            stay = (dated[-1] - dated[0]).total_seconds()
    state.urls, state.read = urls, now
    wait = FEED_MAX_S if stay is None else min(FEED_MAX_S, max(FEED_MIN_S, stay / 3))
    state.due = now + wait
    return state.due


class Scanner:
    def __init__(self, client, fetcher: Optional[Fetcher] = None,
                 interval: Optional[float] = None, concurrency: Optional[int] = None,
                 clock: Optional[Callable[[], datetime]] = None):
        self.client = client
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self._fetcher = fetcher
        self.interval = float(config.SCAN_INTERVAL_S if interval is None else interval)
        self.concurrency = max(1, int(concurrency or config.SCAN_CONCURRENCY))
        self._seen: Dict[int, _Seen] = {}
        self._feeds: Dict[str, _Feed] = {}
        self._listed: List[dict] = []
        # Feed items by URL: the item, when it was last listed, its number and its words.
        self._items: Dict[str, Tuple[Item, float, int, Tuple]] = {}
        self._count = 0
        self._gate: Optional[asyncio.Semaphore] = None
        self._next = 0.0
        self._task: Optional[asyncio.Task] = None
        self._refresh: Optional[asyncio.Task] = None

    @property
    def fetcher(self) -> Fetcher:
        if self._fetcher is None:
            self._fetcher = Fetcher(fetch_config())
        return self._fetcher

    @property
    def gate(self) -> asyncio.Semaphore:
        if self._gate is None:
            self._gate = asyncio.Semaphore(self.concurrency)
        return self._gate

    def _now(self) -> float:
        return self.clock().timestamp()

    def tick(self) -> None:
        """Start a scan in the background when one is due and none is running; while one runs,
        read the feeds that are due."""
        if self._task is not None and not self._task.done():
            if self._listed and (self._refresh is None or self._refresh.done()) and any(
                    self._feeds.get(str(f.get("url")), _Feed()).due <= self._now()
                    for f in self._listed):
                self._refresh = asyncio.create_task(self._read_feeds())
            return
        now = time.monotonic()
        if now < self._next:
            return
        self._next = now + self.interval
        self._task = asyncio.create_task(self._safe())

    async def _safe(self) -> None:
        try:
            await self.run()
        except AuthorizationError:
            self._next = time.monotonic() + OFF_S
            self._listed = []
            bt.logging.info("[SCAN] not a listed validator; scan paused")
        except Exception as e:
            bt.logging.warning(f"[SCAN] failed: {e!r}")

    async def _read(self, url: str) -> Optional[str]:
        async with self.gate:
            try:
                return await self.fetcher.get_text(url)
            except Exception:
                return None

    async def _feed(self, raw: dict) -> None:
        url, kind = str(raw.get("url", "")), raw.get("kind")
        state = self._feeds.setdefault(url, _Feed())
        state.due = float("inf")
        try:
            text = await self._read(url)
            now = self._now()
            if not text:
                return
            if kind == "page":
                items = page_items(text, url)
            else:
                items = parse_sitemap(text) if kind == "sitemap" else parse_feed(text)
                items = [i for i in items if i.published]
            next_read(state, items, now)
            for i in items:
                had = self._items.get(i.url)
                if had is None:
                    self._count += 1
                    had = (i, now, self._count, item_keys(i))
                self._items[i.url] = (had[0], now, had[2], had[3])
        finally:
            if state.due == float("inf"):
                state.due = self._now() + FEED_MIN_S

    async def _read_feeds(self) -> None:
        now = self._now()
        due = [f for f in self._listed
               if self._feeds.get(str(f.get("url")), _Feed()).due <= now]
        await asyncio.gather(*(self._feed(f) for f in due), return_exceptions=True)
        old = now - ITEMS_KEEP_S
        for url, (item, seen, _, _) in list(self._items.items()):
            if seen < old or (item.published and item.published.timestamp() < old):
                del self._items[url]
        while len(self._items) > ITEMS_MAX:
            del self._items[next(iter(self._items))]

    async def run(self) -> Dict[str, int]:
        got = await self.client.scoops_open()
        scoops = [s for s in map(parse_scoop, got.get("scoops") or []) if s]
        self._seen = {s.id: self._seen.get(s.id, _Seen()) for s in scoops}
        scoops = [s for s in scoops if not self._seen[s.id].done]
        self._listed = [f for f in got.get("feeds") or [] if isinstance(f, dict)
                        and f.get("url")] if scoops else []
        if not scoops:
            return {}

        async def sec(cik: str, after: datetime) -> List[Item]:
            text = await self._read(submissions_url(cik))
            try:
                return filings(json.loads(text), cik, after) if text else []
            except ValueError:
                return []

        await self._read_feeds()
        items = list(self._items.values())
        own = {s.id: await asyncio.gather(*(sec(c, s.since) for c in s.ciks)) for s in scoops}

        # Matching is CPU only and grows with scoops times items: a worker thread keeps the
        # event loop, and with it judging and weights, running meanwhile.
        upto, count = {s.id: self._seen[s.id].upto for s in scoops}, self._count
        # Scoops furthest behind go first, so a new scoop is not starved by the budget.
        order = sorted(scoops, key=lambda s: upto[s.id])
        found = await asyncio.to_thread(match_items, order, [(i, k) for i, _, _, k in items],
                                        upto, [n for _, _, n, _ in items], MATCH_BUDGET_S)
        work = []
        last = datetime.max.replace(tzinfo=timezone.utc)
        for s in scoops:
            seen = self._seen[s.id]
            if s.id in found:
                seen.named.update(found[s.id])
                seen.upto = count
            pool = {i.url: i.published for g in own[s.id] for i in g}
            pool.update((u, p) for u, p in seen.named.items() if u not in seen.urls)
            fresh = [u for u in sorted(pool, key=lambda u: pool[u] or last)
                     if u not in seen.urls]
            seen.urls.update(fresh[:CHECKS_PER_RUN])
            work += [(s, u) for u in fresh[:CHECKS_PER_RUN]]

        async def one(scoop: Scoop, url: str) -> bool:
            async with self.gate:
                page = await self.fetcher.fetch(url)
            if not page.ok:
                return False
            found = self.clock()

            def passes() -> bool:
                claim = held_claim(scoop, page.final_url or url, page.text)
                return claim is not None and check_claim(claim, page, found).passed

            return await asyncio.to_thread(passes)

        passed = await asyncio.gather(*(one(s, u) for s, u in work), return_exceptions=True)
        posted, new = 0, {s.id: 0 for s in scoops}
        for (s, url), ok in zip(work, passed):
            seen = self._seen[s.id]
            if isinstance(ok, BaseException):
                seen.urls.discard(url)
            if ok is not True or seen.done:
                continue
            if new[s.id] >= POSTS_PER_PASS:
                seen.urls.discard(url)
                continue
            try:
                status = (await self.client.post_confirm(s.id, url)).get("status")
            except NotFoundError:
                seen.done = True
                continue
            except AlpharidgeAPIError as e:
                seen.urls.discard(url)
                bt.logging.warning(f"[SCAN] scoop {s.id} post failed: {e.message}")
                continue
            if status == "queued":
                new[s.id] += 1
                posted += 1
            seen.done = status in ("already_confirmed", "closed")
        bt.logging.info(f"[SCAN] open={len(scoops)} items={len(items)} matched={len(found)} "
                        f"checked={len(work)} posted={posted}")
        return {"open": len(scoops), "checked": len(work), "posted": posted}
