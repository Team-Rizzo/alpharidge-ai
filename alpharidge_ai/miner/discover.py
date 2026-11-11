"""Discovery: poll public RSS/Atom feeds and official listing pages for new article links.

Feeds are read under the same robots.txt, user agent and per-domain pacing as page fetches.
"""
from __future__ import annotations

import asyncio
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html import unescape
from pathlib import Path
from typing import Iterable, List, Optional, Sequence
from urllib.parse import urljoin, urlsplit

from alpharidge_ai.verify.dates import parse_datetime
from alpharidge_ai.verify.fetch import Fetcher

FEEDS_PATH = Path(__file__).with_name("data") / "feeds.json"

_HREF = re.compile(r"<a\s[^>]*href\s*=\s*[\"']([^\"'#]+)", re.I)
_TAGS = re.compile(r"<[^>]+>")
SUMMARY_MAX = 1000


def _plain(text: str) -> str:
    return " ".join(unescape(_TAGS.sub(" ", text or "")).split())[:SUMMARY_MAX]


@dataclass(frozen=True)
class Source:
    url: str
    category: str = ""
    # "feed" (RSS/Atom) or "page" (an official listing page; links matching `pattern`).
    kind: str = "feed"
    pattern: Optional[str] = None


@dataclass(frozen=True)
class Item:
    url: str
    title: str = ""
    published: Optional[datetime] = None
    category: str = ""
    # The feed's description or summary of the item, as plain text.
    summary: str = ""


def load_sources(path: Optional[str] = None) -> List[Source]:
    with open(path or FEEDS_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    return [Source(url=s["url"], category=s.get("category", ""), kind=s.get("kind", "feed"),
                   pattern=s.get("pattern")) for s in raw.get("sources", [])]


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _child(node, name: str):
    return next((c for c in node if _local(c.tag) == name), None)


def _text(node, *names: str) -> str:
    for name in names:
        c = _child(node, name)
        if c is not None and (c.text or "").strip():
            return c.text.strip()
    return ""


def _atom_link(entry) -> str:
    links = [c for c in entry if _local(c.tag) == "link" and c.get("href")]
    best = next((c for c in links if c.get("rel", "alternate") == "alternate"), None)
    return (best if best is not None else links[0]).get("href") if links else ""


def parse_feed(body: str, base: str = "", category: str = "") -> List[Item]:
    """Items of an RSS 2.0, RSS 1.0 or Atom document; an unreadable feed gives none."""
    if "<!ENTITY" in body:
        return []
    try:
        root = ET.fromstring(body.encode("utf-8"))
    except ET.ParseError:
        return []
    out = []
    for node in root.iter():
        name = _local(node.tag)
        if name == "item":
            link = _text(node, "link") or _text(node, "guid")
            when = _text(node, "pubdate", "date", "published", "updated")
        elif name == "entry":
            link = _atom_link(node)
            when = _text(node, "published", "updated")
        else:
            continue
        url = urljoin(base, link) if link else ""
        if url.startswith(("http://", "https://")):
            out.append(Item(url=url, title=_text(node, "title"), published=parse_datetime(when),
                            category=category,
                            summary=_plain(_text(node, "description", "summary"))))
    return out


def page_links(html: str, base: str, pattern: Optional[str], category: str = "") -> List[Item]:
    """Same-site links on a listing page, optionally only those matching `pattern`."""
    host = urlsplit(base).netloc
    rx = re.compile(pattern) if pattern else None
    seen, out = set(), []
    for href in _HREF.findall(html):
        url = urljoin(base, href.strip())
        if urlsplit(url).netloc != host or url in seen or (rx and not rx.search(url)):
            continue
        seen.add(url)
        out.append(Item(url=url, category=category))
    return out


class Discoverer:
    def __init__(self, sources: Sequence[Source], fetcher: Fetcher, max_age_h: float = 48,
                 categories: Iterable[str] = ()):
        cats = set(categories)
        self.sources = [s for s in sources if not cats or s.category in cats]
        self.fetcher = fetcher
        self.max_age = timedelta(hours=max_age_h)

    async def _read(self, source: Source) -> List[Item]:
        body = await self.fetcher.get_text(source.url)
        if not body:
            return []
        if source.kind == "page":
            return page_links(body, source.url, source.pattern, source.category)
        return parse_feed(body, source.url, source.category)

    async def poll(self, now: Optional[datetime] = None) -> List[Item]:
        """Fresh items from every source, newest first, one per URL."""
        now = now or datetime.now(timezone.utc)
        found = await asyncio.gather(*(self._read(s) for s in self.sources))
        items = {}
        for item in (i for batch in found for i in batch):
            if item.published is not None and now - item.published > self.max_age:
                continue
            items.setdefault(item.url, item)
        oldest = datetime.min.replace(tzinfo=timezone.utc)
        return sorted(items.values(), key=lambda i: i.published or oldest, reverse=True)
