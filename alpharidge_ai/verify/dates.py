"""Publish dates: JSON-LD and meta tags, official release lines, SEC acceptance times."""
from __future__ import annotations

import email.utils
import json
import re
from datetime import date, datetime, timedelta, timezone
from html import unescape
from typing import Dict, Iterable, Optional, Tuple
from zoneinfo import ZoneInfo

from alpharidge_ai.verify import official
from alpharidge_ai.verify.markup import last_end, sub_upto, tag_matches, tags

_LD_JSON = re.compile(
    r"<script[^>]+type\s*=\s*[\"']application/ld\+json[\"'][^>]*>(.*?)</script>", re.I | re.S)
_SCRIPT, _SCRIPT_END = re.compile(r"<script[^>]*", re.I), re.compile(r"</script>", re.I)
_ATTR = re.compile(r"(?<![\w:.-])([\w:.-]+)\s*=\s*([\"'])(.*?)\2", re.S)
_TIME_TAG = re.compile(r"<time[^>]*", re.I)
_DATETIME = re.compile(r"<time[^>]*(?P<at>datetime)\s*=\s*[\"'](?P<value>[^\"']+)", re.I)
_PUBLISHED = re.compile(r"itemprop\s*=\s*[\"']datePublished[\"']", re.I)
_TAG = re.compile(r"<[^>]+>")
_ACCEPTED = re.compile(r'infoHead">\s*Accepted\s*</div>\s*<div class="info">\s*'
                       r'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)')
_EDGAR_TZ = ZoneInfo("America/New_York")
_DATE_META = ("article:published_time", "og:published_time", "datepublished", "pubdate",
              "publishdate", "publish-date", "dc.date.issued", "dcterms.created", "date")


def parse_datetime(value) -> Optional[datetime]:
    """An ISO 8601 or RFC 2822 timestamp as aware UTC; a bare date is midnight UTC."""
    if not isinstance(value, str) or not value.strip():
        return None
    s = value.strip()
    iso = re.sub(r"Z$", "+00:00", s)
    iso = re.sub(r"([+-]\d\d)(\d\d)$", r"\1:\2", iso)
    iso = re.sub(r"(:\d\d)\.(\d+)", lambda m: f"{m.group(1)}.{m.group(2)[:6]:0<6}", iso)
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        try:
            dt = email.utils.parsedate_to_datetime(s)
        except (TypeError, ValueError):
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def ld_nodes(html: str) -> Iterable[dict]:
    for m in tag_matches(html, _SCRIPT, _LD_JSON, last_end(html, _SCRIPT_END)):
        try:
            data = json.loads(m.group(1).strip())
        except ValueError:
            continue
        stack = [data]
        while stack:
            node = stack.pop(0)
            if isinstance(node, list):
                stack.extend(node)
            elif isinstance(node, dict):
                yield node
                if "@graph" in node:
                    stack.append(node["@graph"])


def meta_tags(html: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for tag in tags(html, r"<meta\s"):
        attrs = dict((k.lower(), v) for k, _, v in _ATTR.findall(tag))
        key = (attrs.get("property") or attrs.get("name") or attrs.get("itemprop") or "").lower()
        if key and "content" in attrs:
            out.setdefault(key, attrs["content"])
    return out


def _latest_day() -> date:
    """The latest calendar date anywhere (UTC+14), so a page dated in its own time zone is
    never read as future, whatever the host's zone."""
    return (datetime.now(timezone.utc) + timedelta(hours=14)).date()


def publish_date(html: str, url: str = "") -> Tuple[Optional[datetime], bool]:
    """(publish time, inferred): inferred when only trafilatura found a date. On an
    official site the date the page states in its body wins over that guess."""
    rule = official.rule(url)
    if rule is not None and rule.unix:
        return _first_post(html), False
    dt = _structured_date(html)
    if dt is not None:
        return dt, False
    latest = _latest_day()
    if rule is not None:
        dt = _release_date(html, latest, rule)
        if dt is not None:
            return dt, rule.guess
    try:
        from trafilatura.metadata import extract_metadata
        found = extract_metadata(html, date_config={
            "original_date": True, "extensive_search": True, "max_date": latest.isoformat()})
        dt = parse_datetime(getattr(found, "date", None)) if found else None
    except Exception:
        dt = None
    return dt, dt is not None


_MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august",
           "september", "october", "november", "december")
_MON = "|".join(m[:3] + r"\w*" for m in _MONTHS)
_WORDS = (rf"(?P<mon>{_MON})\.?\s+(?P<day>\d{{1,2}}),?\s+(?P<year>20\d\d)"
          rf"|(?P<day2>\d{{1,2}})\s+(?P<mon2>{_MON})\.?,?\s+(?P<year2>20\d\d)")
# A site's own rule also reads "Monday 9th November 2026".
_RULE_WORDS = _WORDS.replace(r"(?P<day2>\d{1,2})", r"(?P<day2>\d{1,2})(?:st|nd|rd|th)?")
_NUMERIC = (r"|(?P<iso_y>20\d\d)-(?P<iso_m>\d\d)-(?P<iso_d>\d\d)"
            r"|(?P<n1>\d{1,2})[/.](?P<n2>\d{1,2})[/.](?P<n_y>20\d\d)")


def _labelled(labels, words: str = _RULE_WORDS + _NUMERIC) -> str:
    alt = "|".join(re.escape(x) for x in sorted(labels, key=len, reverse=True))
    return rf"(?<!next )(?:{alt})(?!\w)[^<>]{{0,60}}?(?<!\w)(?:{words})(?!\d)"


# "Embargoed until release at 8:30 a.m. EDT, Tuesday, October 6, 2026", "Release date: 22
# October 2026". Numeric dates are left out: 10/6/26 reads either way round.
_RELEASED = re.compile(_labelled(official.DATE_LABELS, _WORDS), re.I)
_RULE_RES: dict = {}


def _rule_res(rule: official.Rule):
    if rule not in _RULE_RES:
        out = []
        if rule.marks:
            alt = "|".join(re.escape(x) for x in rule.marks)
            out.append((False, re.compile(rf"(?:{alt})[^<>]{{0,60}}?(?<![\w-])"
                                          rf"(?:{_RULE_WORDS}{_NUMERIC})(?!\d)", re.I)))
        if rule.labels:
            out.append((True, re.compile(_labelled(rule.labels), re.I)))
        _RULE_RES[rule] = out
    return _RULE_RES[rule]


def _day_of(m, order: str) -> Optional[date]:
    g = m.groupdict()
    try:
        if g.get("iso_y"):
            return date(int(g["iso_y"]), int(g["iso_m"]), int(g["iso_d"]))
        if g.get("n_y"):
            if not order:
                return None
            a, b = int(g["n1"]), int(g["n2"])
            return date(int(g["n_y"]), *((a, b) if order == "mdy" else (b, a)))
        mon = (g["mon"] or g["mon2"]).lower()[:3]
        return date(int(g["year"] or g["year2"]),
                    next(i for i, n in enumerate(_MONTHS, 1) if n.startswith(mon)),
                    int(g["day"] or g["day2"]))
    except (StopIteration, ValueError):
        return None


def _release_date(html: str, latest: date,
                  rule: official.Rule = official.Rule()) -> Optional[datetime]:
    """The first date the page states where the site states it; marks are read in the
    markup, labels in the page text."""
    text = None
    res = [(True, _RELEASED)] if rule.guess else _rule_res(rule)
    for in_text, pattern in res:
        if in_text and text is None:
            text = re.sub(r"\s+", " ", unescape(sub_upto(_TAG, " ", html, ">")))
        for m in pattern.finditer(text if in_text else html):
            day = _day_of(m, rule.order)
            if day is not None and day <= latest:
                return datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    return None


_UNIX_MS = re.compile(r'data-datetime-unix="(\d{12,14})"')


def _first_post(html: str) -> Optional[datetime]:
    """The earliest update time on a status-page incident."""
    now_ms = datetime.now(timezone.utc).timestamp() * 1000
    stamps = [int(m) for m in _UNIX_MS.findall(html) if int(m) <= now_ms]
    return datetime.fromtimestamp(min(stamps) / 1000, timezone.utc) if stamps else None


def _structured_date(html: str) -> Optional[datetime]:
    for node in ld_nodes(html):
        dt = parse_datetime(node.get("datePublished"))
        if dt:
            return dt
    meta = meta_tags(html)
    for key in _DATE_META:
        dt = parse_datetime(meta.get(key))
        if dt:
            return dt
    # <time ... itemprop="datePublished" ... datetime="...">
    for tag in _TIME_TAG.finditer(html):
        m = _DATETIME.match(html, tag.start())
        if m and _PUBLISHED.search(html, tag.start() + 6, m.start("at")):
            return parse_datetime(m.group("value"))
    return None


def accepted_at(index_html: str) -> Optional[datetime]:
    """A filing's acceptance time, from its index page."""
    m = _ACCEPTED.search(index_html or "")
    if not m:
        return None
    local = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=_EDGAR_TZ)
    return local.astimezone(timezone.utc)
