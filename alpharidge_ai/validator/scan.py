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
from functools import lru_cache
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
from alpharidge_ai.verify.names import _FILLER, _TAIL, abbreviates, name_words, named_in

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


# Words that mark a page as about an event of the type: phrases, with "*" for any word ending,
# "~" for up to three words between and a space or hyphen matching either. Words common in
# other news are left out ("change", "result", "round", "rate", "case", "transfer", a bare
# "CEO"), so a page about something else does not pass.
_ROLES = ("ceo", "cfo", "coo", "cto", "chief executive*", "chief financial officer",
          "chief operating officer", "chief technology officer", "chair*", "president",
          "general counsel")
_ROLE_LEADS = ("named ~", "names ~", "hire* ~", "promot* ~", "interim", "next", "become*")
# "as" and "new" lead only the executive roles: "such as President Smith", "its new chair".
_EXEC_LEADS, _EXEC_ROLES = ("as", "new"), tuple(r for r in _ROLES if r not in ("chair*",
                                                                              "president"))
TYPE_WORDS: Dict[str, Tuple[str, ...]] = {
    "monetary_policy": (
        "interest rate*", "policy rate", "benchmark rate", "federal funds rate", "cash rate",
        "bank rate", "rate cut*", "rate hike*", "rate increase", "cut* rates", "rais* rates",
        "hike* rates", "hold* rates", "held rates", "left rates", "kept rates", "rate decision",
        "monetary policy", "basis points", "key rate*", "leave* rates", "keep* rates",
        "maintain* rates", "lower* rates", "rates unchanged", "rates on hold"),
    "economic_data": (
        "inflation rate", "annual inflation", "consumer price*", "producer price*", "cpi", "ppi",
        "payrolls",
        "jobs report", "unemployment rate", "jobless claims", "gdp", "gross domestic product",
        "retail sales", "housing starts", "home sales", "industrial production",
        "durable goods", "trade balance", "trade deficit", "trade surplus", "pmi",
        "purchasing managers", "consumer confidence", "seasonally adjusted", "year on year",
        "month on month"),
    "earnings": (
        "earnings per share", "per diluted share", "eps", "net income", "net profit",
        "net loss", "operating income", "quarterly results", "quarterly earnings",
        "quarterly profit", "quarterly revenue", "earnings report", "earnings call",
        "adjusted earnings", "analysts expected", "consensus estimate*"),
    "guidance": (
        "guidance", "reaffirm*", "profit warning", "full year outlook", "full year forecast",
        "annual outlook", "annual forecast", "raised its outlook", "raised its forecast",
        "lowered its outlook", "lowered its forecast", "cut its outlook", "cut its forecast"),
    "merger_acquisition": (
        "acquir*", "acquisition*", "merger*", "takeover*", "buyout*", "agreed to buy",
        "deal to buy", "offer to buy", "tender offer", "all cash deal", "all stock deal"),
    "funding": (
        "funding round", "round of funding", "in funding", "series a", "series b", "series c",
        "series d", "series e", "seed round", "seed funding", "pre seed", "venture capital",
        "round led by", "investment led by", "post money", "pre money", "valuation",
        "rais* $", "rais* €", "rais* £"),
    "bankruptcy": (
        "bankrupt*", "chapter 7", "chapter 11", "chapter 15", "insolven*", "receivership",
        "creditor protection", "administrators", "debtor in possession",
        "restructuring support agreement"),
    "personnel": tuple(f"{lead} {role}" for lead in _ROLE_LEADS for role in _ROLES) + tuple(
        f"{lead} {role}" for lead in _EXEC_LEADS for role in _EXEC_ROLES) + (
        "appoint*", "steps down", "step down", "stepping down", "stepped down", "resign*",
        "retires", "to retire", "will retire", "successor", "succeeds", "to succeed",
        "will succeed"),
    "layoffs": (
        "layoff*", "lay off", "laying off", "laid off", "job cuts", "jobs cut", "cut jobs",
        "cuts jobs", "cutting jobs", "staff cuts", "job losses", "redundanc*", "headcount",
        "workforce reduction", "reduction in force", "of its workforce"),
    "regulatory": (
        "regulator*", "fined", "civil penalty", "consent order", "cease and desist",
        "enforcement action", "settled charges", "warning letter", "antitrust", "probe", "fda"),
    "legal": (
        "lawsuit*", "sued", "sues", "suing", "class action", "verdict", "jury", "plaintiff*",
        "defendant*", "settlement", "injunction", "litigation", "indict*", "court ruling",
        "judge ruled", "district court", "appeals court", "court of appeals", "supreme court"),
    "product_launch": (
        "product launch*", "launch event", "launch* its", "launch* new", "launches", "unveil*",
        "rolls out", "rolled out", "rolling out", "debut*", "now available", "available now",
        "pre order*", "preorder*", "goes on sale", "release date"),
    "partnership": (
        "partnership*", "partners with", "partnered with", "partnering with", "team* up",
        "alliance", "collaborat*", "joint venture", "memorandum of understanding", "mou",
        "multiyear agreement", "multi year agreement"),
    "security_incident": (
        "data breach*", "security breach*", "breached", "hacker*", "hacked", "cyberattack*",
        "cyber attack*", "ransomware", "data leak*", "leaked data", "compromised",
        "unauthori?ed access", "threat actor*", "malware", "phishing", "stolen data",
        "security incident*", "cybersecurity incident*", "cyber incident*"),
    "breakthrough": (
        "breakthrough*", "world first", "world's first", "world record", "first ever",
        "peer reviewed", "published in", "study published", "preprint", "arxiv",
        "researchers", "scientists"),
    "outage": (
        "outage*", "service disruption", "went down", "is down", "status page",
        "service restored", "restored service", "downtime", "downdetector",
        "degraded performance"),
    "vulnerability": (
        "vulnerabilit*", "cve", "zero day", "exploit*", "patched", "security patch*",
        "security update*", "security advisory", "cvss", "remote code execution",
        "privilege escalation", "security flaw", "proof of concept", "bug bounty"),
    "tech_regulation": (
        "antitrust", "fined", "competition authority", "competition commission",
        "regulator*", "digital markets act", "digital services act", "gdpr", "ftc",
        "european commission", "lawsuit*", "sued", "sues", "settlement", "monopoly",
        "anticompetitive", "anti competitive", "probe", "data protection"),
    "infrastructure_investment": (
        "data center*", "data centre*", "datacenter*", "chip plant*", "chip factory",
        "semiconductor plant*", "fab", "fabs", "megawatt*", "gigawatt*", "mw", "gw",
        "hyperscale*", "compute capacity", "ai infrastructure"),
    "supply_chain": (
        "supply chain*", "supply disruption*", "shortage*", "product recall*",
        "voluntary recall", "safety recall", "issued a recall", "issues a recall",
        "halted production", "halts production", "suspend* production", "production halt*",
        "plant shutdown", "factory shutdown", "force majeure", "shipping disruption*"),
    "ipo_listing": (
        "ipo", "ipos", "initial public offering", "go public", "goes public", "going public",
        "went public", "direct listing", "public debut", "trading debut", "market debut",
        "files to list", "spac", "nasdaq debut", "nyse debut", "exchange debut",
        "stock debut"),
    "delisting": (
        "delist*", "suspended from trading", "non compliance", "noncompliance",
        "listing standards", "listing rule*", "form 25", "deregist*"),
    "trading_halt": (
        "trading halt*", "halted trading", "halts trading", "halt trading", "trading suspended",
        "suspended trading", "suspends trading", "trading pause", "circuit breaker",
        "news pending", "trading resumed", "resume trading", "trading will resume",
        "was halted", "were halted", "shares halted"),
    "index_change": (
        "index change*", "rebalanc*", "reconstitut*", "constituent*", "quarterly review",
        "will replace", "will join the", "added to the s&p", "removed from the s&p",
        "index inclusion", "join* the s&p", "join* s&p", "join* the ftse", "join* the nasdaq 100",
        "join* the dow", "added to the ftse", "added to the nasdaq 100"),
    "match_result": (
        "defeat*", "victory", "final score", "final whistle", "full time", "scoreline",
        "hat trick", "stoppage time", "extra time", "shootout", "innings", "touchdown*",
        "overtime", "half time", "clean sheet", "match report"),
    "transfer": (
        "transfer fee", "transfer window", "free transfer", "signs for", "signed for",
        "completes move", "completed move", "on loan", "loan deal", "loan move",
        "free agent", "contract until", "year contract", "year deal", "release clause",
        "traded to", "trade deadline"),
    "injury": (
        "injur*", "sidelined", "ruled out", "out for the season", "surgery",
        "fracture*", "sprain*", "hamstring", "acl", "concussion", "achilles", "day to day",
        "expected to miss", "will miss"),
    "award": (
        "award*", "mvp", "player of the year", "trophy", "prize", "laureate", "nobel",
        "golden boot", "ballon d'or", "hall of fame"),
    "election_result": (
        "election*", "elected", "re elected", "reelected", "electoral", "vote share",
        "of the vote", "ballots", "runoff", "polls closed", "declared winner",
        "projected winner", "conced*", "landslide", "constituency", "turnout", "referendum",
        "exit poll*"),
    "legislation_passed": (
        "legislation", "legislat*", "lawmakers", "signed into law", "royal assent", "ratif*",
        "enact*", "third reading", "second reading", "final passage", "floor vote",
        "voted to pass", "passed the house", "passed the senate"),
    "appointment": (
        "appoint*", "nominat*", "nominee", "sworn in", "takes office", "took office",
        "confirmed as", "confirmation vote", "confirmation hearing", "resign*", "steps down",
        "stepped down", "step down", "sacked", "ousted", "impeach*", "cabinet", "ambassador"),
    "sanction_or_tariff": (
        "sanction*", "tariff*", "asset freeze", "export control*", "embargo*", "entity list",
        "blacklist*", "trade restrictions", "import duties", "duties on", "levies",
        "anti dumping", "countervailing", "ofac", "specially designated"),
}


def _word_pattern(word: str) -> str:
    gap = r"(?:\s\S+){0,3}\s"
    out = "".join(r"\w*" if c == "*" else gap if c == "~" else r"[\s-]?" if c in " -"
                  else "[sz]" if c == "?" else re.escape(c) for c in word.replace(" ~ ", "~"))
    tail = r"(?!\w)" if re.search(r"[\w*]$", word) else ""
    if re.match(r"\w", word):
        # The first letter, then no word character before it: a pattern that opens on a
        # literal is searched far faster than one that opens on a look-behind.
        return out[0] + r"(?<!\w\w)" + out[1:] + tail
    return out + tail


_TYPE_RES = {t: re.compile("|".join(map(_word_pattern, words))) for t, words in TYPE_WORDS.items()}
# Each word's pattern with the literal pieces it cannot match without.
_TYPE_PARTS = {t: tuple((tuple(p for p in re.split(r"[\s*~?-]+", w) if p), _word_pattern(w))
                        for w in words) for t, words in TYPE_WORDS.items()}


@lru_cache(maxsize=4096)
def _alternation(patterns: Tuple[str, ...]) -> re.Pattern:
    return re.compile("|".join(patterns))


def type_word(event_type: EventType, norm: str) -> bool:
    """The normalised text has a word that marks an event of the type (TYPE_WORDS). Only the
    words whose literal pieces are all in the text are tried, which keeps a long page cheap."""
    present: Dict[str, bool] = {}

    def has(piece: str) -> bool:
        if piece not in present:
            present[piece] = piece in norm
        return present[piece]

    live = tuple(p for parts, p in _TYPE_PARTS[event_type.value] if all(map(has, parts)))
    return bool(live) and _alternation(live).search(norm) is not None


# String fact values that say nothing on their own.
_BARE_VALUES = frozenset("yes no true false none unknown other n/a na".split())
# Slots (or kinds) whose values are stock words or not written as text on a page.
_NOT_TEXT = frozenset("qualifier date exchange chamber".split())
# Slots whose one-word values are common words ("president", "tariff", "knee").
_STOCK_SLOTS = frozenset("office action measure incident_kind injury season term result "
                         "research_field regions reason round".split())
# Words that add nothing to a name they extend: "Premier League" says no more than "Premier".
_STOCK_NAME_WORDS = frozenset(
    "league cup club united city bank group holdings exchange fund trust association "
    "federation union council party committee ministry department agency office university "
    "institute foundation partners capital ventures systems technologies labs".split()) | (
    _TAIL | _FILLER)
# A figure with a one-letter scale glued on and no currency: "3M" (the company), "100m".
_LETTER_SCALE = re.compile(r"[-+]?\d[\d.,]*\s?[kmbt](?!\w)")
_MONTHS = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
           r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
_MONTH_BEFORE = re.compile(r"(?<!\w)" + _MONTHS + r"\.?\s$")
_MONTH_AFTER = re.compile(r"\d{1,2}(?:st|nd|rd|th)?\s(?:of\s)?" + _MONTHS + r"(?!\w)")
# Words a registry's or the SEC's name of a company adds to its short name: "Meta Platforms,
# Inc.", "Amazon.com, Inc.", "JPMorgan Chase & Co.".
_CORP_WORDS = frozenset("co com corp inc ltd llc plc ag sa nv se the company platforms "
                        "chase".split())
# Words two parties' names may share without being one party: "General Motors" and "General
# Electric".
_SHARED_WORDS = frozenset("american national international general first global new "
                          "north south east west".split())
_NAME_TOKEN = re.compile(r"[^\W_]+")
_SCALES = (1e3, 1e6, 1e9, 1e12)
# Figures without a unit that may be written with a scale word ("1.2" as "$1.2 billion").
_MONEY_FIELDS = frozenset("amount deal_value fee liabilities net_income raise_amount revenue "
                          "valuation".split())
_GROUPED = re.compile(r"[-+]?\d{1,3}[,.  ]\d{3}(?!\d)")


def _unit_kind(unit: Optional[str]) -> Optional[str]:
    u = (unit or "").lower()
    if u.startswith("usd"):
        return "currency"
    if u in ("percent", "percent_change", "basis_points", "pct", "%"):
        return "pct"
    return None


def _acronym(name: str) -> bool:
    return name.isupper() and name.isalpha() and len(name) <= 5


def _name_shown(name: str, norm: str, text: str) -> bool:
    """The name on the page; a short all-capitals one only as written, so "ICE" is not
    "ice", or for four or five letters also in title case ("Opec", "Nato")."""
    if _acronym(name):
        return ticker_on(name, text) or (len(name) >= 4 and ticker_on(name.title(), text))
    return any(find(norm, f) for f in name_forms(name))


def _initialism(short: str, long: str) -> bool:
    """`short` is capitals taken in order from `long`'s main words, each word giving its first
    letter and maybe more of its letters: "ICE" for Intercontinental Exchange."""
    if not (_acronym(short) and len(short) >= 2):
        return False
    words = [w for w in name_words(long) if w not in _FILLER and w not in _TAIL]
    s = short.lower()

    def inside(part: str, word: str) -> bool:
        it = iter(word)
        return all(c in it for c in part)

    def go(i: int, j: int) -> bool:
        if j == len(words):
            return i == len(s)
        if i == len(s) or s[i] != words[j][0]:
            return False
        return any(inside(s[i + 1:k], words[j][1:]) and go(k, j + 1)
                   for k in range(i + 1, len(s) + 1))

    return len(words) >= 2 and go(0, 0)


def _alias(a: str, b: str, norm: str) -> bool:
    """`a` and `b` name one party: one is a short form or initialism of the other, or the
    page writes one in brackets after the other ("ICE (Intercontinental Exchange)")."""
    if abbreviates(a, b) or abbreviates(b, a) or _initialism(a, b) or _initialism(b, a):
        return True
    return any(find(norm, f"{x} ({y})") for p, q in ((a, b), (b, a))
               for x in name_forms(p) for y in name_forms(q))


@lru_cache(maxsize=8192)
def _own_words(name: str) -> frozenset:
    """The name's words, camel case split and as written ("north", "strive", "northstrive")."""
    return frozenset(name_words(name)) | frozenset(
        _NAME_TOKEN.findall(floor.normalize(name).text))


def _main_words(name: str) -> frozenset:
    """The name's words less company, stock and common words."""
    return _own_words(name) - _STOCK_NAME_WORDS - _CORP_WORDS - _SHARED_WORDS


def _same_party(a: str, b: str, norm: str) -> bool:
    """`a` and `b` name one party: an alias (`_alias`), or names that share a main word, as
    the registry's and the SEC's names of a subject do ("Meta", "Meta Platforms, Inc.";
    "Disney", "The Walt Disney Company")."""
    return bool(_main_words(a) & _main_words(b)) or _alias(a, b, norm)


def _added(value: str, name: str) -> Optional[str]:
    """What `value` says beyond the subject `name`, as normalised text: the longest run of its
    words that are not the name's or company words, when it is two words or more, holds a
    digit, or is one word of four letters or more that is not a stock name word ("watch series
    11" of "Apple Watch Series 11", not "league" of "Premier League"); else None."""
    own = _own_words(name)
    skip = own | _TAIL | _FILLER | _CORP_WORDS
    norm = floor.normalize(value).text
    runs, run = [], []
    for m in _NAME_TOKEN.finditer(norm):
        if m.group(0) in skip or set(name_words(m.group(0))) <= skip:
            runs.append(run)
            run = []
        else:
            run.append(m)
    runs.append(run)
    best = max(runs, key=lambda r: r[-1].end() - r[0].start() if r else -1)
    if not best:
        return None
    phrase = norm[best[0].start():best[-1].end()]
    words = [m.group(0) for m in best if m.group(0) not in _FILLER]
    if len(phrase) < 3 or not words:
        return None
    if len(words) == 1 and not any(c.isdigit() for c in phrase) and (
            len(words[0]) < 4 or words[0] in _STOCK_NAME_WORDS):
        return None
    return phrase


def _ticker_shown(ticker: str, text: str) -> bool:
    """The ticker as written; one of one or two characters only in brackets, after a colon
    or after a dollar sign."""
    if len(ticker.strip()) > 2:
        return ticker_on(ticker, text)
    return re.search(r"(?:\(\s*|:\s*|\$)" + re.escape(ticker.strip()) + r"(?![A-Za-z0-9])",
                     text) is not None


def _figure_shown(value: float, unit: Optional[str], numbers, page: str,
                  field_name: Optional[str] = None) -> bool:
    """The page prints the figure in a form that marks it: with its currency or percent sign,
    or as a number of 10 or more that is not a bare year. A money figure without a unit
    (`_MONEY_FIELDS`) may be printed with a scale word ("1.2" as "$1.2 billion"); a count may
    not ("5" is not "5,000 square feet")."""
    size = abs(value) * floor.unit_magnitude(unit)
    if size == 0:
        return False
    kind = _unit_kind(unit)
    scaled = not unit and field_name in _MONEY_FIELDS
    sizes = [size] + ([size * m for m in _SCALES] if scaled else [])
    for n in numbers:
        got = abs(n.value)
        if not any(floor.value_grounded(s, got) for s in sizes):
            continue
        if (unit or "").startswith("usd") and n.unit.startswith("currency:") and \
                n.unit != "currency:USD":
            continue
        if kind and n.unit.startswith(kind):
            return True
        if kind and got < 1000:
            continue
        if n.unit == "count" and (kind == "currency" or not unit) and _LETTER_SCALE.match(
                page, n.offset):
            continue
        if got == int(got) and 1900 <= got <= 2100 and not _GROUPED.match(page, n.offset):
            continue
        if 1 <= got < 32 and n.unit == "count" and (
                _MONTH_BEFORE.search(page, max(0, n.offset - 12), n.offset)
                or _MONTH_AFTER.match(page, n.offset)):
            continue
        if got >= 10 or (not kind and round(got, 1) != got):
            return True
    return False


def second_mark(scoop: Scoop, name: str, norm: str, text: str,
                page: Optional[str] = None) -> bool:
    """Besides the subject `name`, the page (`text`; normalised, `norm`, and with its line
    breaks kept, `page`) shows one more mark of the scoop: another party's name, a ticker, a
    fact it gives as text or figure, or a word of its type (TYPE_WORDS). Another subject, or a
    fact that is one, is a mark only when it names another party (`_same_party`); the API
    serves the subject's other names with it. A text fact that is another name for the
    subject (`_alias`) is not a mark; one that shares words with `name` counts by what it adds
    (`_added`); an initialism must be written in capitals."""
    own = _own_words(name)
    forms = {s for s in scoop.subjects if s == name or _same_party(s, name, norm)}

    def on_page(value: str) -> bool:
        if not own & _own_words(value):
            return _name_shown(value, norm, text)
        added = _added(value, name)
        return added is not None and bool(find(norm, added))

    def shown(value: str) -> bool:
        return value not in forms and not _alias(value, name, norm) and on_page(value)

    # `forms` already holds every subject that is an alias of `name`.
    if any(on_page(s) for s in scoop.subjects if s not in forms):
        return True
    if any(_ticker_shown(t, text) for t in scoop.tickers):
        return True
    if type_word(scoop.event_type, norm):
        return True
    fields = TYPE_SPEC[scoop.event_type].fields
    numbers = None
    for f in scoop.facts:
        value, kind = f.get("value"), (fields.get(f.get("field")) or {}).get("kind")
        if isinstance(value, str):
            if kind == "ticker":
                if _ticker_shown(value, text):
                    return True
                continue
            v = floor.normalize(value).text
            if kind in _NOT_TEXT or f.get("field") in _NOT_TEXT or len(v) < 3 or \
                    v in _BARE_VALUES:
                continue
            if f.get("field") in _STOCK_SLOTS and len(v.split()) < 2:
                continue
            if shown(value):
                return True
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if numbers is None:
                page = floor.normalize(text, lines=True).text if page is None else page
                numbers = floor.parse_numbers(page, scale_next=True)
            if _figure_shown(value, f.get("unit"), numbers, page, f.get("field")):
                return True
    return False


def held_claim(scoop: Scoop, url: str, text: str) -> Optional[Claim]:
    """The scoop as checked on a page: the subject name the page shows, its other facts less the
    figures the page does not show; None when a figure its type needs is missing."""
    # One pass: the text without its line breaks is the plain normalised text.
    page = floor.normalize(text, lines=True).text
    norm = page.replace("\n", " ")
    name = next((s for s in scoop.subjects if _name_shown(s, norm, text)), None)
    if name is None or not second_mark(scoop, name, norm, text, page):
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
