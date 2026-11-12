"""The deterministic check every event claim runs on its own fetched page, before the page
judge. Gates run in order and the first failure decides.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from alpharidge_ai.schema import TYPE_SPEC, canonical_json, sha256_hex
from alpharidge_ai.schema.models import QUOTE_MAX, QUOTE_MIN
from alpharidge_ai.verify import floor, official
from alpharidge_ai.verify.fetch import FetchedPage
from alpharidge_ai.verify.language_detector import detect_language

SUPPORTED, REJECTED = "supported", "rejected"
UNREAD = "unread"
# The slot that names a type's subject when it is a fact rather than an actor.
PIVOT_SLOTS = {t.value: s.subject_slot for t, s in TYPE_SPEC.items() if s.subject_slot}
# Other name slots the page must show; they do not set identity.
NAME_SLOTS = {t.value: s.name_slots for t, s in TYPE_SPEC.items() if s.name_slots}


# The oldest event date and page publish time accepted, before receipt.
MAX_AGE_D = 30
FUTURE_SLACK_D = 1
PREPUBLISH_SLACK_S = 0
LANGUAGE = "en"


@dataclass(frozen=True)
class CheckResult:
    verdict: str
    reason: Optional[str] = None
    spans: Tuple[Tuple[int, int], ...] = ()
    date_gate: Optional[bool] = None
    # What failed, in the claim's own terms: {"missing": ["actor:<name>", "fact:<field>"]},
    # {"date": "event_date" | "published"} or {"quote": "missing" | "not_on_page" |
    # "no_subject"}.
    detail: Optional[Dict[str, object]] = None
    # A scoop on a page that could not be read, held on its event date alone (check_unread).
    unread: bool = False

    @property
    def passed(self) -> bool:
        return self.verdict == SUPPORTED


def _fail(reason: str, **kw) -> CheckResult:
    return CheckResult(REJECTED, reason, **kw)


# ---- type rules ----------------------------------------------------------------------

# Claims received from this time on: a breakthrough's subject is an organisation, never a
# paper's author list, and its kind is one the type lists.
BREAKTHROUGH_RULES_FROM = datetime(2026, 11, 12, 21, 58, tzinfo=timezone.utc)
_ET_AL = re.compile(r"\bet\s+al\b", re.I)
_AUTHOR = re.compile(r"\b[A-Z][\w'’-]+,\s*(?:[A-Z]{1,2}\.\s*-?\s*)+")


def author_list(name: Optional[str]) -> bool:
    """A name written as a paper's authors: "et al." or two or more "Surname, A." entries."""
    return bool(name) and bool(_ET_AL.search(name) or len(_AUTHOR.findall(name)) >= 2)


def type_rule(claim, received_at: datetime) -> Optional[CheckResult]:
    if getattr(claim.event_type, "value", claim.event_type) != "breakthrough" or (
            received_at < BREAKTHROUGH_RULES_FROM):
        return None
    spec = TYPE_SPEC[claim.event_type]
    for a in claim.actors:
        if (not a.role or a.role in spec.subject_roles) and author_list(a.name or a.id):
            return _fail("not_an_event", detail={"subject": a.name or a.id})
    listed = spec.fields.get("kind", {}).get("values", {})
    if any(f.field == "kind" and str(f.value) not in listed for f in claim.facts):
        return _fail("not_an_event", detail={"fact": "kind"})
    return None


# ---- names ---------------------------------------------------------------------------

_SUFFIXES = frozenset(
    "inc corp corporation co ltd limited plc llc sa ag nv se group holdings company".split())


_WORD = re.compile(r"\w")


def find(text: str, phrase: str) -> List[Tuple[int, int]]:
    """Whole-word occurrences of `phrase` in normalised text, without overlaps."""
    if not phrase:
        return []
    out, at, n = [], text.find(phrase), len(phrase)
    while at >= 0:
        end = at + n
        if not ((at and _WORD.match(text, at - 1)) or (end < len(text) and _WORD.match(text, end))):
            out.append((at, end))
            at = text.find(phrase, end)
        else:
            at = text.find(phrase, at + 1)
    return out


def name_forms(name: str) -> List[str]:
    """A name normalised, with and without its company suffix."""
    full = floor.normalize(name).text
    words = full.replace(",", " ").split()
    while len(words) > 1 and words[-1].strip(".") in _SUFFIXES:
        words.pop()
    short = " ".join(words).rstrip(".,")
    return [f for f in dict.fromkeys((full, short)) if f]


def names_of(claim) -> List[str]:
    """The names the page must show: every actor (an actor given without a name must show
    its id) and the type's subject and name slots when facts."""
    names = [a.name or a.id for a in claim.actors]
    slots = (PIVOT_SLOTS.get(claim.event_type.value),) + NAME_SLOTS.get(claim.event_type.value, ())
    names += [f.value for f in claim.facts if f.field in slots and isinstance(f.value, str)]
    return names


# ---- tickers and exchanges ----------------------------------------------------------

EXCHANGES = tuple(tuple(floor.normalize(n).text for n in g) for g in json.loads(
    (Path(__file__).with_name("data") / "exchanges.json").read_text(encoding="utf-8"))["groups"])


def exchange_names(value: str) -> List[str]:
    """The names a claimed exchange may go by on a page: its group's (data/exchanges.json),
    by the longest name it starts with, else the value itself."""
    v = floor.normalize(value).text.removeprefix("the ")
    best = max(((len(n), g) for g in EXCHANGES for n in g if v == n or v.startswith(n + " ")),
               default=(0, (v,)))
    return list(best[1])


def ticker_on(ticker: str, text: str) -> bool:
    """The ticker exactly as written, as a whole token."""
    t = r"\s+".join(map(re.escape, str(ticker).split()))
    return bool(t) and re.search(r"(?<![A-Za-z0-9])" + t + r"(?![A-Za-z0-9])", text) is not None


def listing_missing(claim, text: str, norm: Optional[str] = None) -> List[str]:
    """Ticker and exchange facts the page does not show; they are checked here, not judged."""
    spec = TYPE_SPEC.get(claim.event_type)
    norm = floor.normalize(text).text if norm is None else norm
    out = []
    for f in claim.facts:
        kind = spec.fields.get(f.field, {}).get("kind") if spec else None
        if kind == "ticker" and not ticker_on(f.value, text):
            out.append(f"fact:{f.field}")
        elif kind == "exchange" and not any(find(norm, n) for n in exchange_names(str(f.value))):
            out.append(f"fact:{f.field}")
    return out


# ---- the quote -----------------------------------------------------------------------

_ENDS = re.compile(r"[.!?][\"')\]]*$")
_BULLETS = "*•▪◦‣●■◆○"
# What a quote may follow besides a line start: a sentence end, a bullet, a dash with a
# space before it, or a dateline's dash; not a colon, semicolon, opening quote mark or word
# hyphen.
_OPENS = re.compile(r"(?:[.!?][\"')\]]*|[" + _BULLETS + r"]|\s-)\s*$")
_DATELINE = re.compile(r"(?:—|--)\s*$")
# A bullet item with no full stop ends at the next bullet, a line break or the page end.
_ITEM_END = re.compile(r"[^\S\n]*(?:[" + _BULLETS + r"\n]|\Z)")
# A line break inside a quote must follow a hard-wrapped line.
WRAP_MIN_CHARS = 60


def _opens(text: str, a: int) -> bool:
    line = text[max(0, a - 200):a].rsplit("\n", 1)[-1]
    head = floor.normalize(text[max(0, a - 8):a]).text
    return (not line.strip() or bool(_OPENS.search(" " + head))
            or bool(_DATELINE.search(text[max(0, a - 4):a])))


def _bullet_item(text: str, a: int, b: int) -> bool:
    """The span is a whole bullet item: a bullet (or a dash starting the line) before it and
    the item's end after it."""
    line = text[max(0, a - 200):a].rsplit("\n", 1)[-1].strip()
    mark = line.endswith(tuple(_BULLETS)) or line in ("-", "–")
    return mark and bool(_ITEM_END.match(text, b))


def _wrapped(text: str, a: int, b: int) -> bool:
    for i in (i for i in range(a, b) if text[i] == "\n"):
        line = text[text.rfind("\n", 0, i) + 1:i].strip()
        if len(line) < WRAP_MIN_CHARS or line.endswith(tuple(".!?:;")):
            return False
    return True


def quote_span(quote: Optional[str], text: str,
               norm: Optional[floor.Normalized] = None) -> Optional[Tuple[int, int]]:
    """Where the quote stands in the page's visible text (`norm`: the text normalised) as
    whole sentences of one paragraph, or None."""
    q = floor.normalize(quote).text
    if not q or "|" in q:
        return None
    ends = bool(_ENDS.search(q))
    if not ends and any(c in q for c in _BULLETS[1:]):
        return None
    norm = floor.normalize(text) if norm is None else norm
    for s, e in find(norm.text, q):
        a, b = norm.to_original(s, e)
        if (ends or _bullet_item(text, a, b)) and _opens(text, a) and _wrapped(text, a, b):
            return a, b
    return None


# A term the page defines right after a name, in one of up to three brackets in a row:
# 'Atkore Inc. (NYSE: ATKR) ("Atkore" or the "Company")', "Monetary Policy Committee (MPC)".
_DEFINED = re.compile(r"[^()\"]{0,40}((?:\([^()]{1,80}\)[\s,]*){1,3})")
_BRACKET = re.compile(r"\(([^()]{1,80})\)")
_TERM = re.compile(r"\"(?:the )?([^\"]{2,40})\"")
_REFERENTS = frozenset("company registrant issuer bank committee board fund trust bureau "
                       "agency ministry department office group".split())
_SMALL = frozenset("of the and for on in".split())


def _initials(form: str, inner: str) -> List[str]:
    words = form.split()
    ok = {"".join(w[0] for w in words), "".join(w[0] for w in words if w not in _SMALL)}
    return [inner.strip()] if inner.strip() in ok and len(words) > 1 else []


def subject_forms(names, page_norm: str) -> List[str]:
    """The forms of the subject names a quote may use: the names, terms the page defines
    for them, and "the <word>" for a generic word of a name."""
    out = [f for n in names for f in name_forms(n)]
    for f in list(out):
        for _, e in find(page_norm, f):
            m = _DEFINED.match(page_norm, e)
            for inner in _BRACKET.findall(m.group(1)) if m else ():
                out += [t.strip() for t in _TERM.findall(inner)] + _initials(f, inner)
    words = {w for n in names for w in floor.normalize(n).text.split()}
    return list(dict.fromkeys(out + [f"the {w}" for w in sorted(_REFERENTS & words)]))


def quote_problem(claim, text: str) -> Optional[str]:
    """Why the claim's quote fails, or None."""
    if not claim.quote:
        return "missing"
    if quote_span(claim.quote, text) is None:
        return "not_on_page"
    q = floor.normalize(claim.quote).text
    forms = subject_forms(names_of(claim), floor.normalize(text).text)
    return None if any(find(q, f) for f in forms) else "no_subject"


def verdict_hash(record: dict) -> str:
    """The hash of a stored page-judge record, as signed in the ledger."""
    return sha256_hex(canonical_json(record))


# ---- dates ---------------------------------------------------------------------------

def _day(dt) -> datetime:
    return datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)


def _date_gate(claim, page: FetchedPage, received: datetime) -> Optional[Tuple[str, str]]:
    """The failed gate and the date it failed on, the event's or the page's."""
    gate = timedelta(days=MAX_AGE_D)
    slack = timedelta(days=FUTURE_SLACK_D)
    day = _day(claim.event_date)
    if day > _day(received) + slack:
        return "future_date", "event_date"
    body_dated = official.body_dated(page.final_url or page.url)
    pub = None if page.date_inferred and not body_dated else page.published_at
    if pub is None:
        return "no_publish_date", "published"
    if received < pub - timedelta(seconds=PREPUBLISH_SLACK_S):
        return "pre_published", "published"
    if received - pub > gate:
        return "stale_date", "published"
    # The event happened by the page's publish day in the latest time zone (UTC+14).
    if day > _day(pub + timedelta(hours=14)):
        return "future_date", "event_date"
    if received - (day + timedelta(days=1)) > gate:
        return "stale_date", "event_date"
    return None


def check_claim(claim, page: FetchedPage, received_at: datetime,
                excluded: Optional[str] = None) -> CheckResult:
    # 1-2. Page rules, then a page that can't be read.
    if excluded:
        return _fail("page_type")
    ruled = type_rule(claim, received_at)
    if ruled:
        return ruled
    if not page.ok:
        return _fail(page.reason or "fetch_failed")
    lang = detect_language(page.text[:5000])
    if lang.confidence > 0 and lang.code != LANGUAGE:
        return _fail("language")

    # 3. Dates.
    late = _date_gate(claim, page, received_at)
    if late:
        return _fail(late[0], date_gate=False, detail={"date": late[1]})

    # 4. Every name the claim gives is on the page, and every figure's size.
    norm = floor.normalize(page.text)
    text = norm.text
    spans, missing = [], []
    actors = {a.name or a.id for a in claim.actors}
    fields = {f.value: f.field for f in claim.facts if isinstance(f.value, str)}
    for name in names_of(claim):
        hits = [h for form in name_forms(name) for h in find(text, form)]
        if hits:
            spans.append(norm.to_original(*min(hits)))
        else:
            missing.append(f"actor:{name}" if name in actors else f"fact:{fields.get(name)}")
    numbers = floor.parse_numbers(floor.normalize(page.text, lines=True).text)
    spec = TYPE_SPEC.get(claim.event_type)
    for fact in claim.facts:
        judged = spec and spec.fields.get(fact.field, {}).get("judged")
        if not judged and not floor.visible(fact, numbers):
            missing.append(f"fact:{fact.field}")
    missing += listing_missing(claim, page.text, text)
    if missing:
        return _fail("unsupported", date_gate=True,
                     detail={"missing": list(dict.fromkeys(missing))})

    # 5. The quote.
    why = quote_problem(claim, page.text)
    if why:
        return _fail("unsupported", date_gate=True, detail={"quote": why})
    return CheckResult(SUPPORTED, spans=(*spans, quote_span(claim.quote, page.text)),
                       date_gate=True)


def check_unread(claim, received_at: datetime, excluded: Optional[str] = None) -> CheckResult:
    """A scoop whose page could not be read: the page rules, then the event date's gates."""
    if excluded:
        return _fail("page_type")
    ruled = type_rule(claim, received_at)
    if ruled:
        return ruled
    day, gate = _day(claim.event_date), timedelta(days=MAX_AGE_D)
    if day > _day(received_at) + timedelta(days=FUTURE_SLACK_D):
        why = "future_date"
    elif received_at - (day + timedelta(days=1)) > gate:
        why = "stale_date"
    else:
        return CheckResult(SUPPORTED, date_gate=True, unread=True)
    return _fail(why, date_gate=False, detail={"date": "event_date"}, unread=True)


def check_scoop(claim, page: FetchedPage, received_at: datetime,
                excluded: Optional[str] = None) -> CheckResult:
    """A scoop's page: check_unread when it could not be read, else check_claim."""
    if page.unread:
        return check_unread(claim, received_at, excluded=excluded)
    return check_claim(claim, page, received_at, excluded=excluded)


# ---- a quote for a page the claim did not come from ----------------------------------

# A sentence ends at a stop before a line end or a capital, digit or opening mark.
_SENT_END = re.compile(r"[.!?][\"')\]]*(?=\s*$|\s+[\"'(\[A-Z0-9])", re.M)


def _sentences(text: str) -> List[Tuple[int, int]]:
    out = []
    for line in re.finditer(r"[^\n]+", text):
        a = line.start()
        for m in _SENT_END.finditer(text, line.start(), line.end()):
            while text[a].isspace():
                a += 1
            out.append((a, m.end()))
            a = m.end()
    return out


# Distinct sentences pick_quote looks for on the whole page, at most.
PICK_TRIES_MAX = 1000


def pick_quote(claim, text: str) -> Optional[str]:
    """A quote for the claim from a page's visible text: the first sentence (else two in a
    row) that stands as a quote and names a subject, the first that also shows one of the
    claim's figures when any does."""
    page = floor.normalize(text)
    forms = subject_forms(names_of(claim), page.text)
    figures = [f for f in claim.facts if not isinstance(f.value, str)]
    sents = _sentences(text)
    first = None
    # Per distinct sentence: None when it is no quote naming a subject, else whether it shows a
    # figure.
    seen: Dict[str, Optional[bool]] = {}
    tries = 0
    for width in (1, 2):
        for i in range(len(sents) - width + 1):
            a, b = sents[i][0], sents[i + width - 1][1]
            q = text[a:b].strip()
            if not QUOTE_MIN <= len(q) <= QUOTE_MAX or "\n" in q:
                continue
            if q not in seen:
                seen[q] = None
                if not any(find(floor.normalize(q).text, f) for f in forms):
                    continue
                if tries >= PICK_TRIES_MAX:
                    return first
                tries += 1
                if quote_span(q, text, page) is None:
                    continue
                nums = floor.parse_numbers(floor.normalize(q, lines=True).text)
                seen[q] = any(f.value and floor.visible(f, nums) for f in figures)
            if seen[q]:
                return q
            if seen[q] is not None:
                first = first or q
        if first:
            return first
    return None


def unstated_figures(claim, text: str) -> List[str]:
    """The claim's figures, tickers and exchanges the page does not show (left to the judge
    when the type says so)."""
    spec = TYPE_SPEC.get(claim.event_type)
    numbers = floor.parse_numbers(floor.normalize(text, lines=True).text)
    out = [f.field for f in claim.facts if not isinstance(f.value, str)
           and not (spec and spec.fields.get(f.field, {}).get("judged"))
           and not floor.visible(f, numbers)]
    return out + [m.split(":", 1)[1] for m in listing_missing(claim, text)]
