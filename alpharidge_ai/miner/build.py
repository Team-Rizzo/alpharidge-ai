"""Claim building: a fetched page to event claims whose every field is read from the page.

The type comes from `TYPE_SPEC` keywords in a sentence that names a linked actor; facts
come from that sentence and the next. Each claim then runs the shared deterministic check
on the same page, and anything that wouldn't pass is dropped rather than sent. The page
judge is not run here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from alpharidge_ai.schema import (
    OPEN_TYPES, TYPE_SPEC, Actor, Claim, ClaimKind, EventType, Fact, IdScheme, Place, Stage,
    has_required, headline, text_problem,
)
from alpharidge_ai.schema.models import QUOTE_MAX, QUOTE_MIN
from alpharidge_ai.verify import floor
from alpharidge_ai.verify.check import LANGUAGE, check_claim
from alpharidge_ai.verify.fetch import FetchedPage
from alpharidge_ai.verify.language_detector import detect_language
from alpharidge_ai.miner.link import Linker, Mention, PlaceFinder

MAX_ACTORS = 4

# Sentence ends (not "Inc. (NYSE: ABC)"), line breaks, bullets and dateline dashes
# ("2026—Acme", "(WIRE) -- Acme").
_SENTENCE = re.compile(r"(?<=[.!?])[\"'”’)]?\s+(?=[\"'“‘]?[A-Z0-9]|\((?![\w ]+:)[A-Z0-9])|\n+"
                       r"|\s*[•▪◦‣●■◆○]\s*|\s*(?:—|--)\s*(?=[A-Z])")

# Slot -> (kind, cue words that must precede the figure in the sentence).
_CURRENCY, _PERCENT, _COUNT = "currency", "percent", "count"
_SLOTS: Dict[str, Tuple[str, Optional[str]]] = {
    "deal_value": (_CURRENCY, r"deal|valued|value|for|acquire|buy|purchase"),
    "raise_amount": (_CURRENCY, r"rais|offering|proceeds"),
    "revenue": (_CURRENCY, r"revenue|sales"),
    "net_income": (_CURRENCY, r"net income|profit|earnings of"),
    "liabilities": (_CURRENCY, r"liabilities|debts?"),
    "offer_price": (_CURRENCY, r"priced|per share|offer price"),
    "new_rate": (_PERCENT, r"rate|to|at"),
    "stake_pct": (_PERCENT, r"stake"),
    "pct_workforce": (_PERCENT, r"workforce|staff|employees"),
    "jobs_cut": (_COUNT, r"jobs|employees|workers|positions|staff|roles"),
    "records_affected": (_COUNT, r"records|accounts|customers|users"),
}
_COUNT_AFTER = {
    "jobs_cut": re.compile(r"^\s*(?:jobs|employees|workers|positions|staff|roles)\b"),
    "records_affected": re.compile(r"^\s*(?:\w+\s)?(?:records|accounts|customers|users)\b"),
}
# A head count offered something, not cut.
_NOT_CUT = re.compile(r"(?:buyout|voluntary|offer|eligib)\w*\W+(?:\S+\s+){0,4}$")
# Wording of a preview of a scheduled release rather than the release.
_PREVIEW = re.compile(r"\b(?:set|expected|scheduled|due) to (?:report|announce|post|release)|"
                      r"\bwill (?:report|announce|post|release)\b|\bahead of\b|\bpreview",
                      re.I)
_RELEASES = (EventType.EARNINGS, EventType.MONETARY_POLICY, EventType.ECONOMIC_DATA)
_YEAR = re.compile(r"\b(?:19|20)\d\d\b")
_BPS = re.compile(r"(?<![\w.])(\d{1,3})\s*(?:basis points?|bps|bp)\b", re.I)
_PERIOD = re.compile(r"\b(?:(?:first|second|third|fourth)[- ]quarter|Q[1-4](?: (?:FY)?20\d\d)?|"
                     r"(?:fiscal|full)[- ]year(?: 20\d\d)?)\b", re.I)
_CHAPTER = re.compile(r"\bChapter (7|11|15)\b")
# Words before the company being bought, and before the buyer.
_TAKES = re.compile(r"(?:acquir\w*|acquisition of|buy\w*|bought|purchas\w*|takeover of|"
                    r"merge\w* with)\W+(?:\S+\s+){0,2}$", re.I)
_BY = re.compile(r"\bby\s+(?:the\s+)?$", re.I)
# An exchange named only for a listing: "Nasdaq-listed", "(NYSE: ABC)".
_LISTING = re.compile(r"-listed\b|\s*:\s*[A-Z]")
# The verb that gives a rate change its sign.
_DOWN = re.compile(r"(?<!\w)(?:cut(?:s|ting)?|lower(?:s|ed|ing)?|reduc(?:e|es|ed|ing)|"
                   r"trim(?:s|med|ming)?|slash(?:es|ed|ing)?|eas(?:e|es|ed|ing))(?!\w)", re.I)
_UP = re.compile(r"(?<!\w)(?:rais(?:e|es|ed|ing)|hik(?:e|es|ed|ing)|increas(?:e|es|ed|ing)|"
                 r"lift(?:s|ed|ing)?|boost(?:s|ed|ing)?|tighten(?:s|ed|ing)?)(?!\w)", re.I)


@dataclass(frozen=True)
class Candidate:
    """One event read off a page, before novelty is known."""
    claim: Claim
    sentence: str


def sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE.split(text or "") if s and s.strip()]


def _has(text: str, phrase: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text) is not None


def keyword_hits(sentence: str, event_type: EventType) -> int:
    text = floor.normalize(sentence).text
    return sum(1 for k in TYPE_SPEC[event_type].keywords if _has(text, k))


def stage_of(sentence: str, event_type: EventType) -> Optional[Stage]:
    """The stage the sentence's wording names; on a tie, the later one."""
    spec = TYPE_SPEC[event_type]
    text = floor.normalize(sentence).text
    hits = [(sum(1 for k in spec.stage_keywords.get(s, ()) if _has(text, k)), i, s)
            for i, s in enumerate(spec.all_stages)]
    best = max(hits, default=(0, 0, None))
    return best[2] if best[0] else None


def _money(value: float, unit: str) -> Tuple[float, str]:
    code = unit.split(":", 1)[1] if ":" in unit else ""
    if code != "USD":
        return value, code or "USD"
    for scale, name in ((1e12, "usd_trillions"), (1e9, "usd_billions"), (1e6, "usd_millions")):
        if abs(value) >= scale:
            return round(value / scale, 6), name
    return value, "usd_per_unit"


def _sign(text: str, at: int) -> int:
    """-1 or 1 for the cut or hike verb of the figure at `at`: the last one before it, else
    the first after it; 1 when there is none."""
    verbs = sorted((m.start(), d) for d, pat in ((-1, _DOWN), (1, _UP))
                   for m in pat.finditer(text))
    before = [d for o, d in verbs if o < at]
    return before[-1] if before else next((d for o, d in verbs if o > at), 1)


def _cued(text: str, offset: int, cue: Optional[str]) -> bool:
    if cue is None:
        return True
    return re.search(r"(?:" + cue + r")\w*\W+(?:\S+\s+){0,4}\S*$", text[max(0, offset - 60):offset]) \
        is not None


def extract_facts(event_type: EventType, window: str, linker: Linker) -> List[Fact]:
    """Figures and names for the type's slots, found in `window`."""
    spec = TYPE_SPEC[event_type]
    norm = floor.normalize(window).text
    numbers = floor.parse_numbers(norm)
    currency_slots = [s for s in spec.slots if _SLOTS.get(s, ("",))[0] == _CURRENCY]
    used, facts = set(), []

    def take(slot: str, kind: str, cue: Optional[str]) -> Optional[Fact]:
        if kind == _CURRENCY and len(currency_slots) == 1:
            cue = None
        for n in numbers:
            if n.offset in used or not n.literal:
                continue
            if kind == _CURRENCY and n.unit.startswith("currency:") and _cued(norm, n.offset, cue):
                used.add(n.offset)
                value, unit = _money(n.value, n.unit)
                return Fact(field=slot, value=value, unit=unit)
            if kind == _PERCENT and n.unit == "pct" and not n.step \
                    and _cued(norm, n.offset, cue):
                used.add(n.offset)
                return Fact(field=slot, value=n.value, unit="percent")
            if kind == _COUNT and n.unit == "count" and n.value >= 1:
                tail = norm[n.offset:].split(None, 1)
                before = norm[max(0, n.offset - 60):n.offset]
                offered = slot == "jobs_cut" and _NOT_CUT.search(before)
                if len(tail) == 2 and _COUNT_AFTER[slot].match(" " + tail[1]) and not offered:
                    used.add(n.offset)
                    return Fact(field=slot, value=n.value, unit="count")
        return None

    for slot in spec.slots:
        fact = None
        if slot in _SLOTS:
            fact = take(slot, *_SLOTS[slot])
        elif slot == "change_bps":
            m = _BPS.search(window)
            sign = m and _sign(window, m.start())
            fact = m and Fact(field=slot, value=sign * float(m.group(1)), unit="basis_points")
        elif slot == "period":
            m = _PERIOD.search(window)
            fact = m and Fact(field=slot, value=m.group(0))
        elif slot == "chapter":
            m = _CHAPTER.search(window)
            fact = m and Fact(field=slot, value=m.group(0))
        elif slot == "exchange":
            name = next((n for n in linker.names(slot) if _has(window, n)), None)
            fact = name and Fact(field=slot, value=name)
        if fact:
            facts.append(fact)
    return facts


def _old_year(sentence: str, published: datetime) -> bool:
    """A year before the page's, other than last year's early in the year (annual and
    fourth-quarter results)."""
    late = published.year - (1 if published.month <= 3 else 0)
    return any(int(y) < late for y in _YEAR.findall(sentence))


def _not_an_event(event_type: EventType, sentence: str, window: Sequence[str],
                  mentions: Sequence[Mention], published: datetime) -> bool:
    """Wording the reference builder reads as no new event of the type."""
    if event_type in _RELEASES and any(_PREVIEW.search(s) for s in window):
        return True
    if event_type == EventType.IPO_LISTING:
        # Already listed ("Nasdaq-listed"), or a move between exchanges.
        return bool(re.search(r"\w-listed\b", sentence)) or len(
            {m.entity.id for m in mentions if m.entity.kind == "exchange"}) > 1
    return event_type == EventType.EARNINGS and _old_year(sentence, published)


def _with(claim: Claim, **fields) -> Claim:
    return Claim(**{**claim.model_dump(), **fields})


def _actors(mentions: Sequence[Mention], roles: Mapping[str, str] = ()) -> List[Actor]:
    """Actors in mention order. With roles, known kinds the type doesn't list are left out."""
    roles = dict(roles)
    seen, out = set(), []
    for m in mentions:
        key, kind = (m.entity.scheme, m.entity.id), m.entity.kind
        if key in seen or (roles and kind and kind not in roles):
            continue
        seen.add(key)
        out.append(Actor(id_scheme=m.entity.scheme, id=m.entity.id, role=roles.get(kind),
                         name=m.surface))
    return out[:MAX_ACTORS]


def _target(sentence: str, mentions: Sequence[Mention], actors: Sequence[Actor]) -> List[Actor]:
    """Deal actors with the company being bought marked `target`, when the sentence says which."""
    def key(a: Actor) -> Tuple[str, str]:
        return IdScheme(a.id_scheme).value, a.id

    named = {key(a) for a in actors}
    ms = [m for m in mentions if (m.entity.scheme, m.entity.id) in named]
    hit = next((m for m in ms if _TAKES.search(sentence[max(0, m.start - 40):m.start])), None)
    if hit is None:
        by = [m for m in ms if _BY.search(sentence[max(0, m.start - 12):m.start])]
        hit = next((m for m in ms if by and m not in by), None)
    if hit is None:
        return list(actors)
    pick = (hit.entity.scheme, hit.entity.id)
    return [Actor(**{**a.model_dump(), "role": "target"}) if key(a) == pick else a
            for a in actors]


class ClaimBuilder:
    def __init__(self, linker: Linker, categories: Iterable[str] = (), lead_sentences: int = 6,
                 places: Optional[PlaceFinder] = None):
        self.linker = linker
        self.places = places or PlaceFinder()
        cats = set(categories)
        self.types = [t for t in OPEN_TYPES
                      if not cats or TYPE_SPEC[t].category.value in cats]
        self.lead = lead_sentences

    def passes(self, claim: Claim, page: FetchedPage, now: datetime) -> bool:
        return check_claim(claim, page, now).passed

    def _claim(self, event_type: EventType, actors: List[Actor], facts: List[Fact],
               page: FetchedPage, **kw) -> Claim:
        return Claim(
            headline=headline(event_type, [a.name for a in actors], kw.get("stage")),
            event_type=event_type, actors=actors, facts=facts,
            event_date=page.published_at.date(), source_url=page.url, **kw)

    def place(self, event_type: EventType, window: Sequence[str]) -> Optional[Place]:
        """The country the window names, for types keyed by place; none when it names several."""
        if TYPE_SPEC[event_type].place_level == "none":
            return None
        for sentence in window:
            found = self.places.countries_in(sentence)
            if found:
                return Place(iso2=found[0]) if len(found) == 1 else None
        return None

    def supported_facts(self, claim: Claim, page: FetchedPage, now: datetime) -> List[Fact]:
        """The claim's facts the page states, one at a time."""
        keep = []
        for fact in claim.facts:
            probe = _with(claim, facts=[fact])
            if self.passes(probe, page, now):
                keep.append(fact)
        return keep

    def build(self, page: FetchedPage, now: Optional[datetime] = None) -> List[Candidate]:
        """At most one event claim per page: the first lead sentence that reads as one."""
        now = now or datetime.now(timezone.utc)
        if not page.ok or page.published_at is None:
            return []
        lang = detect_language(page.text[:5000])
        if lang.confidence > 0 and lang.code != LANGUAGE:
            return []
        lead = sentences(page.text)[:self.lead]
        for i, sentence in enumerate(lead):
            mentions = self.linker.mentions(sentence)
            if not mentions:
                continue
            ranked = sorted(((keyword_hits(sentence, t), -n, t) for n, t in enumerate(self.types)),
                            reverse=True)
            for hits, _, event_type in ranked:
                if hits == 0:
                    break
                claim = self._candidate(event_type, mentions, sentence, lead[i:i + 2], page, now)
                if claim is not None:
                    return [Candidate(claim, sentence)]
        return []

    def _candidate(self, event_type, mentions, sentence, window, page, now) -> Optional[Claim]:
        if _not_an_event(event_type, sentence, window, mentions, page.published_at) or not (
                QUOTE_MIN <= len(sentence) <= QUOTE_MAX):
            return None
        named = [m for m in mentions
                 if not (m.entity.kind == "exchange" and _LISTING.match(sentence, m.end))]
        actors = _actors(named, TYPE_SPEC[event_type].actor_roles)
        if not actors:
            return None
        if "target" in TYPE_SPEC[event_type].required_slots:
            actors = _target(sentence, mentions, actors)
        facts = extract_facts(event_type, " ".join(window), self.linker)
        claim = self._claim(event_type, actors, facts, page, kind=ClaimKind.EVENT,
                            stage=stage_of(sentence, event_type),
                            place=self.place(event_type, window), quote=sentence)
        if facts:
            claim = _with(claim, facts=self.supported_facts(claim, page, now))
        if not has_required(claim) or text_problem(claim):
            return None
        return claim if self.passes(claim, page, now) else None
