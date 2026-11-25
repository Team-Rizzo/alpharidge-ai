"""Per-type rules. The category of a claim comes from its type, never from its label.

The rules live in `data/types.json` so miners can read them without importing this package.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple
from urllib.parse import urlsplit

from alpharidge_ai.schema.enums import Category, EventType, Stage

TYPES_PATH = Path(__file__).with_name("data") / "types.json"
PLACE_LEVELS = ("none", "country", "admin1")
# Price tiers: "high" for types that usually break as news before any filing or release.
TIERS = ("low", "high")
# Points per tier, before trust; the API and validators must agree.
TIER_PRICE = {"low": 3, "high": 12}
# Pay settings; the API and validators must agree. Both tables live here: "today" applies until
# the API serves a switch to "new" with its start (`set_pay_switch`), and each claim is priced
# by the table in force at its receipt, so every side prices an epoch that straddles the start
# alike. A local EL_<KEY> in the environment still wins, read when used rather than when this
# module is imported, so env files loaded later still apply.
# Event route points are the tier price × EVENT_PAY_X. A confirmed scoop pays SCOOP_PREMIUM × the
# tier price when no trusted page about it was out before its receipt, else the event rate; under
# a LEAD_TABLES table the premium scales with that lead (`lead_multiple`).
# A scoop that closes unconfirmed or that a trusted page contradicts costs SCOOP_MISS_DEBT_X × its
# 1x pay, at SCOOP_DEBT_TRUST: the highest trust the scoop route pays at.
PAY_TABLES = {
    "today": {"EVENT_PAY_X": 1.0, "SCOOP_PREMIUM": 5.0, "SCOOP_MISS_DEBT_X": 2.0},
    "new": {"EVENT_PAY_X": 0.25, "SCOOP_PREMIUM": 10.0, "SCOOP_MISS_DEBT_X": 1.0},
    "lead": {"EVENT_PAY_X": 0.25, "SCOOP_PREMIUM": 10.0, "SCOOP_MISS_DEBT_X": 1.0},
}
PAY_DEFAULTS = PAY_TABLES["today"]
LEAD_TABLES = frozenset({"lead"})
# Lead below SCOOP_LEAD_MIN_M pays the event rate, from SCOOP_LEAD_FULL_M the full premium, and in
# between a straight line from SCOOP_LEAD_START_X to the full premium.
SCOOP_LEAD_MIN_M, SCOOP_LEAD_FULL_M = 5, 30
SCOOP_LEAD_START_X = 1.0
# Release wires and investor-relations pages: official, so a claim citing one is never a scoop.
WIRE_NAMES = frozenset({"prnewswire", "globenewswire", "businesswire", "stocktitan",
                        "accessnewswire", "accesswire", "newsfile", "newsfilecorp"})
IR_HOST_LABELS = frozenset({"ir", "investor", "investors"})
IR_PATH = "/investor-relations"
SCOOP_DEBT_TRUST = 1.0
# An operator-confirmed misconduct penalty, in multiples of the claim's full value (or of a
# missed scoop's debt).
MISCONDUCT_PENALTY_X = 5


class PaySettingError(ValueError):
    """A pay setting in the environment that cannot be used; the message names the variable."""


# The served switches to other tables: ((table, from_ms), ...) in start order, None until served.
Switch = Tuple[Tuple[str, int], ...]
_switch: Optional[Switch] = None


def _step(raw) -> Tuple[str, int]:
    if not isinstance(raw, dict):
        raise PaySettingError(f"pay switch {raw!r} is not an object")
    name, start = raw.get("pay_table"), raw.get("from_ms")
    if name not in PAY_TABLES:
        raise PaySettingError(f"pay switch table {name!r} is not one of {sorted(PAY_TABLES)}")
    if isinstance(start, bool) or not isinstance(start, (int, float)) or start != int(start) \
            or start <= 0:
        raise PaySettingError(f"pay switch from_ms {start!r} is not a time in ms")
    return name, int(start)


def parse_pay_switch(raw) -> Optional[Switch]:
    """{"pay_table": name, "from_ms": start}, or a list of them in rising start order, as
    ((name, start), ...); None for no switch. A table this code does not have, a start that is
    not a positive whole number of ms or starts out of order raise PaySettingError."""
    if raw is None:
        return None
    steps = tuple(_step(r) for r in raw) if isinstance(raw, list) and raw else (_step(raw),)
    if any(a[1] >= b[1] for a, b in zip(steps, steps[1:])):
        raise PaySettingError(f"pay switch starts {[s[1] for s in steps]} are not in rising order")
    return steps


def _env_switch() -> Optional[Switch]:
    names = os.getenv("EL_PAY_TABLE")
    if names is None:
        return None
    names = names.split(",")
    try:
        starts = [int(s) for s in os.getenv("EL_PAY_TABLE_FROM_MS", "").split(",")]
    except ValueError:
        starts = []
    if len(starts) != len(names):
        raise PaySettingError("EL_PAY_TABLE is set but EL_PAY_TABLE_FROM_MS is not one time in "
                              "ms per table")
    return parse_pay_switch([{"pay_table": n.strip(), "from_ms": s}
                             for n, s in zip(names, starts)])


def set_pay_switch(raw) -> Optional[Switch]:
    """Apply a served switch (None clears it). A bad one raises PaySettingError and the last
    known switch stays. Returns the switch now in force."""
    global _switch
    _switch = parse_pay_switch(raw)
    return pay_switch()


def pay_switch() -> Optional[Switch]:
    """The switch in force: EL_PAY_TABLE / EL_PAY_TABLE_FROM_MS when set locally, else the
    served one."""
    return _env_switch() or _switch


def pay_table(at_ms: Optional[int] = None) -> str:
    """The table in force at at_ms (now when None)."""
    at = int(datetime.now(timezone.utc).timestamp() * 1000) if at_ms is None else at_ms
    out = "today"
    for name, start in pay_switch() or ():
        if at >= start:
            out = name
    return out


def _ms(at) -> Optional[int]:
    if isinstance(at, datetime):
        return int(at.timestamp() * 1000)
    return None if at is None else int(at)


def _pay(key: str, at=None) -> float:
    """EL_<key> from the environment when set, else the value of the table in force at `at`
    (a datetime or ms; now when None). EL_<key> must be a finite number of 0 or more."""
    var = f"EL_{key}"
    raw = os.getenv(var)
    if raw is None:
        return PAY_TABLES[pay_table(_ms(at))][key]
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not math.isfinite(value) or value < 0:
        raise PaySettingError(f"{var}={raw!r} is not a pay setting: set a finite number of 0 or "
                              f"more, or unset it for the table's value")
    return value


def pay_settings(at=None) -> Dict[str, float]:
    """The pay settings in force at `at` (now when None), as the API serves them on
    /config/subnet. A scoop premium below the event rate is refused: a confirmed scoop would
    pay less than an event."""
    out = {k: _pay(k, at) for k in PAY_DEFAULTS}
    if out["SCOOP_PREMIUM"] < out["EVENT_PAY_X"]:
        raise PaySettingError(
            f"EL_SCOOP_PREMIUM={out['SCOOP_PREMIUM']:g} is below EL_EVENT_PAY_X="
            f"{out['EVENT_PAY_X']:g}: a confirmed scoop would pay less than an event")
    return {**out, "SCOOP_DEBT_TRUST": SCOOP_DEBT_TRUST}


def __getattr__(name: str):
    # types.EVENT_PAY_X, types.SCOOP_PREMIUM and types.SCOOP_MISS_DEBT_X, read when asked for.
    if name in PAY_DEFAULTS:
        return _pay(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def event_price(tier: str, received=None) -> Optional[float]:
    """The event route's base points for a tier at the claim's receipt, None for an unknown
    tier."""
    price = TIER_PRICE.get(tier)
    return None if price is None else price * _pay("EVENT_PAY_X", received)


def lead_pay(received) -> bool:
    """A scoop received then is paid by its lead."""
    return pay_table(_ms(received)) in LEAD_TABLES


def official_release(url: Optional[str]) -> bool:
    """A release wire or investor-relations page."""
    parts = urlsplit(url or "")
    labels = (parts.hostname or "").lower().removeprefix("www.").split(".")
    return (bool(WIRE_NAMES.intersection(labels)) or labels[0] in IR_HOST_LABELS
            or IR_PATH in parts.path.lower())


def lead_multiple(lead_ms: float, received=None) -> float:
    """The tier price multiple for a scoop this far ahead of the first trusted page, to 4 dp."""
    low, high = _pay("EVENT_PAY_X", received), _pay("SCOOP_PREMIUM", received)
    lead_m = lead_ms / 60_000
    if lead_m < SCOOP_LEAD_MIN_M:
        return low
    if lead_m >= SCOOP_LEAD_FULL_M:
        return high
    share = (lead_m - SCOOP_LEAD_MIN_M) / (SCOOP_LEAD_FULL_M - SCOOP_LEAD_MIN_M)
    start = max(low, min(SCOOP_LEAD_START_X, high))
    return round(start + (high - start) * share, 4)


def scoop_multiple(published: Optional[datetime], received: Optional[datetime],
                   contested: bool = False, first_seen: Optional[datetime] = None,
                   source_url: Optional[str] = None) -> float:
    """The tier price multiple of a confirmed scoop: SCOOP_PREMIUM when the confirming page was
    published after the scoop's receipt and no trusted page about it was seen before
    (`first_seen`), else the event rate (it was first in, not first out); the event rate when
    another scoop named a conflicting outcome. Under a LEAD_TABLES table the premium follows
    the lead over the first of those (`lead_multiple`), and a scoop citing an official release
    (`official_release`) pays the event rate."""
    out = min((d for d in (published, first_seen) if d is not None), default=None)
    if contested or not (out and received and out > received):
        return _pay("EVENT_PAY_X", received)
    if lead_pay(received):
        if official_release(source_url):
            return _pay("EVENT_PAY_X", received)
        return lead_multiple((out - received).total_seconds() * 1000, received)
    return _pay("SCOOP_PREMIUM", received)


def reverify_rate() -> float:
    """Share of paid claims each validator re-verifies, which also sizes the API's results
    cap; default 1.0. The API and validators must agree."""
    return float(os.getenv("LEDGER_REVERIFY_RATE", "1.0"))


# Pay by age, counted from the end of the event day (UTC), or of the page's publish day (UTC)
# when that is earlier: full for AGE_FULL_H, then down in a straight line to AGE_FLOOR at
# AGE_END_D. The API and validators must agree.
AGE_FULL_H, AGE_END_D, AGE_FLOOR = 48, 30, 0.25


def age_factor(event_date: date, received: datetime,
               published: Optional[datetime] = None) -> float:
    if published is not None:
        event_date = min(event_date, published.astimezone(timezone.utc).date())
    end = datetime.combine(event_date, time(), timezone.utc) + timedelta(days=1)
    age_h = (received - end).total_seconds() / 3600
    if age_h <= AGE_FULL_H:
        return 1.0
    slide = (age_h - AGE_FULL_H) / (AGE_END_D * 24 - AGE_FULL_H)
    return max(AGE_FLOOR, 1.0 - (1.0 - AGE_FLOOR) * slide)


def scoop_debt(event_type, event_date: date, received: datetime,
               published: Optional[datetime] = None) -> float:
    """A missed scoop's debt, as negative points: SCOOP_MISS_DEBT_X times the type's tier price
    x SCOOP_DEBT_TRUST x the age factor, rounded to 6 dp."""
    price = TIER_PRICE.get(getattr(TYPE_SPEC.get(EventType(event_type)), "tier", ""), 0)
    pts = (_pay("SCOOP_MISS_DEBT_X", received) * price * SCOOP_DEBT_TRUST
           * age_factor(event_date, received, published))
    return -round(pts, 6) if pts else 0.0


@dataclass(frozen=True)
class TypeSpec:
    category: Category
    group: str = ""
    label: str = ""
    window_days: int = 3
    place_level: str = "none"
    tier: str = ""
    # Main path in order; exits end the event from any non-final stage.
    stages: Tuple[Stage, ...] = ()
    exits: Tuple[Stage, ...] = ()
    slots: Tuple[str, ...] = ()
    # Each entry is met by any of its slots; an event claim must meet them all.
    required: Tuple[Tuple[str, ...], ...] = ()
    keywords: Tuple[str, ...] = ()
    stage_keywords: Dict[Stage, Tuple[str, ...]] = field(default_factory=dict)
    # Entity kind -> actor role. When set, other known kinds are not actors of this type.
    actor_roles: Dict[str, str] = field(default_factory=dict)
    # The roles that are the event's subject (default: every role); notes.whole is for them.
    subject_roles: Tuple[str, ...] = ()
    # The most actors in subject roles (or with no role) one claim may name; 0 for no limit.
    max_subjects: int = 0
    # What the judge is asked, as templates over the claim's values (see docs/miner.md).
    definition: str = ""
    stage_text: Dict[str, str] = field(default_factory=dict)
    # What the judge picks the page's event type by: one text, or one per stage.
    about: Dict[str, str] = field(default_factory=dict)
    notes: Dict[str, str] = field(default_factory=dict)
    # Per slot: "kind" (figure, text, date, name, qualifier), "text", "ask", "values",
    # "default", and "judged" for a figure the fixed check leaves to the judge.
    fields: Dict[str, dict] = field(default_factory=dict)
    office: Dict[str, object] = field(default_factory=dict)
    # The slot that names the subject when it is a fact rather than an actor, and other name
    # slots the page must show.
    subject_slot: str = ""
    name_slots: Tuple[str, ...] = ()
    # The SEC 8-K item codes that report this type, by stage ("" when it has none).
    filing_items: Dict[str, Tuple[str, ...]] = field(default_factory=dict)

    @property
    def required_slots(self) -> Tuple[str, ...]:
        return tuple(dict.fromkeys(s for group in self.required for s in group))

    @property
    def all_stages(self) -> Tuple[Stage, ...]:
        return self.stages + self.exits


def _about(raw) -> Dict[str, str]:
    return {"": raw} if isinstance(raw, str) else {Stage(s).value: v for s, v in raw.items()}


def _items(raw) -> Dict[str, Tuple[str, ...]]:
    if isinstance(raw, list):
        return {"": tuple(raw)}
    return {Stage(s).value: tuple(v) for s, v in raw.items()}


def _key(raw) -> Tuple[Tuple[str, ...], ...]:
    return tuple((k,) if isinstance(k, str) else tuple(k) for k in raw)


def _spec(raw: dict) -> TypeSpec:
    stages = tuple(Stage(s) for s in raw.get("stages", ()))
    exits = tuple(Stage(s) for s in raw.get("exits", ()))
    slots = tuple(raw.get("slots", ()))
    required = _key(raw.get("required", ()))
    if raw.get("place_level", "none") not in PLACE_LEVELS:
        raise ValueError(f"bad place_level {raw.get('place_level')!r}")
    if not {s for g in required for s in g} <= set(slots):
        raise ValueError("required slots must be slots")
    if raw.get("tier", "") not in ("", *TIERS):
        raise ValueError(f"bad tier {raw.get('tier')!r}")
    if exits and not stages:
        raise ValueError("exits need a main path")
    named = {*raw.get("fields", {}), *raw.get("name_slots", ())} | (
        {raw["subject_slot"]} if raw.get("subject_slot") else set())
    if not named <= set(slots):
        raise ValueError("fields, subject_slot and name_slots must be slots")
    roles = tuple(dict.fromkeys(raw.get("actor_roles", {}).values()))
    subject_roles = tuple(raw.get("subject_roles", roles))
    if not set(subject_roles) <= set(roles):
        raise ValueError("subject_roles must be actor roles")
    return TypeSpec(
        category=Category(raw["category"]), group=raw["group"], label=raw.get("label", ""),
        window_days=int(raw["window_days"]), place_level=raw.get("place_level", "none"),
        tier=raw.get("tier", ""), stages=stages, exits=exits, slots=slots, required=required,
        keywords=tuple(k.lower() for k in raw.get("keywords", ())),
        stage_keywords={Stage(s): tuple(k.lower() for k in kws)
                        for s, kws in raw.get("stage_keywords", {}).items()},
        actor_roles=dict(raw.get("actor_roles", {})), subject_roles=subject_roles,
        max_subjects=int(raw.get("max_subjects", 0)),
        definition=raw.get("definition", ""), stage_text=dict(raw.get("stage_text", {})),
        about=_about(raw.get("about", {})),
        notes=dict(raw.get("notes", {})), fields=dict(raw.get("fields", {})),
        office=dict(raw.get("office", {})), subject_slot=raw.get("subject_slot", ""),
        name_slots=tuple(raw.get("name_slots", ())),
        filing_items=_items(raw.get("filing_items", {})),
    )


def load_types(path: Path = TYPES_PATH) -> Dict[EventType, TypeSpec]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {EventType(t): _spec(v) for t, v in raw.items()}


TYPE_SPEC: Dict[EventType, TypeSpec] = load_types()


def category_of(event_type) -> Category:
    return TYPE_SPEC[EventType(event_type)].category


# Types the API accepts unless configured otherwise.
OPEN_TYPES: Tuple[EventType, ...] = tuple(EventType(t) for t in (
    "monetary_policy", "economic_data", "earnings", "merger_acquisition", "bankruptcy",
    "personnel", "layoffs", "delisting", "ipo_listing", "trading_halt",
    "product_launch", "funding", "partnership", "security_incident", "breakthrough", "outage",
    "vulnerability", "tech_regulation", "infrastructure_investment",
    "election_result", "legislation_passed", "appointment", "sanction_or_tariff",
    "transfer", "injury", "award", "match_result"))


def kind_key(event_type, stage=None) -> str:
    """A type, or a type and stage, as one page-judge choice: "earnings",
    "merger_acquisition:completed"."""
    t = EventType(event_type).value
    return f"{t}:{Stage(stage).value}" if stage else t


def parse_kind(key: str) -> Optional[Tuple[EventType, Optional[Stage]]]:
    t, _, s = str(key).partition(":")
    try:
        spec = TYPE_SPEC[EventType(t)]
        stage = Stage(s) if s else None
    except (KeyError, ValueError):
        return None
    if (stage is None) != ("" in spec.about) or (stage and stage.value not in spec.about):
        return None
    return EventType(t), stage


def kinds(types: Iterable = OPEN_TYPES) -> Dict[str, str]:
    """The page-judge choices for the given types: key -> text."""
    return {kind_key(t, s or None): text for t in types
            for s, text in TYPE_SPEC[EventType(t)].about.items()}


def filled(claim) -> set:
    """Slots a claim fills: its fact fields, its actors' roles and, when named, its stage."""
    got = {f.field for f in claim.facts} | {a.role for a in claim.actors if a.role}
    return got | {"stage"} if getattr(claim, "stage", None) is not None else got


def has_required(claim) -> bool:
    """The claim meets every required entry of its type."""
    got = filled(claim)
    return all(any(s in got for s in group)
               for group in TYPE_SPEC[EventType(claim.event_type)].required)


def headline(event_type, actor_names: Sequence[str], stage: Optional[Stage] = None,
             place: Optional[str] = None) -> str:
    """Our own headline, built from fields only."""
    spec = TYPE_SPEC[EventType(event_type)]
    names = _names(actor_names)
    text = f"{names}: {spec.label}" if names else spec.label.capitalize()
    if stage is not None and len(spec.all_stages) > 1:
        text += f" {Stage(stage).value}"
    if place:
        text += f" ({place})"
    return text[:200]


def _names(names: Iterable[str]) -> str:
    names = [n.strip() for n in dict.fromkeys(names) if n and n.strip()]
    if len(names) > 3:
        return ", ".join(names[:3]) + " and others"
    return " and ".join(names) if len(names) < 3 else f"{names[0]}, {names[1]} and {names[2]}"
