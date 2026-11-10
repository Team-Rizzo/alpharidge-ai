"""Claim, event and fact models."""
from __future__ import annotations

import re
from datetime import date
from typing import List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from alpharidge_ai.schema.enums import (
    Category, ClaimKind, EventType, IdScheme, NumericUnit, Stage,
)
from alpharidge_ai.schema.hashing import model_hash
from alpharidge_ai.schema.types import TYPE_SPEC, has_required, parse_kind


QUOTE_MIN, QUOTE_MAX = 20, 500
NAME_MAX, TEXT_MAX = 80, 120
# Letters, digits, spaces and the marks names use; other text values may add a few marks for
# amounts and times.
_NAME = re.compile(r"(?:[^\W_]|[ .,'’\"“”&()/+–-])+")
_TEXT = re.compile(r"(?:[^\W_]|[ .,'’\"“”&()/+–:$%€£¥-])+")
_UNITS = frozenset(u.value for u in NumericUnit)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Actor(_Model):
    id_scheme: IdScheme
    id: str = Field(..., min_length=1, max_length=64)
    role: Optional[str] = Field(None, max_length=64)
    # Surface name as the source writes it; what the page is matched against.
    name: Optional[str] = Field(None, max_length=200)


class Place(_Model):
    iso2: str = Field(..., pattern=r"^[A-Z]{2}$")
    admin1: Optional[str] = Field(None, max_length=64)
    geonames: Optional[int] = Field(None, ge=0)


class Fact(_Model):
    field: str = Field(..., min_length=1, max_length=100)
    value: Union[float, str]
    unit: Optional[str] = Field(None, max_length=32)

    @field_validator("value")
    @classmethod
    def _value(cls, v):
        if isinstance(v, str) and not (0 < len(v.strip()) <= 200):
            raise ValueError("value must be a number or a short non-empty string")
        return v


class Claim(_Model):
    kind: ClaimKind
    headline: str = Field(..., min_length=1, max_length=200)
    event_type: EventType
    # Optional and ignored: the category always comes from the type.
    category: Optional[Category] = None
    stage: Optional[Stage] = None
    actors: List[Actor] = Field(default_factory=list, max_length=20)
    place: Optional[Place] = None
    event_date: date
    facts: List[Fact] = Field(default_factory=list, max_length=40)
    source_url: str = Field(..., max_length=2048)
    # The page's own sentence (or two in a row) that states the event, as its visible text
    # writes it.
    quote: Optional[str] = Field(None, min_length=QUOTE_MIN, max_length=QUOTE_MAX)
    # The event a fact or stage claim attaches to, and any official identifier.
    event_id: Optional[str] = Field(None, max_length=64)
    hard_id: Optional[str] = Field(None, max_length=128)

    @field_validator("source_url")
    @classmethod
    def _url(cls, v: str) -> str:
        if not v.startswith(("https://", "http://")):
            raise ValueError("source_url must be http(s)")
        return v

    @model_validator(mode="after")
    def _by_kind(self):
        if self.kind is ClaimKind.STAGE and self.stage is None:
            raise ValueError("a stage claim needs a stage")
        if self.kind is ClaimKind.FACT and not self.facts:
            raise ValueError("a fact claim needs facts")
        if self.kind is ClaimKind.EVENT and self.event_id is not None:
            raise ValueError("an event claim cannot name an event_id")
        return self

    def claim_hash(self) -> str:
        return model_hash(self)


def _is(pattern, text, cap: int) -> bool:
    return len(text) <= cap and bool(pattern.fullmatch(text))


def text_problem(claim: Claim) -> Optional[str]:
    """Why a submitted claim's text is refused, or None. Actor names and ids, the place and
    name slots hold names (`NAME_MAX`); other string facts plain text (`TEXT_MAX`); roles and
    fields are the type's; units are `NumericUnit` values; a type's `max_subjects` caps its
    subject actors."""
    spec = TYPE_SPEC.get(claim.event_type)
    if spec is None:
        return "event_type"
    names = {spec.subject_slot, *spec.name_slots}
    for a in claim.actors:
        if not all(_is(_NAME, t, NAME_MAX) for t in (a.id, a.name) if t is not None):
            return "actor"
        if a.role and a.role not in {*spec.slots, *spec.actor_roles.values()}:
            return "role"
    if spec.max_subjects and sum(not a.role or a.role in spec.subject_roles
                                 for a in claim.actors) > spec.max_subjects:
        return "subjects"
    if claim.place and claim.place.admin1 and not _is(_NAME, claim.place.admin1, NAME_MAX):
        return "place"
    for f in claim.facts:
        if f.field not in spec.slots or (f.unit is not None and f.unit not in _UNITS):
            return "fact"
        if isinstance(f.value, str) and not (
                _is(_NAME, f.value, NAME_MAX) if f.field in names else
                _is(_TEXT, f.value, TEXT_MAX)):
            return "fact"
    return None


def retype(claim: Claim, event_type, stage=None) -> Optional[Claim]:
    """The claim as another type or stage: facts the new type has no slot for are dropped (and
    figures, at an exit stage such as cancelled), and a role it lacks becomes its only role, or
    none. None when the result is not a valid claim of that type with its required slots."""
    spec = TYPE_SPEC[EventType(event_type)]
    held = set(spec.actor_roles.values())
    only = next(iter(held)) if len(held) == 1 else None
    roles = held | set(spec.slots)
    ended = stage is not None and Stage(stage) in spec.exits
    body = claim.model_dump(mode="json")
    body.update(event_type=EventType(event_type).value, stage=Stage(stage).value if stage else None,
                facts=[f for f in body["facts"] if f["field"] in spec.slots
                       and not (ended and not isinstance(f["value"], str))],
                actors=[{**a, "role": a["role"] if a["role"] in roles else only}
                        for a in body["actors"]])
    try:
        out = Claim.model_validate(body)
    except ValueError:
        return None
    return out if text_problem(out) is None and has_required(out) else None


def judged(claim: Claim, record: dict) -> Optional[Claim]:
    """The claim as its page-judge record has it: retyped when the record says so (None when
    that type cannot hold it)."""
    to = (record or {}).get("retyped")
    if not to:
        return claim
    kind = parse_kind(to.get("kind", "")) if isinstance(to, dict) else None
    return retype(claim, *kind) if kind else None


def renamed(claim: Claim, page_text: str, names: Optional[dict] = None) -> Claim:
    """The claim with each actor name, and the type's subject when it is a fact, that the page
    does not show as written replaced by the first other form of the same entity in `names`
    (name as written -> its other forms) that the page shows. IDs and roles are kept."""
    from alpharidge_ai.verify import floor
    from alpharidge_ai.verify.check import PIVOT_SLOTS, find, name_forms

    if not names:
        return claim
    norm = floor.normalize(page_text).text

    def shown(name: str) -> bool:
        return any(find(norm, f) for f in name_forms(name))

    def other(name: str) -> Optional[str]:
        if not name or shown(name):
            return None
        return next((n for n in names.get(name, ()) if n and n != name and shown(n)), None)

    body = claim.model_dump(mode="json")
    for a in body["actors"]:
        new = other(a.get("name") or a["id"])
        if new:
            a["name"] = new[:200]
    pivot = PIVOT_SLOTS.get(claim.event_type.value)
    for f in body["facts"]:
        new = other(f["value"]) if f["field"] == pivot and isinstance(f["value"], str) else None
        if new:
            f["value"] = new
    return Claim.model_validate(body)


def confirm_claim(claim: Claim, url: str, quote: Optional[str], page_text: str,
                  names: Optional[dict] = None) -> Claim:
    """A held claim as checked on another page: that page's URL and quote, its names as the
    page writes them (`renamed`), without the figures the page does not show (they stay
    unconfirmed) unless the type needs them."""
    from alpharidge_ai.verify.check import unstated_figures

    claim = renamed(claim, page_text, names)
    drop = set(unstated_figures(claim, page_text))
    body = claim.model_dump(mode="json")
    body.update(source_url=url, quote=quote)
    facts = list(body["facts"])
    for f in body["facts"]:
        if f["field"] in drop:
            rest = [x for x in facts if x is not f]
            try:
                if has_required(Claim.model_validate({**body, "facts": rest})):
                    facts = rest
            except ValueError:
                pass
    return Claim.model_validate({**body, "facts": facts})
