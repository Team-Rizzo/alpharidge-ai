"""The mechanism profile: one signed, versioned object holding the economic values.

A validator holds `current` and `next` and resolves by block height, so config and code
land together at a single activation block and a half-applied mechanism state cannot be
represented.

Pure: no I/O, no clock, no chain access. Signature verification is injected by the
caller so this module stays importable without the wallet stack.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

from alpharidge_ai.schema.hashing import canonical_json

SCHEMA_VERSION = "2.0.0"
SECONDS_PER_BLOCK = 12

_SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


class ProfileError(ValueError):
    """A profile that must not be applied. The message is the reason."""


def min_lead_blocks(refresh_seconds: int) -> int:
    """Blocks a publish must lead its activation by.

    Every validator has to have fetched `next` before any validator activates it, so the
    lead must cover a full config refresh interval.
    """
    return max(1, int(refresh_seconds) // SECONDS_PER_BLOCK)


def signing_payload(raw: dict) -> str:
    """The canonical string a profile's signature covers: everything but the signature."""
    return canonical_json({k: v for k, v in raw.items() if k != "signature"})


# ---- range helpers ----------------------------------------------------------------

def _num(section: str, d: dict, key: str, lo: float, hi: float,
         *, lo_open: bool = False) -> float:
    if key not in d:
        raise ProfileError(f"{section}.{key} missing")
    try:
        v = float(d[key])
    except (TypeError, ValueError):
        raise ProfileError(f"{section}.{key} not a number: {d[key]!r}")
    if v != v:
        raise ProfileError(f"{section}.{key} is NaN")
    if (v <= lo if lo_open else v < lo) or v > hi:
        bound = "(" if lo_open else "["
        raise ProfileError(f"{section}.{key}={v} outside {bound}{lo}, {hi}]")
    return v


def _int(section: str, d: dict, key: str, lo: int, hi: int) -> int:
    if key not in d:
        raise ProfileError(f"{section}.{key} missing")
    try:
        v = int(d[key])
    except (TypeError, ValueError):
        raise ProfileError(f"{section}.{key} not an integer: {d[key]!r}")
    if v < lo or v > hi:
        raise ProfileError(f"{section}.{key}={v} outside [{lo}, {hi}]")
    return v


# ---- sections ---------------------------------------------------------------------

@dataclass(frozen=True)
class SettlementSection:
    C: float

    @staticmethod
    def parse(d: dict) -> "SettlementSection":
        return SettlementSection(C=_num("settlement", d, "C", 0.0, 1e15, lo_open=True))


def _hotkeys(section: str, d: dict, key: str) -> Tuple[str, ...]:
    raw = d.get(key)
    if not isinstance(raw, list) or not raw or not all(
            isinstance(h, str) and 40 <= len(h) <= 64 and h.isalnum() for h in raw):
        raise ProfileError(f"{section}.{key} must be a non-empty list of hotkeys")
    return tuple(sorted(set(raw)))


@dataclass(frozen=True)
class Ledger:
    """Event ledger values: the window and, when set, the share of paid claims each validator
    re-verifies, the hours from receipt to settle and the validators whose posted page
    verdicts count. Other keys are ignored."""
    window_epochs: int
    reverify_rate: Optional[float] = None
    settle_delay_h: Optional[float] = None
    judges: Optional[Tuple[str, ...]] = None

    @staticmethod
    def parse(d: dict) -> "Ledger":
        return Ledger(
            window_epochs=_int("ledger", d, "window_epochs", 1, 10_000),
            reverify_rate=(_num("ledger", d, "reverify_rate", 0.0, 1.0, lo_open=True)
                           if "reverify_rate" in d else None),
            settle_delay_h=(_num("ledger", d, "settle_delay_h", 0.0, 24.0 * 365, lo_open=True)
                            if "settle_delay_h" in d else None),
            judges=_hotkeys("ledger", d, "judges") if "judges" in d else None,
        )


@dataclass(frozen=True)
class MechanismProfile:
    version: int
    publish_block: int
    activation_block: int
    schema_version: str
    settlement: SettlementSection
    ledger: Ledger
    signature: str = ""
    raw: Dict[str, Any] = field(default_factory=dict, repr=False, compare=False)


def parse(raw: dict) -> MechanismProfile:
    """Parse and range-check a profile. Raises ProfileError with the reason."""
    if not isinstance(raw, dict):
        raise ProfileError("profile must be an object")

    schema_version = raw.get("schema_version")
    if not isinstance(schema_version, str) or not _SEMVER.match(schema_version):
        raise ProfileError(f"schema_version malformed: {schema_version!r}")
    if schema_version != SCHEMA_VERSION:
        raise ProfileError(f"schema_version {schema_version} not supported by this validator")

    for section in ("settlement", "ledger"):
        if not isinstance(raw.get(section), dict):
            raise ProfileError(f"{section} section missing")

    return MechanismProfile(
        version=_int("profile", raw, "version", 1, 2**63 - 1),
        publish_block=_int("profile", raw, "publish_block", 0, 2**63 - 1),
        activation_block=_int("profile", raw, "activation_block", 0, 2**63 - 1),
        schema_version=schema_version,
        settlement=SettlementSection.parse(raw["settlement"]),
        ledger=Ledger.parse(raw["ledger"]),
        signature=str(raw.get("signature") or ""),
        raw=dict(raw),
    )


# ---- resolution -------------------------------------------------------------------

VerifyFn = Callable[[str, str], bool]  # (canonical_message, signature_hex) -> bool


class ProfileResolver:
    """Holds `current` and `next`, and picks between them by block height.

    A profile is only ever replaced forward. Rolling back is publishing the old values
    under a higher version, so the version number never decreases and there is no edit
    path that could differ between validators.
    """

    def __init__(self, current: Optional[MechanismProfile] = None,
                 refresh_seconds: int = 3600):
        self.current = current
        self.next: Optional[MechanismProfile] = None
        self.refresh_seconds = int(refresh_seconds)

    @property
    def min_lead(self) -> int:
        return min_lead_blocks(self.refresh_seconds)

    def adopt(self, raw: dict, block: int,
              verify: Optional[VerifyFn] = None) -> Tuple[bool, str]:
        """Take a profile already in force, rather than staging it as the next one.

        A validator starting up is handed both the active profile and any staged
        successor. Routing both through offer() would put the active one in the single
        `next` slot and then overwrite it, leaving nothing in force at all.
        """
        try:
            candidate = parse(raw)
        except ProfileError as e:
            return False, f"invalid: {e}"
        if verify is not None:
            if not candidate.signature:
                return False, "unsigned"
            if not verify(signing_payload(raw), candidate.signature):
                return False, "bad_signature"
        if candidate.activation_block > int(block):
            return False, f"not_yet_active(at={candidate.activation_block})"
        if self.current is not None and candidate.version <= self.current.version:
            return False, f"stale_version(have={self.current.version})"

        self.current = candidate
        if self.next is not None and self.next.version <= candidate.version:
            self.next = None
        return True, f"adopted(version={candidate.version})"

    def offer(self, raw: dict, verify: Optional[VerifyFn] = None) -> Tuple[bool, str]:
        """Consider a fetched profile. Returns (accepted, reason).

        A rejected profile changes nothing; whatever is in force stays in force.

        The lead is measured from the profile's own signed publish block, not from
        whenever this validator happened to fetch it. Measuring from the fetch would
        make an already-active profile unadoptable, which is exactly what a restarting
        or previously-offline validator needs to pick up.
        """
        try:
            candidate = parse(raw)
        except ProfileError as e:
            return False, f"invalid: {e}"

        if verify is not None:
            if not candidate.signature:
                return False, "unsigned"
            if not verify(signing_payload(raw), candidate.signature):
                return False, "bad_signature"

        highest = max([p.version for p in (self.current, self.next) if p is not None],
                      default=0)
        if candidate.version <= highest:
            return False, f"stale_version(have={highest}, got={candidate.version})"

        lead = candidate.activation_block - candidate.publish_block
        if lead < self.min_lead:
            return False, (f"insufficient_lead(blocks={lead}, "
                           f"required={self.min_lead})")

        self.next = candidate
        return True, f"staged(version={candidate.version}, at={candidate.activation_block})"

    def resolve(self, block: int) -> Optional[MechanismProfile]:
        """The profile in force at `block`. Promotes `next` once its block is reached."""
        if self.next is not None and int(block) >= self.next.activation_block:
            self.current = self.next
            self.next = None
        return self.current
