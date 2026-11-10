"""Pull signed points from the event ledger and check them against the leaves.

The signature is checked against the pinned API key, the window's Merkle root is rebuilt
from the leaves, the epoch's carry leaves are recomputed from the window before it, and the
points are recomputed locally. Any mismatch fails.
Closed epochs never change, so leaves are cached per epoch; a mismatch drops the cached
window and pulls it once more.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Tuple

from alpharidge_ai import config
from alpharidge_ai.ledger import attestation as ac
from alpharidge_ai.ledger.verifier import MAX_POINTS_PER_UID, verify_points
from alpharidge_ai.mechanism.pools import (WINDOW_EPOCHS, carried, category_of, coldkeys,
                                           window_points)
from alpharidge_ai.schema import Claim, canonical_json, judged, sha256_hex
from alpharidge_ai.schema import category_of as type_category
from alpharidge_ai.schema.types import MISCONDUCT_PENALTY_X, scoop_debt

TOLERANCE = 1e-6
# Reasons a fresh copy of the window's leaves may clear.
RETRY = ("root_mismatch", "points_mismatch", "malformed_leaf", "carry_mismatch")
# A missed scoop's debt leaf, and the closing reasons that write one.
DEBT = "debt"
SCOOP_MISSES = ("scoop_unconfirmed", "scoop_mismatch")


class LedgerSource(Protocol):
    async def ledger_points(self, epoch: int) -> dict: ...
    async def ledger_leaves(self, epoch: int, cursor: Optional[str] = None,
                            limit: int = 5000) -> dict: ...


@dataclass(frozen=True)
class LedgerPoints:
    epoch: int
    points: Dict[str, float]
    capacity: float
    leaf_count: int
    # Signed coldkey at receipt per hotkey, where the leaves carry one.
    coldkeys: Dict[str, str] = field(default_factory=dict)


def _same(a: Mapping[str, float], b: Mapping[str, float]) -> bool:
    keys = set(a) | set(b)
    return all(math.isclose(float(a.get(k, 0.0)), float(b.get(k, 0.0)), abs_tol=TOLERANCE)
               for k in keys)


class LedgerPull:
    def __init__(self, client: LedgerSource, pinned_pubkey: Optional[str] = None,
                 window: int = WINDOW_EPOCHS):
        self.client = client
        self.pubkey = pinned_pubkey if pinned_pubkey is not None else config.API_ATTESTATION_PUBKEY
        self.window = window
        self._leaves: Dict[int, List[dict]] = {}
        # Epochs whose cached leaves rebuilt a signed root.
        self._checked: set = set()

    async def leaves(self, epoch: int) -> List[dict]:
        if epoch not in self._leaves:
            out, cursor = [], None
            while True:
                page = await self.client.ledger_leaves(epoch, cursor)
                out.extend(page.get("leaves") or [])
                cursor = page.get("next_cursor")
                if not cursor:
                    break
            self._leaves[epoch] = out
        return self._leaves[epoch]

    def _prune(self, epoch: int, window: int) -> None:
        for e in [e for e in self._leaves if e < epoch - window]:
            del self._leaves[e]
        self._checked = {e for e in self._checked if e >= epoch - window}

    def drop(self, epoch: int, window: int) -> None:
        for e in range(epoch - window, epoch + 1):
            self._leaves.pop(e, None)
            self._checked.discard(e)

    async def pull(self, epoch: int, hotkey_to_uid: Mapping[str, int],
                   window: Optional[int] = None
                   ) -> Tuple[Optional[LedgerPoints], str]:
        """(points, "accepted") or (None, reason). The window is the signed profile's when
        given, else this pull's default."""
        window = int(self.window if window is None else window)
        args = (epoch, hotkey_to_uid, window)
        got, reason = await self._pull(*args)
        if reason in RETRY:
            self.drop(epoch, window)
            got, reason = await self._pull(*args)
        return got, reason

    async def _pull(self, epoch, hotkey_to_uid, window):
        body = await self.client.ledger_points(epoch)
        att = body.get("attestation") or {}
        verified, reason = verify_points(
            attestation=att, signature=str(body.get("signature") or ""),
            validator_hotkey=ac.LEDGER_SIGNER, hotkey_to_uid=dict(hotkey_to_uid),
            pinned_pubkey=self.pubkey, epoch=epoch)
        # "empty_payload" is signed, with nothing payable to a registered hotkey: it all burns.
        if verified is None and reason != "empty_payload":
            return None, reason

        self._prune(epoch, window)
        leaves: List[dict] = []
        for e in range(epoch - window + 1, epoch + 1):
            leaves.extend(await self.leaves(e))
        try:
            root = ac.merkle_root(leaves)
        except (KeyError, TypeError, ValueError):
            return None, "malformed_leaf"
        if root != att.get("merkleRoot"):
            return None, "root_mismatch"
        self._checked.update(range(epoch - window + 1, epoch + 1))
        try:
            if not await self._carry_ok(epoch, window):
                return None, "carry_mismatch"
        except (KeyError, TypeError, ValueError):
            return None, "malformed_leaf"
        try:
            served = float(body["capacity"])
        except (KeyError, TypeError, ValueError):
            return None, "no_capacity"
        points = window_points(leaves)
        if not _same(points, att.get("perMinerPoints") or {}):
            return None, "points_mismatch"
        if any(p > MAX_POINTS_PER_UID for p in points.values()):
            return None, "points_bound"
        return LedgerPoints(epoch, points, served, len(leaves), coldkeys(leaves)), "accepted"

    async def _carry_ok(self, epoch: int, window: int) -> bool:
        """The epoch's carry leaves are exactly those the window before it gives."""
        def key(c: Mapping) -> tuple:
            return (str(c["miner_hotkey"]), str(c.get("coldkey") or ""),
                    category_of(c) if "categorical_key" in c else c["category"],
                    round(float(c["points_awarded"]), 6))

        got = sorted(key(x) for x in await self.leaves(epoch)
                     if x.get("validator_verdict") == "carry")
        prev: List[dict] = []
        for e in range(max(0, epoch - window), epoch):
            prev.extend(await self.leaves(e))
        leaving = await self.leaves(epoch - window) if epoch - window >= 0 else []
        if leaving and not await self.signed_epoch(epoch - window, epoch, window):
            return False
        want = sorted(key(c) for c in carried(prev, leaving))
        return got == want

    async def signed_epoch(self, paid_epoch: int, epoch: int, window: int) -> bool:
        """The epoch's leaves rebuild the root signed for a window that covers it."""
        if paid_epoch in self._checked:
            return True
        at = min(paid_epoch + window - 1, epoch)
        body = await self.client.ledger_points(at)
        att = body.get("attestation") or {}
        _, reason = verify_points(
            attestation=att, signature=str(body.get("signature") or ""),
            validator_hotkey=ac.LEDGER_SIGNER, hotkey_to_uid={},
            pinned_pubkey=self.pubkey, epoch=at)
        if reason != "empty_payload":
            return False
        leaves: List[dict] = []
        for e in range(at - window + 1, at + 1):
            leaves.extend(await self.leaves(e))
        try:
            if ac.merkle_root(leaves) != att.get("merkleRoot"):
                return False
        except (KeyError, TypeError, ValueError):
            return False
        self._checked.update(range(at - window + 1, at + 1))
        return True

    async def unbound_settles(self, epoch: int, settles: Iterable[Mapping],
                              window: Optional[int] = None,
                              signed: Optional[Callable[[Mapping], bool]] = None
                              ) -> Optional[str]:
        """None when every settle, clawback and penalty leaf of the epoch names the claim's
        paid leaf, and that leaf is in the signed leaves of the epoch it names. A penalty may
        instead name an unpaid scoop's debt leaf, which must hold for the signed claim served
        with it (`debt_ok`), and is then MISCONDUCT_PENALTY_X times it."""
        window = int(self.window if window is None else window)
        by_row = {}
        for s in settles:
            by_row.setdefault(str(s.get("ledger_id")), []).append(s)
        used, hashes = set(), {}
        for leaf in await self.leaves(epoch):
            kind = leaf.get("validator_verdict")
            if kind not in ("settle", "clawback", "penalty"):
                continue
            found = by_row.get(str(leaf.get("resource_id")), [])
            if len(found) != 1:
                return "unbound_settle"
            paid, paid_epoch = found[0].get("paid") or {}, found[0].get("paid_epoch")
            try:
                paid_epoch, pts = int(paid_epoch), float(leaf["points_awarded"])
                h = ac.leaf_hash(paid)
            except (KeyError, TypeError, ValueError):
                return "unbound_settle"
            if not 0 <= paid_epoch <= epoch:
                return "unbound_settle"
            if paid_epoch not in hashes:
                hashes[paid_epoch] = ({ac.leaf_hash(x) for x in await self.leaves(paid_epoch)}
                                      if await self.signed_epoch(paid_epoch, epoch, window)
                                      else set())
            same_key = (paid.get("categorical_key") == leaf.get("categorical_key")
                        if kind == "settle" else category_of(paid) == category_of(leaf))
            key = (kind, h if kind == "settle" else str(leaf.get("claim_id")))
            on_debt = kind == "penalty" and paid.get("validator_verdict") == DEBT
            if on_debt and (signed is None or not isinstance(found[0].get("debt"), dict)
                            or not debt_ok(paid, found[0]["debt"], signed)
                            or not math.isclose(
                                pts, MISCONDUCT_PENALTY_X * _float(paid.get("points_awarded")),
                                abs_tol=TOLERANCE)):
                return "unbound_settle"
            if ((paid.get("validator_verdict") != "provisional" and not on_debt)
                    or h not in hashes[paid_epoch]
                    or key in used or not same_key or leaf.get("claim_id") is None
                    or str(paid.get("claim_id")) != str(leaf.get("claim_id"))
                    or paid.get("miner_hotkey") != leaf.get("miner_hotkey")
                    or (pts <= 0 if kind == "settle" else pts >= 0)):
                return "unbound_settle"
            used.add(key)
        return None

    async def unbound_debts(self, epoch: int, settles: Iterable[Mapping],
                            signed: Callable[[Mapping], bool],
                            window: Optional[int] = None) -> Optional[str]:
        """None when every debt leaf of the epoch is the miss debt of a miner-signed scoop
        (`debt_problem`), the only debt and with no paid leaf for its claim in the window."""
        window = int(self.window if window is None else window)
        mine = [x for x in await self.leaves(epoch) if x.get("validator_verdict") == DEBT]
        if not mine:
            return None
        by_row: Dict[str, List[Mapping]] = {}
        for s in settles:
            if isinstance(s.get("debt"), dict):
                by_row.setdefault(str(s.get("ledger_id")), []).append(s["debt"])
        rows: List[dict] = []
        for e in range(max(0, epoch - window + 1), epoch + 1):
            rows.extend(await self.leaves(e))
        for leaf in mine:
            found = by_row.get(str(leaf.get("resource_id")), [])
            if len(found) != 1 or not debt_ok(leaf, found[0], signed):
                return "unbound_debt"
            cid = str(leaf.get("claim_id"))
            if sum(1 for x in rows if str(x.get("claim_id")) == cid
                   and x.get("validator_verdict") in (DEBT, "provisional")) != 1:
                return "unbound_debt"
        return None


def debt_ok(leaf: Mapping, debt: Mapping, signed: Callable[[Mapping], bool]) -> bool:
    """The leaf is SCOOP_MISS_DEBT_X times the 1x pay of the claim the miner signed, received
    when the leaf says, typed as its page-judge record has it (whose hash the leaf carries)."""
    try:
        received = _received_ms(debt["received_at"])
        if (leaf.get("leaf_version") != ac.LEAF_V2 or received is None
                or abs(received - int(leaf["received_ms"])) >= 1
                or str(debt["hotkey"]) != str(leaf["miner_hotkey"])
                or str(debt["claim_id"]) != str(leaf.get("claim_id"))
                or debt.get("reason") not in SCOOP_MISSES or not signed(debt)):
            return False
        record = debt.get("judge") or None
        if bool(record) != bool(leaf.get("verdict_hash")) or (
                record and sha256_hex(canonical_json(record)) != leaf["verdict_hash"]):
            return False
        claim = Claim.model_validate_json(debt["body"])
        claim = judged(claim, record) if record else claim
        if claim is None or leaf.get("categorical_key") != (
                f"{type_category(claim.event_type).value}|scoop"):
            return False
        published = (_received_ms(debt.get("published_at"))
                     if record and debt.get("published_at") else None)
        want = scoop_debt(claim.event_type, claim.event_date,
                          datetime.fromtimestamp(received / 1000, timezone.utc),
                          datetime.fromtimestamp(published / 1000, timezone.utc)
                          if published is not None else None)
        return want < 0 and math.isclose(float(leaf["points_awarded"]), want, abs_tol=TOLERANCE)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _received_ms(value) -> Optional[float]:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp() * 1000
    except (TypeError, ValueError):
        return None


def unbound(leaves: Iterable[Mapping], candidates: Iterable[Mapping],
            signed: Callable[[Mapping], bool]) -> Optional[str]:
    """None when the epoch's provisional leaves and its re-verify candidates pair up
    exactly, each with the same hotkey, category and route, points, page-judge record hash
    and, on a v2 leaf, receipt time, and a valid miner signature that no other candidate
    carries."""
    by_row: Dict[str, List[Mapping]] = {}
    candidates = list(candidates)
    if len({str(c.get("signature")) for c in candidates}) != len(candidates):
        return "unsampled_leaf"
    paired: set = set()
    for c in candidates:
        for rid in c.get("ledger_ids") or ():
            by_row.setdefault(str(rid), []).append(c)
    for leaf in leaves:
        if leaf.get("validator_verdict") != "provisional":
            continue
        found = by_row.get(str(leaf.get("resource_id")), [])
        if len(found) != 1:
            return "unsampled_leaf"
        c = found[0]
        paired.add(id(c))
        if (str(c.get("hotkey")) != str(leaf.get("miner_hotkey"))
                or f"{c.get('category')}|{c.get('route')}" != leaf.get("categorical_key")
                or (leaf.get("claim_id") is not None
                    and str(c.get("claim_id")) != str(leaf["claim_id"]))
                or c.get("verdict_hash") != leaf.get("verdict_hash")
                or not _same({"p": _float(c.get("points"))}, {"p": leaf["points_awarded"]})
                or not signed(c)):
            return "unsampled_leaf"
        if leaf.get("leaf_version") == ac.LEAF_V2:
            got = _received_ms(c.get("received_at"))
            if got is None or abs(got - int(leaf["received_ms"])) >= 1:
                return "unsampled_leaf"
    return None if len(paired) == len(candidates) else "unsampled_leaf"
