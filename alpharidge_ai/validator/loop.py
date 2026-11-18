"""One pass per closed epoch: signed ledger points to weights, then the keyed re-verify.

Weights come only from points that carry the API's signature, rebuild to the signed
Merkle root, recompute to the signed totals with the signed profile's values, carry the
debt the window before gives, whose paid leaves are all re-verify candidates, and whose
settle, clawback and penalty leaves each name an earlier paid leaf. Anything else keeps the
last weights, and once no points have been accepted for a while the weights go to the burn
UID.
"""
from __future__ import annotations

import asyncio
import math
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

import bittensor as bt
import numpy as np

from alpharidge_ai import config
from alpharidge_ai.ledger.api_client import AlpharidgeAPIError
from alpharidge_ai.ledger.attestation import BLOCK_LENGTH, CLOSE_GRACE_BLOCKS
from alpharidge_ai.ledger.pull import LedgerPoints, LedgerPull, QualityUnavailable, unbound
from alpharidge_ai.mechanism import quality
from alpharidge_ai.mechanism.pools import CAPACITY, net_by_coldkey
from alpharidge_ai.mechanism.profile import SECONDS_PER_BLOCK
from alpharidge_ai.mechanism.settlement import settle, weight_vector
from alpharidge_ai.validator.reverify import Reverifier, miner_signed, sample_pages

# Closed epochs without accepted points before the weights move to the burn UID, and
# before a warning is logged.
STALE_EPOCHS = 18
STALE_WARN_EPOCHS = 3
# Unchecked epochs are caught up while inside the API's re-verify deadline, a few per tick.
REVERIFY_DEADLINE_H = 24
REVERIFY_EPOCHS_PER_TICK = 3
# Longest one epoch's pull and weights may take; past it the last weights are kept and the
# epoch is tried again on the next step.
WEIGHTS_STEP_S = 240
# Claims whose page was held back are re-verified once they settle, inside the same span.
# A signed profile's ledger.settle_delay_h overrides it.
SETTLE_DELAY_H = 72


def _now() -> datetime:
    return datetime.now(timezone.utc)


class LedgerLoop:
    def __init__(self, client, metagraph, block: Callable[[], int], *,
                 reverifier: Optional[Reverifier] = None, pull: Optional[LedgerPull] = None,
                 profile=None, burn_uid: Optional[int] = None,
                 capacity: Optional[float] = None, stale_epochs: Optional[int] = None,
                 accepted_epoch: Optional[int] = None, bind: Optional[bool] = None,
                 reverified: Iterable[int] = (),
                 owner: Optional[Callable[[], Optional[str]]] = None):
        self.client, self.metagraph, self.block = client, metagraph, block
        self.pull = pull or LedgerPull(client)
        self.reverifier = reverifier
        self.profile = profile
        self.burn_uid = int(config.BURN_UID if burn_uid is None else burn_uid)
        # The subnet owner's hotkey, which must hold the burn UID; None skips the check.
        self.owner = owner
        self._owner: Optional[str] = None
        self._owner_checked: Tuple[Optional[int], bool] = (None, False)
        self.default_capacity = float(CAPACITY if capacity is None else capacity)
        self.stale_epochs = int(STALE_EPOCHS if stale_epochs is None else stale_epochs)
        self.stale_warn = STALE_WARN_EPOCHS
        self.bind = True if bind is None else bool(bind)
        self._candidates: Tuple[Optional[int], Optional[List[dict]]] = (None, None)
        # Signed leaves of re-verified epochs that still have held-back claims.
        self._leaves: Dict[int, List[dict]] = {}
        self._warn_epoch: Optional[int] = None
        # The last epoch whose signed points were accepted, kept across restarts.
        self.accepted_epoch = accepted_epoch
        self._since: Optional[int] = None
        self._burn_epoch: Optional[int] = None
        self._weights_epoch: Optional[int] = None
        # Epochs whose re-verify was posted, kept across restarts.
        self.reverified: Set[int] = {int(e) for e in reverified}
        self._task: Optional[asyncio.Task] = None
        # The last quality answer: (epoch, (scaled, locks)).
        self._served: Tuple[Optional[int], Tuple[bool, Dict[str, float]]] = (None, (False, {}))

    def burn_ok(self, epoch: int) -> bool:
        """The burn UID is held by the subnet owner's hotkey, looked up once an epoch."""
        if self.owner is None:
            return True
        if self._owner_checked[0] == epoch:
            return self._owner_checked[1]
        try:
            self._owner = self.owner() or self._owner
        except Exception as e:
            bt.logging.warning(f"[LEDGER] subnet owner lookup failed: {e!r}")
        hotkeys = list(self.metagraph.hotkeys)
        ok = (self._owner is not None and 0 <= self.burn_uid < len(hotkeys)
              and hotkeys[self.burn_uid] == self._owner)
        if not ok:
            bt.logging.error(f"[LEDGER] ALARM burn UID {self.burn_uid} is not the subnet "
                             f"owner's hotkey; setting no weights")
        self._owner_checked = (epoch, ok)
        return ok

    @property
    def held(self) -> bool:
        """The last owner check of the burn UID did not pass: no weights go on chain."""
        return self.owner is not None and not self._owner_checked[1]

    def epoch(self) -> int:
        """The latest closed epoch the API serves."""
        head = max(0, int(self.block()) - CLOSE_GRACE_BLOCKS)
        return head // BLOCK_LENGTH - 1

    def _profile(self):
        if self.profile is None:
            return None
        try:
            return self.profile.resolve(int(self.block()))
        except Exception as e:
            bt.logging.debug(f"[LEDGER] profile unavailable: {e}")
            return None

    def params(self, profile) -> Tuple[float, int]:
        """Capacity and window: the signed profile's, else local defaults."""
        if profile is None:
            return self.default_capacity, int(self.pull.window)
        return float(profile.settlement.C), int(profile.ledger.window_epochs)

    def _check_capacity(self, capacity: float, pulled: LedgerPoints, profile) -> None:
        if not math.isclose(capacity, pulled.capacity, rel_tol=1e-9):
            source = "local" if profile is None else "profile"
            bt.logging.warning(f"[LEDGER] {source} capacity {capacity:.0f} differs from the "
                               f"ledger's {pulled.capacity:.0f}")

    async def weights(self, epoch: int) -> Optional[np.ndarray]:
        hotkeys = list(self.metagraph.hotkeys)
        profile = self._profile()
        capacity, window = self.params(profile)
        pulled, reason = await self.pull.pull(epoch, {hk: uid for uid, hk in enumerate(hotkeys)},
                                              window=window, carry_locks=self.carry_locks,
                                              before_retry=self.forget_quality)
        if reason == "quality_unavailable":
            # Tried again on the next tick, as a failed pull is.
            bt.logging.error(f"[LEDGER] ALARM epoch={epoch} carry quality unavailable; "
                             f"keeping the last weights")
            raise QualityUnavailable(f"epoch {epoch}")
        if pulled is not None and self.bind:
            candidates, settles = await sample_pages(self.client, epoch)
            reason = (unbound(await self.pull.leaves(epoch), candidates, miner_signed)
                      or await self.pull.unbound_settles(epoch, settles, window, miner_signed)
                      or await self.pull.unbound_debts(epoch, settles, miner_signed, window)
                      or reason)
            if reason != "accepted":
                pulled = None
            else:
                self._candidates = (epoch, candidates)
        if pulled is None:
            bt.logging.error(f"[LEDGER] ALARM epoch={epoch} points rejected ({reason}); "
                             f"keeping the last weights")
            return None
        if not self.burn_ok(epoch):
            return None
        self._check_capacity(capacity, pulled, profile)
        coldkeys = {**dict(zip(hotkeys, getattr(self.metagraph, "coldkeys", None) or hotkeys)),
                    **pulled.coldkeys}
        points = pulled.points
        # Quality scales the epoch when its points were signed from QUALITY_FROM, by the
        # signing time the API serves; well before the start it is off without asking.
        if quality.may_be_active(_now()):
            try:
                on, locks = await self._quality(epoch)
            except Exception as e:
                bt.logging.error(f"[LEDGER] ALARM epoch={epoch} quality unavailable ({e!r}); "
                                 f"keeping the last weights")
                raise
            if on:
                leaves = []
                for ep in range(epoch - window + 1, epoch + 1):
                    leaves.extend(await self.pull.leaves(ep))
                points = quality.scaled(leaves, locks)
                cut = sum(1 for m in locks.values() if m < 1)
                bt.logging.info(f"[QUALITY] epoch={epoch} paid={len(locks)} cut={cut}")
                missing = quality.unlocked(leaves, locks)
                if missing:
                    bt.logging.error(f"[QUALITY] ALARM epoch={epoch} {missing} paid claims "
                                     f"received from the start have no lock; paid at 1")
        net = net_by_coldkey(points, coldkeys)
        result = settle(net, capacity)
        size = int(getattr(self.metagraph, "n", len(hotkeys)))
        vector = weight_vector(result.shares, hotkeys, self.burn_uid, size)
        bt.logging.info(
            f"[LEDGER] epoch={epoch} leaves={pulled.leaf_count} W={result.work:.2f} "
            f"C={result.capacity:.2f} burn={result.burn:.4f} paid={result.paid:.4f}")
        return np.asarray(vector, dtype=np.float32)

    async def _quality(self, epoch: int) -> Tuple[bool, Dict[str, float]]:
        """Whether the epoch is scaled, and the locks served for it. The last answer is kept
        for the next epoch's carry."""
        if self._served[0] == epoch:
            return self._served[1]
        served = await self.client.ledger_quality(quality.from_ms(), epoch)
        if served.get("until_ms") is None:
            raise ValueError("no signing time")
        owners = served["owners"]
        got = (quality.active(served["until_ms"]),
               quality.locked((owners[int(i)], bool(p), c) for i, p, c in served["outcomes"]))
        self._served = (epoch, got)
        return got

    def forget_quality(self) -> None:
        """Drops the kept quality answer, so the next read asks again."""
        self._served = (None, (False, {}))

    async def carry_locks(self, epoch: int) -> Optional[Dict[str, float]]:
        """The locks the carry into the epoch is valued with: those of the epoch before it
        when that one was scaled, else None (raw points). The API values it the same way,
        since an epoch's own locks are known only once it is signed. An epoch before it that
        was never signed (the first the API signed) leaves the carry raw."""
        if epoch < 1 or not quality.may_be_active(_now()):
            return None
        try:
            on, locks = await self._quality(epoch - 1)
        except AlpharidgeAPIError as e:
            if e.status_code == 409 and e.detail == "epoch_unsigned":
                return None
            raise
        return locks if on else None

    def reverify_due(self, epoch: int) -> List[int]:
        """Unchecked epochs still inside the deadline, newest first, at most a tick's worth."""
        span = self._span(REVERIFY_DEADLINE_H)
        oldest = max(0, epoch - span + 1)
        self.reverified = {e for e in self.reverified if e >= oldest}
        due = [e for e in range(epoch, oldest - 1, -1) if e not in self.reverified]
        return due[:REVERIFY_EPOCHS_PER_TICK]

    def _span(self, hours: float) -> int:
        return max(1, int(hours * 3600 / (BLOCK_LENGTH * SECONDS_PER_BLOCK)))

    def settled_due(self, epoch: int) -> List[int]:
        """Epochs with held-back claims that have settled since, oldest first; ones past the
        span are dropped."""
        held = getattr(self.reverifier, "withheld", None) or {}
        profile = self._profile()
        hours = SETTLE_DELAY_H
        if profile is not None and profile.ledger.settle_delay_h is not None:
            hours = profile.ledger.settle_delay_h
        settle = self._span(hours) + 1
        for e in [e for e in held if epoch - e >= settle + self._span(REVERIFY_DEADLINE_H)]:
            bt.logging.warning(f"[REVERIFY] epoch={e} {len(held.pop(e))} withheld claims "
                               f"not re-verified before the deadline")
        return sorted(e for e in held if epoch - e >= settle)[:REVERIFY_EPOCHS_PER_TICK]

    async def _bound(self, epoch: int) -> Optional[List[dict]]:
        """The epoch's re-verify candidates, checked against its signed leaves (the run
        pulls them itself when binding is off)."""
        if not self.bind:
            return None
        leaves = self._leaves.get(epoch)
        if leaves is None:
            _, window = self.params(self._profile())
            if not await self.pull.signed_epoch(epoch, self.epoch(), window):
                raise RuntimeError("leaves_unsigned")
            leaves = await self.pull.leaves(epoch)
        held, candidates = self._candidates
        if held != epoch or candidates is None:
            candidates, _ = await sample_pages(self.client, epoch)
            why = unbound(leaves, candidates, miner_signed)
            if why:
                bt.logging.error(f"[REVERIFY] ALARM epoch={epoch} candidates rejected ({why})")
                raise RuntimeError(why)
        self._leaves[epoch] = leaves
        return candidates

    async def _reverify(self, epochs: List[int], settled: Iterable[int] = ()) -> None:
        profile = self._profile()
        if profile is not None and profile.ledger.reverify_rate is not None:
            self.reverifier.rate = profile.ledger.reverify_rate
        if profile is not None and getattr(profile.ledger, "judges", None):
            self.reverifier.judges = set(profile.ledger.judges)
        for epoch in settled:
            only = self.reverifier.withheld.pop(epoch, set())
            try:
                await self.reverifier.run(epoch, await self._bound(epoch), only=only)
            except Exception as e:
                self.reverifier.withheld.setdefault(epoch, set()).update(only)
                bt.logging.warning(f"[REVERIFY] epoch={epoch} settled pass failed: {e!r}")
        for epoch in epochs:
            try:
                await self.reverifier.run(epoch, await self._bound(epoch))
                self.reverified.add(epoch)
            except Exception as e:
                bt.logging.warning(f"[REVERIFY] epoch={epoch} failed: {e!r}")
        self._leaves = {e: v for e, v in self._leaves.items() if e in self.reverifier.withheld}

    def burn_all(self) -> np.ndarray:
        size = int(getattr(self.metagraph, "n", len(self.metagraph.hotkeys)))
        return np.asarray(weight_vector({}, list(self.metagraph.hotkeys),
                                        self.burn_uid, size), dtype=np.float32)

    def _stale(self, epoch: int) -> Optional[np.ndarray]:
        """The burn vector once accepted points are too old, else None."""
        last = self.accepted_epoch if self.accepted_epoch is not None else self._since
        if last is None or epoch - last < self.stale_epochs:
            if (last is not None and epoch - last >= self.stale_warn
                    and self._warn_epoch != epoch):
                self._warn_epoch = epoch
                bt.logging.warning(f"[LEDGER] epoch={epoch} no accepted points for "
                                   f"{epoch - last} epochs (since epoch {last})")
            return None
        if self._burn_epoch != epoch:
            self._burn_epoch = epoch
            bt.logging.error(f"[LEDGER] ALARM epoch={epoch} no accepted points since "
                             f"epoch {last}; weights to the burn UID")
        return self.burn_all() if self.burn_ok(epoch) else None

    async def tick(self) -> Optional[np.ndarray]:
        """New weights when a closed epoch's points were accepted, the burn vector when
        none have been for too long, else None (keep the last weights)."""
        epoch = self.epoch()
        if epoch < 0:
            return None
        if self._since is None:
            self._since = epoch
        self.burn_ok(epoch)
        weights = None
        if epoch != self._weights_epoch:
            try:
                weights = await asyncio.wait_for(self.weights(epoch), WEIGHTS_STEP_S)
            except Exception as e:
                bt.logging.warning(f"[LEDGER] epoch={epoch} pull failed: {e!r}")
                return self._stale(epoch)
            self._weights_epoch = epoch
            if weights is not None:
                self.accepted_epoch = epoch
        if weights is None:
            weights = self._stale(epoch)
        if self.reverifier is not None and (self._task is None or self._task.done()):
            due, settled = self.reverify_due(epoch), self.settled_due(epoch)
            if due or settled:
                self._task = asyncio.create_task(self._reverify(due, settled))
        return weights
