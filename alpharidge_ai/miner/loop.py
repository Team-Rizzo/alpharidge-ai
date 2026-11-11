"""One reference-miner pass: discover, read, build, submit, then follow up on outcomes."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional

import bittensor as bt

from alpharidge_ai import config
from alpharidge_ai.ledger import receipts as rc
from alpharidge_ai.ledger.api_client import AlpharidgeAPIError
from alpharidge_ai.miner.build import ClaimBuilder
from alpharidge_ai.miner.discover import Discoverer
from alpharidge_ai.miner.store import MinerStore
from alpharidge_ai.schema import TYPE_SPEC, Claim


class MinerLoop:
    def __init__(self, client, store: MinerStore, discoverer: Discoverer, builder: ClaimBuilder,
                 fetcher, items_per_tick: int = 40,
                 backoff_s: float = 3600.0):
        self.client, self.store = client, store
        self.backoff_s = backoff_s
        self.discoverer, self.builder, self.fetcher = discoverer, builder, fetcher
        self.items_per_tick = items_per_tick
        self.paused_until: Optional[datetime] = None

    async def known_event(self, claim: Claim, now: datetime) -> Optional[dict]:
        """A published event of the same type naming one of the claim's actors."""
        since = now - timedelta(days=TYPE_SPEC[claim.event_type].window_days)
        for actor in claim.actors:
            try:
                found = await self.client.list_events(
                    entity=f"{actor.id_scheme.value}:{actor.id}", type=claim.event_type.value,
                    since=since.isoformat(), limit=5)
            except AlpharidgeAPIError as e:
                bt.logging.debug(f"[MINER] events lookup failed: {e.detail or e}")
                return None
            for item in found.get("items", []):
                if item.get("type") == claim.event_type.value and item.get("id"):
                    return item
        return None

    async def submit(self, claim: Claim) -> Optional[dict]:
        if self.store.sent(claim) or self.paused(datetime.now(timezone.utc)):
            return None
        try:
            response = await self.client.submit_claim(claim)
        except AlpharidgeAPIError as e:
            if e.status_code in (429, 503):
                wait = e.retry_after if e.retry_after is not None else self.backoff_s
                self.paused_until = datetime.now(timezone.utc) + timedelta(seconds=max(wait, 1))
            bt.logging.warning(f"[MINER] submit failed ({e.status_code}): {e.detail or e}")
            return None
        self.store.record(claim, response)
        self.keep_receipt(claim, response)
        bt.logging.info(f"[MINER] sent {claim.kind.value} {claim.event_type.value} "
                    f"id={response.get('id')} {claim.source_url}")
        return response

    def keep_receipt(self, claim: Claim, response: dict) -> None:
        """Keep the API's signed receipt for the claim, once it checks out."""
        r = response.get("receipt")
        if not r:
            return
        try:
            received = datetime.fromisoformat(str(response["received_at"]).replace("Z", "+00:00"))
            ok = rc.valid(r, config.API_ATTESTATION_PUBKEY) and rc.matches(
                r, response["id"], self.client.ss58_address, claim.claim_hash(),
                round(received.timestamp() * 1000))
        except (KeyError, TypeError, ValueError, AttributeError):
            ok = False
        if not ok:
            bt.logging.warning(f"[MINER] receipt for claim {response.get('id')} does not check out")
            return
        self.store.keep_receipt(r)

    async def process(self, url: str, now: datetime) -> int:
        page = await self.fetcher.fetch(url)
        candidates = await asyncio.to_thread(self.builder.build, page, now) if page.ok else []
        sent = 0
        for candidate in candidates:
            # Skip events the API already lists.
            if await self.known_event(candidate.claim, now) is not None:
                continue
            if await self.submit(candidate.claim):
                sent += 1
        if not self.paused(datetime.now(timezone.utc)):
            self.store.mark_seen(url, page.reason or ("claim" if candidates else "none"))
        return sent

    def paused(self, now: datetime) -> bool:
        return self.paused_until is not None and now < self.paused_until

    async def tick(self, now: Optional[datetime] = None) -> int:
        now = now or datetime.now(timezone.utc)
        sent = 0
        if not self.paused(now):
            items = [i for i in await self.discoverer.poll(now) if not self.store.seen(i.url)]
            for item in items[:self.items_per_tick]:
                if self.paused(datetime.now(timezone.utc)):
                    break
                try:
                    sent += await self.process(item.url, now)
                except Exception as e:
                    bt.logging.warning(f"[MINER] {item.url} failed: {e!r}")
                    self.store.mark_seen(item.url, "error")
        return sent
