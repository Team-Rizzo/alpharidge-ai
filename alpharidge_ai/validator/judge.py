"""Listed validators judge the pages of queued claims. Each pulls `GET /judge/pending`, fetches
the page itself, runs the shared fixed check and then the page judge the operator gave it, and
posts the answers as returned with the fixed check's result (`POST /judge/verdicts`, signed).
The API combines the listed validators' posts into one verdict. Each validator pays for its own
judge calls and stops pulling at its daily cap.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Callable, Dict, Optional, Set
from urllib.parse import urlsplit

import bittensor as bt

from alpharidge_ai import config
from alpharidge_ai.ledger.api_client import AlpharidgeAPIError, AuthorizationError
from alpharidge_ai.schema import Claim, confirm_claim, judged, kind_key
from alpharidge_ai.schema.hashing import canonical_json
from alpharidge_ai.validator.reverify import fetch_config
from alpharidge_ai.verify.check import check_claim
from alpharidge_ai.verify.dates import parse_datetime
from alpharidge_ai.verify.fetch import Fetcher

OFF_S = 6 * 3600
FETCH_TRIES = 3
# A run stops calling the judge after this many failed calls in a row.
JUDGE_TRIP = 3
# Longest one item (fetch, check, judge, post) and one pull may take; what is left over is
# served again on the next pull.
ITEM_TIMEOUT_S = 180
BATCH_TIMEOUT_S = 900


def _retype(claim: Optional[Claim], kind: Optional[str]) -> Optional[Claim]:
    if claim is None or not kind:
        return claim
    return judged(claim, {"retyped": {"kind": kind}})


def claim_of(item: dict, text: str) -> Optional[Claim]:
    """The claim the API judges for an item, built from the miner-signed body as the API
    built it: retyped, as a confirm claim on the found page, then judged as another type."""
    try:
        claim = _retype(Claim.model_validate_json(item["body"]), item.get("retyped"))
        if claim is not None and item.get("kind") == "confirm":
            claim = confirm_claim(claim, item["url"], item.get("quote"), text,
                                  {k: [v] for k, v in (item.get("names") or {}).items()})
        return _retype(claim, item.get("judged_as"))
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


def verdict_body(item: dict, claim: Optional[Claim], content_hash: str, answers: dict,
                 passed: bool, reason: Optional[str]) -> bytes:
    """The exact bytes posted and signed for an item."""
    return canonical_json({
        "key": item["key"], "claim_id": str(item["claim_id"]), "kind": item.get("kind"),
        "as": kind_key(claim.event_type, claim.stage) if claim is not None else None,
        "content_hash": content_hash, "answers": answers,
        "fixed_check": {"passed": passed, "reason": reason}}).encode("utf-8")


class Spend:
    """This validator's judge spend in the current UTC day."""

    def __init__(self, cap_usd: float, clock: Callable[[], datetime]):
        self.cap, self.clock = float(cap_usd), clock
        self.day, self.usd = None, 0.0

    def _roll(self) -> None:
        today = self.clock().date()
        if today != self.day:
            self.day, self.usd = today, 0.0

    def room(self, est: float) -> bool:
        self._roll()
        return self.usd + est <= self.cap

    def add(self, usd: float) -> None:
        self._roll()
        self.usd += usd


class JudgeWorker:
    def __init__(self, client, judge=None, fetcher: Optional[Fetcher] = None,
                 interval: Optional[float] = None, cap_usd: Optional[float] = None,
                 concurrency: Optional[int] = None,
                 clock: Optional[Callable[[], datetime]] = None):
        self.client, self.judge = client, judge
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self._fetcher = fetcher
        self.interval = float(config.JUDGE_POLL_S if interval is None else interval)
        self.spend = Spend(config.JUDGE_DAILY_CAP_USD if cap_usd is None else cap_usd,
                           self.clock)
        self.concurrency = max(1, int(concurrency or config.REVERIFY_CONCURRENCY))
        self.done: Set[str] = set()
        self.tries: Dict[str, int] = {}
        # Answers by item key and page content, so a failed post never pays twice.
        self.answers: Dict[tuple, dict] = {}
        self.fails = 0
        self._next = 0.0
        # When the last pull finished (monotonic), for the watchdog.
        self.last_pull = time.monotonic()
        self._capped: Optional[object] = None
        self._task: Optional[asyncio.Task] = None

    @property
    def fetcher(self) -> Fetcher:
        if self._fetcher is None:
            self._fetcher = Fetcher(fetch_config())
        return self._fetcher

    def tick(self) -> None:
        """Start a pull in the background when one is due and none is running."""
        if self.judge is None or (self._task is not None and not self._task.done()):
            return
        now = time.monotonic()
        if now < self._next:
            return
        self._next = now + self.interval
        self._task = asyncio.create_task(self._safe())

    async def _safe(self) -> None:
        try:
            await self.run()
            self.last_pull = time.monotonic()
        except AuthorizationError:
            self._next = time.monotonic() + OFF_S
            bt.logging.info("[JUDGE] not a listed validator; judging paused")
        except Exception as e:
            bt.logging.warning(f"[JUDGE] failed: {e!r}")

    async def run(self) -> Dict[str, int]:
        if self.judge is None:
            return {}
        if not self.spend.room(self.judge.est_usd):
            if self._capped != self.spend.day:
                self._capped = self.spend.day
                bt.logging.warning(f"[JUDGE] daily cap ${self.spend.cap:.2f} reached; "
                                   f"pulling again tomorrow (UTC)")
            return {"capped": 1}
        got = await self.client.judge_pending()
        types = got.get("types") or None
        served = [i for i in got.get("items") or [] if isinstance(i, dict) and i.get("key")]
        keys = {i["key"] for i in served}
        self.done &= keys
        self.tries = {k: n for k, n in self.tries.items() if k in keys}
        self.answers = {k: a for k, a in self.answers.items() if k[0] in keys}
        items = [i for i in served if i["key"] not in self.done]
        self.fails = 0
        gate = asyncio.Semaphore(self.concurrency)

        async def one(item):
            async with gate:
                try:
                    return await asyncio.wait_for(self.judge_item(item, types), ITEM_TIMEOUT_S)
                except asyncio.TimeoutError:
                    key = item["key"]
                    self.tries[key] = self.tries.get(key, 0) + 1
                    if self.tries[key] >= FETCH_TRIES:
                        self.done.add(key)
                    bt.logging.warning(f"[JUDGE] item {item.get('claim_id')} timed out after "
                                       f"{ITEM_TIMEOUT_S}s")
                    return "timeout"
                except Exception as e:
                    bt.logging.warning(f"[JUDGE] item {item.get('claim_id')} failed: {e!r}")
                    return "error"

        tasks = [asyncio.ensure_future(one(i)) for i in items]
        finished, late = await asyncio.wait(tasks, timeout=BATCH_TIMEOUT_S) if tasks else (
            set(), set())
        for t in late:
            t.cancel()
        if late:
            bt.logging.warning(f"[JUDGE] pull ran over {BATCH_TIMEOUT_S}s; {len(late)} items "
                               f"left for the next pull")
        outcomes = [t.result() for t in finished] + ["late"] * len(late)
        counts: Dict[str, int] = {}
        for o in outcomes:
            counts[o] = counts.get(o, 0) + 1
        if items:
            bt.logging.info(f"[JUDGE] pending={len(items)} {dict(sorted(counts.items()))} "
                            f"spent=${self.spend.usd:.4f}")
        return counts

    async def judge_item(self, item: dict, types=None) -> str:
        key = item["key"]
        if item.get("unread") and item.get("kind") != "confirm":
            self.done.add(key)
            return "unread"
        page = await self.fetcher.fetch(str(item.get("final_url") or item.get("url") or ""))
        if not page.ok:
            self.tries[key] = self.tries.get(key, 0) + 1
            if self.tries[key] >= FETCH_TRIES:
                self.done.add(key)
            # Why, in the pull's summary: the fetch reason, its cause or HTTP status, the host.
            host = urlsplit(page.final_url or page.url).hostname or "-"
            return f"unreachable:{page.reason}:{page.blocked_by or page.status or '-'}:{host}"
        claim = claim_of(item, page.text)
        at = parse_datetime(item.get("at"))
        result = check_claim(claim, page, at) if claim is not None and at else None
        passed = bool(result and result.passed)
        answers: dict = {}
        seen = (key, page.content_hash)
        if passed and seen in self.answers:
            answers = self.answers[seen]
        elif passed:
            if self.fails >= JUDGE_TRIP or not self.spend.room(self.judge.est_usd):
                return "waiting"
            published = page.published_at.isoformat() if page.published_at else None
            try:
                answers = await asyncio.to_thread(self.judge.judge_page, claim, page.text,
                                                  published, types)
            except Exception as e:
                self.fails += 1
                self.spend.add(self.judge.est_usd)
                bt.logging.warning(f"[JUDGE] judge call failed: {e!r}")
                return "judge_failed"
            cost = getattr(self.judge, "last_cost", None)
            self.spend.add(cost if cost is not None else self.judge.est_usd)
            if not answers:
                self.fails += 1
                return "judge_failed"
            self.fails = 0
            self.answers[seen] = answers
        reason = None if passed else (result.reason if result else "invalid_claim")
        body = verdict_body(item, claim, page.content_hash, answers, passed, reason)
        try:
            status = (await self.client.post_verdict(body)).get("status")
        except AlpharidgeAPIError as e:
            if e.status_code in (404, 409):
                self.done.add(key)
                return "closed"
            raise
        self.done.add(key)
        return status or "posted"
