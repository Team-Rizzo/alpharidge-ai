"""Re-verify this validator's keyed share of the ledger's paid claims.

Each kept candidate's body is checked against the miner's signature, its stored page-judge
record against the hash the signed leaf carries and for a yes to every check and field, and the
record's verdict against the listed validators' signed posts it keeps, recomputed from their
answers (an event the page's 8-K confirmed is read again from the filer's public submissions
list), and its points against the type's price, trust and the event's age. A claim the judge
retyped is checked as that type. Its page is fetched independently, at the URL the API's fetch
ended at: the shared deterministic check, the miner's quote included, runs on that page, and
the category must be the type's. A paid scoop is checked on the trusted page that confirmed it,
as the API built its claim there, not on its own page; one whose page could not be read has no
page-judge record of its own, and its age is from its event date alone. No model is asked here.
Results go back to the API; weights still follow the signed ledger, so validators stay in
consensus.
"""
from __future__ import annotations

import asyncio
import json
import math
from collections import Counter
from dataclasses import replace
from typing import Dict, Iterable, List, Optional, Set, Tuple

import bittensor as bt
from bittensor_wallet import Keypair

from alpharidge_ai import config
from alpharidge_ai.judge.combine import bodies_of, combined, post_signed
from alpharidge_ai.ledger import request_auth
from alpharidge_ai.mechanism.pools import PROVISIONAL_SHARE
from alpharidge_ai.schema import (
    TYPE_SPEC, Claim, category_of, judged, kind_key, text_problem,
)
from alpharidge_ai.schema.types import TIER_PRICE, age_factor, scoop_multiple
from alpharidge_ai.validator.selector import Selector
from alpharidge_ai.verify.check import PIVOT_SLOTS, check_claim, verdict_hash
from alpharidge_ai.verify.dates import parse_datetime
from alpharidge_ai.verify.fetch import BLOCKED, FETCH_FAILED, FetchConfig, Fetcher, user_agent
from alpharidge_ai.verify.filing import confirms, filing_row, submissions_url

AGREE, DISAGREE, UNREACHABLE = "agree", "disagree", "unreachable"
DOMAIN = "reverify"
POST_MAX = 5000
DISAGREE_ALARM = 0.05


def fetch_config() -> FetchConfig:
    return FetchConfig(user_agent=config.REVERIFY_USER_AGENT
                       or user_agent(config.USER_AGENT_CONTACT),
                       timeout_s=config.REVERIFY_FETCH_TIMEOUT_S,
                       retries=config.REVERIFY_FETCH_RETRY,
                       domain_rps=config.REVERIFY_DOMAIN_RPS,
                       public_only=config.REVERIFY_FETCH_PUBLIC_ONLY,
                       parse_timeout_s=config.REVERIFY_PARSE_TIMEOUT_S,
                       pace_max_wait_s=config.REVERIFY_PACE_MAX_WAIT_S)


def keyed(items: Iterable[dict], selector: Selector, rate: float) -> List[dict]:
    """The candidates this validator's key keeps."""
    out = []
    for item in items:
        try:
            fraction = selector.fraction(int(item["claim_id"]), DOMAIN)
        except (KeyError, TypeError, ValueError):
            continue
        if fraction < rate:
            out.append(item)
    return out


def miner_signed(item: dict) -> bool:
    """The miner's signature over the body, or over its hash while the body is withheld."""
    try:
        digest = (request_auth.body_sha256(str(item["body"]).encode("utf-8"))
                  if "body" in item else str(item["body_sha256"]))
        message = request_auth.auth_message(
            "POST", "/claims", digest, int(item["signed_at"]), str(item["nonce"]))
        return Keypair(ss58_address=str(item["hotkey"])).verify(
            message.encode("utf-8"), bytes.fromhex(str(item["signature"])))
    except Exception:
        return False


def unread(record: dict) -> bool:
    """A scoop held without reading its page: no page-judge record of its own."""
    return record.get("unread") is True


def paid_right(item: dict, claim: Claim, received) -> bool:
    """The paid points are the paid share of base × T × A, with T in [0, 1], A the age factor
    at receipt of the event date or the page's publish date, the earlier, and base the type's
    tier price on the event route and that price × `scoop_multiple` of the confirming page's
    publish time on the scoop route (1 when the record marks it contested)."""
    f = item.get("factors") or {}
    try:
        base, t, points = float(f["base"]), float(f["T"]), float(item["points"])
        a = float(f.get("A", 1.0))
    except (KeyError, TypeError, ValueError):
        return False
    record = item.get("judge") or {}
    published = None if unread(record) else parse_datetime(item.get("published_at"))
    expected = age_factor(claim.event_date, received, published)
    if not 0.0 <= t <= 1.0 or not math.isclose(a, expected, abs_tol=1e-6):
        return False
    tier = getattr(TYPE_SPEC.get(claim.event_type), "tier", "")
    route = item.get("route")
    if route in ("event", "scoop"):
        price = TIER_PRICE.get(tier)
        if route == "scoop" and price is not None:
            confirm = record.get("confirm") or {}
            if not isinstance(confirm, dict):
                return False
            price = price * scoop_multiple(parse_datetime(confirm.get("published_at")), received,
                                           bool(confirm.get("contested")))
        if price is None or not math.isclose(base, price):
            return False
    return math.isclose(points, PROVISIONAL_SHARE * base * t * a, rel_tol=1e-5, abs_tol=1e-6)


def consistent(record: dict, claim: Claim) -> bool:
    """A pass record must say yes to every check and every claimed field."""
    answers, fields = record.get("answers"), record.get("fields")
    if not isinstance(answers, dict) or not isinstance(fields, dict) or not answers:
        return False
    return (set(fields) == {f.field for f in claim.facts}
            and all(v == "yes" for v in (*answers.values(), *fields.values())))


def panel_problem(record: dict, claim_id, claim: Claim, judges, kind: str = "claim"
                  ) -> Optional[str]:
    """None when every post the record keeps is signed by a listed validator, is about this
    claim as judged, and the record's verdict is the one those posts combine to."""
    bodies = bodies_of(record)
    if bodies is None:
        return "no_posts"
    posts = record["posts"]
    if len({str(p.get("validator")) for p in posts}) != len(posts):
        return "post_mismatch"
    want = kind_key(claim.event_type, claim.stage)
    for p, b in zip(posts, bodies):
        if not post_signed(p):
            return "bad_post_signature"
        if p.get("validator") not in judges:
            return "not_a_judge"
        if (str(b.get("claim_id")) != str(claim_id) or b.get("kind") != kind
                or b.get("as") != want):
            return "post_mismatch"
    expected = combined(bodies, claim, record.get("filing"))
    return None if all(record.get(k) == v for k, v in expected.items()) else "verdict_mismatch"


async def filing_problem(record: dict, claim: Claim, fetcher) -> Optional[str]:
    """For a record whose event the page's 8-K confirmed: None when the filer's public
    submissions list gives the same row and it confirms the claim, else why not."""
    row = record.get("filing")
    if not isinstance(row, dict) or str(row.get("accession")) not in claim.source_url.replace(
            "-", ""):
        return "filing_mismatch"
    text = await fetcher.get_text(submissions_url(row.get("cik", "")))
    if text is None:
        return UNREACHABLE
    try:
        got = filing_row(json.loads(text), row.get("cik", ""), row["accession"])
    except ValueError:
        got = None
    return None if got == row and confirms(claim, got) else "filing_mismatch"


def _renamed_only(found: Claim, claim: Claim) -> bool:
    """`found` is `claim` with, at most, actor names and the type's subject written as another
    form: actors keep their ID, scheme and role in order."""
    mine = [(a.id, a.id_scheme, a.role) for a in claim.actors]
    if [(a.id, a.id_scheme, a.role) for a in found.actors] != mine:
        return False
    pivot = PIVOT_SLOTS.get(claim.event_type.value)
    subject = {f.field: f for f in claim.facts if f.field == pivot and isinstance(f.value, str)}
    return all(f in claim.facts or (f.field in subject and isinstance(f.value, str)
                                    and f.unit == subject[f.field].unit)
               for f in found.facts)


def stored_confirm(claim: Claim, confirm: dict) -> Optional[Claim]:
    """The claim the API judged on the confirming page, as its record keeps it, when it is the
    scoop on that page with that quote, some of the scoop's facts and its names, or other
    forms of them the page writes (`schema.renamed`); else None."""
    try:
        found = Claim.model_validate(confirm["claim"])
    except (KeyError, TypeError, ValueError):
        return None
    rest = {"source_url", "quote", "facts", "actors"}
    same = (found.model_dump(mode="json", exclude=rest) == claim.model_dump(mode="json", exclude=rest)
            and found.source_url == confirm.get("url") and found.quote == confirm.get("quote")
            and _renamed_only(found, claim))
    return found if same else None


async def confirm_problem(item: dict, claim: Claim, received, fetcher,
                          judges=()) -> Tuple[Optional[str], dict]:
    """For a scoop: None when the claim the API built on the page that confirmed it (stored in
    the record) has a judge record that says yes to it, and that page, fetched again, still
    passes the shared check on that claim (its quote and figures), is dated on the same side
    of the scoop's receipt as the record says and has the trust it was paid at; else why
    not."""
    confirm = (item.get("judge") or {}).get("confirm")
    if not isinstance(confirm, dict):
        return "no_confirm", {}
    found = parse_datetime(confirm.get("found_at"))
    url, record = confirm.get("url"), confirm.get("judge") or {}
    try:
        trust = float(confirm["trust"])
        paid_at = float((item.get("factors") or {})["T"])
    except (KeyError, TypeError, ValueError):
        return "no_confirm", {}
    if not isinstance(url, str) or found is None:
        return "no_confirm", {}
    if not math.isclose(trust, paid_at, abs_tol=1e-6):
        return "points_mismatch", {}
    page = await fetcher.fetch(url)
    if page.reason in (FETCH_FAILED, BLOCKED):
        return UNREACHABLE, {"confirm_status": page.status}
    seen = {"confirm_content_hash": page.content_hash}
    found_claim = stored_confirm(claim, confirm)
    if found_claim is None:
        return "confirm_inconsistent", seen
    if record.get("outcome") != "pass" or not consistent(record, found_claim):
        return "confirm_inconsistent", seen
    why = panel_problem(record, item.get("claim_id"), found_claim, judges, "confirm")
    if why:
        return why, seen
    stored = parse_datetime(confirm.get("published_at"))
    if stored is None or page.published_at is None or (
            (page.published_at > received) != (stored > received)):
        return "published_mismatch", seen
    got = check_claim(found_claim, page, found)
    return (None if got.passed else got.reason or "unsupported"), seen


def _result(item: dict, outcome: str, **observed) -> dict:
    return {"claim_id": str(item.get("claim_id")), "verdict": outcome, "observed": observed}


async def reverify(item: dict, fetcher: Fetcher, judges=()) -> dict:
    """One candidate's result, as posted to the API. `judges` are the listed validators whose
    posted verdicts count; with none known the record's posts cannot be checked."""
    if not miner_signed(item):
        return _result(item, DISAGREE, verdict="rejected", reason="bad_signature")
    try:
        claim = Claim.model_validate_json(item["body"])
    except Exception:
        claim = None
    if claim is None or text_problem(claim):
        return _result(item, DISAGREE, verdict="rejected", reason="invalid_schema")
    received = parse_datetime(item.get("received_at"))
    if received is None:
        return _result(item, UNREACHABLE, reason="no_receipt")
    record = item.get("judge") or {}
    scoop = item.get("route") == "scoop"
    held = scoop and unread(record)
    if ((record.get("outcome") != "pass" and not held) or not item.get("verdict_hash")
            or verdict_hash(record) != item["verdict_hash"]):
        return _result(item, DISAGREE, verdict="rejected", reason="no_judge_record")
    claim = judged(claim, record)
    if claim is None or not (held or consistent(record, claim)):
        return _result(item, DISAGREE, verdict="rejected", reason="judge_inconsistent")
    if not paid_right(item, claim, received):
        return _result(item, DISAGREE, verdict="rejected", reason="points_mismatch")
    if not judges:
        return _result(item, UNREACHABLE, reason="no_judge_list")
    why = None if held else panel_problem(record, item.get("claim_id"), claim, judges)
    if why:
        return _result(item, DISAGREE, verdict="rejected", reason=why)
    if record.get("event_by") == "filing":
        why = await filing_problem(record, claim, fetcher)
        if why == UNREACHABLE:
            return _result(item, UNREACHABLE, reason="filing_unreachable")
        if why:
            return _result(item, DISAGREE, verdict="rejected", reason=why)
    expected = item.get("category")
    if scoop:
        if expected and category_of(claim.event_type).value != expected:
            return _result(item, DISAGREE, reason="category_mismatch")
        why, seen = await confirm_problem(item, claim, received, fetcher, judges)
        if why == UNREACHABLE:
            return _result(item, UNREACHABLE, reason="confirm_unreachable", **seen)
        if why:
            return _result(item, DISAGREE, reason=why, **seen)
        return _result(item, AGREE, **seen)

    page = await fetcher.fetch(item.get("final_url") or claim.source_url)
    if page.reason in (FETCH_FAILED, BLOCKED):
        return _result(item, UNREACHABLE, reason=page.reason, status=page.status)
    # A page restamped since the API read it (an update) is checked at the date the API saw.
    stored = parse_datetime(item.get("published_at"))
    if stored is not None and page.published_at is not None and page.published_at > stored:
        page = replace(page, published_at=stored, date_inferred=False)
    got = check_claim(claim, page, received)
    observed = {"verdict": got.verdict, "reason": got.reason, "content_hash": page.content_hash}
    if not got.passed:
        return _result(item, DISAGREE, **observed)
    if expected and category_of(claim.event_type).value != expected:
        return _result(item, DISAGREE, **{**observed, "reason": "category_mismatch"})
    return _result(item, AGREE, **observed)


async def sample_pages(client, epoch: int) -> Tuple[List[dict], List[dict]]:
    """Every re-verify candidate of a closed epoch, and the paid leaf behind each of its
    settle, clawback and penalty rows, all pages."""
    out, settles, cursor = [], [], None
    while True:
        page = await client.ledger_sample(epoch, cursor)
        out.extend(page.get("items") or [])
        settles.extend(page.get("settles") or [])
        cursor = page.get("next_cursor")
        if not cursor:
            return out, settles


class Reverifier:
    def __init__(self, client, selector: Selector, fetcher: Optional[Fetcher] = None,
                 rate: Optional[float] = None,
                 alarm: Optional[float] = None, concurrency: Optional[int] = None,
                 judges: Optional[Iterable[str]] = None):
        self.client, self.selector = client, selector
        self._fetcher = fetcher
        self.rate = config.LEDGER_REVERIFY_RATE if rate is None else float(rate)
        self.alarm = DISAGREE_ALARM if alarm is None else float(alarm)
        self.concurrency = max(1, int(concurrency or config.REVERIFY_CONCURRENCY))
        # The signed profile's ledger.judges replaces these when it names any.
        self.judges: Set[str] = set(config.JUDGE_VALIDATORS if judges is None else judges)
        # Claim ids per epoch whose page the API holds back until they settle.
        self.withheld: Dict[int, Set[str]] = {}

    @property
    def fetcher(self) -> Fetcher:
        if self._fetcher is None:
            self._fetcher = Fetcher(fetch_config())
        return self._fetcher

    async def candidates(self, epoch: int) -> List[dict]:
        return (await sample_pages(self.client, epoch))[0]

    async def run(self, epoch: int, candidates: Optional[List[dict]] = None,
                  only: Optional[Set[str]] = None) -> Dict[str, int]:
        """Re-verify the epoch's keyed sample, or of it only the claims in `only`."""
        if candidates is None:
            candidates = await self.candidates(epoch)
        if only is not None:
            candidates = [c for c in candidates if str(c.get("claim_id")) in only]
        items = keyed(candidates, self.selector, self.rate)
        held = {str(i.get("claim_id")) for i in items if i.get("withheld")}
        if held:
            self.withheld.setdefault(epoch, set()).update(held)
        items = [i for i in items if not i.get("withheld")]
        gate = asyncio.Semaphore(self.concurrency)

        async def one(item):
            async with gate:
                try:
                    return await reverify(item, self.fetcher, self.judges)
                except Exception as e:
                    bt.logging.warning(f"[REVERIFY] claim {item.get('claim_id')} failed: {e!r}")
                    return _result(item, UNREACHABLE, reason="error")

        results = await asyncio.gather(*(one(i) for i in items))
        # An empty post still reports the epoch as done.
        for i in range(0, max(1, len(results)), POST_MAX):
            await self.client.post_reverify(epoch, results[i:i + POST_MAX])

        counts = Counter(r["verdict"] for r in results)
        checked = counts[AGREE] + counts[DISAGREE]
        share = counts[DISAGREE] / checked if checked else 0.0
        bt.logging.info(
            f"[REVERIFY] epoch={epoch} candidates={len(candidates)} sampled={len(items)} "
            f"agree={counts[AGREE]} disagree={counts[DISAGREE]} "
            f"unreachable={counts[UNREACHABLE]} withheld={len(held)}")
        if share > self.alarm:
            bt.logging.error(
                f"[REVERIFY] ALARM epoch={epoch} disagreement {share:.1%} of {checked} "
                f"checked claims")
        return dict(counts)
