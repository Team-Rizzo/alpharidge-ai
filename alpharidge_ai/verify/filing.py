"""An SEC 8-K's item codes as evidence for the event question: the filer's own current report,
read from its public submissions list, whose items include one that reports the claim's type
and stage (types.json `filing_items`), on the claimed day. The API and validators read the
same public list and must agree."""
from __future__ import annotations

from typing import Optional

from alpharidge_ai.schema import TYPE_SPEC, EventType

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
FORMS = ("8-K", "8-K/A")
# Items any current report may add (Reg FD, other events, exhibits).
PLAIN_ITEMS = frozenset(("7.01", "8.01", "9.01"))
# At most this many items besides those: a report of many changes at once is left to the judge.
ITEMS_MAX = 2


def submissions_url(cik: str) -> str:
    return SUBMISSIONS_URL.format(cik=str(cik).zfill(10))


def filing_row(submissions: dict, cik: str, accession: str) -> Optional[dict]:
    """The filing's form, items and dates from a submissions list, or None."""
    try:
        recent = submissions["filings"]["recent"]
        acc = [a.replace("-", "") for a in recent["accessionNumber"]]
        i = acc.index(accession.replace("-", ""))
        return {"cik": str(cik).zfill(10), "accession": acc[i], "form": recent["form"][i],
                "items": [x for x in str(recent["items"][i]).split(",") if x],
                "report_date": recent["reportDate"][i] or None,
                "accepted": str(recent["acceptanceDateTime"][i])[:10]}
    except (KeyError, TypeError, ValueError, IndexError, AttributeError):
        return None


def confirms(claim, row: Optional[dict]) -> bool:
    """The filing is a current report with an item for the claim's type and stage and at most
    ITEMS_MAX items besides PLAIN_ITEMS, and the claimed day is its report date or the day it
    was accepted."""
    if not isinstance(row, dict) or row.get("form") not in FORMS:
        return False
    spec = TYPE_SPEC.get(EventType(claim.event_type))
    stage = getattr(claim.stage, "value", claim.stage) or ""
    mine = set(spec.filing_items.get(stage, ())) if spec else set()
    items = set(row.get("items") or ())
    day = claim.event_date.isoformat()
    return (bool(mine & items) and len(items - PLAIN_ITEMS) <= ITEMS_MAX
            and day in (row.get("report_date"), row.get("accepted")))
