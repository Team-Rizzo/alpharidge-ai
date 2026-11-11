import asyncio
import json

from alpharidge_ai.schema import Claim
from alpharidge_ai.validator.reverify import filing_problem
from alpharidge_ai.verify.filing import confirms, filing_row

URL = "https://www.sec.gov/Archives/edgar/data/1/000000000126000001/a8k.htm"
SUBS = {"filings": {"recent": {
    "accessionNumber": ["0000000001-26-000002", "0000000001-26-000001"],
    "form": ["10-Q", "8-K"], "items": ["", "2.02,9.01"], "reportDate": ["", "2026-10-13"],
    "acceptanceDateTime": ["2026-10-14T20:00:00.000Z", "2026-10-13T20:05:00.000Z"]}}}


def earnings(**kw):
    base = dict(kind="event", headline="Acme results", event_type="earnings", category="finance",
                event_date="2026-10-13", source_url=URL, quote="Acme Inc. reported results.",
                actors=[{"id_scheme": "other", "id": "a", "role": "company", "name": "Acme Inc."}],
                facts=[{"field": "revenue", "value": 1.2, "unit": "usd_billions"}])
    base.update(kw)
    return Claim(**base)


class Fetcher:
    def __init__(self, body):
        self.body = body

    async def get_text(self, url):
        return self.body


def test_an_8k_item_for_the_type_on_the_day_confirms_and_validators_reread_it():
    row = filing_row(SUBS, "1", "000000000126000001")
    assert row == {"cik": "0000000001", "accession": "000000000126000001", "form": "8-K",
                   "items": ["2.02", "9.01"], "report_date": "2026-10-13",
                   "accepted": "2026-10-13"}
    assert confirms(earnings(), row) and not confirms(earnings(event_date="2026-10-11"), row)
    assert not confirms(earnings(event_type="layoffs", facts=[]), row)
    assert not confirms(earnings(), filing_row(SUBS, "1", "000000000126000002"))
    record = {"event_by": "filing", "filing": row}
    run = asyncio.run
    assert run(filing_problem(record, earnings(), Fetcher(json.dumps(SUBS)))) is None
    assert run(filing_problem({**record, "filing": {**row, "items": ["2.02", "5.02"]}},
                              earnings(), Fetcher(json.dumps(SUBS)))) == "filing_mismatch"
    assert run(filing_problem(record, earnings(), Fetcher(None))) == "unreachable"
