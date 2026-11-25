from datetime import date, datetime, timedelta, timezone

import pytest

from alpharidge_ai.schema import Claim
from alpharidge_ai.schema import types
from alpharidge_ai.validator import reverify

NEW = datetime(2026, 11, 19, 9, 30, tzinfo=timezone.utc)
LEAD = datetime(2026, 12, 1, 12, 0, tzinfo=timezone.utc)
STEPS = [{"pay_table": "new", "from_ms": int(NEW.timestamp() * 1000)},
         {"pay_table": "lead", "from_ms": int(LEAD.timestamp() * 1000)}]
RECEIVED = LEAD + timedelta(hours=1)
NEWS = "https://news.example.org/acme-beta"


@pytest.fixture(autouse=True)
def _steps(monkeypatch):
    for k in ("EL_PAY_TABLE", "EL_PAY_TABLE_FROM_MS", "EL_EVENT_PAY_X", "EL_SCOOP_PREMIUM",
              "EL_SCOOP_MISS_DEBT_X"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(types, "_switch", None)
    types.set_pay_switch(STEPS)


def _m(lead_min, received=RECEIVED, url=NEWS, first_seen=None):
    return types.scoop_multiple(received + timedelta(minutes=lead_min), received,
                                first_seen=first_seen, source_url=url)


@pytest.mark.parametrize("lead_min, want", [
    (-5, 0.25), (0, 0.25), (9.9, 0.25), (10, 0.25), (35, 5.125), (60, 10.0), (61, 10.0),
    (19 * 60, 10.0)])
def test_the_premium_follows_the_lead(lead_min, want):
    assert _m(lead_min) == pytest.approx(want, abs=1e-12)


def test_the_curve_is_a_straight_line_rounded_to_4_dp():
    assert _m(12.5) == round(0.25 + 9.75 * 2.5 / 50, 4) == 0.7375
    assert _m(53.5) == round(0.25 + 9.75 * 43.5 / 50, 4)
    assert types.lead_multiple(10 * 60_000 + 1) == 0.25
    assert types.lead_multiple(60 * 60_000 - 1) == 10.0


def test_the_lead_counts_from_the_first_trusted_sighting():
    assert _m(120, first_seen=RECEIVED + timedelta(minutes=35)) == 5.125
    assert _m(120, first_seen=RECEIVED - timedelta(minutes=2)) == 0.25
    assert _m(120, first_seen=RECEIVED) == 0.25


@pytest.mark.parametrize("url", [
    "https://www.prnewswire.com/news-releases/acme-301.html",
    "https://www.prnewswire.co.uk/news-releases/acme.html",
    "https://www.globenewswire.com/news-release/2026/12/01/1/0/en/acme.html",
    "https://www.businesswire.com/news/home/2026/acme/en/",
    "https://www.stocktitan.net/news/ACME/acme-to-acquire.html",
    "https://www.accessnewswire.com/newsroom/en/acme",
    "https://www.accesswire.com/viewarticle.aspx?id=1",
    "https://www.newsfilecorp.com/release/1/acme",
    "https://ir.acme.com/news-releases/news-release-details/acme",
    "https://investors.acme.com/news/2026/acme",
    "https://investor.acme.com/press",
    "https://www.acme.com/investor-relations/news/acme"])
def test_a_wire_or_ir_page_is_official_and_pays_the_event_rate(url):
    assert types.official_release(url)
    assert _m(120, url=url) == 0.25


@pytest.mark.parametrize("url", [
    NEWS, "https://www.mlbtraderumors.com/2026/11/x.html", "https://finance.yahoo.com/news/x",
    "https://www.wire.com/en/blog", "https://ir-news.example.org/x", "https://www.acme.com/ir"])
def test_other_pages_are_not_official(url):
    assert not types.official_release(url)


def test_nothing_changes_before_the_lead_start():
    before = LEAD - timedelta(minutes=1)
    assert types.pay_table(int(before.timestamp() * 1000)) == "new"
    assert _m(12.5, received=before) == 10.0
    assert _m(120, received=before, url="https://www.stocktitan.net/news/A/a.html") == 10
    assert _m(-1, received=before) == 0.25
    old = NEW - timedelta(minutes=1)
    assert types.pay_table(int(old.timestamp() * 1000)) == "today"
    assert _m(12.5, received=old) == 5.0
    assert types.pay_table(int(RECEIVED.timestamp() * 1000)) == "lead"


def test_the_event_rate_and_debt_are_the_new_tables():
    assert types.event_price("high", RECEIVED) == types.event_price("high", LEAD - timedelta(1))
    assert (types.scoop_debt("merger_acquisition", date(2026, 12, 1), RECEIVED)
            == types.scoop_debt("merger_acquisition", date(2026, 12, 1), LEAD - timedelta(1)))
    assert types.pay_settings(RECEIVED) == types.pay_settings(LEAD - timedelta(minutes=1))


def test_steps_parse_from_the_env_and_out_of_order_ones_are_refused(monkeypatch):
    monkeypatch.setenv("EL_PAY_TABLE", "new,lead")
    monkeypatch.setenv("EL_PAY_TABLE_FROM_MS", f"{STEPS[0]['from_ms']},{STEPS[1]['from_ms']}")
    assert types.pay_switch() == (("new", STEPS[0]["from_ms"]), ("lead", STEPS[1]["from_ms"]))
    monkeypatch.setenv("EL_PAY_TABLE_FROM_MS", str(STEPS[0]["from_ms"]))
    with pytest.raises(types.PaySettingError):
        types.pay_switch()
    monkeypatch.delenv("EL_PAY_TABLE")
    with pytest.raises(types.PaySettingError):
        types.set_pay_switch(STEPS[::-1])
    assert types.pay_table(int(RECEIVED.timestamp() * 1000)) == "lead"


CLAIM = Claim(
    kind="event", headline="Acme to acquire Beta Systems", event_type="merger_acquisition",
    category="finance", event_date="2026-12-01",
    actors=[{"id_scheme": "other", "id": "a", "role": "acquirer", "name": "Acme Inc."},
            {"id_scheme": "other", "id": "b", "role": "target", "name": "Beta Systems"}],
    facts=[], source_url=NEWS,
    quote="Acme Inc. agreed on Tuesday to acquire Beta Systems, the companies said.")


def _scoop_item(multiple, published, first_seen=None, t=0.7):
    base = round(12 * multiple, 6)
    confirm = {"published_at": published.isoformat().replace("+00:00", "Z")}
    if first_seen is not None:
        confirm["first_seen"] = first_seen.isoformat().replace("+00:00", "Z")
    return {"route": "scoop", "points": round(0.4 * base * t, 6),
            "factors": {"base": base, "T": t, "A": 1.0},
            "judge": {"confirm": confirm}, "published_at": None}


def test_a_validator_rebuilds_the_lead_price_from_the_confirm_record():
    published = RECEIVED + timedelta(minutes=12, seconds=28)
    m = _m(12 + 28 / 60)
    assert reverify.paid_right(_scoop_item(m, published), CLAIM, RECEIVED)
    assert not reverify.paid_right(_scoop_item(10.0, published), CLAIM, RECEIVED)
    seen = RECEIVED - timedelta(minutes=9)
    assert reverify.paid_right(_scoop_item(0.25, published, seen), CLAIM, RECEIVED)
    wire = CLAIM.model_copy(update={"source_url": "https://www.stocktitan.net/news/A/a.html"})
    late = RECEIVED + timedelta(hours=3)
    assert reverify.paid_right(_scoop_item(0.25, late), wire, RECEIVED)
    assert not reverify.paid_right(_scoop_item(10.0, late), wire, RECEIVED)
