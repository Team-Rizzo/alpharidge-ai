"""check_claim(): the deterministic gates every event claim runs on its own page."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from alpharidge_ai.schema import Claim, confirm_claim
from alpharidge_ai.verify.check import (
    check_claim, check_scoop, pick_quote, quote_problem,
)
from alpharidge_ai.verify.content_extractor import ContentExtractor
from alpharidge_ai.verify.fetch import FetchConfig, FetchedPage, read_page

PAGE = Path(__file__).parent / "fixtures" / "pages" / "acme_deal.html"
RECEIVED = datetime(2026, 10, 13, 12, 40, tzinfo=timezone.utc)
PUBLISHED = datetime(2026, 10, 13, 12, 0, tzinfo=timezone.utc)
DEAL = "Acme Inc. agreed to acquire Beta Systems for $1.2 billion."
SAVED = ("Acme Inc. agreed on Tuesday to acquire Beta Systems in a deal valued at $1.2 billion, "
         "the companies said in a joint statement.")


def claim(**kw):
    base = dict(
        kind="event", headline="Acme to acquire Beta Systems", event_type="merger_acquisition",
        category="finance", event_date="2026-10-13",
        actors=[{"id_scheme": "wikidata", "id": "Q1", "role": "acquirer", "name": "Acme Inc."},
                {"id_scheme": "other", "id": "beta", "role": "target", "name": "Beta Systems"}],
        facts=[{"field": "deal_value", "value": 1.2, "unit": "usd_billions"}],
        source_url="https://news.example.org/acme-beta", quote=DEAL)
    base.update(kw)
    return Claim(**base)


def page(text, published=PUBLISHED, **kw):
    return FetchedPage(url="https://news.example.org/a", final_url="https://news.example.org/a",
                       status=200, text=text, published_at=published, content_hash="h", **kw)


def test_saved_page_is_supported():
    html = PAGE.read_text()
    got = read_page("https://news.example.org/acme-beta", "https://news.example.org/acme-beta",
                    200, html.encode(), html, FetchConfig())
    assert got.published_at == PUBLISHED
    assert got.canonical_url == "https://news.example.org/acme-beta"
    result = check_claim(claim(quote=SAVED), got, RECEIVED)
    assert result.passed and result.spans


@pytest.mark.parametrize("text", [
    "Acme Inc. agreed to acquire Beta for $1.2 billion.",
    "Acme Inc. agreed to acquire Beta Systems for $900 million.",
    "Acme Inc. agreed to acquire Beta Systems for 1.2 billion shares.",
])
def test_every_name_and_figure_must_be_on_the_page(text):
    got = check_claim(claim(), page(text), RECEIVED)
    assert (got.verdict, got.reason) == ("rejected", "unsupported")


def test_a_rejection_names_what_is_missing():
    got = check_claim(claim(), page("Acme Inc. agreed to acquire Beta for $900 million."),
                      RECEIVED)
    assert got.detail == {"missing": ["actor:Beta Systems", "fact:deal_value"]}
    data = claim(event_type="economic_data", actors=[{"id_scheme": "other", "id": "bls",
                                                      "name": "Bureau of Labor Statistics"}],
                 facts=[{"field": "indicator", "value": "CPI"}])
    got = check_claim(data, page("The Bureau of Labor Statistics said prices rose."), RECEIVED)
    assert got.detail == {"missing": ["fact:indicator"]}
    assert check_claim(claim(), page(DEAL), RECEIVED).detail is None


@pytest.mark.parametrize("facts,text", [
    ([{"field": "revenue", "value": 94.9, "unit": "usd_billions"}], "revenue was $94,930 million"),
    ([{"field": "revenue", "value": 94930, "unit": "usd_millions"}], "revenue was $94.9B"),
    ([{"field": "eps", "value": 1.2, "unit": "usd_per_unit"}], "earnings per share were $1.20"),
    ([{"field": "change_bps", "value": -25, "unit": "basis_points"}], "rates were cut by a "
     "quarter point"),
    ([{"field": "change_bps", "value": 0, "unit": "basis_points"}], "rates were left unchanged"),
    ([{"field": "actual", "value": 5.1, "unit": "percent"}], "California | 5.1"),
    ([{"field": "net_income", "value": -116, "unit": "usd_millions"}], "it lost $(116) million"),
    ([{"field": "new_rate", "value": 4.5, "unit": "percent"}], "the target range is 4-1/4 to "
     "4-1/2 percent"),
])
def test_figures_in_the_forms_a_page_prints(facts, text):
    c = claim(event_type="earnings", headline="h", facts=facts, quote="Acme Inc. reported its results.",
              actors=[{"id_scheme": "other", "id": "x", "role": "company", "name": "Acme Inc."}])
    assert check_claim(c, page(f"Acme Inc. reported its results.\nIt said on Tuesday that {text}."),
                       RECEIVED).passed


def test_a_dollar_figure_is_not_another_currency():
    c = claim(facts=[{"field": "deal_value", "value": 1.26, "unit": "usd_billions"}])
    assert not check_claim(c, page(f"{DEAL} It is worth C$1.26 billion."), RECEIVED).passed
    assert check_claim(c, page(f"{DEAL} It is worth 1.26 billion."), RECEIVED).passed


def test_a_rate_change_is_left_to_the_judge():
    c = claim(event_type="monetary_policy", headline="h", quote="Acme Bank raised the rate.",
              actors=[{"id_scheme": "other", "id": "x", "role": "central_bank", "name": "Acme Bank"}],
              facts=[{"field": "change_bps", "value": 25, "unit": "basis_points"}])
    text = "Acme Bank raised the rate.\nIt is 1.25 per cent, from 1 per cent."
    assert check_claim(c, page(text), RECEIVED).passed


def _earnings(value, unit, text):
    c = claim(event_type="earnings", headline="h",
              facts=[{"field": "eps" if unit == "usd_per_unit" else "net_income", "value": value,
                      "unit": unit}],
              actors=[{"id_scheme": "other", "id": "x", "role": "company", "name": "Acme Inc."}],
              quote=f"Acme Inc. said on Tuesday that {text}.")
    return check_claim(c, page(c.quote), RECEIVED).passed


@pytest.mark.parametrize("text,ok", [
    ("it earned 26 cents a share", True),
    ("it earned 26 cents per diluted share", True),
    ("earnings per share: 26 cents adjusted vs. 31 cents expected", True),
    ("earnings per share were (26) cents", True),
    ("its fuel cost rose 26 cents a gallon", False),
])
def test_cents_count_as_dollars_only_per_share(text, ok):
    assert _earnings(0.26, "usd_per_unit", text) is ok


@pytest.mark.parametrize("value,unit,text", [
    (-71, "usd_millions", "it posted a $71 million net loss"),
    (-0.26, "usd_per_unit", "it reported a net loss of $116 million, or 26 cents a share"),
    (71, "usd_millions", "it shrank its loss to $71 million"),
])
def test_a_figure_matches_its_size_whatever_its_sign(value, unit, text):
    assert _earnings(value, unit, text)


def test_a_table_figure_reads_at_the_table_scale():
    c = claim(event_type="earnings", headline="h",
              facts=[{"field": "revenue", "value": 49526, "unit": "usd_millions"}],
              actors=[{"id_scheme": "other", "id": "x", "role": "company", "name": "Acme Inc."}],
              quote="Acme Inc. reported its results.")
    row = "Acme Inc. reported its results.\nTotal revenues $ 50,534 $ 49,526"
    head = "INCOME STATEMENTS (in millions, except per share amounts)"
    assert check_claim(c, page(f"{row} {head} {row}"), RECEIVED).passed
    assert not check_claim(c, page(row), RECEIVED).passed
    head = "Financial Highlights | in millions, except per share data"
    assert check_claim(c, page(f"{row} {head} {row}"), RECEIVED).passed


def test_a_fact_subject_must_be_on_the_page():
    deal = claim(actors=claim().actors[:1],
                 facts=[{"field": "target", "value": "Gamma Corp"}])
    assert not check_claim(deal, page(DEAL), RECEIVED).passed
    gamma = "Acme Inc. agreed to acquire Gamma Corp."
    assert check_claim(deal.model_copy(update={"quote": gamma}), page(gamma), RECEIVED).passed
    unnamed = claim(actors=[{"id_scheme": "wikidata", "id": "Q312", "role": "acquirer"}],
                    facts=[{"field": "target", "value": "Beta Systems"}])
    assert not check_claim(unnamed, page(DEAL), RECEIVED).passed


def test_date_gate():
    assert check_claim(claim(), page(DEAL, published=None), RECEIVED).reason == "no_publish_date"
    # Events and pages up to 30 days old are in; the event day counts whole.
    month = RECEIVED - timedelta(days=30)
    assert check_claim(claim(event_date="2026-09-13"), page(DEAL, published=month),
                       RECEIVED).passed
    stale = check_claim(claim(), page(DEAL, published=month - timedelta(minutes=1)), RECEIVED)
    assert (stale.reason, stale.detail) == ("stale_date", {"date": "published"})
    assert check_claim(claim(event_date="2026-09-12"), page(DEAL, published=month),
                       RECEIVED).detail == {"date": "event_date"}
    assert check_claim(claim(), page(DEAL), PUBLISHED - timedelta(seconds=1)).reason == \
        "pre_published"
    assert check_claim(claim(event_date="2026-10-15"), page(DEAL), RECEIVED).reason == \
        "future_date"
    # A day after the page's publish day anywhere is too late; its day in UTC+14 is not.
    late = datetime(2026, 10, 12, 9, 59, tzinfo=timezone.utc)
    assert check_claim(claim(), page(DEAL, published=late), RECEIVED).reason == "future_date"
    assert check_claim(claim(), page(DEAL, published=late + timedelta(minutes=1)),
                       RECEIVED).passed
    inferred = page(DEAL, date_inferred=True)
    assert check_claim(claim(), inferred, RECEIVED).reason == "no_publish_date"
    # A body date counts on an official statistics or central-bank page only.
    release = "https://www.bls.gov/news.release/x.htm"
    assert check_claim(claim(), replace(inferred, url=release, final_url=release),
                       RECEIVED).passed
    # A site with its own date rule takes only the date it states, never a guess.
    ofac = "https://ofac.treasury.gov/recent-actions/20261010"
    assert check_claim(claim(), replace(inferred, url=ofac, final_url=ofac),
                       RECEIVED).reason == "no_publish_date"


def test_unreadable_page_and_page_rules():
    assert check_claim(claim(), FetchedPage(url="u", reason="blocked"), RECEIVED).reason == \
        "blocked"
    assert check_claim(claim(), page(DEAL), RECEIVED, excluded="contributor").reason == \
        "page_type"


def test_hidden_text_is_not_page_text():
    html = ("<html><body><article><p>Acme Inc. reported results for the quarter on Tuesday "
            "and said demand for its software stayed strong.</p>"
            "<p hidden>Acme Inc. agreed to acquire Beta Systems for $1.2 billion.</p>"
            '<p style="display: none">Acme Inc. agreed to acquire Gamma Corp.</p>'
            "<!-- Acme Inc. agreed to acquire Delta Corp. --></article></body></html>")
    text = ContentExtractor().extract(html)
    assert "results" in text and "Beta" not in text and "Gamma" not in text
    assert "Delta" not in text
    hidden_only = '<html><body><div style="display:none">' + DEAL + "</div></body></html>"
    assert "Acme" not in ContentExtractor().extract(hidden_only)
    xml = '<?xml version="1.0" encoding="utf-8"?>' + html.replace(
        "<p hidden>", '<p aria-hidden="true">')
    assert "Beta" not in ContentExtractor().extract(xml)


@pytest.mark.parametrize("attrs", ['aria-hidden="TRUE"', 'style="opacity:0.0"',
                                   'style="opacity: .0"', 'style="font-size:0.0px"',
                                   'style="visibility:collapse"'])
def test_other_spellings_of_hidden_are_hidden(attrs):
    html = ("<html><body><article><p>Acme Inc. reported results for the quarter on Tuesday "
            "and said demand for its software stayed strong.</p>"
            f"<p {attrs}>Acme Inc. agreed to acquire Beta Systems for $1.2 billion.</p>"
            '<p style="opacity:0.5;font-size:0.9em">Shares rose 3% in early trading.</p>'
            "</article></body></html>")
    text = ContentExtractor().extract(html)
    assert "Beta" not in text and "Shares rose" in text


QUOTED = ("Deals\n"
          "Acme Inc. agreed on Tuesday to acquire Beta Systems for $1.2 billion, the companies "
          "said. Shares rose 3%.\n"
          "Analysts said Acme Inc. may also cut 2,000 jobs.\n"
          "Short seller Bob wrote: \u201cAcme Inc. is insolvent.\u201d\n"
          "\u201cAcme Inc. will close the deal soon,\u201d its chief executive said.\n"
          "Acme Inc. said the deal, which was approved by both boards, is expected to close\n"
          "in the first quarter of next year.\n"
          "Name | Value\nAcme Inc. | 1.2 billion dollars.\n")
SAID = "Acme Inc. agreed on Tuesday to acquire Beta Systems for $1.2 billion, the companies said."


@pytest.mark.parametrize("quote,why", [
    (SAID, None),
    (SAID.replace("1.2 billion", "1.2\u00a0billion") + " Shares rose 3%.", None),
    ('"Acme Inc. will close the deal soon," its chief executive said.', None),
    ("Acme Inc. said the deal, which was approved by both boards, is expected to close in the "
     "first quarter of next year.", None),
    ("Acme Inc. may also cut 2,000 jobs.", "not_on_page"),
    ('"Acme Inc. is insolvent."', "not_on_page"),
    ("Deals " + SAID, "not_on_page"),
    ("Acme Inc. | 1.2 billion dollars.", "not_on_page"),
    ("Acme Inc. agreed on Tuesday to acquire Beta Systems", "not_on_page"),
    (None, "missing"),
])
def test_the_quote_is_whole_sentences_of_the_visible_text(quote, why):
    assert quote_problem(claim(quote=quote), QUOTED) == why
    hidden = ContentExtractor().extract(f"<html><body><article><p>{SAID}</p><p hidden>{DEAL}"
                                        "</p></article></body></html>")
    assert quote_problem(claim(), hidden) == "not_on_page"


RELEASE = ("SEATTLE--(WIRE)--Acme Inc. (NYSE: ACME) today reported results for its third "
           "quarter. \u25aa Acme Inc. revenue was $1.2 billion, up 4% \u25aa Acme Inc. cash rose "
           "8%, a record \u2022 Net income was $90 million\nSEATTLE, Oct. 1, 2026\u2014Acme Inc. "
           "also said it would cut 2,000 jobs.\nNotes: Acme Inc. revenue was $1.2 billion, up 4%")


@pytest.mark.parametrize("quote,why", [
    ("Acme Inc. (NYSE: ACME) today reported results for its third quarter.", None),
    ("Acme Inc. revenue was $1.2 billion, up 4%", None),
    ("Acme Inc. cash rose 8%", "not_on_page"),
    ("Acme Inc. also said it would cut 2,000 jobs.", None),
    ("Acme Inc. revenue was $1.2 billion, up 4% \u25aa Acme Inc. cash rose 8%, a record",
     "not_on_page"),
])
def test_press_release_bullets_and_datelines(quote, why):
    assert quote_problem(claim(quote=quote), RELEASE) == why


NAMED = ('Vandelay Inc., a Delaware corporation ("Vandelay" or the "Company"), filed a report.\n'
         "The Company will cut 500 jobs.\n"
         'Gamma Corp. (NYSE: GMA) ("Gamma Holdings") filed a report.\n'
         "Gamma Holdings will cut 100 jobs.\n"
         'Beta Corp ("Beta") filed a report.\nBeta will cut 300 jobs.\n'
         "The Federal Open Market Committee (FOMC) met.\nThe Committee decided to cut rates.\n"
         "The FOMC said inflation eased.")


@pytest.mark.parametrize("name,quote,why", [
    ("Vandelay Inc.", "The Company will cut 500 jobs.", None),
    ("Vandelay Inc.", "Beta will cut 300 jobs.", "no_subject"),
    ("Federal Open Market Committee", "The Committee decided to cut rates.", None),
    ("Federal Open Market Committee", "The FOMC said inflation eased.", None),
    ("Beta Corp", "The Company will cut 500 jobs.", "no_subject"),
    ("Gamma Corp.", "Gamma Holdings will cut 100 jobs.", None),
])
def test_the_quote_names_the_subject(name, quote, why):
    c = claim(event_type="layoffs", quote=quote, facts=[],
              actors=[{"id_scheme": "other", "id": "x", "role": "company", "name": name}])
    assert quote_problem(c, NAMED) == why


def test_ticker_and_exchange_are_checked_on_the_page():
    acme = [{"id_scheme": "other", "id": "a", "role": "subject", "name": "Acme Inc."}]
    quote = "Trading in Acme Inc. shares was halted on the New York Stock Exchange."

    def halt(exchange, ticker):
        return claim(event_type="trading_halt", actors=acme, quote=quote,
                     facts=[{"field": "exchange", "value": exchange},
                            {"field": "ticker", "value": ticker}])
    text = quote + " The stock (NYSE: ACME) last traded at $12."
    assert check_claim(halt("NYSE", "ACME"), page(text), RECEIVED).passed
    assert check_claim(halt("The New York Stock Exchange", "ACME"), page(text), RECEIVED).passed
    for c, field in ((halt("Nasdaq", "ACME"), "exchange"), (halt("NYSE", "ACM"), "ticker"),
                     (halt("NYSE", "acme"), "ticker")):
        r = check_claim(c, page(text), RECEIVED)
        assert r.detail == {"missing": [f"fact:{field}"]}


FOUND = ("Markets were quiet on Tuesday.\n"
         "Acme Inc. agreed to buy Beta Systems, people familiar with the matter said. The deal "
         "values Beta Systems at $1.2 billion. Acme Inc. will pay $1.2 billion in cash for "
         "Beta Systems.\nShares rose.")


def test_a_quote_is_picked_from_the_found_page():
    held = claim(facts=[{"field": "deal_value", "value": 1.2, "unit": "usd_billions"},
                        {"field": "stake_pct", "value": 80, "unit": "pct"}])
    q = pick_quote(held, FOUND)
    assert q == "The deal values Beta Systems at $1.2 billion."
    assert pick_quote(held, FOUND) == q
    found = confirm_claim(held, "https://wire.example.com/acme", q, FOUND)
    assert found.source_url == "https://wire.example.com/acme" and found.quote == q
    assert [f.field for f in found.facts] == ["deal_value"]
    assert check_claim(found, page(FOUND), RECEIVED).passed
    assert pick_quote(held, "Markets were quiet on Tuesday.") is None


def test_a_confirm_claim_keeps_figures_its_type_needs():
    held = claim(event_type="layoffs", headline="h", quote="Acme Inc. will cut 900 jobs.",
                 actors=[{"id_scheme": "other", "id": "a", "role": "company", "name": "Acme Inc."}],
                 facts=[{"field": "jobs_cut", "value": 900, "unit": "count"}])
    text = "Acme Inc. will cut 100 jobs, the company said."
    found = confirm_claim(held, "https://wire.example.com/acme", pick_quote(held, text), text)
    assert [f.field for f in found.facts] == ["jobs_cut"]
    assert not check_claim(found, page(text), RECEIVED).passed


UNREAD = FetchedPage(url="https://news.example.org/a", status=403, reason="blocked",
                     unread=True)


@pytest.mark.parametrize("kw,want", [
    ({}, (True, None)),
    ({"event_date": "2026-10-15"}, (False, "future_date")),
    ({"event_date": "2026-09-01"}, (False, "stale_date")),
])
def test_an_unread_scoop_is_held_on_its_event_date(kw, want):
    assert not check_claim(claim(**kw), UNREAD, RECEIVED).passed
    r = check_scoop(claim(**kw), UNREAD, RECEIVED)
    assert (r.passed, r.reason, r.unread) == (*want, True)


def test_a_readable_scoop_page_is_checked_as_any_page():
    got = page(SAVED)
    assert check_scoop(claim(), got, RECEIVED) == check_claim(claim(), got, RECEIVED)
    assert check_scoop(claim(), UNREAD, RECEIVED, excluded="hub").reason == "page_type"


def test_find_matches_whole_words_without_overlaps():
    from alpharidge_ai.verify.check import find
    assert find("aa aaa aa", "aa") == [(0, 2), (7, 9)]
    assert find("beta_x beta-x xbeta beta", "beta") == [(7, 11), (20, 24)]
    assert find("anything", "") == []


def test_pick_quote_stays_fast_on_a_long_page():
    import time
    lines = "".join(f"Beta Systems agreed on day {i} to a plan with the board.\n"
                    for i in range(900))
    text = lines + "Acme Inc. will buy Beta Systems for $1.2 billion, it said.\n"
    t = time.perf_counter()
    got = pick_quote(claim(), text)
    assert time.perf_counter() - t < 2
    assert got == "Acme Inc. will buy Beta Systems for $1.2 billion, it said."


def test_a_status_page_incident_is_dated_by_its_first_post():
    from datetime import datetime, timezone
    from alpharidge_ai.verify.dates import publish_date
    html = ('<div class="update"><small data-datetime-unix="1794269152000"></small></div>'
            '<div class="update"><small data-datetime-unix="1794268784000"></small></div>')
    dt, inferred = publish_date(html, "https://www.githubstatus.com/incidents/abc")
    assert dt == datetime.fromtimestamp(1794268784, timezone.utc) and not inferred
    assert publish_date("<p>On 2026-11-18</p>", "https://www.githubstatus.com/incidents/m")[0] is None
