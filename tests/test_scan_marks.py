from datetime import date, datetime, timezone

import pytest

from alpharidge_ai.schema import EventType
from alpharidge_ai.validator.scan import TYPE_WORDS, Scoop, held_claim, second_mark, type_word
from alpharidge_ai.verify import floor

RECEIVED = datetime(2026, 10, 13, 12, 0, tzinfo=timezone.utc)
URL = "https://wire.example.com/x"


def _scoop(event_type="personnel", subjects=("ICE",), **kw):
    base = dict(id=1, event_type=EventType(event_type), stage=None,
                event_date=date(2026, 10, 13), received_at=RECEIVED, subjects=subjects,
                ciks=(), facts=())
    base.update(kw)
    return Scoop(**base)


def _mark(scoop, text, name=None):
    return second_mark(scoop, name or scoop.subjects[0], floor.normalize(text).text, text)


def test_a_common_word_is_not_a_word_of_the_type():
    assert not _mark(_scoop(), "ICE agents left and witnesses described the change in mood.")
    assert not _mark(_scoop(), "ICE CEO Jeff Sprecher said markets were calm.")
    s = _scoop(subjects=("Premier",))
    assert not _mark(s, "Premier League leaders Arsenal made a change in goal.")
    assert not _mark(_scoop("funding", subjects=("Premier",)),
                     "Premier League clubs played the next round of fixtures.")


def test_a_real_personnel_page_shows_a_word_of_the_type():
    text = "ICE named Jane Doe chief financial officer on Monday."
    assert _mark(_scoop(), text)
    assert _mark(_scoop(), "Jeff Sprecher will step down as CEO of ICE.")
    s = _scoop(facts=({"field": "person", "value": "Jane Doe"},
                      {"field": "role", "value": "cfo"}))
    claim = held_claim(s, URL, text)
    assert claim is not None and claim.actors[0].name == "ICE"


def test_a_short_capitals_subject_is_found_only_in_capitals():
    text = "Ice storms closed roads as the board named Jane Doe chief financial officer."
    assert held_claim(_scoop(), URL, text) is None
    assert held_claim(_scoop(), URL, text.replace("Ice storms", "ICE")) is not None
    # Another subject the page does show is used instead.
    s = _scoop(subjects=("ICE", "Intercontinental Exchange"))
    claim = held_claim(s, URL, "Intercontinental Exchange " + text)
    assert claim is not None and claim.actors[0].name == "Intercontinental Exchange"


def test_every_type_has_words_and_none_is_common_news():
    assert set(TYPE_WORDS) == {t.value for t in EventType}
    common = ("the change in data, the release of results, a rate, the case, a round, breaking "
              "news, the transfer, security, action, a product, leaders, the deal, the market, "
              "a report, a statement, the season, the company said, the new plan, an office, "
              "the match, the price, a policy, the chain, the index, a launch party")
    norm = floor.normalize(common).text
    assert [t for t in EventType if type_word(t, norm)] == []


@pytest.mark.parametrize("event_type,text", [
    ("merger_acquisition", "the acquisition closed"),
    ("funding", "it raised $20 million in a series b round"),
    ("bankruptcy", "it filed for chapter 11 protection"),
    ("layoffs", "it will lay off 300 workers"),
    ("monetary_policy", "it cut interest rates by 25 basis points"),
    ("vulnerability", "a zero-day flaw, cve-2026-1234"),
    ("security_incident", "a ransomware attack"),
    ("ipo_listing", "its initial public offering"),
    ("match_result", "a hat-trick in stoppage time"),
    ("transfer", "he signed a three-year contract"),
    ("election_result", "she was re-elected"),
    ("legislation_passed", "the bill was signed into law"),
])
def test_type_words_find_their_type(event_type, text):
    assert type_word(EventType(event_type), floor.normalize(text).text)


@pytest.mark.parametrize("value,unit,text", [
    (1_500_000, "count", "It raised $1,500,000 from investors."),
    (1.5, "usd_millions", "It raised $1.5m from investors."),
    (1.5, "usd_millions", "It raised 1.5 million dollars."),
    (1.5, "usd_millions", "It raised US$1,500,000."),
    (1.5, "usd_billions", "It raised $1.5bn."),
    (1.5, "usd_billions", "It raised 1.5 billion dollars."),
    (2400, "count", "It cut 2,400 jobs."),
    (51, "percent", "It took a 51% stake."),
    (25, "basis_points", "It moved by 25 basis points."),
    (4.25, "percent", "The rate is now 4.25%."),
])
def test_a_figure_counts_in_its_usual_written_forms(value, unit, text):
    s = _scoop("partnership", subjects=("Beta Systems",),
               facts=({"field": "deal_value", "value": value, "unit": unit},))
    assert _mark(s, "Beta Systems said this. " + text)


@pytest.mark.parametrize("value,unit,text", [
    (3, "count", "Beta Systems said 3 people met."),
    (0, "count", "Beta Systems said 0 people met."),
    (2026, "count", "Beta Systems said in 2026 it would grow."),
    (51, "percent", "Beta Systems spoke for 51 minutes."),
    (20, "usd", "Beta Systems hired 20 people."),
])
def test_a_small_or_common_figure_is_not_a_mark(value, unit, text):
    s = _scoop("partnership", subjects=("Beta Systems",),
               facts=({"field": "deal_value", "value": value, "unit": unit},))
    assert not _mark(s, text)


def test_a_text_fact_must_be_more_than_the_subject_or_a_stock_word():
    s = _scoop("match_result", subjects=("Premier",),
               facts=({"field": "competition", "value": "Premier League"},))
    assert not _mark(s, "Premier League clubs met on Tuesday.")
    s = _scoop("funding", subjects=("Beta Systems",),
               facts=({"field": "lead_investor", "value": "Sequoia Capital"},))
    assert _mark(s, "Beta Systems said Sequoia Capital put money in.")
    s = _scoop("vulnerability", subjects=("Beta Systems",),
               facts=({"field": "exploited", "value": "yes"},))
    assert not _mark(s, "Beta Systems said yes to the plan.")
    s = _scoop("breakthrough", subjects=("Beta Systems",),
               facts=({"field": "kind", "value": "first"},))
    assert not _mark(s, "Beta Systems first opened in Ohio.")
    s = _scoop("ipo_listing", subjects=("Beta Systems",),
               facts=({"field": "exchange", "value": "Nasdaq"},))
    assert not _mark(s, "Beta Systems shares fell on the Nasdaq.")


def test_another_name_must_be_another_party_written_as_such():
    s = _scoop("partnership", subjects=("Beta Systems", "ICE"))
    assert not _mark(s, "Beta Systems said ice storms hit.")
    assert _mark(s, "Beta Systems and ICE met on Tuesday.")
    s = _scoop("partnership", subjects=("International Business Machines", "IBM"))
    assert not _mark(s, "International Business Machines (IBM) opened an office.")


def test_a_short_ticker_counts_only_where_written_as_one():
    s = _scoop("partnership", subjects=("Beta Systems",), tickers=("A",))
    assert not _mark(s, "A Beta Systems office opened.")
    assert _mark(s, "Beta Systems (NYSE: A) opened an office.")


@pytest.mark.parametrize("event_type,subject,text", [
    ("product_launch", "Acme Corp", "Acme Corp launches Widget 2, its first folding phone."),
    ("security_incident", "Acme Corp",
     "Item 1.05 Material Cybersecurity Incidents. On October 10, Acme Corp detected a "
     "cybersecurity incident involving unauthorised activity on its network."),
    ("security_incident", "Acme Corp", "Acme Corp disclosed a security incident on Friday."),
    ("trading_halt", "Widget Labs", "Trading in Widget Labs shares was halted on Tuesday."),
    ("index_change", "Widget Labs", "Widget Labs is set to join the S&P 500 next week."),
    ("index_change", "Widget Labs", "Widget Labs joins S&P 500 in quarterly rebalance."),
    ("ipo_listing", "Widget Labs", "Widget Labs shares soared in their Nasdaq debut."),
    ("ipo_listing", "Widget Labs", "Widget Labs jumps in NYSE debut."),
    ("monetary_policy", "Federal Reserve", "Federal Reserve trims key rate to 3.75%-4%"),
    ("monetary_policy", "Bank of England", "Bank of England leaves rates unchanged at 4%"),
    ("monetary_policy", "Bank of England", "Bank of England keeps rates on hold"),
    ("monetary_policy", "Bank of England", "Bank of England maintains rates for a third month"),
])
def test_realistic_confirming_text_has_a_word_of_its_type(event_type, subject, text):
    assert _mark(_scoop(event_type, subjects=(subject,)), text)


def test_a_hold_decision_is_marked_by_its_words_not_its_zero_change():
    s = _scoop("monetary_policy", subjects=("Bank of England",),
               facts=({"field": "change_bps", "value": 0, "unit": "basis_points"},))
    assert _mark(s, "Bank of England leaves rates unchanged at 4%")
    assert not _mark(s, "Bank of England said 0 people attended.")


def test_a_name_that_shares_a_word_with_the_subject_counts_for_what_it_adds():
    s = _scoop("product_launch", subjects=("Apple", "Apple Inc."),
               facts=({"field": "product", "value": "Apple Watch Series 11"},))
    assert _mark(s, "Apple shows Watch Series 11 to buyers in Cupertino.", "Apple")
    assert held_claim(s, URL, "Apple shows Watch Series 11 to buyers in Cupertino.") is not None
    assert _mark(_scoop("vulnerability", subjects=("Microsoft",),
                        facts=({"field": "product", "value": "Microsoft Copilot Vision"},)),
                 "Microsoft said Copilot Vision could leak files.")
    # A remainder of one stock word, or of under three characters, is not another mark.
    s = _scoop("match_result", subjects=("Premier",),
               facts=({"field": "competition", "value": "Premier League"},))
    assert not _mark(s, "Premier League clubs met on Tuesday.")
    s = _scoop("partnership", subjects=("Beta Systems", "Beta Systems AG"))
    assert not _mark(s, "Beta Systems AG opened an office.", "Beta Systems")


def test_a_long_capitals_subject_may_be_written_in_title_case():
    for name, event_type, text in (
            ("OPEC", "sanction_or_tariff", "Opec members agreed sanctions on Iran."),
            ("UEFA", "award", "Uefa named him player of the year."),
            ("NATO", "sanction_or_tariff", "Nato allies agreed new sanctions on Moscow."),
            ("NASA", "breakthrough", "Nasa scientists reported a world first.")):
        s = _scoop(event_type, subjects=(name,))
        assert held_claim(s, URL, text) is not None, name
        assert held_claim(s, URL, text.replace(name.title(), name.lower())) is None, name
    # Three letters stay capitals only: "Ice" is not ICE.
    assert held_claim(_scoop(), URL, "Ice storms closed roads as the board named Jane Doe "
                                     "chief financial officer.") is None


def test_a_percent_change_is_not_read_as_millions():
    s = _scoop("economic_data", subjects=("Statistics Canada",),
               facts=({"field": "prior", "value": 300_000, "unit": "count"},))
    assert not _mark(s, "Statistics Canada said prices rose 0.3% m/m in September.")
    s = _scoop("economic_data", subjects=("Statistics Canada",),
               facts=({"field": "actual", "value": 0.3, "unit": "percent"},))
    assert _mark(s, "Statistics Canada said prices rose 0.3% m/m in September.")
    # The shared parse is unchanged: the API's claim check and re-verify read as before.
    nums = floor.parse_numbers(floor.normalize("up 0.3% m/m").text)
    assert [(n.value, n.unit) for n in nums] == [(300_000.0, "count")]


@pytest.mark.parametrize("event_type,text", [
    ("merger_acquisition", "Acme Corp's founder was acquitted of fraud."),
    ("merger_acquisition", "Acme Corp acquiesced to the board's demands."),
    ("personnel", "Acme Corp sponsored the event; guests such as President Smith attended."),
    ("personnel", "Acme Corp, led by its new president, opened a shop."),
])
def test_a_near_word_of_the_type_is_not_a_mark(event_type, text):
    assert not _mark(_scoop(event_type, subjects=("Acme Corp",)), text)


def test_acquiring_and_naming_still_mark():
    assert _mark(_scoop("merger_acquisition", subjects=("Acme Corp",)),
                 "Acme Corp agreed to acquire Widget Inc.")
    assert _mark(_scoop("merger_acquisition", subjects=("Acme Corp",)),
                 "Acme Corp completed its acquisition of Widget Inc.")
    assert _mark(_scoop("personnel", subjects=("Acme Corp",)),
                 "Acme Corp's new CEO starts on Monday.")
    assert _mark(_scoop("personnel", subjects=("Acme Corp",)),
                 "Acme Corp named Jane Doe as president.")


def test_a_day_of_the_month_is_not_a_count():
    s = _scoop("layoffs", subjects=("Acme Corp",),
               facts=({"field": "jobs_cut", "value": 25, "unit": "count"},))
    assert not _mark(s, "NEW YORK, October 25 (Reuters) - Acme Corp shares fell.")
    assert not _mark(s, "Acme Corp shares fell on 25 October.")
    assert not _mark(s, "Acme Corp shares fell on Oct. 25, 2026.")
    assert _mark(s, "Acme Corp said 25 staff would go.")


def test_a_company_called_3m_is_not_three_million_dollars():
    s = _scoop("partnership", subjects=("Beta Systems",),
               facts=({"field": "deal_value", "value": 3, "unit": "usd_millions"},))
    assert not _mark(s, "Beta Systems signed a supply deal with 3M on Monday.")
    assert _mark(s, "Beta Systems signed a $3m supply deal on Monday.")
    assert _mark(s, "Beta Systems signed a 3 million dollar supply deal on Monday.")


def test_a_one_word_stock_value_is_not_a_mark():
    s = _scoop("legislation_passed", subjects=("Farm Bill",),
               facts=({"field": "chamber", "value": "House"},))
    assert not _mark(s, "The Farm Bill was discussed at the White House.")
    s = _scoop("election_result", subjects=("Jane Doe",),
               facts=({"field": "office", "value": "president"},))
    assert not _mark(s, "Jane Doe met the president on Monday.")
    s = _scoop("election_result", subjects=("Jane Doe",),
               facts=({"field": "office", "value": "Mayor of London"},))
    assert _mark(s, "Jane Doe was declared Mayor of London on Friday.")


def test_another_name_for_the_same_party_is_not_a_second_party():
    s = _scoop("personnel", subjects=("ICE", "Intercontinental Exchange"))
    assert not _mark(s, "ICE (Intercontinental Exchange) futures were calm on Monday.")
    assert not _mark(s, "Intercontinental Exchange (ICE) futures were calm on Monday.",
                     "Intercontinental Exchange")
    s = _scoop("partnership", subjects=("Alphabet", "Google"))
    assert not _mark(s, "Alphabet (Google) opened an office on Monday.")
    assert _mark(s, "Alphabet and Google staff met on Monday.")


@pytest.mark.parametrize("short,full", [
    ("Meta", "Meta Platforms, Inc."), ("Meta", "Meta Platforms"), ("Amazon", "Amazon.com, Inc."),
    ("JPMorgan", "JPMorgan Chase & Co."), ("Disney", "The Walt Disney Company"),
    ("Disney", "Walt Disney"), ("NorthStrive", "NorthStrive Energy"),
    ("Microsoft", "Microsoft Corporation"), ("Berkshire", "Berkshire Hathaway Inc."),
])
def test_another_name_of_the_subject_served_with_it_is_not_a_second_mark(short, full):
    # The API serves a scoop's subjects with the registry's and the SEC's names of the same
    # entity; a page that only names the subject in full says nothing of the event.
    s = _scoop("layoffs", subjects=(short, full))
    text = f"{full} had a quiet Monday. {short} shares were flat."
    assert not _mark(s, text, short)
    assert not _mark(s, text, full)
    assert held_claim(s, URL, text) is None
    # Its event's words still mark it.
    assert held_claim(s, URL, f"{full} said it would lay off 3,600 staff.") is not None


def test_a_filing_under_the_registrants_full_name_is_not_a_second_mark():
    s = _scoop("layoffs", subjects=("Meta", "Meta Platforms, Inc."))
    text = ("FORM 8-K. Meta Platforms, Inc. (Exact name of registrant as specified in its "
            "charter). Item 5.07 Submission of Matters to a Vote of Security Holders.")
    assert not _mark(s, text)
    assert held_claim(s, "https://www.sec.gov/x", text) is None
    s = _scoop("personnel", subjects=("Amazon", "Amazon.com, Inc."))
    assert not _mark(s, "Amazon.com, Inc. announced the date of its annual shareholder meeting.")


def test_the_subject_as_a_fact_is_not_a_mark_for_another_of_its_names():
    s = _scoop("transfer", subjects=("Cristiano Ronaldo", "Ronaldo"),
               facts=({"field": "player", "value": "Cristiano Ronaldo"},))
    assert not _mark(s, "Ronaldo trained alone. Cristiano Ronaldo smiled.", "Ronaldo")


def test_another_party_still_marks_the_page():
    s = _scoop("merger_acquisition", subjects=("Widget Co", "Acme", "Acme Inc."),
               facts=({"field": "target", "value": "Widget Co"},))
    assert _mark(s, "Acme Inc. and Widget Co staff met on Monday.", "Acme")
    s = _scoop("partnership", subjects=("Bank of America", "Bank of England"))
    assert _mark(s, "Bank of America and Bank of England staff met on Monday.")


def test_a_count_without_a_unit_is_not_read_at_a_thousand_times_its_size():
    s = _scoop("layoffs", subjects=("Acme Corp",),
               facts=({"field": "jobs_cut", "value": 5, "unit": None},))
    assert not _mark(s, "Acme Corp opened 5,000 square feet of offices.")
    assert not _mark(s, "Acme Corp opened a 5 million dollar office.")
    # A money figure without a unit may still be written at its scale.
    s = _scoop("partnership", subjects=("Beta Systems",),
               facts=({"field": "deal_value", "value": 1.2, "unit": None},))
    assert _mark(s, "Beta Systems signed a $1.2 billion supply contract on Monday.")
