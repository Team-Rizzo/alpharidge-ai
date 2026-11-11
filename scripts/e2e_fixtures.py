"""Fixture pages and hand-built claims for the local end-to-end run (e2e_local.py)."""
from __future__ import annotations

import os
from dataclasses import dataclass

SITE_PORT = int(os.getenv("E2E_SITE_PORT", "8899"))


def site(host: str) -> str:
    return f"http://{host}.localhost:{SITE_PORT}"


# ---- fixture pages ---------------------------------------------------------------------

@dataclass(frozen=True)
class Page:
    key: str
    host: str
    path: str
    category: str
    title: str
    paragraphs: tuple
    age_h: float = 1.0
    phase: int = 1
    # Unlisted pages are left out of the feeds; only the hand-built claims point at them.
    listed: bool = True
    # The scripted page judge's answers for claims on this page, where a model would say no.
    judge: tuple = ()

    @property
    def url(self) -> str:
        return site(self.host) + self.path


PAGES = (
    # An official-style filing page: the first trusted report of a deal.
    Page("deal", "filings", "/news/globex-initech", "finance",
         "Globex to acquire Initech",
         ("Globex Inc. agreed to acquire Initech Inc. in an all-stock transaction, "
          "the companies said in a joint statement.",
          "The boards of both companies approved the merger agreement on the same day.",
          "The transaction is expected to close in the first half of next year."),
         age_h=1.0),
    # The same deal on a wire site, read later by the miner, with the deal value.
    Page("duplicate", "wire", "/markets/globex-initech-deal", "finance",
         "Globex agrees to buy Initech for $12 billion",
         ("Globex Inc. agreed to buy Initech Inc. in a deal valued at $12 billion, "
          "according to a statement from the companies.",
          "The boards of both companies approved the terms of the transaction.",
          "Globex shareholders will own most of the combined company."),
         age_h=3.0),
    Page("rate", "centralbank", "/press/monetary-policy", "finance",
         "Federal Reserve raises interest rates to 4.75%",
         ("The Federal Reserve raised its benchmark interest rate by 25 basis points to 4.75% "
          "after its policy meeting, the central bank said in a statement.",
          "The Federal Open Market Committee said inflation had stayed above its target over "
          "the summer.",
          "The decision was approved by a majority of the committee."),
         age_h=2.0),
    # An index change on a wire site: a type that is off.
    Page("off", "wire", "/markets/acme-joins-index", "finance",
         "Acme Corp. to join the S&P 500",
         ("Acme Corp. will join the S&P 500 index before the open next Monday, S&P Dow Jones "
          "Indices said in a statement.",
          "Acme replaces a company that is being acquired.",
          "Shares of companies added to the index often see heavy trading on the day."),
         age_h=2.0, listed=False),
    # A page that reports a plan, not an event.
    Page("hedged", "wire", "/companies/umbrella-wonka-talks", "finance",
         "Umbrella plans to acquire Wonka",
         ("Umbrella Corp. plans to acquire Wonka Industries in an all-stock deal, people "
          "familiar with the matter said.",
          "The companies are expected to announce terms next month.",
          "Neither company commented on the talks."),
         age_h=1.0, listed=False, judge=(("status", "no"), ("not_news_reason", "preview"))),
    # A page that denies the event; the reference miner leaves it alone.
    Page("negated", "wire", "/companies/northwind-denies-layoffs", "finance",
         "Northwind Traders denies layoff report",
         ("Northwind Traders denied a report that it would announce layoffs of 10,000 jobs, "
          "saying no job cuts were planned.",
          "A spokesperson said the report was false.",
          "The company said its hiring plans were unchanged."),
         age_h=2.5, judge=(("event", "no"), ("not_news_reason", "denied"))),
    # A first report on a site with no track record: held as a scoop until a trusted page
    # published later is found.
    Page("scoop", "blog", "/posts/hooli-layoffs", "finance",
         "Hooli announces layoffs",
         ("Hooli Inc. announced layoffs that eliminate 4,000 jobs across its chip "
          "plants, the company said in a memo to staff.",
          "The cuts affect manufacturing sites in several states.",
          "The company said the changes take effect next month."),
         age_h=1.5),
    # Another low-trust first report, of an event a trusted page had already reported: first
    # in, paid the tier price once that page is found.
    Page("scoop2", "blog", "/posts/stark-motors-layoffs", "finance",
         "Stark Motors announces layoffs",
         ("Stark Motors announced layoffs that eliminate 1,500 jobs at its assembly plants, the "
          "company said in a memo to staff.",
          "The cuts affect two plants in the Midwest.",
          "The company said the changes take effect next quarter."),
         age_h=0.5),
    # A scheduled data release reported on a low-trust site before its official release time.
    Page("prerelease", "blog", "/posts/consumer-prices", "finance",
         "Consumer prices rise 0.3%",
         ("The consumer price index rose 0.3% last month, the Bureau of Labor Statistics said "
          "in its monthly report.",
          "Inflation figures showed higher prices for food and shelter.",
          "The next report is due in four weeks."),
         age_h=0.5, listed=False),
    # Phase 2: the rate decision again, on another site, once the API knows the event.
    Page("known", "wire", "/markets/fed-raises-rates", "finance",
         "Fed lifts benchmark rate to 4.75%",
         ("The Federal Reserve raised its benchmark interest rate by 25 basis points to 4.75%, "
          "the central bank said in a statement.",
          "Policymakers pointed to inflation that has stayed above target.",
          "The decision was approved by a majority of the committee."),
         age_h=0.0, phase=2),
    # Phase 2: the same event on a trusted site, published after the low-trust report.
    Page("confirm", "wire", "/companies/hooli-layoffs", "finance",
         "Hooli to cut 4,000 jobs in layoffs",
         ("Hooli Inc. announced layoffs that eliminate 4,000 jobs across its chip "
          "plants, the company said in a statement.",
          "The cuts affect manufacturing sites in several states.",
          "Hooli employs more than 100,000 people worldwide."),
         age_h=0.0, phase=2),
    # Phase 2: a trusted page on the second low-trust report, published before it.
    Page("early", "wire", "/companies/stark-motors-layoffs", "finance",
         "Stark Motors to cut 1,500 jobs",
         ("Stark Motors announced layoffs that eliminate 1,500 jobs at its assembly plants, the "
          "company said in a statement.",
          "The cuts affect two plants in the Midwest.",
          "Stark Motors employs about 40,000 people."),
         age_h=1.5, phase=2),
    # Phase 2: the deal moves to its next stage on a trusted site.
    Page("stage", "filings", "/news/initech-shareholders-vote", "finance",
         "Globex acquisition of Initech approved by shareholders",
         ("Initech Inc. shareholders approved the acquisition by Globex Inc. at a special "
          "meeting, the companies said in a statement.",
          "Regulators cleared the transaction last month.",
          "Initech will become a unit of Globex."),
         age_h=0.0, phase=2),
    # Phase 2: a trusted page gives a different deal value; the fact claim on it is not paid.
    Page("dispute", "filings", "/news/globex-initech-terms", "finance",
         "Globex to pay $14 billion for Initech",
         ("Globex Inc. agreed to acquire Initech Inc. for $14 billion under the merger "
          "agreement, according to a filing by the companies.",
          "The boards of both companies approved the terms of the transaction.",
          "Globex shareholders will own most of the combined company."),
         age_h=0.0, phase=2, listed=False),
    # Phase 2: a page type the API does not accept as a source.
    Page("excluded", "wire", "/sponsored/soylent-layoffs", "finance",
         "Soylent announces layoffs",
         ("Soylent Corp. announced layoffs that eliminate 2,000 jobs at its headquarters, the "
          "company said in a statement.",
          "The cuts affect several corporate teams.",
          "The company said the changes take effect next month."),
         age_h=0.0, phase=2, listed=False),
)
PAGE = {p.key: p for p in PAGES}


# Hand-built claims a second hotkey sends: one the reference miner declined (the page denies
# it), one from a page that only reports a plan, a re-report of an event the API already
# knows, a fact claim with a different deal value, a claim on a page type that is
# not accepted, a scheduled release sent before its official release time, and a type that
# is off.
RAW = {"negated": ("event", "layoffs", "finance", "Q99999005", "Northwind Traders",
                   (("jobs_cut", 10000.0, "count"),)),
       "hedged": ("event", "merger_acquisition", "finance", "Q99999004", "Umbrella Corp.",
                  (("target", "Wonka Industries", None),)),
       "known": ("event", "monetary_policy", "finance", "Q53536", "Federal Reserve",
                 (("change_bps", 25.0, "basis_points"),)),
       "dispute": ("fact", "merger_acquisition", "finance", "Q99999002", "Globex",
                   (("deal_value", 14.0, "usd_billions"),)),
       "excluded": ("event", "layoffs", "finance", "Q99999007", "Soylent",
                    (("jobs_cut", 2000.0, "count"),)),
       "prerelease": ("event", "economic_data", "finance", "Q2928428",
                      "Bureau of Labor Statistics",
                      (("indicator", "Consumer Price Index", None), ("actual", 0.3, "percent"))),
       "off": ("event", "index_change", "finance", "Q99999009", "Acme Corp.", ())}
# The page whose event a fact claim attaches to.
ATTACH = {"dispute": "deal"}
