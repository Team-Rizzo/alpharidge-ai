# Running a miner

A miner finds new events in finance, technology, politics and sports on the open web and sends each one to the coordination API as a signed claim, with the URL of the page that states it. There is no axon: the miner only makes outbound requests.

The bundled **reference miner** will get you started: it covers finance news from a few sources, and shows the claim format, signing and the pre-send check.

## Upgrading from 3.x

Miners on the 3.x code stop earning at the switch to 4.0. On an existing checkout:

```bash
git stash                          # only if you edited tracked files such as run_miner.sh
git fetch && git checkout v4       # before the switch; after it: git checkout main && git pull
./install.sh miner
cp miner_env.example .miner_env    # 3.x settings are not used: set it up again (section 1)
```

Then register if needed and run the miner as in section 1. No GPU or models are needed any more. The first start after the upgrade replaces an old bittensor dependency once and restarts itself.

## First hour

1. Install and run the reference miner (section 1).
2. Run `scripts/e2e_local.sh` on your machine. It starts fixture sites, a local API, the reference miner and a validator on a mock chain, and walks a claim through every step, so you can try changes without spending anything. It needs a checkout of the coordination API ([alpharidge-ai-api](https://github.com/Team-Rizzo/alpharidge-ai-api)) next to this repo and a local Postgres with pgvector.
3. Read the trusted list: `GET /dashboard/sources` on the coordination API. Only pages from those sources win directly (section 4).
4. Read the page judge's questions in `alpharidge_ai/judge/prompt.py`, with each type's `definition` and `about` in `alpharidge_ai/schema/data/types.json`.
5. Send claims and keep the signed receipts. Follow your results on the dashboard (`GET /dashboard/items?hotkey=<your hotkey>`), or one claim with `GET /claims/{id}`: `reason` and `reason_detail` say why a claim was not paid (section 7). The reference miner does not poll claim status.

## Launch day

On launch day, events from the past 30 days are claimable, for everyone at once. Pay per window is capped by capacity (README, "Where the rest of the emission goes"): once total points reach it, the launch backlog and fresh claims share the same emission, so a large backlog does not add pay, it divides it. Older events also pay less (the age factor, section 3).

## Propose a source

The trusted list covers all four categories and grows as data, without a release. To propose a source, open a GitHub issue on this repo with its domain, its category and links to a few events it reported before other trusted sources did. Proposals are checked (readable to the validators' fetch, original reporting, the right paths) and reviewed weekly.

## 1. Install, configure, register, run

Python 3.10 or newer, CPU only (no GPU, no models).

```bash
git clone https://github.com/Team-Rizzo/alpharidge-ai.git && cd alpharidge-ai
./install.sh miner                 # .venv with the shared [verify] check
cp miner_env.example .miner_env    # set USER_AGENT_CONTACT; MINER_FEEDS for your own sources
btcli subnet register --netuid 45 --wallet.name <coldkey> --wallet.hotkey <hotkey>
.venv/bin/python -m neurons.miner --netuid 45 --subtensor.network finney \
  --wallet.name <coldkey> --wallet.hotkey <hotkey> --logging.info
```

`run_miner.sh` wraps the last command (set `WALLET_NAME` and `WALLET_HOTKEY`). `MINER_FEEDS` points at a JSON file such as `{"sources": [{"url": "https://www.example.gov/rss.xml", "category": "finance"}]}`; RSS and Atom feeds are read directly, and `"kind": "page"` reads links from a listing page. Keep robots.txt and per-domain pacing on.

## 2. Open types

Four categories are open: finance, technology, politics and sports. Only the types below are open; any other type is rejected `type_off` without a check. The category always comes from the type. `personnel` pays only for the top offices: a claim whose `role` is not `ceo`, `cfo`, `chair`, `president` or `coo`, or a new event without one, is rejected `role_off`, also without a check. Only `event` claims are taken; a `stage` or `fact` claim is rejected `kind_off` without a check. `appointment` is a public office; a company officer is `personnel`. Facts use the type's slot names (`alpharidge_ai/schema/data/types.json`); an actor's `role` also fills a slot of that name. An event claim needs at least one actor and its required facts.

**Finance**

| Type | Tier | Official record | Required | Sample claim |
|---|---|---|---|---|
| `monetary_policy` | low | Fed | change_bps | Fed raises rates: `change_bps=25`, `new_rate=4.75`; a hold is `change_bps=0` |
| `economic_data` | low | BLS (BEA release times) | indicator, actual | US CPI: `indicator=CPI`, `actual=0.3`, `measure=mom`, `period=September 2026` |
| `earnings` | low | EDGAR 8-K 2.02 | period; eps or revenue | Acme results: `period=Q3 2026`, `revenue=12.1`, `eps=1.05`, `eps_basis=adjusted` |
| `merger_acquisition` | high | EDGAR 8-K 2.01 | target, acquirer | Acme to buy Widget Co: actor roles `acquirer` and `target`, `stage=announced`, `deal_value=1.2`. A named bidder's public offer or bid (binding or not) and a deal proposal filed with an exchange or regulator are `stage=announced` |
| `bankruptcy` | high | EDGAR 8-K 1.03 | chapter | Widget Co files: `chapter=Chapter 11`, `court=District of Delaware` |
| `personnel` | low | EDGAR 8-K 5.02 (CEO, CFO, chair, president, COO only) | person, role | Acme names a CFO: `person=Jane Doe`, `role=cfo`, `direction=appointed` |
| `layoffs` | high | EDGAR 8-K 2.05 | jobs_cut or pct_workforce | Acme cuts jobs: `jobs_cut=1200`, `pct_workforce=8` |
| `delisting` | low | EDGAR 8-K 3.01 | exchange | Widget Co leaves Nasdaq: `exchange=Nasdaq`, `ticker=WDGT`, `effective_date=2026-11-03`. A delisting the company or exchange announces (a decision to delist, a Form 25 filed or to be filed, a liquidation that ends the listing) counts, dated the day it was announced |
| `ipo_listing` | low | EDGAR S-1 | exchange | Acme prices IPO: `exchange=NYSE`, `ticker=ACME`, `offer_price=18`. Filing for an IPO is `stage=announced` |
| `trading_halt` | low | — | exchange | Halt in WDGT: `exchange=Nasdaq`, `ticker=WDGT`, `reason=news pending` |

**Technology**

| Type | Tier | Official record | Required | Sample claim |
|---|---|---|---|---|
| `product_launch` | low | — | product | Acme launches a chip: actor role `company`, `product=Acme X1`, `price=499`, `release_date=2026-11-20`. A product the company releases, makes available or formally unveils; a leak, a teaser, a rumoured launch, or an existing product brought to new places or users, is not a launch |
| `funding` | high | — | amount | Widget AI raises a round: `amount=40`, `round=Series B`, `lead_investor=Example Ventures`, `valuation=400`. A round the company or its lead investor confirms; talks to raise are not |
| `partnership` | low | — | — | Acme and Widget Co partner: two actors with role `party`, `term=five years`, `deal_value=2` |
| `security_incident` | high | EDGAR 8-K 1.05 | — | Acme breach: actor role `company`, `incident_kind=ransomware`, `records_affected=2000000`. An incident the organisation, its operator or a regulator acknowledges is fact even when the page also quotes the attackers; malware spread through its own marketplace and theft by its own staff are its incidents |
| `breakthrough` | high | — | result | Acme Labs sets a record: actor role `organisation`, `result=a 47.6% efficient solar cell`, `kind=record`, `value=47.6`, `research_field=solar cells`; or a university paper: `kind=published`, `journal=Nature`, `doi=10.1038/…`. The organisation or a peer-reviewed journal reports the result as achieved; a goal, a plan, a grant, an expected result or what it could lead to is not this, and a product going on sale is `product_launch` |
| `outage` | high | — | — | Widget Cloud goes down: actor role `provider`, `service=Widget Storage`, `regions=US East`, `started_at=2026-11-12`. A widespread outage the provider acknowledges, or that the page reports as observed from user reports or monitoring; planned maintenance or one customer's problem is not this |
| `vulnerability` | low | — | cve_id or product | Critical flaw in Acme Server: actor role `vendor`, `cve_id=CVE-2026-12345`, `product=Acme Server`, `cvss=9.8`, `exploited=yes`. Only a vulnerability rated CVSS 9.0 or higher, or reported as exploited in the wild; the vendor is the maker of the flawed product. Two claims with different CVE IDs are two events |
| `tech_regulation` | low | — | — | Regulator fines Acme: actor role `company`, `regulator=European Commission`, `action=fine`, `amount=1.8`, `law=Digital Markets Act`. A fine, a lawsuit filed by a regulator or government, a court ruling or a settlement; an investigation opened, a rival's complaint, a proposed rule, a law or a trade measure is not this |
| `infrastructure_investment` | low | — | — | Acme builds a data centre: actor role `company`, `facility=datacenter`, place `{"iso2": "US"}`, `amount=10`, `capacity_mw=300`. An investment the company announces or commits to, with a site or an amount; a reported plan, a site search or a funding round is not this |

**Politics**

| Type | Tier | Official record | Required | Sample claim |
|---|---|---|---|---|
| `election_result` | low | — | office | Jane Doe wins: actor role `candidate` (or `party` for a party), `office=Governor of Example`, `vote_share=52.1`. A result the election authority declares, the loser concedes or a news organisation calls; an exit poll, a lead in a partial count or a candidate's own claim of victory is not |
| `legislation_passed` | low | — | bill | The Senate passes the Example Act: actor role `legislature`, `bill=Example Act`, `stage=approved`, `chamber=Senate`, `votes_for=60`. Signed, given assent or promulgated is `stage=completed`; voted down, vetoed or withdrawn is `stage=cancelled` |
| `appointment` | low | — | office, appointee | Jane Doe named ambassador: actor role `appointee`, `office=Ambassador to Example`, `stage=announced`, `appointed_by=President`. Confirmed is `approved`, sworn in is `completed`; a resignation or removal is `announced` |
| `sanction_or_tariff` | low | — | targets | Example sanctions a shipper: actor role `imposer`, `targets=Example Shipping Co`, `measure=asset freeze`, `stage=announced`; a tariff gives `rate=25`. Taking effect is `stage=completed`; lifted or suspended is `stage=cancelled` |

**Sports**

| Type | Tier | Official record | Required | Sample claim |
|---|---|---|---|---|
| `transfer` | high | — | player, to_team | Jane Doe joins Example FC: actor role `player`, `to_team=Example FC`, `from_team=Other FC`, `fee=50`, `contract_years=4`. A move the club, the league or the player confirms; agreed terms, talks or a reporter's word that a deal is done are not |
| `injury` | low | — | player | Jane Doe ruled out: actor role `player`, `injury=torn ACL`, `expected_return=next season` |
| `award` | low | — | award, recipient | Jane Doe named MVP: actor role `recipient`, `award=MVP`, `season=2026`. Not a match result or a title won on the field |
| `match_result` | low | — | score | Example FC 2-1 Other FC: two actors with role `team`, `score=2-1`, `competition=Premier League`. The final score of that match, not a series, aggregate, half-time or live score |

An `event` claim without its type's required fields is refused at intake (`400 missing_required`); a required entry with two slots ("eps or revenue") is met by either. A required field can be a fact or an actor role (`target`, `acquirer`). Names must be written as the page writes them: company, agency, indicator, person, team and bill names alike. A company's suffix (`Inc.`, `Corp.`) may be dropped; nothing else is translated, and there is no alias list. `indicator=CPI` fails on a page that only says "Consumer Price Index", and "Labor Department" fails on a page that only says "Bureau of Labor Statistics". A claim with a name that isn't on the page is rejected as `unsupported`.

Some slots take a word, not a figure, and say which figure a value is: `eps_basis` (`adjusted` or `gaap`), `measure` (`mom`, `qoq` or `yoy`), `direction` (`appointed` or `departing`). The page judge checks every field as the claimed quantity: the same basis, measure, period, direction, person and office. Leave a qualifier out and either figure counts. `role` (`personnel`) is `ceo`, `cfo`, `chair` (an executive chair counts), `president` or `coo`; vice and deputy titles and unit heads are not these roles. `change_bps` is signed (a cut is negative); a hold is `change_bps=0`. It is not matched against the page's numbers; the page judge reads it, so a page that gives only the old and new rate ("to 1.25 per cent from 1 per cent") can state it. A figure for a state or region needs `place` with its `admin1` (`{"iso2": "US", "admin1": "South Dakota"}`). A `usd_*` figure next to another currency's mark (C$, A$, €) doesn't match. The sign, the period, the basis, the currency of a plain `$` and which row of a table a figure is on are the judge's: a figure a change starts from ("up from $88.1B") is not the claim's, and a `deal_value` is the deal's value, not an earn-out or other contingent payment alone. An `effective_date` is the day the page gives, at any time of that day (at the open or at the close). An `ipo_listing` with `stage=announced` (plans or a filing to go public) is an event of its own and pays.

Values written another way on the page match: `$94.9B` and `$94,930 million`, `1.2` and `$1.20`, `4.5` and `4-1/2 percent`, `-25` basis points and "a quarter point".

Actors carry a `wikidata`, `cik` or `lei` ID whose registry record has the name the page uses, or `other` with just the name. People (officials, athletes and others), teams, countries and governments, and private companies and startups take their Wikidata ID. An ID that can't be confirmed under that name makes the claim count as known (`unclear_entity`). A name-only actor is allowed and is checked and paid as usual; an event about a person who is not a linked public figure is labelled unverified in the product until trusted, independent sources confirm it.

## 3. Pay

| Tier | Points | Types |
|---|---|---|
| Low | 3 × source trust × age | earnings, economic_data, monetary_policy, trading_halt, delisting, personnel, ipo_listing, product_launch, partnership, vulnerability, tech_regulation, infrastructure_investment, election_result, legislation_passed, appointment, sanction_or_tariff, injury, award, match_result |
| High | 12 × source trust × age | merger_acquisition, layoffs, bankruptcy, security_incident, funding, transfer, breakthrough, outage |

- Only the first verified claim for a new event is paid, once per event, scoops included (section 4). A later stage or extra facts for a known event earn nothing. A type is high tier when it usually breaks as news before any filing or scheduled release (`tier` in `types.json`).
- **Source trust** is the trust of the page's source: 1 for official and company pages, lower for news outlets.
- **A scoop** pays 5 × the tier price × the trust of the trusted page that confirmed it × age at the scoop's receipt when that page was published after the API received the scoop: the premium, because it gives the event earlier (`schema.types.SCOOP_PREMIUM`). Confirmed by a trusted page published at or before the receipt, it pays 1 × the tier price: you were first in, not first out. A contested scoop (section 4) also pays 1 ×.
- **A missed scoop costs** 2 × the tier price × 1 (the highest trust a scoop pays at) × age, as a debt on your coldkey (`schema.types.scoop_debt`, `SCOOP_MISS_DEBT_X`).
- **Age** is how old the event is when the API receives the claim, counted from the end of `event_date` (UTC), or of the page's publish day (UTC) when that is earlier: 1 up to 48 h, then down in a straight line to 0.25 at 30 days (about 0.65 at 15 days). Send fresh events first; older ones still pay (`schema.types.age_factor`).
- **Pay is per event.** Every claim is priced the same way in every category: there is no per-category pool, share or cap. The only limit is the total capacity (README, "Where the rest of the emission goes"), over which all points scale together, so a busy day in one category is never capped by category.
- Nothing else multiplies pay. The scoop premium shifts share toward scoops within capacity; it adds no emission. 40% is credited when the claim is verified and 60% at +72 h, if nothing has proved it wrong.

## 4. Trusted sources

Pages from trusted sources win an event directly: official sites (central banks, statistics agencies, SEC filings; governments, legislatures, election authorities and regulators; leagues, governing bodies and clubs), the company's own site and newsroom, and reviewed news outlets, each trusted for the categories it covers. The live list is `GET /dashboard/sources` on the coordination API: each source with its trust per category, its kind (`official`, `outlet` or `wire`), whom a company's or club's site is official for (`official_for`), and the page rules that apply on any source. It grows (see "Propose a source"); the starting list is `app/data/trust_seed.json` and `app/data/catalogue.csv` in the coordination API repo. A company's newsroom or a club's own site is official only for claims that name that company or club among their parties; for any other claim it counts as a reviewed outlet. A page on an untrusted site can still win as a scoop (below). Contributor pages, self-serve press-release wires, tag and index pages and user-generated pages never win. A press release on the company's own site does, and so does one on PR Newswire, Business Wire or GlobeNewswire's own release pages (a copy republished elsewhere does not). A wire page can win a claim but never settles one: it does not prove another claim or a scoop wrong on its own. An SEC filing counts as the company's own source when the claim's subject (an actor in one of the type's subject roles; for a deal the acquirer or the target) is the filer on the filing's index page: by its CIK (`"id_scheme": "cik"`), or without a CIK by the filer's registered name, its first word, its initials ("UPS" for United Parcel Service), another shorter form of it ("FHLBank Chicago" for Federal Home Loan Bank of Chicago, "SST VI" for Strategic Storage Trust VI, Inc.) or its ticker. A filing that names it only as the subject company (another holder's 13D, for instance) does not.

**Scoops.** A claim that passes the page checks on a page outside the trusted sources is held as a scoop for up to 72 h. For a scoop the page is your citation: pay rests on the API's receipt time and the trusted page that confirms it. A scoop page the API can't read is still accepted, unread: paywalled or login-only, disallowed by robots.txt, refused by the server (any 4xx other than 404 and 410), not answering (a timeout, a connection error, a 5xx), or a body that is not HTML or plain text, is over the size limit or has no text the extractor finds. Then only the claim's `event_date` is checked (within 30 days of receipt, not in the future), there is no page check or page judge, its age is counted from `event_date` alone, and `GET /claims/{id}` marks it `unread`. Still rejected: a page that is gone (404 or 410), a URL outside the page rules, a host that is not public or a URL that is not http(s), and a page the text extractor refuses (`fetch_failed`). A page the API can read still runs the page checks and the page judge, a quick free rejection when something is off. Pay waits for a trusted page exactly as for any scoop, and your claim is judged on that page. An unreadable page on the event route stays `blocked`. It is paid only when a trusted page confirms the same event: 5 × the tier price when that page was published after the API received the scoop, 1 × when it was published at or before (section 3). Listed validators look for that page in the trusted sources' feeds, sitemaps and listing pages and the subject's SEC filings and send it to the API, which runs the same fixed check and page judge on it (the page judge as for any claim, below), with a quote taken from that page. Another miner's claim never confirms a scoop: a trusted report's page, not the report, can confirm it.

- The confirming page must state the subject and the event. It may write a name another way than your claim, as another form of the same entity (the registry's name or alias, the SEC filer's name for a CIK or a ticker, the ticker): the confirmed record then uses the page's form, with your actor's ID and role. Figures it does not state are left out of the confirmed record (they stay unconfirmed and are not served as verified); pay does not depend on how many facts are confirmed. A figure it states differently fails the scoop (`scoop_mismatch`), and so does a trusted page that reports another outcome of the same story, or the same event dated earlier than your `event_date` (the scoop's date was wrong).
- No confirming page within 72 h closes it `scoop_unconfirmed`, except when another claim took the event first with the same outcome: a correct scoop that only lost on timing closes `lost_line`, with no debt. When that claim names another outcome, the scoop closes `scoop_unconfirmed` as usual.
- **Miss debt.** Every scoop that closes `scoop_unconfirmed`, and every `scoop_mismatch`, adds a debt of 2 × its tier price at trust 1 (section 3), read or unread: anyone can put a guess on a page they control, so only a trusted page proves a scoop. The debt is netted across every hotkey on the coldkey like a claw, and carried forward until later pay covers it. Guessing loses money.
- **Contested scoops.** When another scoop of the same type and subject (a hire sent as `personnel` or as `appointment` counts as one type), from any coldkey (yours included), names a conflicting outcome (another counterparty or value where the type decides it) within the confirmed scoop's 72 h, the confirmed one pays 1 ×, not 5 ×, and the others take their miss debt. These cases are logged for the operator, who reviews them for misconduct (section 6).
- Once confirmed, the trusted page and the scoop's signed receipt time are the proof: the scoop's own page is not checked again.
- One payment per event, whatever name the claims give the company: a ticker written as the company (`GBLI`) is the same company as the filer's name for it (Global Indemnity Group). While a scoop is held, a later trusted claim for the same event waits behind it, and its page is also tried as the scoop's confirming page: when the scoop is confirmed the later claim is `known_event`; when the scoop fails, the later claim is checked and paid as usual, with age from its own receipt.
- **A conflicting scoop is held too.** While an event has only held scoops, a new scoop that names a different outcome from every held one (another acquirer, another person for the post, another value where the type decides it) is held beside them, from any coldkey, yours included. The first one confirmed takes the event and pays 1 × (contested); the others close `scoop_unconfirmed` or `scoop_mismatch` and take their miss debt. A later scoop with the same outcome as a held one is `known_event`: the first finder keeps it. Once a scoop is confirmed, any later claim on the event is `known_event`.
- For `sanction_or_tariff` and `legislation_passed`, claims that both give the targets (or the bill) in a firm form (a bill number, a country, a linked entity) are one event only when those match: two different tariffs or two bills are two events. A name alone never splits an event, since one bill or target goes by several.
- `monetary_policy`, `economic_data`, `earnings`, `trading_halt`, `election_result` and `match_result` never take scoops: they come from scheduled releases, official notices and declared results (`pre_release` or `low_trust`). Only a trusted page pays for them.
- **The `confirmed` label.** `GET /events` gives each event a `confidence`: `confirmed` once an official record matches it or at least two different trusted owners have reported it, `reported` before that. A confirmed scoop's confirming page counts as one owner (the scoop's own page never does). A later trusted claim on a taken event is still `known_event` and unpaid, but its page may be fetched and run through the fixed check (no judge); when it passes, its owner counts too, once per owner.
- **Accuracy gate.** A coldkey (all its hotkeys together) with at least 10 closed (judged) scoops in the last 30 days, fewer than 10% of them confirmed, is refused new scoops (`scoop_limit`, before the page judge). Only `scoop_unconfirmed` and `scoop_mismatch` count as misses; `lost_line` does not. Otherwise there is no limit. It lifts on its own as old misses pass 30 days. On top of it each miss carries its debt; there is no strike.

## 5. What gets a claim paid

- **First by receipt.** The API's receipt time decides, not your clock. Send as soon as the page is live. Claims received in the same 250 ms bucket are ordered by a public draw (see the README's "Who is first?").
- **The quote.** Every event claim carries `quote`: the page's own sentence (or two sentences in a row) that states the event, 20 to 500 characters. Take it from `alpharidge_ai.verify.fetch.read_page(...).text`, the visible text the API and validators read; another extractor moves sentence and line boundaries. It must be, after folding case, curly quotes, dashes and odd spaces: (1) word for word on the page, with no `...` cuts or brackets; (2) whole sentences, ending in `.`, `!` or `?` (a closing quote mark or bracket may follow) and starting at a line start, after a sentence end, after a bullet (• ▪ ● and the like), after a spaced dash or after a dateline's dash ("October 29, 2026—Acme Inc. today…", "(BUSINESS WIRE)--Acme…"), not after a colon, semicolon, opening quote mark or hyphen, so a quote keeps the "analysts expect" or "X wrote:" that frames it; a whole bullet item with no full stop also counts, from its bullet to the next bullet or line break; (3) one paragraph: a line break inside it only after a hard-wrapped line of 60 or more characters that does not end in `.!?:;`; (4) no table cells (`|`); (5) naming the subject: an actor, the type's subject (a deal's `target`, a release's `indicator`) or a personnel claim's `person`, as the page writes it, a term the page defines for it right after its name (`Atkore Inc. (NYSE: ATKR) ("Atkore" or the "Company")`, `Monetary Policy Committee (MPC)`), or "the Company", "the Bank", "the Committee" and the like when that word is in its name. A claim without a quote is refused at intake (`400 missing_required`).
- **The page states it.** Two checks run on the page. First a fixed one, on the API's fetch and again on each listed validator's: every actor name (an actor sent without a `name` must show its `id`), the type's subject when it is a fact (a deal's `target`, a release's `indicator`, as the page names it) and a personnel claim's `person` must appear anywhere in the page's visible text, and so must each figure's size, in any standard form (`$94.9B` or `$94,930 million`; a bare figure under a table heading `(in millions)`, `(in thousands)` or `in millions, except per share data` reads at that scale; percent or per cent; basis or percentage points; cents a share; with or without a currency mark; a zero may be stated in words), a `ticker` exactly as written (case and all) as a whole token, and an `exchange` by one of the names it goes by (`alpharidge_ai/verify/data/exchanges.json`: `Nasdaq Capital Market` is found as Nasdaq, `NYSE` or `New York Stock Exchange` as either; an exchange not listed there must be on the page as written); then the quote as above. Hidden text, scripts and comments don't count. Run it before you send (`alpharidge_ai.verify.check.check_claim`), as the reference miner does. Then the page judge, run by the listed validators (each fetches the page itself and makes one model call per claim; the API averages their answers), reads the page with your quote marked (weekday names and words like "yesterday" in it followed by the date they mean from the publish date, `Tuesday [2026-10-20]`) and answers the same questions for every type: does the page, around the marked passage, report this event (the passage may be the report's lede, a summary or a figures line; the type's text is in `types.json`, `definition`); is it stated as fact (not planned, requested or conditional, rumour, another party's unconfirmed claim, an estimate or preview, or denied); did it happen or was it announced on `event_date` (a page may report an older event; the date is still the day it happened or was announced, not a later one, nor another step of the story); is each party the whole company or body in its role; and for each figure, is it the subject's own and the claimed quantity (not another period, basis, measure, currency or value, nor only contingent or offered). Each must be yes. Tickers and exchanges are not asked: the fixed check reads them. The event question counts as yes, whatever the judge answers, when the page is the subject's own SEC 8-K (or its exhibit; the filer check below) whose items, from the filer's public submissions list, include one for the claim's type (`filing_items` in `types.json`: 2.02 earnings, 5.02 personnel, 2.01 a deal completed, 1.03 bankruptcy, 2.05 layoffs, 1.05 security incident; not 1.01, which is any material agreement, nor 3.01, which also carries listing-rule notices), with at most two items besides 7.01, 8.01 and 9.01, and `event_date` is its report date or the day it was accepted. A filing answers only the event question, never status, timing or figures. Its verdict is stored once and never re-asked; a new quote is a new claim.
- **A wrong type or stage is fixed, not failed.** The judge also says which open type (and stage) the page reports around the quote (`about` in `types.json`). When it does not pass a claim but picks another open type or stage with confidence, the claim is judged once more as that type: facts with no slot in it are dropped (and figures, when the pick is a called-off stage), an actor role it lacks becomes its only role (or none), and the stage is the one picked. A claim whose page dates the event earlier than `event_date` is not retyped: a type does not fix a date. If that passes, the claim is checked, deduplicated, priced and paid as that type, and `GET /claims/{id}` shows it. When that type needs a slot the claim lacks, or cannot hold it (a deal's two parties as one company's results), the claim is `judge_unclear`. The signed body stays as sent; validators re-check the claim as the type in its stored judge record.
- **Live and recent.** The page must be readable without a login or paywall (a scoop's page excepted, section 4) within 15 s a request (5 s to connect; the API and validators read robots.txt, the page and feeds this way), in English, and published no more than 30 days before receipt. The publish time is the page's own structured date (JSON-LD `datePublished` or a publish-date meta tag); an SEC filing's is its EDGAR acceptance time; on the official sites in `alpharidge_ai/verify/data/official.json` a release date in the page body also counts (one written out after release wording, "Embargoed until release at 8:30 a.m. EDT, Tuesday, October 6, 2026", is preferred over a numeric one such as 10/6/26); a site listed there with its own date rule is dated only where that rule says, such as a numeric date after "Release Date" read month-first, and a date guessed from elsewhere on its pages does not count). Body dates are read the same on any host: a date up to the latest calendar date anywhere (UTC+14) is not treated as future. A page with no publish time is rejected (`no_publish_date`). `event_date` must be within 30 days of receipt too (the whole event day counts). `event_date` is the day the event happened or was announced, not the day the article was published: a page published on Thursday about results reported "after the close on Wednesday" is dated Wednesday, and the page judge does not pass the publish date for it. A date after the page's publish day in the latest time zone (UTC+14) is rejected (`future_date`).
- **Release rule.** For `monetary_policy`, `economic_data`, `earnings` and `trading_halt`, a claim received before the official release time is not paid (`pre_release`). A page from before the release is a preview or estimate.
- **Required facts filled**, and one event prize per document: a roundup page wins one event.

## 6. Wrong claims, misconduct and check budget

- A paid claim is **wrong** when an official figure contradicts a figure it claims under the same label (for percent figures, the same `measure`) within 72 h, the operator retracts or corrects it, or an official page settles it otherwise (an official page that speaks for the claim: a newsroom or club site only for its own company or club, never a wire page). It loses its pay, whatever the source. On a source below the trusted bands, a page that later changes or goes away can lose its pay (not a confirmed scoop, section 4). A trusted page that later changes is reviewed by the operator. A trusted outlet that disagrees is reviewed by the operator; it does not by itself change your pay.
- **Check budget.** Every claim of an open type uses one check from a rolling 24-hour budget: 200 checks, plus 2 for each of your claims that passed in the last 7 days. A claim rejected `judge_unavailable` does not count. When it is used up, `POST /claims` returns `429 budget_exhausted` with `Retry-After`; nothing is stored.
- **Page-judge allowance.** Each claim sent to the page judge counts to the hotkey's daily allowance: 2,666 claims a day, or 266 for a hotkey with no passing claim in the last 7 days. It resets at 00:00 UTC. Once it is spent, each further claim that needs the judge that day is rejected `judge_allowance`; it is not kept waiting or judged later. Claims wait rather than fail while the validators have not answered; claims of fresher events (the higher age factor) go to the judge first. A claim no listed validator answered within 24 h is rejected `judge_unavailable`: it gives back its check and does not count toward `judge_repeat`.
- **An honest error loses its pay, nothing more.** A claim proved wrong (above) is clawed back: the claw takes back what the claim still adds to the ledger window. Pay it already earned before the proof stays owed: the debt is carried forward and netted against the other hotkeys on your coldkey until later pay covers it. There is no other penalty (a missed scoop carries its debt, section 4), and it does not change your budget or your place in line.
- **Misconduct.** When the operator confirms misconduct (a page altered to fit a claim, a claim made false on purpose), the claim is clawed back and takes a points penalty of 5 times its full value (an unpaid scoop: 5 times its miss debt), counted across every hotkey on the coldkey. Each penalty is logged with its reason. Repeat offenders are blacklisted.
- A page check that fails (names, figures, the quote, dates or the page judge), fetch failures, known events and stale pages are not wrong claims; they use a check and are not paid.
- Splitting the same claims across hotkeys never pays more: pay adds up per event, with no per-hotkey floors or bonuses.

## 7. Outcomes

`GET /claims/{id}` returns the status, reason, route, event id and points. A claim the fixed page check rejected also carries `reason_detail`, in the claim's own terms: `{"missing": ["actor:Bank of America", "fact:eps"]}` lists the claimed names (`actor:`) and fields (`fact:`) not found in the page's visible text, `{"quote": "missing" | "not_on_page" | "no_subject"}` says the quote is absent, is not whole sentences of the visible text, or names no subject, and `{"date": "event_date"}` or `{"date": "published"}` says which date failed a date gate. A claim the page judge did not pass carries `{"judge": [...]}`: which questions failed, in the same terms (`event`, `status`, `timing`, `actor:<name>`, `fact:<field>`, and `type` when it was judged as another type), never scores. Fix that fact (the date, the figure, the party, the quote) and send it as a new claim; a page and type you already failed twice today is turned away (`judge_repeat`). It is null otherwise. It is not on the public dashboard. Reasons and check results are public at once on `/dashboard/items`; only a claim's source domain and trust band stay hidden from others until it settles (72 h); sign the request with your hotkey to see your own. The common reasons:

| Reason | Meaning |
|---|---|
| `known_event` | The event is already on record |
| `lost_line` | Another claim for the same event was received earlier; also a correct scoop another claim beat with the same outcome (no debt) |
| `kind_off` | A `stage` or `fact` claim; only `event` claims are taken. No check |
| `doc_prize_used` | The document has already won its prize, or is the event's winning document |
| `stale_date` | The page or event is older than the window |
| `fetch_failed` / `blocked` | The page couldn't be read (retried), or is paywalled, login-only or disallowed (a scoop is held unread instead, except a page that is gone or unsafe, section 4) |
| `page_type` | The URL is outside the page rules (tag, index, contributor or user pages, a republished release). No check |
| `language` | The page is not in English |
| `no_publish_date` | The page gives no publish time the API reads (section 5) |
| `pre_published` | The page's publish time is after the API received the claim |
| `future_date` | `event_date` is after the page's publish day, or in the future |
| `unsupported` | A name or a figure's size isn't on the page, or the quote fails (section 5). |
| `judge_reject` / `judge_unclear` | The page judge answered no to a question (event, status, timing, a party, a figure), or could not settle one. |
| `judge_repeat` | The coldkey (or hotkey) already had two claims of this type on this page that the judge did not pass today (UTC); it is not asked again. |
| `judge_unavailable` | No listed validator judged the claim within 24 h. It does not use a check or count toward `judge_repeat`; send it again later |
| `judge_allowance` | The hotkey used its daily page-judge allowance (section 6; resets at 00:00 UTC). |
| `fixed_check` | The page passed the fixed check on the API's fetch but failed it on a listed validator's own fetch (it changed, or reads differently from there). |
| `pre_release` | Received before the official release |
| `contrastive_known` | The page names an event already on record (its subject and type): the claim counts as that known event |
| `unclear_entity` | An actor's ID could not be confirmed under the name given (section 2); the claim counts as known |
| `type_off` / `role_off` | A type that is not open, or a `personnel` role outside the top offices (section 2). No check |
| `low_trust` | The source isn't trusted for this route, and the type takes no scoops |
| `scoop_limit` | The coldkey is under the scoop accuracy gate (section 4); no strike |
| `scoop_unconfirmed` | No trusted page confirmed the scoop within 72 h; adds the miss debt (section 4) |
| `scoop_mismatch` | A trusted page states a figure, outcome or the event's date differently; adds the miss debt (section 4) |
| `budget_expired` | The claim waited a full 24 h budget window without being checked |
| `rate_limited` | Too many requests from your address (HTTP 429 with `Retry-After`); nothing is stored |
| `busy` | The API is taking in as many claims as it can (HTTP 503 with `Retry-After`); nothing is stored, send it again |
| `clawed:*` | A paid claim taken back: `clawed:answer_key` (an official figure contradicts it), `clawed:contradicted` (an official page settles it otherwise), `clawed:retraction` (operator correction or retraction), `clawed:reverify` (validators' re-check disagreed), `clawed:changed` (the page changed), `clawed:ban` (the source was banned), `clawed:misconduct` (section 6) |
| `invalid_schema` / `missing_required` | The body doesn't parse or a text value is not plain (see below), or a required field is missing (400 at intake) |

## 8. Receipts

Every stored claim, including one rejected at intake, gets a `receipt` in the `POST /claims` answer: the claim id, your hotkey, the body's SHA-256, `received_ms` (the time that decides who was first), `seq`, `prev`, `hash` and the API's signature over them with the ledger key (`API_ATTESTATION_PUBKEY` in `config.py`). A re-sent body returns its first receipt. `GET /claims/{id}` and `/dashboard/items` show it too. The reference miner checks each one against its own claim (`alpharidge_ai.ledger.receipts`) and keeps it in its state file.

## 9. Request signing

Every call is signed with your hotkey (sr25519). `POST /claims` sends one claim; `GET /claims/{id}`, `GET /events?entity=&type=&since=` and `GET /events/{id}` read results. `alpharidge_ai/ledger/request_auth.py` is the reference.

| Header | Value |
|---|---|
| `X-Auth-SS58Address` | Your hotkey's SS58 address |
| `X-Auth-Timestamp` | Unix time in whole seconds, within 5 minutes of the server |
| `X-Auth-Nonce` | New random lowercase hex per request (32 characters, at most 64); never reused |
| `X-Auth-Body-SHA256` | Lowercase hex SHA-256 of the body bytes (of the empty string for a GET) |
| `X-Auth-Message` | `alpharidge-ai-auth:v2\|<METHOD>\|<path[?query]>\|<body sha256>\|<timestamp>\|<nonce>` |
| `X-Auth-Signature` | Hex signature (no `0x`) of the message's UTF-8 bytes |

The claim body must be canonical JSON, or it is rejected `invalid_schema`: keys sorted, no spaces (`,` and `:`), UTF-8 rather than `\u` escapes, null fields left out, `actors` and `facts` each sorted by their own canonical text, and numbers with a decimal point (`1200.0`). In Python, `alpharidge_ai.schema.canonical_body(claim)` gives the exact bytes. Re-sending the same body returns the same claim.

Text values must be plain, or the claim is rejected `invalid_schema` (`alpharidge_ai.schema.text_problem(claim)` returns why, or None): an actor's `name` and `id`, `place.admin1` and the type's name slots (`subject_slot`, `name_slots` in `types.json`, such as a deal's `target` or a personnel claim's `person`) are names of at most 80 characters, made of letters, digits, spaces and `. , ' ’ " “ ” & ( ) / + – -`; other string facts are at most 120 characters and may also use `: $ % € £ ¥`. An actor's `role` is one of its type's actor roles or slots, a fact's `field` one of its slots, and a `unit` one of `NumericUnit`. **One subject per claim**: a type with `max_subjects` in `types.json` (earnings, monetary_policy, personnel, layoffs, bankruptcy, security_incident, ipo_listing, delisting, trading_halt and the other single-company types) takes at most that many actors in its subject roles or with no role (`text_problem` returns `subjects`). A page with two companies' results is two claims, one per company. A deal keeps its `acquirer` and its `target`: the target is not a subject role.

```json
{"actors":[{"id":"Q1234","id_scheme":"wikidata","name":"Acme","role":"acquirer"},{"id":"Q5678","id_scheme":"wikidata","name":"Widget Co","role":"target"}],"event_date":"2026-10-13","event_type":"merger_acquisition","facts":[{"field":"deal_value","unit":"usd_billions","value":1.2}],"headline":"Acme agrees to buy Widget Co for $1.2 billion","kind":"event","quote":"Acme Inc. agreed on Tuesday to buy Widget Co for $1.2 billion in cash, the companies said.","source_url":"https://www.example.com/news/acme-to-buy-widget","stage":"announced"}
```
