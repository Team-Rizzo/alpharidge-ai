# Running a validator

A validator doesn't search the web (beyond the trusted sources' feeds, step 5) and serves no axon. It pulls the signed ledger, re-checks a keyed sample and sets weights; a listed validator also judges claims' pages (step 6). Each epoch it:

1. pulls the signed points for the last closed epoch from the coordination API's ledger;
2. checks the signature against the pinned ledger key, downloads the ledger rows, rebuilds their Merkle root and recomputes every hotkey's points from them, including the epoch's carry rows (debt that would otherwise leave the window), which it recomputes from the signed window before;
3. nets each coldkey's hotkeys, so a negative balance on one hotkey counts against the others (by the coldkey signed in the ledger at receipt, so a deregistered hotkey's debt still counts), then passes the points to `settle()` with the network's capacity and sets weights, with anything unclaimed on the burn UID;
4. in the background, re-checks that epoch's paid claims on its own (the share set by the signed mechanism profile's `ledger.reverify_rate`, which the API's results cap reads too; without it, `LEDGER_REVERIFY_RATE`, default 1.0): it verifies the miner's signature and that the claim's text is plain as at intake, fetches the source from its own machine (and, for an SEC filing, its index page, for the acceptance time), runs the same deterministic page check the API runs (names, figure sizes, dates and the miner's quote, which must be whole sentences or a whole bullet item of the page it fetched), checks that the API's stored page-judge record matches the hash in the signed ledger and says yes to every check and field, that every validator answer it keeps is signed by a listed validator (the signed profile's `ledger.judges`; without it, `JUDGE_VALIDATORS`) and is about this claim, and that the record's verdict is the one those answers combine to (when the record's event answer came from the page's SEC 8-K items, `event_by: filing`, it reads the filer's public submissions list on data.sec.gov itself and checks the stored filing row and that its items and dates confirm the claim; a claim the judge retyped to another type or stage, `retyped` in the record, is re-built and checked as that type with the shared `schema.judged`), and that the claim's points are its type's price times its trust times the age factor at receipt of its event date or the page's publish date, the earlier. Re-checking asks no model. It reports agreement or disagreement back to the API. A validator not on the API's re-verify list never gets a claim's body or page, since a claim's source is shown only to the listed validators; it checks the miner's signature over the body's hash, counts those claims as `withheld` and lets them go at the 24 h re-check deadline. Epochs it missed (after a restart or a slow re-check) are caught up a few per tick while they are inside the API's 24 h re-check deadline; the epochs already done are kept in the validator's state file. A paid scoop is checked on the trusted page that confirmed it, stored in its judge record, not on its own page: the validator fetches that page, takes the claim the API judged there as the record keeps it (the scoop on that page, with the stored quote), runs the same page check of that claim on its fetch, and checks that the page was published on the side of the scoop's receipt the record says, that the scoop's trust is that page's and that its price is 5 × the tier price only when that page came out after receipt and the record does not mark it `contested` (else 1 ×). A scoop whose page could not be read has no page-judge record of its own (`unread` in its record): only its confirming page is checked, and its age factor is from its event date alone. A debt row without a page-judge record is aged the same way. Each debt row of the epoch (a missed scoop, `debt`) must be the miss debt of a claim the miner signed: the validator checks the miner's signature over the body the API serves with it, the receipt time, the stored page-judge record's hash when there is one, and that the points are `schema.types.scoop_debt` of that claim; otherwise it keeps its last weights (`unbound_debt`). An operator's misconduct penalty on such an unpaid scoop names its debt leaf instead of a paid leaf and must be `MISCONDUCT_PENALTY_X` (5) times that debt; the debt is checked again against the signed claim served with the penalty, as for a debt row (`unbound_settle` otherwise).
5. on the API's re-verify list, while scoops are open (`GET /scoops/open` lists their type, date, subject names and facts, never the miner's page), reads the trusted sources' feeds, news sitemaps and listing pages from that list and the subjects' SEC filings every `SCAN_INTERVAL_S`, runs the shared page check on pages that name a subject, and sends the ones that pass to the API (`POST /scoops/confirm`), which queues them for the listed validators' page judges (step 6). A validator not on the list skips this step.
6. on the API's re-verify list, every `JUDGE_POLL_S`, pulls the claims and found pages waiting for a page verdict (`GET /judge/pending`: each claim's miner-signed body, the page to fetch and how the API built the claim it judges from the body), fetches each page from its own machine, runs the shared page check and then the page judge, and posts the judge's answers exactly as returned, with the page check's result and its page's content hash (`POST /judge/verdicts`, signed with its hotkey). It judges each item once. When every listed validator has posted, or 10 minutes after the item was queued with at least one post, the API takes the mean of the posted probabilities per question and reads it with the usual thresholds; a failed page check on any posting validator's fetch rejects the claim. The API serves the judge's endpoint and model to listed validators; the validator uses its own key (`API_KEY`) and pays for its own calls, and stops pulling at its daily cap (`EL_JUDGE_DAILY_CAP_USD`) until 00:00 UTC. Without a key, or off the list, a validator skips this step.

Every validator computes the same weights from the same signed data, so validators stay in consensus without broadcasting to each other. A disagreement from a listed validator holds that claim's later payment until an operator reviews it. The sample is chosen with a key private to each validator, so no one, including the API, knows in advance which claims a given validator will re-check.

If the ledger can't be pulled or doesn't verify, the validator keeps its last weights and logs an alarm. If that goes on for too long, it moves its weights to the burn UID rather than keep paying on old data.

## Hardware

No GPU. A small VPS is enough: 2 CPU cores, 4 GB RAM and a few GB of disk. The validator makes outbound HTTPS requests to the API, to the source pages it re-checks and, on the list, to the page judge.

## 1. Install

Python 3.10 or newer.

```bash
git clone https://github.com/Team-Rizzo/alpharidge-ai.git
cd alpharidge-ai
./install.sh validator
```

This creates `.venv` and installs `requirements.lock`, which pins every package (the core, the `[verify]` extra and all their dependencies) by version and sha256 hash, then the package itself without dependencies: the shared fetch rules and check, with the same text-extraction libraries the API uses, so validators and the API read pages identically. No torch, no models. The manual equivalent is:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements.lock
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
```

The lock is built with Python 3.10 by pip-tools from `requirements.txt` and `requirements-verify.txt`, with `constraints.txt` holding the versions the release was tested with, and every package carries its hash.

## 2. Configure

```bash
cp vali_env.example .vali_env
```

Set `API_KEY` (your judge key) and leave the rest as it is. The settings marked *served* below come from the API (`/config/subnet`, refreshed hourly) and need no local value; to pin one locally, set `OVERRIDE_<KEY>` (for example `OVERRIDE_REVERIFY_FETCH_TIMEOUT_S=15`).

| Variable | Default | Meaning |
|---|---|---|
| `API_KEY` | none | Listed validators: your own key for the page judge (a secret) |
| `MINER_API_URL` | set in the example | Coordination API |
| `BATCH_HTTP_TIMEOUT` | `30.0` | HTTP timeout for API calls, seconds |
| `VALIDATION_POLL_SECONDS` | `10` | Seconds between loop passes |
| `AUDIT_KEY` | generated | Your private sample key. Leave it unset to have one generated and kept in `alpharidge_ai/.audit_key`. Never share it, and don't reuse one key across validators |
| `REVERIFY_CONCURRENCY` | `4` | Re-check fetches in flight |
| `PARSE_PROCESSES` | `2` | Worker processes that parse fetched pages |
| `REVERIFY_PARSE_TIMEOUT_S` | `60` | Longest one page may take to parse; past it the page counts as unread and the parse workers restart |
| `REVERIFY_FETCH_TIMEOUT_S` | `15`, served | Seconds per request (robots.txt, page, feed), with a 5 s connect limit; the API reads with the same limit |
| `REVERIFY_FETCH_RETRY` | `1` | Retries of a failed re-check fetch |
| `REVERIFY_DOMAIN_RPS` | `1`, served | Requests per second to any one domain |
| `REVERIFY_FETCH_PUBLIC_ONLY` | `true` | Connect to public addresses only; `false` is refused outside `--mock` and local chains |
| `USER_AGENT_CONTACT` | served | Contact URL or email added to the default `AlphaRidgeBot` user agent |
| `REVERIFY_USER_AGENT` | `AlphaRidgeBot/<version> (+<contact>)` | User agent for re-check fetches |
| `SCAN_INTERVAL_S` | `600`, served | Seconds between scans while scoops are open (listed validators) |
| `SCAN_CONCURRENCY` | `4` | Scan fetches in flight; the scan uses the re-check fetch settings |
| `EL_JUDGE_URL`, `EL_JUDGE_MODEL` | served | Listed validators: the page judge's endpoint and model. Page fetches use the re-check settings |
| `EL_JUDGE_DAILY_CAP_USD` | `10` | Listed validators: page-judge spend per UTC day; pulling pauses once it is reached |
| `JUDGE_POLL_S` | `10`, served | Listed validators: seconds between pulls of the judge queue |
| `LEDGER_REVERIFY_RATE` | `1.0` | Share of paid claims to re-check, used only when the signed profile gives none (`ledger.reverify_rate`). Leave it unset unless the operator asks; another value logs a `[CONFIG] WARNING` at start |
| `JUDGE_VALIDATORS` | empty | Hotkeys whose posted page verdicts re-verify accepts, used only when the signed profile names none (`ledger.judges`); with neither, those claims re-verify as `unreachable` |

Values every validator must share with the API are constants in the shared package (`alpharidge_ai.mechanism.pools`, `alpharidge_ai.ledger.attestation`, `alpharidge_ai.verify.check`), and the burn UID and the ledger's public key are constants in `config.py`.

The open categories (finance, technology, politics, sports) need no setting: the types come from the shared package and the scan feed list from the API. Capacity and the ledger window come from the signed mechanism profile the API serves, which the validator verifies with the same pinned key. See [`profiles/README.md`](../profiles/README.md).

## 3. Run

```bash
.venv/bin/python -m neurons.validator \
  --netuid 45 \
  --subtensor.network finney \
  --wallet.name <coldkey> \
  --wallet.hotkey <hotkey> \
  --logging.info
```

It needs no open port: it only makes outbound requests.

To run under PM2 with the auto-updater:

```bash
.venv/bin/python scripts/start_validator.py --pm2_name sn45vali \
  --netuid 45 --wallet.name <coldkey> --wallet.hotkey <hotkey> --logging.info
```

The auto-updater checks the repository every minute; when `main` has moved, it pulls (`git pull --rebase --autostash`, keeping any local changes), reinstalls `requirements.txt` and the package, and restarts the validator under the same PM2 name.

## 4. Confirm it is working

Look for these log lines once per epoch:

| Log | Meaning |
|---|---|
| `[LEDGER] epoch=… leaves=… W=… C=… burn=… paid=…` | The signed points verified, and weights were computed: total work, capacity, and the split between miners and burn |
| `[REVERIFY] epoch=… candidates=… sampled=… agree=… disagree=… unreachable=… withheld=…` | The re-check pass for that epoch: the signed judge record, the points, and the fixed page check (names, figure sizes, dates, the miner's quote) on your own fetch; the page judge is not asked again. `withheld` counts claims whose body and page go only to the listed validators |
| `[LEDGER] epoch=… no accepted points for … epochs` | A warning that the ledger has not verified for about an hour; weights are left unchanged |
| `[LEDGER] ALARM …` | The ledger didn't verify, no points have been accepted for a while, or the burn UID is not held by the subnet owner's hotkey (it then sets no weights at all until the check passes) |
| `[REVERIFY] ALARM …` | Re-checks are disagreeing with the ledger more often than expected |
| `[SCAN] open=… items=… checked=… posted=…` | Listed validators, while scoops are open: feed items read, pages checked and pages sent to the API |
| `[JUDGE] pending=… {…} spent=$…` | Listed validators: items judged this pull by outcome (`accepted`, `unreachable`, `judge_failed`, …) and today's judge spend |
| `[JUDGE] item … timed out` / `[JUDGE] pull ran over …` | An item took over 3 minutes, or a pull over 15; the rest is served again on the next pull |
| `[JUDGE] daily cap … reached` | Your judge spend hit `EL_JUDGE_DAILY_CAP_USD`; pulling resumes at 00:00 UTC |

A re-check that can't reach the source counts as `unreachable` and has no effect. Disagreements are reported to the API and alarm locally when they become frequent.

## Local test run

`scripts/e2e_local.sh` runs the whole flow on one machine: fixture sites, a local API, the reference miner and a validator on a mock chain. It needs a checkout of the coordination API next to this repo and a local Postgres with pgvector, so it is mainly for development.
