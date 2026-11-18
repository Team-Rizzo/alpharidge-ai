# Quality-preview review: what changed and why

Branch `quality-review` exists in both repos and holds the same work:

- **API**: `Team-Rizzo/alpharidge-ai-api`, branched from `quality-preview` at `dca6f64`.
- **Subnet**: `Team-Rizzo/alpharidge-ai`, branched from `quality-preview` at `0218981`.

This file is identical in both. The deploy runbook is `DEPLOY.md` in the API repo; follow it rather than this file when deploying.

## How the work was done

1. **Full review of `quality-preview`.** Both repos, against the deploy handoff of 2026-11-17.
2. **Four rounds of fix, then independent review.**
   - Separate agents fixed the findings.
   - Fresh reviewers then checked each round without seeing the fixer's reasoning.
   - Every high-severity finding was reproduced with a test or a script before it was fixed, and re-checked with the reviewer's own repro afterwards.
3. **Verification**
   - Unit and contract tests in both repos.
   - The API suite against a real Postgres 16.
   - Migration timing on a 1M-claim table.
   - Concurrency stress tests of the ledger.
   - A differential test against the 4.1.2 validator code (`7f732ab`), which the peer validator runs.

## Results

| Suite | Result |
|---|---|
| Subnet | 545 passed |
| API (memory store) | 407 passed, 48 skipped (the skips are the Postgres-only tests) |
| API on Postgres 16 | 452 passed |

## Compatibility with the 4.1.2 peer validator

With quality off (`QUALITY_FROM` is None, as in prod), nothing the 4.1.2 validator checks changes.

- **Leaves, order, root, points and signed fields:** unchanged.
- **Carry:** 70,000 randomized cases match 4.1.2's own carry check.
  - The API now lists the carry window epoch by epoch.
  - The old order (by row id) would have made 4.1.2 fail the carry check in about 0.1% of realistic cases, so this is also a fix.
- **Concurrency:** about 25,000 concurrent ledger writes, with signing failures injected, left every epoch passing 4.1.2's checks.

## The main changes (API)

### 1. Coldkey quality gives the same answer every time (the original blocker)

Quality used to be cut by `judged_at`. A claim judged before the cut but paid after it, or the reverse, could move between epochs, so two reads of the same epoch could disagree.

- Claims now carry `decided_at`: the moment their pass or fail became final.
- Each epoch's quality cut is fixed when the epoch is sealed (see 2).

**Migration:** four small migrations plus a batched backfill script (`scripts/backfill_decided_at.py`).

- **The column:** added with a short lock timeout and retries, so it never queues behind long transactions.
- **The `claim_touch` trigger:** skips `updated_at` when only `decided_at` changes, so the backfill doesn't replay old claims into miners' `GET /claims` change feed.
- **The index:** built `CONCURRENTLY`.
- **Measured on 1M claims:** the longest lock was about 23 ms. The old single migration held an exclusive lock for 36 s.

### 2. Signing no longer races ledger writes

This race was already in the code before this branch: a ledger row could commit into an epoch after that epoch was signed. The fix:

- **Ledger writes and claim decisions** take a shared Postgres advisory lock (key 4545002).
- **Signing first "seals" the epoch:**
  - It takes the lock exclusively, in one short transaction.
  - That transaction writes the carry rows and an `attestation` row with `capacity` NULL, an empty root and an empty signature. The row's `issued_at` is the quality cut.
- **The window is then read, hashed and signed** with no lock or transaction held, and the signature fills that same row.
- **A sealed epoch whose signing failed** is re-signed from the same rows on the next request.

Measured with a 200k-row window and 40 concurrent workers:

| | Before | After |
|---|---|---|
| Failures | 8 of 8 runs had timeouts | 0 |
| Slowest write | 12.6 s | 1.8 s |
| Slowest decision | timed out | 0.7 s |

Related:

- **Timeouts:** pool and transaction limits are explicit. `app/db.py` sets `TX_TIMEOUT_MS` (60 s), `SIGN_TX_TIMEOUT_MS` (180 s) and `LEDGER_LOCK_TIMEOUT_MS` (20 s).
- **The 5 s settle sleep** in `/ledger` is gone.
- **`/ledger/leaves`** pages through the database instead of loading the window.

**Please review:** migration `20261117000300_ledger_carry_replace` changes the `ledger_entry` append-only trigger.

- It allows exactly one kind of delete: carry rows of an epoch that has no `attestation` row yet.
- Only the seal does it, under the exclusive lock, with a transaction-local setting.
- It exists because the old writer wrote carry and signed in two separate steps. A writer stopped between the two (for example by the deploy restart) left unsigned carry rows, and the new writer would have added a second copy: every 4.1.2 validator would fail that epoch, and the debt would count twice.
- Everything else stays append-only.

### 3. Each status move commits with its ledger rows

Before, a status change and its ledger rows were separate writes. A failure between them could:

- pay a claim whose page had failed its check;
- lose a settle or a scoop's debt;
- pay a claim twice.

Now pay, settle, clawback, penalty, and a scoop rejected with its debt each commit in one transaction, with a row lock on the claim (`lock_claim`).

Other fixes in the same area:

- **Failed checks:** a claim whose check fails goes straight from `novelty` to `rejected`, and the resolver rejects any `fetched` claim whose recorded check did not pass.
- **Stuck pays:** `Resolver.repay` pays verified claims whose pay never committed. It is idempotent: a claim that already has its paid row is moved to `provisional` without a second row.
- **Retried jobs:** confirm and re-check jobs are retried (60 s apart, at most 3 runs). A re-check never claws back or penalises twice.
- **Window closing:** `close_windows` pages through claims, and one bad claim no longer stops the pass.

### 4. Quality scoring details

- **Junk count:** only claims a judge actually judged count as junk. Before, novelty rejects (duplicates) counted against a coldkey.
- **Cache:** the quality cache is bounded (32 entries), and concurrent reads of the same epoch share one computation.
- **Duplicate penalties:** `trust.penalize` skips a source event that was already recorded.

### 5. Scoops and source trust

- **Shared categories:** scoop confirmations trust a source in its paired categories (`SHARED_CATEGORY` in `scoops.py`), not only its own.
- **Catalogue:** 19 rows of `app/data/catalogue.csv` whose category didn't match the outlet's beat are corrected.
- **New command, `scripts/trust.py catalogue`:**
  - It adds missing categories to the live registry and never overwrites trust an operator set.
  - It is a dry run by default; pass `--apply` to write.
  - It must run after the deploy (DEPLOY.md step 6). Until then, some outlets lose confirmations they get today.

### 6. Ops scripts added

| Script | What it does |
|---|---|
| `scripts/backfill_decided_at.py` | Fills `decided_at`; safe to stop and re-run |
| `scripts/clear_seals.py` | Rollback helper: removes unsigned seal rows so the old writer can sign those epochs |
| `scripts/preview_budget.sql` + `scripts/preview_budget.py` | Read-only preview of which coldkeys' check-budget multiplier changes on deploy |

**A test flake that predates this branch is fixed:** `test_api_keys_can_be_revoked` failed about 1 time in 64. A generated key that starts with `-` was read by argparse as a flag, so the test now passes `--key=<key>`.

**Caveat for operators:** the same applies when running `scripts/api_keys.py revoke` by hand. Use `--key=<key>`.

## The main changes (subnet)

### Validator

- **Quality-read failures** now raise, so the validator retries the epoch instead of scoring it without quality.
- **Cached quality** is dropped on a `carry_mismatch`.
- **Ledger pull** has a `before_retry` hook.
- **Pay settings mismatch** (new `settings_check.py`):
  - If a validator's pay settings differ from the API's, it posts price re-checks as unreachable (`pay_settings_differ`), instead of filing disagreements against honest claims.
  - The alarm message now says exactly which setting differs.
- **Pay settings:** the defaults are back to today's prod values, `EVENT_PAY_X`=1, `SCOOP_PREMIUM`=5, `SCOOP_MISS_DEBT_X`=2.
  - Invalid values stop the validator at start-up with a clear `PaySettingError`.
  - Option A (0.25/10/1) is left for a later coordinated release.
- **Env files:** `config.py` logs a line when it skips one.

### Scoop scan (`validator/scan.py`)

- **More real confirmations:**
  - more type words;
  - Titlecase forms of short all-caps names (Opec, Nato);
  - a same-party alias rule (a company and its own brand or unit is one party, not a second source).
- **Fewer false confirmations:** several false-positive patterns removed, and scale words ("million", "bn") only expand in money fields.
- **Faster:** alternations are prefiltered, which makes it several times faster on adversarial pages.
- **Known limit:** "0.3% m/m" style figures are fixed only in the scan path; `verify/floor.py` keeps its old behaviour unless `scale_next` is set.

### Determinism

- `mechanism/pools.py` and `mechanism/quality.py` use `math.fsum` in scaled mode, so the result doesn't depend on summation order.
- Default (unscaled) behaviour is unchanged, which keeps weights identical to 4.1.2.

### Tests and docs

- **Hermetic tests:** both repos' `conftest.py` clear `EL_*` env variables so a local `.env` can't change test results.
- **Docs:** `docs/miner.md`, `README.md` and `.env.example` describe current pay, the junk rule and shared categories.

## Deploying

Follow `DEPLOY.md` in the API repo. In short:

1. **Pre-check queries:**
   - size;
   - unsigned carry;
   - half-paid claims;
   - stuck verified claims, which the first repay pass pays all at once;
   - settled claims with no settle row;
   - the check-budget preview.
2. **Update both checkouts** on the API host, then run `prisma generate`.
3. **`prisma migrate deploy`** (four migrations) while the old API is still running.
4. **Backfill**, first pass.
5. **Restart the writer, then each reader one at a time.** Never `pm2 restart all`.
6. **Backfill**, second pass.
7. **`scripts/trust.py catalogue`**, dry run, then `--apply`.
8. **Checks.**

**Rollback has a required order:**

1. Stop the new writer.
2. Run `scripts/clear_seals.py --apply`.
3. Start the old writer.
4. Restart the readers one at a time.

`clear_seals.py` can't tell whether the writer is still running, so make sure it is stopped first.

**Behaviour that changes on deploy:**

- The check budget uses the new coldkey scoring (last 30 days, start 0.50, half-life 20). Prod's current API uses all history, start 0.30, half-life 50. `preview_budget` shows who is affected.
- Scoop confirmations follow the stricter shared-category rule.
- The repay pass pays stuck claims once, in the epoch current at the restart.

## Decisions still open (owner)

1. **Fresh coldkeys start at q 0.50.** A miner with a bad score can reset it by rotating coldkey.
2. **Move the pay values into the signed profile**, so validators can't silently run different pay. This is not done here.
3. **`/ledger/quality` grows without limit** once quality is on. A checkpoint design is written up but not built; it needs a new table and a validator release.
4. **The one-time repay burst** on first start has no off switch. Count it with the pre-check query first.
5. **Claims the old bug left `settled` with no settle row** are listed by a pre-check query but not repaired automatically.
6. **Trust category mismatch:** `first_seen` uses a source's own-category trust, while confirmations use paired categories.
7. **Claim intake** waiting on the receipt lock can hold a pooled connection for up to 60 s.

**Before turning quality on (`QUALITY_FROM`):**

- The backfill must be finished.
- The API host's subnet checkout and every validator must run the same subnet release with the same `QUALITY_FROM`.
- 4.1.2 validators will fail every carry epoch from then on.
- Never backfill or roll back after `QUALITY_FROM` is set.
