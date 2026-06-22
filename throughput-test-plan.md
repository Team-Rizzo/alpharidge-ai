# Throughput Stress Test — Implementation Plan

## Objective

Push our validator + miner pipeline to maximum throughput using recycled tweets to determine:

1. **Max achievable points per epoch** with our current 5 miners
2. **Where the throughput ceiling is** (tweet supply, LLM grading, dispatch, etc.)
3. **How point volume translates to on-chain miner incentive** via burn.py

This test is **local only** — no code changes pushed subnet-wide. Only our validator settings and our API database are affected.

---

## Background: Why This Matters

Currently our miners accumulate ~24 points/epoch across 5 miners. The burn.py weight calculation gives miners 0.072% and burn gets 99.928%. We need to know whether throughput alone can fix this, or if `USD_PRICE_PER_POINT` also needs adjustment.

| Total Points/Epoch | Miner Weight % | Burn Weight % |
|---------------------|----------------|---------------|
| 24 (current)        | 0.07%          | 99.93%        |
| 500                 | 1.5%           | 98.5%         |
| 1,000               | 3.0%           | 97.0%         |
| 3,333               | 10.0%          | 90.0%         |

---

## Infrastructure

| Component       | Location                   | Details                        |
|-----------------|----------------------------|--------------------------------|
| Validator       | rizzo_2297 (this box)      | PM2: `sn45_vali`               |
| API             | rizzo_2297 (this box)      | PM2: `sn45.api`, port 8000     |
| Database        | rizzo_2297 (this box)      | PostgreSQL, `miner_tweets` db  |
| Miners (5)      | Hetzner VM (188.245.224.138) | UIDs 2, 7, 13, 44, 57       |
| Tweet recycler  | rizzo_2297 (new script)    | PM2: `tweet-recycler`          |

---

## Implementation

### Step 1: Tweet Recycler Script

**File:** `/home/rizzo/sn45-api/tweet_recycler.py`

**What it does:**
- Connects to the `miner_tweets` PostgreSQL database
- Every N seconds, resets completed scoring records back to `pending`
- This creates an infinite supply of tweets from the existing pool without ingesting new ones

**SQL logic:**
```sql
-- Reset completed scorings so tweets become available again
UPDATE scoring
SET status = 'pending',
    start_time = NULL,
    validator_hotkey = NULL
WHERE status = 'completed';

-- Also delete analysis records so validators must re-grade
-- (otherwise the API might skip tweets that already have analysis)
DELETE FROM tweet_analysis
WHERE tweet_id IN (
    SELECT tweet_id FROM scoring WHERE status = 'pending'
);
```

**Configuration (env vars):**
- `RECYCLE_INTERVAL_SECONDS` — how often to recycle (default: 30)
- `DATABASE_URL` — postgres connection string (reuse from API .env)

**Safety:**
- Only recycles tweet scorings, not telegram
- Does NOT modify the tweets table itself (tweet text/metadata stays intact)
- Can be stopped instantly via PM2 with no side effects
- No changes to the API codebase

### Step 2: Validator Throughput Settings

**File:** `/home/rizzo/alpharidge-ai/.vali_env` (local only, not committed)

Changes to apply before starting the test:

| Setting                    | Current | Test Value | Rationale                                    |
|----------------------------|---------|------------|----------------------------------------------|
| `VALIDATION_FETCH_LIMIT`   | 24      | 100        | Fetch enough tweets to feed all 5 miners/cycle |
| `VALIDATION_POLL_SECONDS`  | 10      | 3          | Poll 3x faster for tighter dispatch loop      |
| `VALIDATION_MAX_WORKERS`   | 8       | 16         | Double LLM validation concurrency             |

These are env-var overrides — no code changes needed. Revert by restoring original `.vali_env` values.

### Step 3: Monitoring Script

**File:** `/home/rizzo/sn45-api/throughput_monitor.py`

A lightweight script that runs during the test and prints periodic stats:

**Metrics collected (every 60 seconds):**
- Current reward points per miner (from validator logs or reward store)
- Total points this epoch
- burn.py `total_percent_needed` (from validator logs)
- Validator pending task count
- API response times (is it keeping up?)
- Miner batch pass/fail ratio

**Also collects per-epoch summaries:**
- Points accumulated this epoch vs. previous
- On-chain incentive for our UIDs (from metagraph)
- Burn UID weight vs. miner weight

Output: prints to stdout (PM2 logs) and appends to a CSV for post-analysis.

---

## Test Procedure

### Pre-Test (save baselines)

1. Record current on-chain incentive for UIDs 2, 7, 13, 44, 57
2. Record current burn.py weight split from validator logs
3. Record current epoch number
4. Snapshot current `.vali_env` for rollback

### Start Test

```bash
# 1. Start the tweet recycler
pm2 start /home/rizzo/miniconda3/envs/vllm311/bin/python \
    --name tweet-recycler -- /home/rizzo/sn45-api/tweet_recycler.py

# 2. Update validator throughput settings
#    (edit .vali_env with test values, then restart)
pm2 restart sn45_vali

# 3. Start the monitor
pm2 start /home/rizzo/miniconda3/envs/alpharidge_ai/bin/python3 \
    --name throughput-monitor -- /home/rizzo/sn45-api/throughput_monitor.py
```

### During Test (~1-2 hours, 3-6 epochs)

- Watch `pm2 logs sn45_vali` for:
  - `total_percent_needed` increasing
  - Reward application logs showing higher point counts
  - No error spikes or resource issues
- Watch `pm2 logs throughput-monitor` for periodic summaries
- Check system resources: `htop`, memory, CPU
- Watch for LLM API rate limit errors in validator logs

### End Test

```bash
# 1. Stop the recycler
pm2 stop tweet-recycler

# 2. Restore original validator settings
#    (revert .vali_env, then restart)
pm2 restart sn45_vali

# 3. Stop the monitor
pm2 stop throughput-monitor
```

---

## What to Look For

### Success Criteria
- [ ] Points per epoch significantly higher than 24 (target: 500+)
- [ ] Miner weight % in burn.py visibly increases
- [ ] On-chain incentive for our UIDs becomes non-trivial
- [ ] No crashes, OOMs, or cascading failures

### Failure Modes to Watch
- **LLM rate limits** — validator logs show 429 errors or timeouts during re-validation
- **API bottleneck** — API response times spike, scoring query gets slow with recycled volume
- **Memory growth** — validator or API memory climbing (especially with higher worker count)
- **Miner saturation** — miners can't keep up with dispatch rate (batches timing out)

### Key Questions Answered

1. **What's the max throughput?** — How many points/epoch can we realistically achieve?
2. **Where's the ceiling?** — Is it LLM grading, tweet supply, miner speed, or dispatch rate?
3. **Is throughput alone enough?** — At max throughput, does burn.py give miners meaningful weight, or do we still need to adjust `USD_PRICE_PER_POINT`?

---

## Rollback

Everything is reversible:

| Component | Rollback |
|-----------|----------|
| Tweet recycler | `pm2 stop tweet-recycler && pm2 delete tweet-recycler` |
| Validator settings | Restore original `.vali_env`, `pm2 restart sn45_vali` |
| Database | Recycled tweets naturally settle — completed scorings stay completed once recycler stops. No permanent damage to data. |
| Monitor | `pm2 stop throughput-monitor && pm2 delete throughput-monitor` |

No subnet-wide code was changed. No other validators are affected.
