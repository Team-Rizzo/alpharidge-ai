#!/usr/bin/env bash
# Local end-to-end run: fixture sites, API and worker, reference miner,
# validator on a mock chain. Safe to re-run. A local .e2e_env (not committed) may set API_DIR
# and E2E_DB.
set -euo pipefail

SUBNET_DIR=$(cd "$(dirname "$0")/.." && pwd)
[[ -f $SUBNET_DIR/.e2e_env ]] && source "$SUBNET_DIR/.e2e_env"
API_DIR=$(cd "${API_DIR:-$SUBNET_DIR/../alpharidge-ai-api}" && pwd)
export E2E_DIR=${E2E_DIR:-$SUBNET_DIR/.e2e}
E2E_DB=${E2E_DB:-alpharidge_e2e}
PY=$SUBNET_DIR/.venv/bin/python
API_PY=$API_DIR/.venv/bin/python
DRIVER="$PY $SUBNET_DIR/scripts/e2e_local.py"

step() { printf '\n==> %s\n' "$*"; }

stop_old() {
  if [[ -f $E2E_DIR/pids ]]; then
    while read -r pid; do kill "$pid" 2>/dev/null || true; done < "$E2E_DIR/pids"
    # Wait for them to exit, so a run straight after finds the ports free.
    for _ in $(seq 50); do
      alive=0
      while read -r pid; do kill -0 "$pid" 2>/dev/null && alive=1; done < "$E2E_DIR/pids"
      (( alive )) || break
      sleep 0.2
    done
    rm -f "$E2E_DIR/pids"
  fi
}
trap stop_old EXIT
stop_old
rm -rf "$E2E_DIR"
mkdir -p "$E2E_DIR"

# The database comes from the API .env; only the local scratch database may be reset.
DATABASE_URL=${DATABASE_URL:-$(grep -E '^DATABASE_URL=' "$API_DIR/.env" | tail -1 | cut -d= -f2-)}
case "$DATABASE_URL" in
  *@localhost:*/"$E2E_DB"|*@127.0.0.1:*/"$E2E_DB") ;;
  *) echo "refusing to reset $DATABASE_URL (expected a local $E2E_DB)" >&2; exit 1 ;;
esac
export DATABASE_URL

# Shared clock: 0.1 s blocks, so an epoch of 100 blocks lasts 10 s.
export MOCK_GENESIS_TS=$(date +%s) MOCK_BLOCK_SECONDS=0.1
export E2E_SITE_PORT=${E2E_SITE_PORT:-8899} API_PORT=${API_PORT:-8890} API_HOST=127.0.0.1
export MINER_API_URL=http://127.0.0.1:$API_PORT
# A small check budget, so the over-budget step needs only a few claims.
export EL_CHECK_BUDGET_BASE=${EL_CHECK_BUDGET_BASE:-12}

for port in "$API_PORT" "$E2E_SITE_PORT"; do
  if curl -s -o /dev/null "http://127.0.0.1:$port/" || curl -s -o /dev/null "http://[::1]:$port/"; then
    echo "port $port is already in use; stop whatever is listening there first" >&2; exit 1
  fi
done

step "1. Reset and migrate $E2E_DB"
(cd "$API_DIR" && .venv/bin/prisma migrate reset --force --skip-seed --skip-generate) \
  > "$E2E_DIR/migrate.log" 2>&1 || { cat "$E2E_DIR/migrate.log"; exit 1; }
grep -E "applied|migrations" "$E2E_DIR/migrate.log" | tail -3

step "2. Seed entities and trust"
eval "$($DRIVER keys)"
export EL_ENTITY_OVERRIDES=$SUBNET_DIR/tests/fixtures/entities.json
(cd "$API_DIR" && $API_PY scripts/seed_entities.py load | tail -1 \
  && $API_PY scripts/seed_entities.py load --seed "$EL_ENTITY_OVERRIDES" | tail -1 \
  && $API_PY scripts/trust.py seed \
  && for spec in filings:1.0 centralbank:1.0 parliament:1.0 wire:0.7; do
       $API_PY scripts/trust.py add "${spec%%:*}.localhost" --trust "${spec##*:}" \
         --category finance --category technology --category politics --category sports \
         --reason "local fixture" --by e2e
     done)

step "3. Fixture sites on *.localhost:$E2E_SITE_PORT"
$DRIVER sites > "$E2E_DIR/sites.log" 2>&1 &
echo $! >> "$E2E_DIR/pids"

step "4. API and worker on $MINER_API_URL"
$DRIVER judge-script > "$E2E_DIR/judge_script.json"
$DRIVER scan-feeds > "$E2E_DIR/scan_feeds.json"
(cd "$API_DIR" && exec env SUBTENSOR_NETWORK=mock AUTH_ENABLED=true ALLOW_MANUAL_HOTKEYS=true \
   MANUAL_MINER_HOTKEYS="$E2E_MINER_HOTKEYS" MANUAL_VALIDATOR_HOTKEYS="$E2E_VALIDATOR_HOTKEYS" \
   OPERATOR_HOTKEYS="$E2E_VALIDATOR_HOTKEY" REVERIFY_VALIDATORS="$E2E_VALIDATOR_HOTKEYS" \
   API_ATTESTATION_PRIVKEY="$API_ATTESTATION_PRIVKEY" BLOCKED_HOTKEYS= \
   EL_FETCH_PUBLIC_ONLY=false EL_FETCH_RETRY=0 EL_DOMAIN_RPS=50 EL_WORKER_INTERVAL_S=0.5 \
   EL_SETTLE_DELAY_H=0.0125 EL_SETTLE_EVERY_S=1 EL_PUBLISH_HOLD_S=0 EL_AK_ADAPTERS=bls \
   EL_AK_BLS_URL="http://stats.localhost:$E2E_SITE_PORT/latest.rss" \
   EL_AK_BLS_CALENDAR_URL="http://stats.localhost:$E2E_SITE_PORT/calendar.ics" \
   EL_RATE_LIMIT_ENABLED=false EL_SCOOPS=true EL_SCAN_FEEDS="$E2E_DIR/scan_feeds.json" \
   $API_PY -m app > "$E2E_DIR/api.log" 2>&1) &
echo $! >> "$E2E_DIR/pids"
for _ in $(seq 60); do
  curl -sf "$MINER_API_URL/health" > /dev/null && break
  sleep 0.5
done
curl -sf "$MINER_API_URL/health" > /dev/null || { tail -30 "$E2E_DIR/api.log"; exit 1; }
echo "API is up (settlement delay shortened to 45 s for this run)"

# Miner and validator settings for the local run.
export MINER_DOMAIN_RPS=50 MINER_FETCH_PUBLIC_ONLY=false
export LEDGER_REVERIFY_RATE=0.5 REVERIFY_FETCH_PUBLIC_ONLY=false REVERIFY_FETCH_RETRY=0 \
  REVERIFY_DOMAIN_RPS=50 AUDIT_KEY=$(printf e2e-audit | sha256sum | cut -c1-64) \
  JUDGE_VALIDATORS=$E2E_VALIDATOR_HOTKEYS

step "4b. Two listed validators judge queued pages (scripted judge, different confidence)"
for spec in validator:0.9 validator2:0.8; do
  EL_JUDGE=scripted EL_ALLOW_SCRIPTED_JUDGE=true EL_JUDGE_SCRIPT="$E2E_DIR/judge_script.json" \
    EL_JUDGE_SCRIPTED_YES=${spec##*:} JUDGE_POLL_S=0.5 \
    $DRIVER judge "${spec%%:*}" > "$E2E_DIR/judge_${spec%%:*}.log" 2>&1 &
  echo $! >> "$E2E_DIR/pids"
done

step "5. Reference miner, one cycle; a second hotkey sends claims the miner declined"
$DRIVER miner
$DRIVER submit negated
$DRIVER submit hedged
$DRIVER submit prerelease
$DRIVER submit off

step "5b. A third hotkey sends claims past its check budget"
$DRIVER flood

step "6. Claims through novelty -> fetch -> verify -> ledger"
$DRIVER watch --until verified

step "6b. Next cycle: a trusted report of the low-trust find, a later stage of the deal; known, fact and excluded claims"
$DRIVER phase 2
$DRIVER miner
$DRIVER submit known
$DRIVER submit dispute
$DRIVER submit excluded

step "6c. Validator (--mock): scan the trusted list's feeds for the held scoops"
$DRIVER scan
$DRIVER watch --until verified

step "7. Validator (--mock): keyed re-verify of the closed epochs"
$DRIVER reverify

step "7b. Time runs forward to the shortened settlement"
$DRIVER watch --until settled --timeout 240

step "8. Validator (--mock): signed points -> settle() + burn -> weights"
$DRIVER weights

step "9. Expected outcomes"
$DRIVER check
FIRST=$(psql "$DATABASE_URL" -At -c "SELECT COUNT(*) FROM source_event e JOIN claim c
  ON c.id = e.claim_id WHERE e.role = 'first' AND c.source_url LIKE '%/posts/consumer-prices'")
echo "[check] first-report rows for the early scheduled release: $FIRST"
[[ $FIRST == 0 ]] || { echo "[check] FAIL first credit before the release time"; exit 1; }

step "10. Operator: a correction of a paid claim and a retraction of a trusted one"
$DRIVER wrong

step "11. Store contract: the memory store and the Prisma store agree"
(cd "$API_DIR" && STORE_TEST_DATABASE_URL="$DATABASE_URL" $API_PY -m pytest -q -p no:cacheprovider \
  tests/test_store_contract.py) | tail -1
echo "E2E passed. Logs: $E2E_DIR"
