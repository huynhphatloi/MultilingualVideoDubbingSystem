#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Activate the dubbing workflow in n8n and prove its webhook answers.
#
# Importing a workflow does NOT activate it, and an inactive workflow has no
# production webhook: POST /webhook/dubbing/start returns 404. The web UI then
# uploads the video fine (job reaches 2/13 stages: create_job + store_video)
# and stops there forever, because nothing else ever calls the AI service.
# That is the "stuck at WAITING / 15.4%" symptom.
#
# n8n also refuses to apply `update:workflow --active` to a running instance
# ("Please restart n8n for changes to take effect"), so the container is
# restarted when the flag actually changed.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."

CONTAINER=${CONTAINER:-mvds-n8n}
WORKFLOW_NAME=${WORKFLOW_NAME:-"Multilingual Video Translation & Dubbing Pipeline"}
[ -f .env ] && N8N_PORT=$(grep -E '^N8N_PORT=' .env | cut -d= -f2 || true)
N8N_PORT=${N8N_PORT:-47678}
BASE="http://localhost:$N8N_PORT"

ok()   { printf "\033[1;32m  ok\033[0m  %s\n" "$1"; }
warn() { printf "\033[1;33m  !!\033[0m  %s\n" "$1"; }

webhook_registered() {
  # 404 == not registered. Anything else means n8n owns the route.
  local code
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 10 -X POST \
           "$BASE/webhook/dubbing/start" -H 'Content-Type: application/json' \
           -d '{"_probe":true}' || echo 000)
  [ "$code" != "404" ] && [ "$code" != "000" ]
}

id=$(docker exec "$CONTAINER" n8n list:workflow 2>/dev/null \
     | grep -F "$WORKFLOW_NAME" | head -1 | cut -d'|' -f1 || true)
if [ -z "$id" ]; then
  warn "workflow '$WORKFLOW_NAME' is not in n8n - run ./scripts/import-workflow.sh first"
  exit 1
fi

if webhook_registered; then
  ok "workflow $id is active (webhook /webhook/dubbing/start answers)"
  exit 0
fi

echo "==> activating workflow $id"
docker exec "$CONTAINER" n8n update:workflow --id="$id" --active=true >/dev/null 2>&1 || true

echo "==> restarting n8n so the activation takes effect"
docker restart "$CONTAINER" >/dev/null
for _ in $(seq 1 60); do
  curl -fsS -m 3 "$BASE/healthz" >/dev/null 2>&1 && break
  sleep 2
done
sleep 3

if webhook_registered; then
  ok "workflow $id active - POST $BASE/webhook/dubbing/start is live"
else
  warn "webhook still not registered. Open $BASE, open the workflow, press Save, then toggle Active."
  exit 1
fi
