#!/usr/bin/env bash
# Import (or re-import) the pipeline workflow into the running n8n container.
set -euo pipefail

WORKFLOW=${1:-/workflows/multilingual-dubbing-pipeline.json}
CONTAINER=${CONTAINER:-mvds-n8n}
# host-side port for the closing message; the healthz probe below runs INSIDE
# the container, where n8n still listens on its default 5678.
[ -f .env ] && N8N_PORT=$(grep -E '^N8N_PORT=' .env | cut -d= -f2)

echo "==> waiting for n8n to answer on :5678"
for _ in $(seq 1 60); do
  if docker exec "$CONTAINER" sh -c 'wget -q -O- http://localhost:5678/healthz >/dev/null 2>&1'; then
    break
  fi
  sleep 2
done

echo "==> importing $WORKFLOW"
docker exec "$CONTAINER" n8n import:workflow --input="$WORKFLOW"

echo "==> activating it (an imported workflow is INACTIVE, and an inactive"
echo "    workflow has no /webhook/ URL - jobs then stall at 2/13 stages)"
"$(dirname "$0")/activate-workflow.sh" || true

echo
echo "Done. Open http://localhost:${N8N_PORT:-47678} and look for:"
echo "  'Multilingual Video Translation & Dubbing Pipeline'"
echo
echo "Re-running this script after editing the JSON updates the workflow in place"
echo "(n8n matches on the workflow id embedded in the file)."
