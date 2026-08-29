#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Wipe job state. The model cache (models/) is never touched.
#
#   ./scripts/reset.sh            # delete every job: DB rows + objects + scratch
#   ./scripts/reset.sh --orphans  # only sweep scratch dirs with no job behind them
#
# The scratch directory is intentionally kept between the separate HTTP calls
# that make up one job (it is the cache n8n's stage-per-request design relies
# on), so nothing deletes it on its own. A few dozen runs of a feature-length
# file will happily fill a disk - hence this script.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."

API=${API:-http://localhost:47800}
SCRATCH=${SCRATCH_ROOT:-data/jobs/_scratch}
ORPHANS_ONLY=0
[ "${1:-}" = "--orphans" ] && ORPHANS_ONLY=1

command -v jq >/dev/null || { echo "jq is required (brew install jq)"; exit 1; }

# GET /jobs caps `limit` at 200 (Query(50, le=200)); 500 is rejected with a 422.
JOB_LIMIT=200

# Prints one job id per line. Exits non-zero if the registry could not be read -
# which the caller MUST distinguish from "there are no jobs". Conflating the two
# is how an orphan sweep deletes the scratch of every live job.
live_ids() {
  local body
  body=$(curl -fsS -m 10 "$API/jobs?limit=$JOB_LIMIT") || return 1
  printf '%s' "$body" | jq -r '.jobs[].job_id'
}

if ! LIVE=$(live_ids); then
  echo "cannot read the job registry at $API - is the AI service running?"
  echo "refusing to sweep: every scratch directory would look orphaned."
  exit 1
fi

if [ "$ORPHANS_ONLY" = 0 ]; then
  if [ -z "$LIVE" ]; then
    echo "no jobs registered"
  else
    for id in $LIVE; do
      echo "deleting job $id"
      curl -sS -X DELETE "$API/jobs/$id" >/dev/null
    done
  fi
  # Everything was just deleted, so nothing is live any more.
  LIVE=""
fi

# Scratch directories outlive the jobs that created them whenever a run is
# interrupted, the DB is reset, or the service is restarted mid-pipeline.
echo
echo "sweeping orphaned scratch under $SCRATCH"
remaining=$(printf '%s' "$LIVE" | tr '\n' ' ')
freed=0
for dir in "$SCRATCH"/*/; do
  [ -d "$dir" ] || continue
  id=$(basename "$dir")
  case " $remaining " in
    *" $id "*) continue ;;
  esac
  size=$(du -sk "$dir" | cut -f1)
  freed=$((freed + size))
  echo "  removing $id ($((size / 1024)) MB)"
  rm -rf "$dir"
done
echo "freed $((freed / 1024)) MB"
echo "done"
