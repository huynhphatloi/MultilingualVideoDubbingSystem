#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Run the FastAPI AI service natively (uses the Apple GPU when available).
#
#   ./scripts/run-native.sh            # normal
#   ./scripts/run-native.sh --reload   # auto-reload while editing code
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."

VENV=${VENV:-.venv}
[ -d "$VENV" ] || { echo "No $VENV - run ./scripts/setup-native.sh first"; exit 1; }
# shellcheck disable=SC1091
source "$VENV/bin/activate"

set -a
# shellcheck disable=SC1091
[ -f .env ] && source .env
set +a

# A few ops in pyannote / coqui have no Metal kernel; without this the process
# raises NotImplementedError instead of quietly using the CPU for those ops.
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONPATH="$PWD/ai-service"
export TOKENIZERS_PARALLELISM=false
# transformers downloads a *second* copy of any .bin-only checkpoint in a
# background thread (the community safetensors conversion). NLLB-200 is
# .bin-only, so that is 2.46 GB of pure waste that also starves the next
# stage's download. app/core/paths.py sets this too; exported here as well so
# subprocesses (demucs) inherit it.
export DISABLE_SAFETENSORS_CONVERSION=true

# Model caches must be ABSOLUTE: demucs and friends run as subprocesses whose
# working directory we do not control, and .env ships relative paths so the
# same file works for the container too.
export HF_HOME="${HF_HOME:-./models/huggingface}"
export TORCH_HOME="${TORCH_HOME:-./models/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-./models/cache}"
case "$HF_HOME"        in /*) ;; *) HF_HOME="$PWD/${HF_HOME#./}" ;; esac
case "$TORCH_HOME"     in /*) ;; *) TORCH_HOME="$PWD/${TORCH_HOME#./}" ;; esac
case "$XDG_CACHE_HOME" in /*) ;; *) XDG_CACHE_HOME="$PWD/${XDG_CACHE_HOME#./}" ;; esac
export HF_HOME TORCH_HOME XDG_CACHE_HOME

mkdir -p data/jobs data/samples "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME"

echo "==> checking that the infrastructure is up"
for probe in "postgres:localhost:${POSTGRES_PORT:-47432}" "minio:localhost:${MINIO_API_PORT:-47900}"; do
  name=${probe%%:*}; hostport=${probe#*:}
  host=${hostport%%:*}; port=${hostport##*:}
  if ! nc -z "$host" "$port" 2>/dev/null; then
    echo "    $name is not reachable on $host:$port - run: docker compose up -d"
    exit 1
  fi
  echo "    $name ok"
done

echo "==> model cache: $TORCH_HOME (torch) · $HF_HOME (huggingface)"
echo "==> starting the AI service on http://localhost:${AI_SERVICE_PORT:-47800}"

# No arrays here on purpose: macOS still ships bash 3.2, where expanding an
# EMPTY array under `set -u` aborts with "unbound variable".
if [ "${1:-}" = "--reload" ]; then
  exec uvicorn app.main:app \
    --app-dir ai-service \
    --host 0.0.0.0 \
    --port "${AI_SERVICE_PORT:-47800}" \
    --timeout-keep-alive 300 \
    --reload --reload-dir ai-service/app
fi

exec uvicorn app.main:app \
  --app-dir ai-service \
  --host 0.0.0.0 \
  --port "${AI_SERVICE_PORT:-47800}" \
  --timeout-keep-alive 300
