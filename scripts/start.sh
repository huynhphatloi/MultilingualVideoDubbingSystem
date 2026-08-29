#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Bring the stack up.
#
#   ./scripts/start.sh              hybrid (default): infra in Docker,
#                                   AI service native  -> uses the Apple GPU
#   ./scripts/start.sh --docker-ai  everything in Docker (CPU only on macOS)
#   ./scripts/start.sh --docker-ai --gpu   everything in Docker, NVIDIA GPU
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."

MODE=hybrid
GPU=0
# `while [ $# ]` rather than `for arg in "$@"` - bash 3.2 (still the macOS
# default) is fussy about empty expansions under `set -u`.
while [ $# -gt 0 ]; do
  case "$1" in
    --docker-ai) MODE=docker ;;
    --gpu)       GPU=1 ;;
    *) echo "unknown option: $1"; exit 1 ;;
  esac
  shift
done

COMPOSE="docker compose"
[ "$GPU" = 1 ] && COMPOSE="docker compose -f docker-compose.yml -f docker-compose.gpu.yml"
[ "$MODE" = docker ] && COMPOSE="$COMPOSE --profile docker-ai"

bold() { printf "\n\033[1;36m==> %s\033[0m\n" "$1"; }
ok()   { printf "\033[1;32m  ok\033[0m  %s\n" "$1"; }
warn() { printf "\033[1;33m  !!\033[0m  %s\n" "$1"; }

# .env holds the ports; load it early so every URL below is consistent.
[ -f .env ] && set -a && . ./.env && set +a

bold "checking prerequisites"
command -v docker >/dev/null || { echo "Docker is not installed / not on PATH"; exit 1; }
docker info >/dev/null 2>&1 || { echo "Docker daemon is not running - start Docker Desktop first"; exit 1; }
ok "docker is running"

[ -f .env ] || { cp .env.example .env; ok ".env created from .env.example"; }
mkdir -p data/jobs data/samples

if grep -q '^HUGGINGFACE_TOKEN=hf_' .env; then
  ok "HUGGINGFACE_TOKEN is set (speaker diarization enabled)"
else
  warn "HUGGINGFACE_TOKEN is empty -> diarization falls back to a single speaker"
fi

# The AI service endpoint differs between the two modes; keep .env honest.
if [ "$MODE" = docker ]; then
  sed -i.bak 's|^AI_SERVICE_BASE_URL=.*|AI_SERVICE_BASE_URL=http://ai-service:8000|; s|^AI_SERVICE_UPSTREAM=.*|AI_SERVICE_UPSTREAM=ai-service:8000|' .env && rm -f .env.bak
  ok "n8n + frontend will target the containerised AI service"
else
  sed -i.bak 's|^AI_SERVICE_BASE_URL=.*|AI_SERVICE_BASE_URL=http://host.docker.internal:${AI_SERVICE_PORT:-47800}|; s|^AI_SERVICE_UPSTREAM=.*|AI_SERVICE_UPSTREAM=host.docker.internal:${AI_SERVICE_PORT:-47800}|' .env && rm -f .env.bak
  ok "n8n + frontend will target the native AI service on the host"
fi

bold "starting containers"
$COMPOSE up -d --build

bold "waiting for services"
wait_for() {
  local name=$1 url=$2 tries=${3:-120}
  printf "  %-12s" "$name"
  for _ in $(seq 1 "$tries"); do
    if curl -fsS -o /dev/null "$url" 2>/dev/null; then printf "\033[1;32mup\033[0m\n"; return 0; fi
    printf "."; sleep 3
  done
  printf "\033[1;31mtimeout\033[0m\n"; return 1
}

wait_for "minio"    "http://localhost:${MINIO_API_PORT:-47900}/minio/health/live" 40 || true
wait_for "n8n"      "http://localhost:${N8N_PORT:-47678}/healthz"           80
wait_for "frontend" "http://localhost:${FRONTEND_PORT:-47300}/"            40 || true
[ "$MODE" = docker ] && wait_for "ai-service" "http://localhost:${AI_SERVICE_PORT:-47800}/health" 120

bold "importing the n8n workflow"
docker exec mvds-n8n n8n import:workflow \
  --input=/workflows/multilingual-dubbing-pipeline.json 2>&1 | tail -3

# Importing does not activate. An inactive workflow has no production webhook,
# so the web UI uploads the video and the job then sits at 2/13 stages forever.
bold "activating the workflow"
./scripts/activate-workflow.sh || warn "activate it by hand in the n8n editor before uploading"

echo
echo "  ---------------------------------------------------------------"
echo "  Web UI     http://localhost:${FRONTEND_PORT:-47300}"
echo "  n8n        http://localhost:${N8N_PORT:-47678}"
echo "  FastAPI    http://localhost:${AI_SERVICE_PORT:-47800}/docs"
echo "  MinIO      http://localhost:${MINIO_CONSOLE_PORT:-47901}   (minioadmin / minioadmin123)"
echo "  ---------------------------------------------------------------"

if [ "$MODE" = hybrid ]; then
cat <<'EOF'

  Mode: HYBRID - the AI service is NOT running yet. In a second terminal:

      ./scripts/setup-native.sh     # once, creates .venv and installs torch
      ./scripts/run-native.sh       # every time you want the pipeline up

  Then check http://localhost:47800/health/device - it should report
  "resolved": "mps" on Apple Silicon.
EOF
else
cat <<'EOF'

  Mode: ALL-DOCKER. On macOS this cannot reach the Apple GPU, so every model
  runs on CPU. Use the default hybrid mode unless you specifically want a
  self-contained container demo.
EOF
fi

cat <<'EOF'

  Next:
    1. Cut demo clips that actually contain dialogue:  ./scripts/make-clip.sh
    2. Upload data/samples/dialogue_37s.mp4 at http://localhost:47300
       (or run: ./scripts/smoke-test.sh data/samples/dialogue_37s.mp4 vi)

  The workflow was activated above. If a job ever stops at 2/13 stages, its
  webhook is gone - re-run ./scripts/activate-workflow.sh

EOF
