#!/bin/bash
# ---------------------------------------------------------------------------
# Double-click this file in Finder to bring the whole system up.
#
#   1. creates the native Python env (first run only, downloads ~1 GB of torch)
#   2. starts n8n + Postgres + MinIO + the web UI in Docker
#   3. opens the four URLs in your browser
#   4. runs the AI service natively (keeps this window open)
#
# Close this window (or Ctrl-C) to stop the AI service.
# The Docker containers keep running - stop them with:  make down
# ---------------------------------------------------------------------------
cd "$(dirname "$0")" || exit 1

printf '\033[1;36m'
cat <<'BANNER'
  ============================================================
   Multilingual Video Translation & Dubbing System
  ============================================================
BANNER
printf '\033[0m\n'

fail() { printf '\n\033[1;31m%s\033[0m\n\n' "$1"; echo "Press Return to close."; read -r _; exit 1; }

# --- 1. native python environment -------------------------------------------
if [ ! -x .venv/bin/python ]; then
  printf '\033[1;36m[1/4] Setting up the native Python environment (one time, ~10 min)\033[0m\n'
  ./scripts/setup-native.sh || fail "setup-native.sh failed - see the error above."
else
  printf '\033[1;32m[1/4] Native environment already present (.venv)\033[0m\n'
fi

# --- 2. infrastructure -------------------------------------------------------
printf '\n\033[1;36m[2/4] Starting Docker infrastructure\033[0m\n'
./scripts/start.sh || fail "start.sh failed. Is Docker Desktop running?"

# --- 3. open the URLs --------------------------------------------------------
printf '\n\033[1;36m[3/4] Opening the interfaces in your browser\033[0m\n'
open "http://localhost:47300"    # web UI
open "http://localhost:47678"    # n8n
open "http://localhost:47901"    # MinIO console
open "http://localhost:47800/docs" 2>/dev/null || true

# --- 4. the AI service (blocking) -------------------------------------------
printf '\n\033[1;36m[4/4] Starting the AI service (leave this window open)\033[0m\n\n'
./scripts/run-native.sh

printf '\n\033[1;33mAI service stopped.\033[0m Press Return to close this window.\n'
read -r _
