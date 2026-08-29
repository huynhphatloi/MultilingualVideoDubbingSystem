#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Create the native Python environment for the AI service.
#
# Why native: Docker Desktop on macOS runs a Linux VM with no Metal
# passthrough, so a containerised torch can only ever use the CPU. Running here
# lets torch use the `mps` backend (Demucs / XTTS / NLLB ~4-6x faster) and also
# sidesteps the missing linux-arm64 wheels (sphn, torchcodec).
#
#   ./scripts/setup-native.sh
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN=${PYTHON_BIN:-python3.11}
VENV=${VENV:-.venv}

bold() { printf "\n\033[1;36m==> %s\033[0m\n" "$1"; }
ok()   { printf "\033[1;32m  ok\033[0m  %s\n" "$1"; }
die()  { printf "\033[1;31m  !!\033[0m  %s\n" "$1"; exit 1; }

bold "checking prerequisites"

if ! command -v "$PYTHON_BIN" >/dev/null; then
  if python3 -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info < (3,14) else 1)' 2>/dev/null; then
    PYTHON_BIN=python3
  else
    die "Python 3.11 not found. Install it:  brew install python@3.11
      (3.10-3.13 all work; coqui-tts requires <3.15)"
  fi
fi
ok "$($PYTHON_BIN --version)"

command -v ffmpeg >/dev/null || die "ffmpeg not found. Install it:  brew install ffmpeg"
ok "$(ffmpeg -version | head -1 | cut -d' ' -f1-3)"

case "$(uname -s)/$(uname -m)" in
  Darwin/arm64) TORCH_REQ=ai-service/requirements-native-macos.txt; PLATFORM="macOS Apple Silicon (Metal available)";;
  Darwin/*)     TORCH_REQ=ai-service/requirements-native-macos.txt; PLATFORM="macOS Intel (CPU only)";;
  *)            TORCH_REQ=ai-service/requirements-native-linux.txt; PLATFORM="$(uname -s) $(uname -m)";;
esac
ok "$PLATFORM"

bold "creating virtualenv at $VENV"
[ -d "$VENV" ] || "$PYTHON_BIN" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel --quiet
ok "$(python --version) in $VENV"

bold "installing torch (this is the big one, ~1 GB)"
pip install -r "$TORCH_REQ"

bold "installing the AI/media stack"
pip install -r ai-service/requirements.txt

bold "installing the test tooling (pytest, ruff)"
pip install --quiet pytest==8.3.4 ruff==0.9.2

bold "verifying the install"
python - <<'PY'
import importlib, sys
mods = ["torch", "torchaudio", "torchcodec", "faster_whisper", "transformers",
        "pyannote.audio", "demucs", "TTS", "fastapi", "minio", "sqlalchemy",
        # XTTS language tokenizers. Missing ones do not fail at install time -
        # they fail at SYNTHESIS time, after ASR and translation have already
        # run, and only for the affected language. Catch them here instead.
        "cutlet", "fugashi", "pypinyin"]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
        print(f"  ok   {m}")
    except Exception as exc:
        missing.append(m)
        print(f"  FAIL {m}: {exc}")

if missing:
    print("\n  Missing language support -> reinstall with the extras:")
    print("      pip install 'coqui-tts[languages]==0.27.5'")

import torch
print(f"\n  torch {torch.__version__}")
mps = getattr(torch.backends, "mps", None)
if mps and mps.is_available():
    print("  Metal (MPS): AVAILABLE  <- models will use the Apple GPU")
elif torch.cuda.is_available():
    print(f"  CUDA: {torch.cuda.get_device_name(0)}")
else:
    print("  no GPU backend - running on CPU")
sys.exit(1 if missing else 0)
PY

mkdir -p data/jobs data/samples

cat <<'EOF'

  ---------------------------------------------------------------
  Native environment ready.

  1. Start the infrastructure (n8n + Postgres + MinIO + web UI):
         docker compose up -d

  2. Start the AI service natively:
         ./scripts/run-native.sh

  3. Import the workflow (first time only):
         make import-workflow
  ---------------------------------------------------------------

EOF
