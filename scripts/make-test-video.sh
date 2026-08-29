#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Builds a synthetic two-speaker video with background music, so you can
# exercise the whole pipeline (diarization included) without hunting for a clip.
#
#   ./scripts/make-test-video.sh  ->  data/samples/test_two_speakers.mp4
#
# Speech is generated with MMS-TTS (free, local). Runs in the native venv when
# there is one, otherwise inside the ai-service container.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."

ROOT="$PWD"
OUT_DIR="$ROOT/data/samples"
CONTAINER=${CONTAINER:-mvds-ai-service}
VENV=${VENV:-.venv}
mkdir -p "$OUT_DIR"

if [ -x "$VENV/bin/python" ]; then
  MODE=native
  WORK="$OUT_DIR"
  echo "==> using the native venv"
elif docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  MODE=docker
  WORK=/data/jobs/_samples
  echo "==> using the $CONTAINER container"
else
  echo "Neither $VENV nor the $CONTAINER container is available."
  echo "Run ./scripts/setup-native.sh first, or start the stack with --docker-ai."
  exit 1
fi

run_py() {
  if [ "$MODE" = native ]; then
    WORK="$WORK" "$VENV/bin/python" -
  else
    docker exec -i -e WORK="$WORK" "$CONTAINER" python -
  fi
}

run_sh() {
  if [ "$MODE" = native ]; then
    WORK="$WORK" bash -s
  else
    docker exec -i -e WORK="$WORK" "$CONTAINER" sh -s
  fi
}

echo "==> generating speech (first run downloads ~150 MB of MMS-TTS)"
run_py <<'PY'
import os, wave, numpy as np, torch
from transformers import AutoTokenizer, VitsModel

work = os.environ["WORK"]
os.makedirs(work, exist_ok=True)

repo = "facebook/mms-tts-eng"
tok = AutoTokenizer.from_pretrained(repo)
model = VitsModel.from_pretrained(repo).eval()

LINES = [
    ("a", "Good morning everyone. Today we are going to look at a multilingual dubbing system."),
    ("b", "That sounds interesting. How does the pipeline keep the timing of the original video?"),
    ("a", "Every sentence keeps its own start and end time, and the generated speech is placed back exactly there."),
    ("b", "And what happens when the translated sentence is much longer than the original one?"),
    ("a", "Then the system rewrites the translation until it fits, and applies only a very light time stretch."),
]

def synth(text, speed):
    with torch.inference_mode():
        wav = model(**tok(text, return_tensors="pt")).waveform.squeeze().cpu().numpy()
    if speed != 1.0:  # crude resample to fake a second voice
        idx = np.clip((np.arange(0, len(wav), speed)).astype(int), 0, len(wav) - 1)
        wav = wav[idx]
    return wav

sr = int(model.config.sampling_rate)
gap = np.zeros(int(sr * 0.6), dtype=np.float32)
parts = []
for speaker, text in LINES:
    parts.append(synth(text, 1.0 if speaker == "a" else 1.18))
    parts.append(gap)

audio = np.concatenate(parts)
audio = np.clip(audio / (np.max(np.abs(audio)) or 1.0) * 0.85, -1, 1)
with wave.open(f"{work}/speech.wav", "wb") as fh:
    fh.setnchannels(1); fh.setsampwidth(2); fh.setframerate(sr)
    fh.writeframes((audio * 32767).astype("<i2").tobytes())
print(f"speech.wav: {len(audio)/sr:.1f}s @ {sr} Hz")
PY

echo "==> compositing video + background music with ffmpeg"
run_sh <<'SH'
set -e
S="$WORK"
DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$S/speech.wav")

# Gentle background pad so the Demucs stage has something real to preserve.
ffmpeg -y -loglevel error \
  -f lavfi -i "sine=frequency=220:sample_rate=44100:duration=$DUR" \
  -f lavfi -i "sine=frequency=277:sample_rate=44100:duration=$DUR" \
  -filter_complex "[0:a][1:a]amix=inputs=2,tremolo=f=0.4:d=0.6,volume=-20dB[bg]" \
  -map "[bg]" "$S/background.wav"

ffmpeg -y -loglevel error -i "$S/speech.wav" -i "$S/background.wav" \
  -filter_complex "[0:a]volume=0dB[a];[a][1:a]amix=inputs=2:duration=first[out]" \
  -map "[out]" -ar 44100 "$S/mixed.wav"

ffmpeg -y -loglevel error \
  -f lavfi -i "testsrc2=size=1280x720:rate=25:duration=$DUR" \
  -i "$S/mixed.wav" \
  -vf "drawtext=text='Multilingual Dubbing Test Clip':fontcolor=white:fontsize=42:x=(w-text_w)/2:y=60" \
  -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p \
  -c:a aac -b:a 160k -shortest "$S/test_two_speakers.mp4"

rm -f "$S/speech.wav" "$S/background.wav" "$S/mixed.wav"
ls -lh "$S/test_two_speakers.mp4"
SH

if [ "$MODE" = docker ]; then
  echo "==> copying the clip out of the container"
  docker cp "$CONTAINER:$WORK/test_two_speakers.mp4" "$OUT_DIR/test_two_speakers.mp4"
fi

echo
echo "Saved: $OUT_DIR/test_two_speakers.mp4"
echo "Next:  open http://localhost:47300 and upload it,"
echo "       or run ./scripts/smoke-test.sh data/samples/test_two_speakers.mp4 vi"
