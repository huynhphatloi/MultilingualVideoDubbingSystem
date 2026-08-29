#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Cut demo clips out of a long source video.
#
#   ./scripts/make-clip.sh                          # the two default demo clips
#   ./scripts/make-clip.sh src.mp4 START LEN NAME   # one custom clip
#
# Pick DIALOGUE, not action. A dubbing demo cut from an action beat produces a
# file with music and effects and almost no voice - which looks exactly like a
# broken pipeline. data/samples/avengers_20s.mp4 used to be such a cut: 20
# seconds containing the single line "Send the rest."
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."

SRC=${1:-data/test/The-Avengers.mp4}
[ -f "$SRC" ] || { echo "No such file: $SRC"; exit 1; }
mkdir -p data/samples

cut_clip() {
  local start=$1 len=$2 name=$3
  local out="data/samples/${name}.mp4"
  ffmpeg -y -v error -ss "$start" -t "$len" -i "$SRC" \
    -vf "scale=-2:720" -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p \
    -c:a aac -b:a 160k -movflags +faststart "$out"
  printf "  %-34s %ss  (from %ss)\n" "$out" \
    "$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$out" | cut -d. -f1)" "$start"
}

if [ $# -ge 3 ]; then
  cut_clip "$2" "$3" "${4:-$(basename "${SRC%.*}")_${3}s}"
else
  echo "cutting the default demo clips (dialogue-dense sections):"
  # 114-150 s: the "Call it, Captain" briefing - eight speakers' worth of
  # back-to-back lines, the best short dubbing demo in this source.
  cut_clip 113 37 "dialogue_37s"
  # 2-62 s: slower two-hander, useful for checking speaker diarization.
  cut_clip 2 60 "dialogue_60s"
fi
