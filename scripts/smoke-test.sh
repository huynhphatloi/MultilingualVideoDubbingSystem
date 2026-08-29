#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# End-to-end pipeline run straight against the FastAPI service (no n8n).
# Useful to prove the backend works before blaming the workflow.
#
#   ./scripts/smoke-test.sh [video.mp4] [target_lang]
#
# NOTE ON QUOTING - do not "simplify" the payload lines back into one call.
# macOS still ships bash 3.2, which mis-parses escaped double quotes nested
# inside "$( ... )". This:
#
#     check "$(post /translation/translate "{\"job_id\":\"$JOB_ID\",\"x\":\"y\"}")"
#
# reaches curl as the single word  "job_id":"98dc..."  - the leading brace is
# eaten and everything after the comma is split off as a separate argument
# (which curl then treats as a second URL and requests as well). Every payload
# with a comma in it silently became invalid JSON, so this script could never
# get past the merge stage on a Mac. Build each payload into its own variable
# first; that parses identically on bash 3.2 and bash 5.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."

API=${API:-http://localhost:47800}
VIDEO=${1:-data/samples/dialogue_37s.mp4}
TARGET=${2:-vi}
OUTDIR=${OUTDIR:-data/output}

command -v jq >/dev/null || { echo "jq is required (brew install jq / apt install jq)"; exit 1; }
if [ ! -f "$VIDEO" ]; then
  echo "No such file: $VIDEO"
  echo "Cut the demo clips first:  ./scripts/make-clip.sh"
  exit 1
fi

step() { printf "\n\033[1;36m==> %s\033[0m\n" "$1"; }

# post <path> <payload>  -> raw response body
post() { curl -sS -X POST "$API$1" -H 'Content-Type: application/json' --data-binary "$2"; }

# show <response> [jq filter] -> print, or abort if the service returned an error
show() {
  if printf '%s' "$1" | jq -e '.error' >/dev/null 2>&1; then
    printf '%s' "$1" | jq .
    echo "FAILED"; exit 1
  fi
  printf '%s' "$1" | jq -c "${2:-.}"
}

# call <path> <payload> [jq filter] -> post, print, and keep the body in $REPLY_BODY
REPLY_BODY=""
call() {
  REPLY_BODY=$(post "$1" "$2")
  show "$REPLY_BODY" "${3:-.}"
}

step "health"
curl -sS "$API/health/ready" | jq .

step "create job (target=$TARGET)"
payload="{\"target_language\":\"$TARGET\",\"source_filename\":\"$(basename "$VIDEO")\"}"
call /jobs "$payload" '{job_id, target_language}'
JOB_ID=$(printf '%s' "$REPLY_BODY" | jq -r .job_id)
echo "job_id = $JOB_ID"

JOB="{\"job_id\":\"$JOB_ID\"}"

step "upload video"
UP=$(curl -sS -X POST "$API/jobs/$JOB_ID/upload" -F "file=@$VIDEO")
show "$UP" '{video_key, duration: .media_info.duration}'

step "extract audio"
call /media/extract-audio "$JOB" '{audio_key, duration_seconds, peak_dbfs}'

step "separate speech / background (demucs - slow on CPU)"
call /media/separate "$JOB" '{speech_key, background_key, model, separated, mode}'

step "speech-to-text"
call /speech/transcribe "$JOB" '{language, language_confidence, segment_count}'

step "speaker diarization"
call /speech/diarize "$JOB" '{speaker_count, enabled}'

step "merge transcript + speaker + time"
call /speech/merge "$JOB" '{segment_count, speaker_count, turn_source}'

step "translate (duration-aware)"
payload="{\"job_id\":\"$JOB_ID\",\"target_language\":\"$TARGET\"}"
call /translation/translate "$payload" '{engine, segment_count, source_language, target_language}'

step "generate speech"
call /speech/synthesize "$JOB" '{synthesized, failed, models_used, ratio_stats, needs_adaptation}'

# The adapt loop, i.e. what the n8n "Duration Within Tolerance?" branch drives.
# SYNC_MAX_RETRANSLATE_ATTEMPTS caps it at two passes.
round=1
while [ "$round" -le 2 ]; do
  NEEDS=$(printf '%s' "$REPLY_BODY" | jq -c '.needs_adaptation // []')
  [ "$NEEDS" = "[]" ] && break

  step "adapt translation, pass $round, for $NEEDS"
  payload="{\"job_id\":\"$JOB_ID\",\"segment_ids\":$NEEDS}"
  call /translation/adapt "$payload" '{adapted, identical}'
  ADAPTED=$(printf '%s' "$REPLY_BODY" | jq -c '.adapted // []')
  [ "$ADAPTED" = "[]" ] && break

  step "re-generate speech, pass $round"
  payload="{\"job_id\":\"$JOB_ID\",\"segment_ids\":$ADAPTED}"
  call /speech/synthesize "$payload" '{synthesized, ratio_stats, needs_adaptation}'
  round=$((round + 1))
done

step "place audio on the original timeline"
call /audio/synchronize "$JOB" \
  '{placed_segments, stretched_segments, overlapping_segments, timeline_duration}'

step "mix dub + background"
call /audio/mix "$JOB" \
  '{final_audio_key, background_used, ducking_applied, speech_auto_gain_db}'

step "generate subtitles"
call /subtitle/generate "$JOB" '{subtitle_keys, cue_count}'

step "render"
call /video/render "$JOB" '{output_key, duration_seconds, size_bytes, video_stream_copied}'

step "final job state"
curl -sS "$API/jobs/$JOB_ID" | jq '{status, progress, output_key}'

step "downloading the result"
# A job whose only trace is an object key is hard to actually watch. Pull the
# finished artifacts onto disk so `open` just works.
mkdir -p "$OUTDIR/$JOB_ID"
curl -sS -o "$OUTDIR/$JOB_ID/dubbed_$TARGET.mp4" "$API/jobs/$JOB_ID/download/video"
curl -sS -o "$OUTDIR/$JOB_ID/dubbed_$TARGET.srt" "$API/jobs/$JOB_ID/download/subtitle"
curl -sS -o "$OUTDIR/$JOB_ID/source.srt" "$API/artifacts/jobs/$JOB_ID/subtitles/source_$(
  curl -sS "$API/jobs/$JOB_ID" | jq -r '.source_language // "en"').srt" || true
ls -lh "$OUTDIR/$JOB_ID"

printf "\n\033[1;32mSUCCESS\033[0m  job %s\n" "$JOB_ID"
echo "  dubbed video : $OUTDIR/$JOB_ID/dubbed_$TARGET.mp4"
echo "  subtitles    : $OUTDIR/$JOB_ID/dubbed_$TARGET.srt"
echo "  source clip  : $VIDEO"
echo
echo "  play it:   open '$OUTDIR/$JOB_ID/dubbed_$TARGET.mp4'"
