"""Stage 5: merge ASR + diarization into the pipeline's working document.

Output (``transcript/segments.json``) is the single source of truth from here
on: one entry per utterance carrying timing, speaker, text and - later -
translation, generated audio and sync decisions.
"""
from __future__ import annotations

import logging
from collections import defaultdict

from app.core.config import settings
from app.services import ffmpeg
from app.services.workspace import JobWorkspace

log = logging.getLogger(__name__)

#: A speaker change inside one ASR segment is only honoured when the minority
#: speaker owns at least this fraction of the words.
_SPLIT_MIN_SHARE = 0.25
_MIN_SPLIT_DURATION = 0.6


def merge(job_id: str, transcript_key: str, diarization_key: str,
          extract_voice_references: bool = True) -> dict:
    ws = JobWorkspace(job_id)
    transcript = ws.pull_json(transcript_key)
    diarization = ws.pull_json(diarization_key)

    turns, turn_source = _pick_turns(diarization)
    asr_segments = transcript.get("segments", [])
    source_language = transcript.get("language")

    merged: list[dict] = []
    for seg in asr_segments:
        merged.extend(_assign_speakers(seg, turns))

    for new_id, seg in enumerate(merged):
        seg["segment_id"] = new_id
        seg["duration"] = round(seg["end"] - seg["start"], 3)
        seg["source_language"] = source_language
        seg.setdefault("translated_text", None)
        seg.setdefault("generated_audio", None)
        seg.setdefault("generated_duration", None)
        seg.setdefault("duration_ratio", None)
        seg.setdefault("sync_action", "pending")
        seg.setdefault("attempts", 0)

    speakers = _speaker_stats(merged)

    references: dict[str, str] = {}
    if extract_voice_references and merged:
        references = _extract_references(ws, merged, speakers)
        for seg in merged:
            seg["voice_reference"] = references.get(seg["speaker_id"])
        for spk in speakers:
            spk["voice_reference"] = references.get(spk["speaker_id"])

    document = {
        "job_id": job_id,
        "source_language": source_language,
        "language_confidence": transcript.get("language_confidence"),
        "target_language": None,
        "speakers": speakers,
        "segment_count": len(merged),
        "diarization_model": diarization.get("model"),
        "turn_source": turn_source,
        "segments": merged,
    }
    key = ws.push_json(document, ws.layout.segments)

    log.info("merged %d segments across %d speakers using %s turns",
             len(merged), len(speakers), turn_source)
    return {
        "segments_key": key,
        "segment_count": len(merged),
        "speaker_count": len(speakers),
        "speakers": speakers,
        "turn_source": turn_source,
    }


def _pick_turns(diarization: dict) -> tuple[list[dict], str]:
    """Prefer *exclusive* diarization turns when the model provides them.

    Exclusive diarization assigns every instant to at most one speaker. Regular
    diarization does not: during overlapped speech two turns cover the same
    milliseconds, and the "which speaker owns this word" question becomes a
    coin flip. Since Whisper timestamps are coarser than diarization
    boundaries anyway, the exclusive view is strictly better for reconciling
    the two - which is precisely why `community-1` added it.
    """
    exclusive = diarization.get("exclusive_turns") or []
    regular = diarization.get("turns") or []

    if settings.diarization_use_exclusive and exclusive:
        return exclusive, "exclusive"
    if regular:
        return regular, "regular"
    return exclusive, "exclusive" if exclusive else "none"


# --------------------------------------------------------------- speakers ---
def _assign_speakers(segment: dict, turns: list[dict]) -> list[dict]:
    """Attach a speaker to an ASR segment, splitting it on real speaker changes."""
    words = segment.get("words") or []

    if not turns:
        return [_clone(segment, "SPEAKER_00")]

    if not words:
        return [_clone(segment, _dominant_speaker(segment["start"], segment["end"], turns))]

    labelled = [(w, _dominant_speaker(w["start"], w["end"], turns)) for w in words]
    distinct = {spk for _, spk in labelled}
    if len(distinct) == 1:
        return [_clone(segment, next(iter(distinct)))]

    counts = defaultdict(int)
    for _, spk in labelled:
        counts[spk] += 1
    minority_share = min(counts.values()) / len(labelled)
    if minority_share < _SPLIT_MIN_SHARE:
        winner = max(counts.items(), key=lambda kv: kv[1])[0]
        return [_clone(segment, winner)]

    # Split into contiguous same-speaker runs.
    chunks: list[dict] = []
    run: list[dict] = []
    run_speaker = labelled[0][1]
    for word, spk in labelled:
        if spk != run_speaker and run:
            chunks.append(_chunk_from_words(segment, run, run_speaker))
            run, run_speaker = [], spk
        run.append(word)
    if run:
        chunks.append(_chunk_from_words(segment, run, run_speaker))

    chunks = [c for c in chunks
              if c["end"] - c["start"] >= _MIN_SPLIT_DURATION and c["source_text"]]
    return chunks or [_clone(segment, run_speaker)]


def _dominant_speaker(start: float, end: float, turns: list[dict]) -> str:
    best, best_overlap = "SPEAKER_00", 0.0
    for turn in turns:
        overlap = min(end, turn["end"]) - max(start, turn["start"])
        if overlap > best_overlap:
            best, best_overlap = turn["speaker_id"], overlap
    if best_overlap <= 0:
        # No overlap at all: fall back to the nearest turn in time.
        nearest = min(
            turns, key=lambda t: min(abs(t["start"] - end), abs(t["end"] - start))
        )
        return nearest["speaker_id"]
    return best


def _clone(segment: dict, speaker_id: str) -> dict:
    item = dict(segment)
    item["speaker_id"] = speaker_id
    item["source_text"] = (segment.get("source_text") or "").strip()
    return item


def _chunk_from_words(segment: dict, words: list[dict], speaker_id: str) -> dict:
    text = "".join(w["word"] for w in words).strip()
    return {
        **{k: v for k, v in segment.items() if k not in ("words", "start", "end", "source_text")},
        "start": round(float(words[0]["start"]), 3),
        "end": round(float(words[-1]["end"]), 3),
        "source_text": text,
        "speaker_id": speaker_id,
        "words": words,
    }


def _speaker_stats(segments: list[dict]) -> list[dict]:
    stats: dict[str, dict] = {}
    for seg in segments:
        entry = stats.setdefault(
            seg["speaker_id"],
            {"speaker_id": seg["speaker_id"], "segment_count": 0,
             "total_seconds": 0.0, "voice_reference": None},
        )
        entry["segment_count"] += 1
        entry["total_seconds"] += seg["end"] - seg["start"]
    for entry in stats.values():
        entry["total_seconds"] = round(entry["total_seconds"], 2)
    return sorted(stats.values(), key=lambda s: -s["total_seconds"])


# ------------------------------------------------------ voice references ----
def _extract_references(ws: JobWorkspace, segments: list[dict],
                        speakers: list[dict]) -> dict[str, str]:
    """Build a clean voice sample per speaker for the cloning TTS models."""
    try:
        speech = ws.pull(ws.layout.speech_audio)
    except Exception:
        speech = ws.pull(ws.layout.original_audio)

    references: dict[str, str] = {}
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for seg in segments:
        by_speaker[seg["speaker_id"]].append(seg)

    for spk in speakers:
        sid = spk["speaker_id"]
        candidates = sorted(
            (s for s in by_speaker[sid] if (s["end"] - s["start"]) >= 1.2),
            key=lambda s: -(s["end"] - s["start"]),
        ) or sorted(by_speaker[sid], key=lambda s: -(s["end"] - s["start"]))

        chunks, total = [], 0.0
        for index, seg in enumerate(candidates):
            if total >= settings.max_speaker_reference_seconds:
                break
            piece = ws.path("speakers", f"{_slug(sid)}_part{index}.wav")
            ffmpeg.slice_audio(speech, piece, seg["start"], seg["end"],
                               sample_rate=settings.tts_sample_rate)
            chunks.append(piece)
            total += seg["end"] - seg["start"]

        if not chunks:
            log.warning("no usable reference audio for %s", sid)
            continue

        ref = ws.path("speakers", f"{_slug(sid)}.wav")
        ffmpeg.concat_audio(chunks, ref, sample_rate=settings.tts_sample_rate)
        key = ws.push(ref, ws.layout.speaker_reference(sid))
        references[sid] = key
        log.info("voice reference for %s: %.1fs", sid, total)

    return references


def _slug(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in value).lower()
