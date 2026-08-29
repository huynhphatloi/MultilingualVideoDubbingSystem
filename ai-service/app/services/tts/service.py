"""Stage 7: speech generation for every segment + duration measurement.

For each segment we synthesise, measure the real spoken duration and classify
the result against the original slot:

    ratio = generated_duration / original_duration

    0.90 <= ratio <= 1.10  -> accept
    0.80..0.90 / 1.10..1.20 -> light time-stretch later
    < 0.80 or > 1.20        -> ask the translator for a shorter/longer line
"""
from __future__ import annotations

import logging
import statistics
from pathlib import Path

from app.core import languages
from app.core.config import settings
from app.core.errors import SynthesisError
from app.services import ffmpeg
from app.services.translation.duration_aware import classify, measured_chars_per_second
from app.services.tts import router
from app.services.tts.base import SynthesisRequest, SynthesisResult
from app.services.workspace import JobWorkspace

log = logging.getLogger(__name__)


def synthesize_job(job_id: str, segments_key: str, target_language: str | None = None,
                   segment_ids: list[int] | None = None,
                   prefer_voice_clone: bool | None = None,
                   force_model: str | None = None) -> dict:
    ws = JobWorkspace(job_id)
    document = ws.pull_json(segments_key)
    segments = document.get("segments", [])
    target = languages.normalize(target_language or document.get("target_language"))
    if not target:
        raise SynthesisError("Target language unknown - translate before synthesising.")

    selected = [
        s for s in segments
        if (segment_ids is None or s["segment_id"] in set(segment_ids))
        and (s.get("translated_text") or "").strip()
        and (segment_ids is not None or not s.get("generated_audio"))
    ]
    if segment_ids is None and not selected:
        # Nothing pending: re-report the current state instead of failing.
        log.info("all segments already synthesised")
        selected = []

    # A bad model name is a configuration error: identical on every segment,
    # and now reachable from a dropdown. Checking it once turns "32 segments
    # failed, job reports 200" into one clear 4xx naming the valid choices.
    if force_model:
        router.resolve(target, has_voice_reference=True, force_model=force_model)

    references = _voice_references(ws, document)
    pitch = _speaker_pitch(document)
    models_used: dict[str, int] = {}
    failures: list[dict] = []
    #: Segments THIS call actually produced audio for. Not the same as "has
    #: generated_audio": on a re-run the previous take is still attached, so
    #: counting that would report a failed re-synthesis as a success and hand
    #: back the old voice as though the new engine had spoken it.
    produced: list[int] = []

    for index, seg in enumerate(selected):
        try:
            _synthesize_segment(ws, seg, target, references, prefer_voice_clone,
                                force_model, models_used, pitch)
        except Exception as exc:  # noqa: BLE001 - one bad line must not kill the job
            detail = _describe(exc)
            log.error("segment %s failed: %s", seg["segment_id"], detail)
            seg["sync_action"] = "failed"
            seg["error"] = detail[:500]
            failures.append({"segment_id": seg["segment_id"], "error": detail[:500]})

            # A configuration problem (missing tokenizer, no checkpoint for this
            # language) fails identically on every line. Grinding through all of
            # them costs minutes and buries the reason under N copies of itself.
            if not models_used and index + 1 >= _FAIL_FAST_AFTER:
                log.error("aborting synthesis: the first %d segment(s) all failed",
                          index + 1)
                break
        else:
            produced.append(seg["segment_id"])

    # Feed the engine's real speaking rate back into the document so the adapt
    # loop can aim at a budget this voice can actually hit.
    rate = measured_chars_per_second(segments)
    if rate:
        document["tts_chars_per_second"] = round(rate, 2)
        log.info("measured %s speaking rate: %.1f chars/s", target, rate)

    # Nothing at all came out. This MUST fail the stage: returning 200 with
    # `synthesized: 0` painted the UI green and let the job march on to
    # synchronisation, which then failed with "No generated audio to
    # synchronise" - a true statement that points at the wrong stage and hides
    # the real reason (which lives in `failures` below).
    if selected and not produced:
        raise SynthesisError(
            f"No speech could be generated for '{target}' - "
            f"all {len(failures)} segment(s) failed.",
            details={
                "target_language": target,
                "attempted": len(selected),
                "distinct_errors": _distinct(failures),
                "hint": _language_hint(target),
            },
        )

    document["segments"] = segments
    out_key = ws.push_json(document, ws.layout.segments)

    ratios = [s["duration_ratio"] for s in segments if s.get("duration_ratio")]
    needs_adaptation = [
        s["segment_id"] for s in segments if s.get("sync_action") == "retranslate"
        and s.get("attempts", 0) < settings.sync_max_retranslate_attempts
    ]

    # The manifest describes the JOB, not this one call. n8n's adapt loop ends
    # with a "Re-generate Speech" call carrying an empty segment_ids list once
    # every off-target line came back identical; that call legitimately does no
    # work, and writing ITS counters here erased the record - a job in which
    # XTTS cloned all 32 lines reported `models_used: {}`.
    manifest = {
        "job_id": job_id,
        "target_language": target,
        "models_used": _tally(segments, "tts_model"),
        "failures": [
            {"segment_id": s["segment_id"], "error": s.get("error")}
            for s in segments if s.get("sync_action") == "failed"
        ],
        "segments": [
            {
                "segment_id": s["segment_id"],
                "speaker_id": s["speaker_id"],
                "start": s["start"],
                "end": s["end"],
                "original_duration": round(s["end"] - s["start"], 3),
                "generated_duration": s.get("generated_duration"),
                "duration_ratio": s.get("duration_ratio"),
                "sync_action": s.get("sync_action"),
                "generated_audio": s.get("generated_audio"),
                "tts_model": s.get("tts_model"),
            }
            for s in segments
        ],
    }
    manifest_key = ws.push_json(manifest, ws.layout.synthesis_manifest)

    # The response, by contrast, reports what THIS call did - that is what the
    # workflow branches on.
    return {
        "segments_key": out_key,
        "manifest_key": manifest_key,
        "synthesized": len(produced),
        "failed": len(failures),
        "models_used": models_used,
        "needs_adaptation": needs_adaptation,
        "within_tolerance": len([s for s in segments if s.get("sync_action") == "accept"]),
        "ratio_stats": _ratio_stats(ratios),
    }


# ---------------------------------------------------------------- internals --
def _synthesize_segment(ws: JobWorkspace, seg: dict, target: str,
                        references: dict[str, Path | None],
                        prefer_voice_clone: bool | None, force_model: str | None,
                        models_used: dict[str, int],
                        pitch: dict[str, float] | None = None) -> None:
    reference = references.get(seg["speaker_id"])
    attempt = seg.get("attempts", 0)
    out_path = ws.path("generated", f"segment_{seg['segment_id']:04d}_v{attempt}.wav")

    chain = router.resolve(
        target, has_voice_reference=reference is not None,
        prefer_voice_clone=prefer_voice_clone, force_model=force_model,
    )

    original = max(0.2, seg["end"] - seg["start"])
    result = _run_chain(chain, seg, target, reference, out_path, speed=1.0)
    result = _trim_padding(result)
    ratio = result.duration / original

    # A take that overruns its slot has three possible remedies, in increasing
    # order of damage: say it faster, stretch it afterwards, or re-translate it
    # shorter. Ask the engine for a faster read FIRST - it is the only one that
    # costs nothing in quality, and for MMS-TTS it is usually enough on its own.
    if settings.tts_speed_adaptation and ratio > settings.sync_ratio_accept_max:
        wanted = _clamp(ratio, 1.0, settings.tts_max_speaking_rate)
        faster_path = out_path.with_name(f"{out_path.stem}_fast.wav")
        try:
            faster = _trim_padding(
                _run_chain(chain, seg, target, reference, faster_path, speed=wanted)
            )
        except Exception as exc:  # noqa: BLE001 - keep the take we already have
            log.debug("faster re-read failed on segment %s: %s", seg["segment_id"], exc)
        else:
            if abs(faster.duration / original - 1.0) < abs(ratio - 1.0):
                log.debug("segment %s re-read at %.2fx: %.2fs -> %.2fs",
                          seg["segment_id"], wanted, result.duration, faster.duration)
                result = faster
                ratio = result.duration / original
                seg["speaking_rate"] = round(wanted, 3)

    # Give this speaker their own voice when the engine could not. Skip it
    # when the engine already did: Edge-TTS casts SPEAKER_00 as a woman and
    # SPEAKER_01 as a man, and shifting that male read down another 2.5
    # semitones buys no separation and costs naturalness.
    shift = (pitch or {}).get(seg["speaker_id"], 0.0)
    if shift and not result.voice_cloned and not result.distinct_voice:
        shifted = out_path.with_name(f"{out_path.stem}_pitch.wav")
        try:
            ffmpeg.pitch_shift(result.path, shifted, shift, result.sample_rate)
        except Exception as exc:  # noqa: BLE001 - a shared voice beats no voice
            log.warning("pitch shift failed for %s: %s", seg["speaker_id"], exc)
        else:
            result = SynthesisResult(
                path=shifted, duration=ffmpeg.duration_of(shifted), model=result.model,
                voice_cloned=result.voice_cloned, sample_rate=result.sample_rate,
                distinct_voice=result.distinct_voice,
            )
            seg["speaker_pitch"] = shift

    key = ws.push(result.path, ws.layout.generated_segment(seg["segment_id"], attempt))
    seg["generated_audio"] = key
    seg["generated_duration"] = round(result.duration, 3)
    seg["duration_ratio"] = round(ratio, 4)
    seg["tts_model"] = result.model
    seg["voice_cloned"] = result.voice_cloned
    seg["sync_action"] = classify(
        ratio,
        settings.sync_ratio_accept_min, settings.sync_ratio_accept_max,
        settings.sync_ratio_stretch_min, settings.sync_ratio_stretch_max,
    )
    if (seg["sync_action"] == "retranslate"
            and seg.get("attempts", 0) >= settings.sync_max_retranslate_attempts):
        # Out of retries: fall back to an aggressive stretch instead of looping.
        seg["sync_action"] = "stretch"
        seg["forced_stretch"] = True

    models_used[result.model] = models_used.get(result.model, 0) + 1
    log.debug("segment %s: %.2fs vs %.2fs (ratio %.2f) -> %s",
              seg["segment_id"], result.duration, original, ratio, seg["sync_action"])




#: Consecutive failures tolerated before we conclude the engine cannot speak
#: this language at all and stop wasting time on the remaining segments.
_FAIL_FAST_AFTER = 3



def _tally(segments: list[dict], field: str) -> dict[str, int]:
    """Count a field's values across every segment that produced audio."""
    counts: dict[str, int] = {}
    for seg in segments:
        value = seg.get(field)
        if value and seg.get("generated_audio"):
            counts[value] = counts.get(value, 0) + 1
    return counts


def _describe(exc: Exception) -> str:
    """Flatten a PipelineError (message + per-engine attempts) into one line."""
    details = getattr(exc, "details", None) or {}
    attempts = details.get("attempts")
    if attempts:
        return f"{exc}; tried -> " + " | ".join(str(a) for a in attempts)
    return str(exc)


def _distinct(failures: list[dict]) -> list[str]:
    """One entry per distinct cause - 32 copies of one error help nobody."""
    seen: dict[str, int] = {}
    for item in failures:
        seen[item["error"]] = seen.get(item["error"], 0) + 1
    return [f"{count}x {error}" for error, count in seen.items()]


def _language_hint(target: str) -> str:
    """Point at the actual remedy for the languages that need extra packages."""
    extras = {
        "ja": "coqui-tts needs its Japanese tokenizer: pip install 'coqui-tts[ja]' "
              "(cutlet + fugashi). MMS-TTS has no 'jpn' checkpoint, so XTTS is the "
              "only engine for Japanese.",
        "zh": "coqui-tts needs its Chinese tokenizer: pip install 'coqui-tts[zh]' "
              "(pypinyin + spacy-pkuseg).",
        "ko": "coqui-tts needs its Korean g2p: pip install 'coqui-tts[ko]'.",
        "bn": "coqui-tts needs its Bengali helpers: pip install 'coqui-tts[bn]'.",
    }
    return extras.get(
        target,
        "Check that a TTS engine covers this language: XTTS-v2 speaks 17 languages "
        "and needs a voice reference; MMS-TTS needs a facebook/mms-tts-<iso3> "
        "checkpoint to exist. GET /languages reports the voice_cloning flag.",
    )


def _run_chain(chain, seg: dict, target: str, reference: Path | None,  # noqa: ANN001
               out_path: Path, speed: float) -> SynthesisResult:
    """First adapter in the chain that can speak this line wins.

    On total failure the message names EVERY engine that was tried and why.
    Reporting only the last error is actively misleading: for Japanese the
    chain is [xtts, mms], XTTS fails on a missing tokenizer and MMS fails
    because facebook/mms-tts-jpn does not exist - and only the MMS error was
    ever shown, which points at the wrong engine entirely.
    """
    errors: list[str] = []
    for adapter in chain:
        try:
            return adapter.synthesize(SynthesisRequest(
                text=seg["translated_text"], language=target, output_path=out_path,
                voice_reference=reference, speaker_id=seg["speaker_id"], speed=speed,
            ))
        except Exception as exc:  # noqa: BLE001 - try the next adapter
            log.warning("tts adapter %s failed on segment %s: %s",
                        adapter.name, seg["segment_id"], exc)
            errors.append(f"{adapter.name}: {exc}")
    raise SynthesisError(
        f"All TTS engines failed for segment {seg['segment_id']}",
        details={"attempts": errors, "language": target},
    )


def _trim_padding(result: SynthesisResult) -> SynthesisResult:
    """Drop the dead air TTS engines wrap every take in. See ffmpeg.trim_silence."""
    if not settings.tts_trim_silence:
        return result
    trimmed = result.path.with_name(f"{result.path.stem}_trim.wav")
    try:
        ffmpeg.trim_silence(result.path, trimmed,
                            threshold_db=settings.tts_trim_threshold_db,
                            keep_ms=settings.tts_trim_keep_ms)
        duration = ffmpeg.duration_of(trimmed)
    except Exception as exc:  # noqa: BLE001 - padding is not worth failing over
        log.warning("could not trim silence from %s: %s", result.path.name, exc)
        return result
    if duration <= 0.05:
        # The whole take was below the threshold: keep the original rather than
        # hand an empty file to the synchroniser.
        return result
    return SynthesisResult(path=trimmed, duration=duration, model=result.model,
                           voice_cloned=result.voice_cloned,
                           sample_rate=result.sample_rate,
                           distinct_voice=result.distinct_voice)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))



def _speaker_pitch(document: dict) -> dict[str, float]:
    """Assign each speaker a semitone offset, ordered by screen time.

    The busiest speaker keeps the natural voice (offset 0) so the bulk of the
    dub is untouched; the others are nudged apart. Deterministic, so re-running
    a job does not reshuffle who sounds like whom.
    """
    if not settings.tts_speaker_pitch_enabled:
        return {}
    speakers = document.get("speakers") or []
    if len(speakers) < 2:
        return {}
    steps = settings.speaker_pitch_steps
    ordered = sorted(speakers, key=lambda s: -(s.get("total_seconds") or 0))
    return {s["speaker_id"]: steps[i % len(steps)] for i, s in enumerate(ordered)}


def _voice_references(ws: JobWorkspace, document: dict) -> dict[str, Path | None]:
    refs: dict[str, Path | None] = {}
    for speaker in document.get("speakers", []):
        key = speaker.get("voice_reference")
        sid = speaker["speaker_id"]
        if not key:
            refs[sid] = None
            continue
        try:
            refs[sid] = ws.pull(key)
        except Exception as exc:  # noqa: BLE001
            log.warning("cannot fetch voice reference for %s: %s", sid, exc)
            refs[sid] = None
    return refs


def _ratio_stats(ratios: list[float]) -> dict[str, float]:
    if not ratios:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(ratios),
        "mean": round(statistics.fmean(ratios), 4),
        "median": round(statistics.median(ratios), 4),
        "min": round(min(ratios), 4),
        "max": round(max(ratios), 4),
    }
