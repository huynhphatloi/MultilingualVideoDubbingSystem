"""Stage 8: temporal synchronisation - the heart of the system.

Rules implemented here:

* generated audio is **never** concatenated back to back;
* every take is placed at its ORIGINAL ``start`` on a silent canvas whose
  length equals the video duration;
* only light time-stretching is used (voice quality degrades quickly);
* a take that would run into the next speaker is compressed harder - up to a
  second, higher ceiling - because two overlapping voices sound worse than one
  fast one. If even that is not enough the overlap is accepted and **reported**
  in ``overlapping_segments``; a non-empty list means the adapt/re-translate
  loop upstream did not do its job.
"""
from __future__ import annotations

import logging

from app.core.config import settings
from app.core.errors import InvalidInput
from app.services import ffmpeg
from app.services.translation.duration_aware import stretch_factor
from app.services.workspace import JobWorkspace

log = logging.getLogger(__name__)

#: Compression limit in normal circumstances. Past ~1.35x the voice starts
#: sounding rushed and metallic.
_MAX_EXTRA_STRETCH = 1.35
#: Allowed only to prevent a take from running over the next speaker - two
#: voices at once is a worse artefact than one noticeably fast voice.
_COLLISION_STRETCH_CEILING = 1.6
#: Slow-down limit. A take that is *shorter* than its slot is usually fine - the
#: gap simply becomes a natural pause - so we only ever stretch it a little.
_MIN_EXTRA_STRETCH = 0.90
#: Only bother slowing a take down when it is noticeably short.
_SLOWDOWN_TRIGGER = 0.85
#: Leave a small breath between consecutive utterances.
_GUARD_SECONDS = 0.06


def synchronize(job_id: str, segments_key: str,
                total_duration: float | None = None) -> dict:
    ws = JobWorkspace(job_id)
    document = ws.pull_json(segments_key)
    segments = sorted(document.get("segments", []), key=lambda s: s["start"])

    usable = [s for s in segments if s.get("generated_audio")]
    if not usable:
        raise InvalidInput("No generated audio to synchronise - run synthesis first.",
                           details={"segments_key": segments_key})

    timeline = total_duration or _timeline_duration(ws, segments)
    canvas = ws.path("audio", "canvas.wav")
    ffmpeg.silence(timeline, canvas, sample_rate=settings.sync_sample_rate, channels=1)

    overlays: list[tuple[str, float]] = []
    stretched = 0
    dropped: list[int] = []
    collisions: list[int] = []

    for index, seg in enumerate(usable):
        start = float(seg["start"])
        slot = max(0.2, float(seg["end"]) - start)
        next_start = _next_start(usable, index, timeline)
        window = max(0.2, next_start - start - _GUARD_SECONDS)

        try:
            local = ws.pull(seg["generated_audio"])
        except Exception as exc:  # noqa: BLE001
            log.error("missing generated audio for segment %s: %s", seg["segment_id"], exc)
            dropped.append(seg["segment_id"])
            continue

        duration = ffmpeg.duration_of(local)
        if duration <= 0.02:
            dropped.append(seg["segment_id"])
            continue

        factor = 1.0
        action = seg.get("sync_action", "accept")

        # Compress against the WINDOW - the room before the next take starts -
        # not against the original slot.
        #
        # Squeezing every take back into its slot is what a subtitle would do,
        # and it is far too aggressive for speech: measured on the sample clip,
        # 9 of 18 takes were compressed harder than anything required, one of
        # them 1.32x while 3.26 s of silence sat unused right after it. The
        # result is the rushed, metallic voice that reads as "garbled".
        #
        # Human dubbing does the opposite: start the line on time and let it run
        # into the pause that follows. The take is still anchored at the
        # original `start`, so it stays with the picture; it simply borrows the
        # silence nobody is using.
        if duration > window:
            needed = duration / window
            factor = min(needed, _COLLISION_STRETCH_CEILING)
        elif ((action in ("stretch", "retranslate", "failed") or seg.get("forced_stretch"))
                and duration / slot < _SLOWDOWN_TRIGGER):
            # Clearly shorter than the line it replaces - slow it a little so it
            # does not sound clipped. Never past _MIN_EXTRA_STRETCH.
            factor = _clamp(stretch_factor(duration / slot), _MIN_EXTRA_STRETCH, 1.0)

        factor = _clamp(factor, _MIN_EXTRA_STRETCH, _COLLISION_STRETCH_CEILING)
        if factor > _MAX_EXTRA_STRETCH:
            log.warning("segment %s compressed %.2fx to avoid overlapping the next take",
                        seg["segment_id"], factor)
            seg["compressed_to_avoid_overlap"] = True

        if abs(factor - 1.0) > 0.01:
            stretched_path = ws.path("generated", f"seg_{seg['segment_id']:04d}_sync.wav")
            ffmpeg.time_stretch(local, stretched_path, factor)
            local = stretched_path
            duration = ffmpeg.duration_of(local)
            stretched += 1

        normalised = ws.path("generated", f"seg_{seg['segment_id']:04d}_48k.wav")
        ffmpeg.resample(local, normalised, settings.sync_sample_rate, channels=1)

        overrun = (start + duration) - next_start
        if overrun > 0.15:
            collisions.append({"segment_id": seg["segment_id"],
                               "overrun_seconds": round(overrun, 2)})
            log.warning("segment %s still overruns the next take by %.2fs - the "
                        "translation is too long for its slot; the adapt loop "
                        "should have shortened it",
                        seg["segment_id"], overrun)

        overlays.append((str(normalised), start))
        seg["final_duration"] = round(duration, 3)
        seg["applied_stretch"] = round(factor, 4)
        seg["placed_at"] = round(start, 3)

    if not overlays:
        raise InvalidInput("Every generated segment was unusable.",
                           details={"dropped": dropped})

    dubbed = ws.path("audio", "dubbed_speech.wav")
    ffmpeg.overlay_segments(canvas, overlays, dubbed, sample_rate=settings.sync_sample_rate)
    dubbed_key = ws.push(dubbed, ws.layout.dubbed_track)

    document["segments"] = segments
    document["timeline_duration"] = round(timeline, 3)
    ws.push_json(document, ws.layout.segments)

    log.info("placed %d segments on a %.1fs timeline (%d stretched, %d dropped)",
             len(overlays), timeline, stretched, len(dropped))

    truncated = [
        s["segment_id"] for s in usable
        if s.get("placed_at") is not None
        and s["placed_at"] + (s.get("final_duration") or 0) > timeline + 0.05
    ]
    if truncated:
        log.warning("%d take(s) run past the end of the video and will be cut: %s",
                    len(truncated), truncated)

    return {
        "dubbed_track_key": dubbed_key,
        "stretched_segments": stretched,
        "placed_segments": len(overlays),
        "dropped_segments": dropped,
        "overlapping_segments": collisions,
        "truncated_at_end": truncated,
        "timeline_duration": round(timeline, 3),
    }


def _next_start(segments: list[dict], index: int, timeline: float) -> float:
    if index + 1 < len(segments):
        return float(segments[index + 1]["start"])
    return timeline


def _timeline_duration(ws: JobWorkspace, segments: list[dict]) -> float:
    for key in (ws.layout.background_audio, ws.layout.original_audio):
        try:
            local = ws.pull(key)
            duration = ffmpeg.duration_of(local)
            if duration > 0:
                return duration
        except Exception:  # noqa: BLE001
            continue
    return max((s["end"] for s in segments), default=1.0) + 1.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
