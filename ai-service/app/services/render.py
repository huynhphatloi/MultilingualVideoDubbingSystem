"""Stage 11: final render - mux the dubbed mix back onto the original video."""
from __future__ import annotations

import logging

from app.core import languages
from app.core.config import settings
from app.core.errors import FFmpegError
from app.services import ffmpeg
from app.services.workspace import JobWorkspace

log = logging.getLogger(__name__)


def render(job_id: str, video_key: str, audio_key: str,
           target_language: str | None = None, burn_subtitles: bool = False,
           subtitle_key: str | None = None) -> dict:
    ws = JobWorkspace(job_id)
    video = ws.pull(video_key)
    audio = ws.pull(audio_key)

    lang = languages.normalize(target_language) or (target_language or "out")

    subtitle_path = None
    if subtitle_key:
        try:
            subtitle_path = ws.pull(subtitle_key)
        except Exception as exc:  # noqa: BLE001
            log.warning("subtitle not available for muxing: %s", exc)

    out = ws.path("output", f"dubbed_{lang}.mp4")
    stream_copied = not burn_subtitles
    try:
        ffmpeg.mux(video, audio, out, subtitle=subtitle_path, burn=burn_subtitles,
                   language=lang)
    except FFmpegError as exc:
        # The source codec may not be stream-copyable into MP4. Re-encoding is
        # slow and lossy, so it is a fallback, never the normal path - if you
        # see this on every job, something is wrong with the mux command.
        log.warning("stream-copy mux failed (%s) - retrying with re-encode",
                    exc.message, extra={"details": exc.details})
        _reencode(video, audio, out, subtitle_path, burn_subtitles)
        stream_copied = False

    info = ffmpeg.media_info(out)
    key = ws.push(out, ws.layout.output_video(lang))
    url = ws.storage.url(key)

    log.info("rendered %s (%.1fs, %.1f MB)", key, info["duration"],
             info["size_bytes"] / 1_048_576)

    # Render is the last stage, and everything in scratch is a copy of an object
    # that is already in storage. Re-running a stage later just re-downloads.
    if settings.cleanup_scratch_after_render:
        ws.cleanup()
    return {
        "output_key": key,
        "output_url": url,
        "duration_seconds": round(info["duration"], 3),
        "size_bytes": info["size_bytes"],
        "burned_subtitles": bool(burn_subtitles and subtitle_path),
        "video_stream_copied": stream_copied,
    }


def _reencode(video, audio, out, subtitle, burn: bool) -> None:  # noqa: ANN001
    from app.core.config import settings

    cmd = [settings.ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
           "-i", str(video), "-i", str(audio)]
    if burn and subtitle:
        escaped = str(subtitle).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
        cmd += ["-vf", f"subtitles='{escaped}'"]
    cmd += [
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "medium", "-crf", "21",
        "-c:a", "aac", "-b:a", "192k",
    ]
    # Same rule as ffmpeg.mux(): pin the output to the source video's length
    # instead of letting -shortest pick whichever stream happens to end first.
    duration = ffmpeg.duration_of(video)
    if duration > 0:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-movflags", "+faststart", str(out)]
    ffmpeg._run(cmd)  # noqa: SLF001 - deliberate reuse of the guarded runner
