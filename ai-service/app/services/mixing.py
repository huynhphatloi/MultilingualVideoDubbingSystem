"""Stage 9: mix dubbed speech back under the preserved background stem."""
from __future__ import annotations

import logging

from app.core.config import settings
from app.services import ffmpeg
from app.services.workspace import JobWorkspace

log = logging.getLogger(__name__)

#: Below this the "background" carries no audible content at all.
_SILENT_DBFS = -60.0


def mix(job_id: str, dubbed_track_key: str, background_key: str | None = None,
        background_gain_db: float | None = None, speech_gain_db: float | None = None,
        loudnorm: bool | None = None, duck: bool | None = None) -> dict:
    ws = JobWorkspace(job_id)
    dubbed = ws.pull(dubbed_track_key)

    speech_gain = settings.speech_gain_db if speech_gain_db is None else speech_gain_db
    bg_gain = settings.background_gain_db if background_gain_db is None else background_gain_db

    # TTS engines disagree wildly about output level (MMS-TTS lands ~10 dB below
    # XTTS), and a quiet dub under a loud score is the single most common way
    # for a "finished" job to be unlistenable. Level the voice first, then let
    # the gains above be a deliberate creative choice rather than a lottery.
    speech_peak = ffmpeg.peak_dbfs(dubbed)
    auto_gain_db = 0.0
    if settings.speech_auto_gain and speech_peak is not None and speech_peak > -60:
        auto_gain_db = _clamp(
            settings.speech_target_peak_dbfs - speech_peak,
            -settings.speech_auto_gain_limit_db, settings.speech_auto_gain_limit_db,
        )
        log.info("dubbed track peaks at %.1f dBFS -> auto gain %+.1f dB",
                 speech_peak, auto_gain_db)

    inputs: list[tuple[str, float]] = [(str(dubbed), speech_gain + auto_gain_db)]
    background_used = False

    key = background_key or ws.layout.background_audio
    background = None
    background_reason = None
    try:
        background = ws.pull(key)
        peak = ffmpeg.peak_dbfs(background)
        if ffmpeg.duration_of(background) <= 0.1:
            background_reason = "stem is empty"
        elif peak is not None and peak < _SILENT_DBFS:
            # A stem can exist, have the right duration, and still be digital
            # silence. Reporting "background_used: true" for that is a lie the
            # UI and the tests both believed.
            background_reason = f"stem is silent ({peak:.0f} dBFS)"
        else:
            inputs.append((str(background), bg_gain))
            background_used = True
    except Exception as exc:  # noqa: BLE001
        background_reason = str(exc)

    if not background_used:
        log.warning("no usable background stem (%s) - the dub will be dry",
                    background_reason)

    # Match the background's layout so the original soundtrack keeps its stereo
    # image; the dubbed speech is mono and gets duplicated across the channels.
    layout = "stereo"
    if background_used:
        channels = (ffmpeg.media_info(background).get("audio") or {}).get("channels")
        layout = {1: "mono", 2: "stereo", 6: "5.1"}.get(channels, "stereo")

    out = ws.path("audio", "final_mix.wav")
    use_loudnorm = settings.loudnorm_enabled if loudnorm is None else loudnorm
    use_duck = (settings.mix_duck_enabled if duck is None else duck) and background_used
    ffmpeg.mix(inputs, out, sample_rate=settings.sync_sample_rate,
               duration_source="longest", loudnorm=use_loudnorm,
               channel_layout=layout, duck=use_duck)

    final_key = ws.push(out, ws.layout.final_audio)
    log.info("mixed %d track(s) as %s, background=%s, duck=%s, loudnorm=%s",
             len(inputs), layout, background_used, use_duck, use_loudnorm)
    return {
        "final_audio_key": final_key,
        "background_used": background_used,
        "background_reason": background_reason,
        "channel_layout": layout,
        "loudnorm_applied": use_loudnorm,
        "ducking_applied": use_duck,
        "background_gain_db": bg_gain,
        "speech_gain_db": speech_gain,
        "speech_auto_gain_db": round(auto_gain_db, 2),
        "speech_peak_dbfs": speech_peak,
    }


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
