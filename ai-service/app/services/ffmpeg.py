"""Thin, well-behaved wrapper around the ffmpeg/ffprobe binaries.

Everything media-related in this project goes through here so that failures
surface as a single typed error (``FFmpegError``) with the real stderr tail.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

from app.core.config import settings
from app.core.errors import FFmpegError

log = logging.getLogger(__name__)

_STDERR_TAIL = 4000


def _run(cmd: list[str], timeout: int = 3600) -> subprocess.CompletedProcess:
    log.debug("exec: %s", " ".join(cmd[:24]))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise FFmpegError(f"Binary not found: {cmd[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError(f"Command timed out after {timeout}s: {cmd[0]}") from exc
    if proc.returncode != 0:
        raise FFmpegError(
            f"{Path(cmd[0]).name} exited with code {proc.returncode}",
            details={"stderr": (proc.stderr or "")[-_STDERR_TAIL:], "cmd": cmd[:24]},
        )
    return proc


def ensure_available() -> None:
    for binary in (settings.ffmpeg_bin, settings.ffprobe_bin):
        if shutil.which(binary) is None:
            raise FFmpegError(f"'{binary}' is not on PATH inside the container")


# ------------------------------------------------------------------ probe ---
def probe(path: str | Path) -> dict:
    proc = _run([
        settings.ffprobe_bin, "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ], timeout=120)
    return json.loads(proc.stdout)


def media_info(path: str | Path) -> dict:
    """Normalised summary: duration, streams, codecs."""
    data = probe(path)
    fmt = data.get("format", {})
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    def _f(value, default=0.0) -> float:  # noqa: ANN001
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    duration = _f(fmt.get("duration"))
    if not duration and video:
        duration = _f(video.get("duration"))

    return {
        "duration": duration,
        "size_bytes": int(fmt.get("size") or 0),
        "format_name": fmt.get("format_name"),
        "has_video": video is not None,
        "has_audio": audio is not None,
        "video": {
            "codec": video.get("codec_name"),
            "width": video.get("width"),
            "height": video.get("height"),
            "fps": _fps(video.get("r_frame_rate")),
        } if video else None,
        "audio": {
            "codec": audio.get("codec_name"),
            "sample_rate": int(audio.get("sample_rate") or 0),
            "channels": int(audio.get("channels") or 0),
        } if audio else None,
    }


def _fps(rate: str | None) -> float | None:
    if not rate or "/" not in rate:
        return None
    num, den = rate.split("/", 1)
    try:
        den_f = float(den)
        return round(float(num) / den_f, 3) if den_f else None
    except ValueError:
        return None


def duration_of(path: str | Path) -> float:
    return media_info(path)["duration"]


def peak_dbfs(path: str | Path) -> float | None:
    """Return peak level in dBFS - used to flag 'audio too quiet' edge cases."""
    try:
        proc = subprocess.run(
            [settings.ffmpeg_bin, "-hide_banner", "-i", str(path),
             "-af", "volumedetect", "-f", "null", "-"],
            capture_output=True, text=True, timeout=600,
        )
    except Exception:  # pragma: no cover
        return None
    for line in (proc.stderr or "").splitlines():
        if "max_volume:" in line:
            try:
                return float(line.split("max_volume:")[1].strip().split(" ")[0])
            except (IndexError, ValueError):
                return None
    return None


# ----------------------------------------------------------------- audio ----
def extract_audio(video: str | Path, out_wav: str | Path,
                  sample_rate: int = 16000, channels: int = 1) -> Path:
    out = Path(out_wav)
    out.parent.mkdir(parents=True, exist_ok=True)
    _run([
        settings.ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video), "-vn", "-sn", "-dn",
        "-acodec", "pcm_s16le", "-ar", str(sample_rate), "-ac", str(channels),
        str(out),
    ])
    return out


def resample(src: str | Path, dst: str | Path, sample_rate: int, channels: int = 1) -> Path:
    out = Path(dst)
    out.parent.mkdir(parents=True, exist_ok=True)
    _run([
        settings.ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src), "-acodec", "pcm_s16le",
        "-ar", str(sample_rate), "-ac", str(channels), str(out),
    ])
    return out


def to_opus(src: str | Path, dst: str | Path, kbps: int = 24,
            sample_rate: int = 16000) -> Path:
    """Compress speech for upload to remote ASR.

    Ten minutes of film is 19 MB as 16 kHz wav and about 1.8 MB at 24 kbps
    Opus. Whisper's accuracy on speech is unaffected at this rate - it resamples
    to 16 kHz mono internally anyway - so the tenfold saving is free. Anything
    lower starts eating consonants, which costs word error where it hurts most.
    """
    _run([
        "ffmpeg", "-y", "-i", str(src),
        "-ac", "1", "-ar", str(sample_rate),
        "-c:a", "libopus", "-b:a", f"{kbps}k", "-application", "voip",
        str(dst),
    ])
    return Path(dst)


def apply_gain(src: str | Path, dst: str | Path, gain_db: float,
               sample_rate: int = 48000, channels: int = 2) -> Path:
    """Copy an audio file at a different level (used for voice-over ducking)."""
    out = Path(dst)
    out.parent.mkdir(parents=True, exist_ok=True)
    _run([
        settings.ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src), "-af", f"volume={gain_db:.2f}dB",
        "-ar", str(sample_rate), "-ac", str(channels),
        "-acodec", "pcm_s16le", str(out),
    ])
    return out


def slice_audio(src: str | Path, dst: str | Path, start: float, end: float,
                sample_rate: int | None = None) -> Path:
    out = Path(dst)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        settings.ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{max(0.0, start):.3f}", "-to", f"{max(start + 0.05, end):.3f}",
        "-i", str(src), "-acodec", "pcm_s16le",
    ]
    if sample_rate:
        cmd += ["-ar", str(sample_rate)]
    cmd.append(str(out))
    _run(cmd)
    return out


def concat_audio(parts: list[str | Path], dst: str | Path,
                 sample_rate: int = 24000) -> Path:
    """Concatenate wavs (used to build a speaker voice reference)."""
    out = Path(dst)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not parts:
        raise FFmpegError("concat_audio called with no parts")
    if len(parts) == 1:
        return resample(parts[0], out, sample_rate)

    listfile = out.parent / f"{out.stem}_concat.txt"
    listfile.write_text(
        "\n".join(f"file '{Path(p).resolve()}'" for p in parts), encoding="utf-8"
    )
    _run([
        settings.ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(listfile),
        "-acodec", "pcm_s16le", "-ar", str(sample_rate), "-ac", "1", str(out),
    ])
    listfile.unlink(missing_ok=True)
    return out


def silence(duration: float, dst: str | Path, sample_rate: int = 48000,
            channels: int = 2) -> Path:
    out = Path(dst)
    out.parent.mkdir(parents=True, exist_ok=True)
    layout = "mono" if channels == 1 else "stereo"
    _run([
        settings.ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"anullsrc=r={sample_rate}:cl={layout}",
        "-t", f"{max(0.01, duration):.3f}", "-acodec", "pcm_s16le", str(out),
    ])
    return out


def atempo_chain(factor: float) -> str:
    """Build an ffmpeg atempo filter chain.

    ``atempo`` only accepts 0.5..2.0 per instance, so extreme factors are
    decomposed into a product of in-range steps.
    """
    factor = max(0.25, min(4.0, factor))
    steps: list[float] = []
    remaining = factor
    while remaining > 2.0:
        steps.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        steps.append(0.5)
        remaining /= 0.5
    steps.append(remaining)
    return ",".join(f"atempo={s:.6f}" for s in steps)


def time_stretch(src: str | Path, dst: str | Path, factor: float) -> Path:
    """Change speed WITHOUT changing pitch. factor > 1 => faster/shorter."""
    out = Path(dst)
    out.parent.mkdir(parents=True, exist_ok=True)
    _run([
        settings.ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src), "-filter:a", atempo_chain(factor),
        "-acodec", "pcm_s16le", str(out),
    ])
    return out


def trim_silence(src: str | Path, dst: str | Path, threshold_db: float = -45.0,
                 keep_ms: int = 40) -> Path:
    """Strip the leading and trailing silence a TTS engine pads its takes with.

    Every MMS-TTS (VITS) take arrives with ~0.41 s of silence in front and
    ~0.28 s behind - roughly 0.7 s of nothing on a line whose slot in the film
    may only be 0.9 s long. Left in place it is counted as speech by the
    duration ratio, which then demands a time-stretch that damages the actual
    words. Trimming first is free and makes every downstream decision honest.

    ``keep_ms`` of silence is deliberately left at each end so consonants are
    not clipped and takes do not start abruptly.
    """
    out = Path(dst)
    out.parent.mkdir(parents=True, exist_ok=True)
    keep = max(0.0, keep_ms / 1000.0)
    one_end = (f"silenceremove=start_periods=1:start_silence={keep:.3f}"
               f":start_threshold={threshold_db:.1f}dB:detection=peak")
    _run([
        settings.ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-af", f"{one_end},areverse,{one_end},areverse",
        "-acodec", "pcm_s16le", str(out),
    ])
    return out



def pitch_shift(src: str | Path, dst: str | Path, semitones: float,
                sample_rate: int = 24000) -> Path:
    """Shift pitch WITHOUT changing duration. Positive = higher voice.

    Used to tell speakers apart when the TTS engine cannot clone: MMS-TTS ships
    exactly one voice per language, so a three-hander comes out as one person
    talking to themselves. A couple of semitones is enough for a listener to
    track who is speaking and is small enough to still sound human.

    There is no `rubberband` in a stock ffmpeg build, so this is the classic
    resample trick: play the samples at a different rate (which moves pitch AND
    duration), then undo the duration change with atempo.
    """
    out = Path(dst)
    out.parent.mkdir(parents=True, exist_ok=True)
    if abs(semitones) < 0.01:
        return resample(src, out, sample_rate, channels=1)

    ratio = 2.0 ** (semitones / 12.0)
    _run([
        settings.ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        # asetrate replays the samples `ratio` times faster: pitch goes UP by
        # `ratio` and duration goes DOWN by the same factor. atempo then has to
        # undo the duration change, so it takes the RECIPROCAL - using `ratio`
        # here squares the effect and leaves the take the wrong length.
        "-af", (f"asetrate={int(round(sample_rate * ratio))},"
                f"aresample={sample_rate},"
                f"{atempo_chain(1.0 / ratio)}"),
        "-ar", str(sample_rate), "-ac", "1", "-acodec", "pcm_s16le", str(out),
    ])
    return out


def mix(inputs: list[tuple[str | Path, float]], dst: str | Path,
        sample_rate: int = 48000, duration_source: str = "longest",
        loudnorm: bool | None = None, channel_layout: str = "stereo",
        duck: bool | None = None) -> Path:
    """Mix N tracks with per-track gain in dB. ``inputs[0]`` is the dubbed voice.

    Every input is forced to ``channel_layout`` *before* amix. Without that,
    amix adopts the layout of the first input: the dubbed speech track is mono
    by design, so it would silently fold the preserved stereo background down
    to mono and throw away the original soundtrack's stereo image.

    With exactly two inputs (voice + background) and ``duck`` enabled, the
    background is side-chain compressed by the voice: it drops out of the way
    while a line is spoken and comes straight back afterwards. A flat
    ``background_gain_db`` cannot do both - either the score is too loud to
    understand the dub, or it is turned down for the whole film.
    """
    out = Path(dst)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not inputs:
        raise FFmpegError("mix called with no inputs")

    cmd = [settings.ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error"]
    for path, _ in inputs:
        cmd += ["-i", str(path)]

    parts = []
    for idx, (_, gain_db) in enumerate(inputs):
        parts.append(
            f"[{idx}:a]volume={gain_db:.2f}dB,aresample={sample_rate},"
            f"aformat=sample_fmts=fltp:channel_layouts={channel_layout}[a{idx}]"
        )

    use_duck = settings.mix_duck_enabled if duck is None else duck
    if use_duck and len(inputs) == 2:
        # [a0] is the dubbed voice: one copy is mixed, one drives the compressor.
        parts.append("[a0]asplit=2[voice][key]")
        parts.append(
            f"[a1][key]sidechaincompress="
            f"threshold={settings.mix_duck_threshold}:ratio={settings.mix_duck_ratio}"
            f":attack={settings.mix_duck_attack_ms}:release={settings.mix_duck_release_ms}"
            f":makeup=1:level_sc=1:mix={settings.mix_duck_mix}[ducked]"
        )
        labels = "[voice][ducked]"
    else:
        labels = "".join(f"[a{i}]" for i in range(len(inputs)))

    parts.append(
        f"{labels}amix=inputs={len(inputs)}:duration={duration_source}"
        f":dropout_transition=0:normalize=0[mixed]"
    )

    final_label = "mixed"
    use_loudnorm = settings.loudnorm_enabled if loudnorm is None else loudnorm
    if use_loudnorm:
        parts.append(
            f"[mixed]loudnorm=I={settings.loudnorm_i}:TP={settings.loudnorm_tp}"
            f":LRA={settings.loudnorm_lra}[out]"
        )
        final_label = "out"

    cmd += [
        "-filter_complex", ";".join(parts),
        "-map", f"[{final_label}]",
        "-ar", str(sample_rate), "-acodec", "pcm_s16le", str(out),
    ]
    _run(cmd)
    return out


#: amix degrades badly past a few dozen inputs, so long videos are mixed in
#: passes of this size instead of one gigantic filter graph.
_OVERLAY_BATCH = 24


def overlay_segments(base: str | Path, overlays: list[tuple[str | Path, float]],
                     dst: str | Path, sample_rate: int = 48000) -> Path:
    """Place each overlay wav at an absolute offset (seconds) on top of ``base``.

    This is the 'audio canvas' step: the dubbed segments keep the timing of the
    original speech instead of being concatenated back to back.
    """
    out = Path(dst)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not overlays:
        return resample(base, out, sample_rate, channels=1)

    current = Path(base)
    batches = [overlays[i:i + _OVERLAY_BATCH] for i in range(0, len(overlays), _OVERLAY_BATCH)]
    for index, batch in enumerate(batches):
        target = out if index == len(batches) - 1 else out.with_name(f"{out.stem}_p{index}.wav")
        _overlay_batch(current, batch, target, sample_rate)
        if index > 0 and current != Path(base):
            current.unlink(missing_ok=True)
        current = target
    return out


def _overlay_batch(base: Path, overlays: list[tuple[str | Path, float]],
                   dst: Path, sample_rate: int) -> Path:
    cmd = [settings.ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-i", str(base)]
    for path, _ in overlays:
        cmd += ["-i", str(path)]

    parts = [f"[0:a]aresample={sample_rate},aformat=sample_fmts=fltp[base]"]
    labels = ["[base]"]
    for idx, (_, offset) in enumerate(overlays, start=1):
        delay_ms = max(0, int(round(offset * 1000)))
        parts.append(
            f"[{idx}:a]aresample={sample_rate},aformat=sample_fmts=fltp,"
            f"adelay={delay_ms}|{delay_ms}[s{idx}]"
        )
        labels.append(f"[s{idx}]")

    parts.append(
        f"{''.join(labels)}amix=inputs={len(labels)}:duration=first"
        f":dropout_transition=0:normalize=0[out]"
    )
    cmd += [
        "-filter_complex", ";".join(parts), "-map", "[out]",
        "-ar", str(sample_rate), "-acodec", "pcm_s16le", str(dst),
    ]
    _run(cmd)
    return dst


# ----------------------------------------------------------------- video ----
def mux(video: str | Path, audio: str | Path, dst: str | Path,
        subtitle: str | Path | None = None, burn: bool = False,
        language: str | None = None) -> Path:
    """Replace the audio track of ``video``. Optionally burn in subtitles.

    The output is pinned to the SOURCE VIDEO's duration with ``-t``.

    It used to use ``-shortest`` instead, and ``-shortest`` looks at every
    stream - including the embedded subtitle track. A subtitle track ends at
    its last cue, so a 20 s clip whose last line of dialogue is at 17.5 s came
    out as a 17.5 s video: the ending was simply cut off, and the shorter the
    dialogue relative to the clip, the more was lost.
    """
    out = Path(dst)
    out.parent.mkdir(parents=True, exist_ok=True)

    # ffmpeg requires EVERY -i before the first output option. Declaring the
    # subtitle input after -map made the whole stream-copy command invalid, so
    # render() silently fell back to a full re-encode on every single job.
    cmd = [settings.ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
           "-i", str(video), "-i", str(audio)]
    embed_subtitle = bool(subtitle) and not burn
    if embed_subtitle:
        cmd += ["-i", str(subtitle)]

    cmd += ["-map", "0:v:0", "-map", "1:a:0"]

    if burn and subtitle:
        escaped = str(subtitle).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
        cmd += [
            "-vf", f"subtitles='{escaped}':force_style='FontSize=18,OutlineColour=&H80000000'",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        ]
    else:
        cmd += ["-c:v", "copy"]
        if embed_subtitle:
            cmd += ["-map", "2:0", "-c:s", "mov_text",
                    "-metadata:s:s:0", f"language={_iso639_2(language)}"]

    cmd += ["-c:a", "aac", "-b:a", "192k"]

    video_duration = duration_of(video)
    if video_duration > 0:
        cmd += ["-t", f"{video_duration:.3f}"]

    cmd += ["-movflags", "+faststart", str(out)]
    _run(cmd)
    return out


#: ISO-639-1 -> the ISO-639-2/B code mp4 metadata wants. Anything unknown stays
#: "und", which is what the container spec asks for.
_ISO639_2 = {
    "en": "eng", "vi": "vie", "ja": "jpn", "ko": "kor", "zh": "chi", "fr": "fre",
    "de": "ger", "es": "spa", "pt": "por", "it": "ita", "ru": "rus", "nl": "dut",
    "pl": "pol", "tr": "tur", "ar": "ara", "hi": "hin", "id": "ind", "th": "tha",
    "cs": "cze", "hu": "hun", "uk": "ukr", "ro": "rum", "sv": "swe", "da": "dan",
    "fi": "fin", "no": "nor", "el": "gre", "he": "heb", "ms": "may", "fa": "per",
}


def _iso639_2(language: str | None) -> str:
    return _ISO639_2.get((language or "").lower(), "und")
