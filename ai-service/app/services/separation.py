"""Stage 2: speech / background separation with Demucs.

The background stem (music, ambience, sound effects) is preserved and mixed
back under the dubbed voice later - we never throw the original soundtrack away.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from app.core.config import settings
from app.core.device import resolve_device_for
from app.core.errors import SeparationError
from app.services import ffmpeg
from app.services.workspace import JobWorkspace

log = logging.getLogger(__name__)

#: htdemucs stems -> we treat "vocals" as speech and sum the rest as background.
_SPEECH_STEM = "vocals"

#: Wrapper that restores torch's pre-2.6 `torch.load` default so demucs 4.0.1
#: can read its checkpoints. See demucs_runner.py.
_RUNNER = Path(__file__).with_name("demucs_runner.py")


def separate(job_id: str, audio_key: str, model: str | None = None) -> dict:
    ws = JobWorkspace(job_id)
    src = ws.pull(audio_key)

    if not settings.demucs_enabled:
        return _passthrough(ws, src, reason="demucs disabled by configuration")

    model_name = model or settings.demucs_model
    out_dir = ws.subdir("demucs")
    device = resolve_device_for("separation")

    proc = _run_demucs(src, out_dir, model_name, device)

    if proc.returncode != 0 and device != "cpu":
        # Demucs 4.0.x predates solid Metal support and a handful of its ops
        # still fall over on `mps`. A CPU retry is slow but reliable, and far
        # better than failing the whole job at stage 4 of 13.
        log.warning("demucs failed on %s, retrying on cpu", device)
        first_error = (proc.stderr or "")[-2000:]
        shutil.rmtree(out_dir, ignore_errors=True)
        out_dir = ws.subdir("demucs")
        proc = _run_demucs(src, out_dir, model_name, "cpu")
        if proc.returncode == 0:
            log.info("demucs succeeded on cpu after the %s failure", device)
        else:
            raise SeparationError(
                f"Demucs failed on {device} and on cpu",
                details={
                    "device": device,
                    "model": model_name,
                    f"stderr_{device}": first_error,
                    "stderr_cpu": (proc.stderr or "")[-4000:],
                    "hint": "Set SEPARATION_DEVICE=cpu, or DEMUCS_ENABLED=false to skip "
                            "separation entirely (the dub then has no background stem).",
                },
            )
    elif proc.returncode != 0:
        raise SeparationError(
            "Demucs separation failed",
            details={
                "device": device,
                "model": model_name,
                "stderr": (proc.stderr or "")[-4000:],
                "stdout": (proc.stdout or "")[-1000:],
                "hint": f"Reproduce it directly: python {_RUNNER} -n {model_name} "
                        f"--two-stems vocals -d cpu -o /tmp/demucs-check <original.wav>",
            },
        )

    vocals = _find(out_dir, f"{_SPEECH_STEM}.wav")
    accompaniment = _find(out_dir, "no_vocals.wav")
    if vocals is None or accompaniment is None:
        raise SeparationError(
            "Demucs produced no usable stems",
            details={"searched": str(out_dir), "stdout": (proc.stdout or "")[-2000:]},
        )

    speech = ws.path("audio", "speech.wav")
    background = ws.path("audio", "background.wav")
    ffmpeg.resample(vocals, speech, sample_rate=16000, channels=1)
    ffmpeg.resample(accompaniment, background, sample_rate=settings.sync_sample_rate, channels=2)

    speech_key = ws.push(speech, ws.layout.speech_audio)
    background_key = ws.push(background, ws.layout.background_audio)
    shutil.rmtree(out_dir, ignore_errors=True)

    return {
        "speech_key": speech_key,
        "background_key": background_key,
        "model": model_name,
        "separated": True,
    }


def _run_demucs(src: Path, out_dir: Path, model_name: str,
                device: str) -> subprocess.CompletedProcess:
    cmd = [
        # sys.executable, not "python": guarantees the interpreter that is
        # running the API, whatever the PATH of the parent shell looks like.
        # demucs_runner.py, not "-m demucs.separate": demucs 4.0.1 cannot load
        # its own checkpoints under torch >= 2.6 (see that file for the detail).
        sys.executable, str(_RUNNER),
        "-n", model_name,
        "--two-stems", _SPEECH_STEM,
        "-o", str(out_dir),
        "--filename", "{stem}.{ext}",
        "-d", device,
        "--shifts", str(settings.demucs_shifts),
    ]
    if settings.demucs_segment:
        cmd += ["--segment", str(settings.demucs_segment)]
    cmd.append(str(src))

    log.info("running demucs (%s) on %s [device=%s]", model_name, src.name, device)
    env = {**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"}
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=7200, env=env)
    except FileNotFoundError as exc:
        raise SeparationError("demucs is not installed in this environment") from exc
    except subprocess.TimeoutExpired as exc:
        raise SeparationError("demucs timed out (>2h)") from exc


def _passthrough(ws: JobWorkspace, src: Path, reason: str) -> dict:
    """Degrade to voice-over style when separation is unavailable.

    Without Demucs we cannot remove the original dialogue, so there are only
    two honest options:

    * background = silence  -> the dub loses the music, the effects and the
      room tone. The result is a voice floating over nothing. This is what this
      function used to do, and it is worse than it sounds: a 60 s action scene
      came out as a bare voice track.
    * background = the original mix, ducked -> the "lektor" / voice-over style
      used for documentaries: music and effects survive intact, the original
      dialogue stays faintly audible underneath the dub.

    The second loses less, so that is what we do. The trade-off is explicit in
    the return value (``mode: "voiceover"``, ``original_dialogue_present``) so
    the UI can say so rather than pretending this is a full dub.
    """
    log.warning("separation unavailable (%s) - falling back to voice-over: the "
                "original dialogue will remain audible under the dub", reason)

    speech = ws.path("audio", "speech.wav")
    background = ws.path("audio", "background.wav")

    # ASR still gets the full mix; Whisper copes with music behind speech.
    ffmpeg.resample(src, speech, sample_rate=16000, channels=1)

    # Pre-duck so the mixer's normal background gain lands in voice-over range.
    ffmpeg.apply_gain(src, background, settings.voiceover_duck_db,
                      sample_rate=settings.sync_sample_rate, channels=2)

    return {
        "speech_key": ws.push(speech, ws.layout.speech_audio),
        "background_key": ws.push(background, ws.layout.background_audio),
        "model": "none",
        "separated": False,
        "mode": "voiceover",
        "original_dialogue_present": True,
        "reason": reason,
    }


def _find(root: Path, filename: str) -> Path | None:
    matches = sorted(root.rglob(filename))
    return matches[0] if matches else None
