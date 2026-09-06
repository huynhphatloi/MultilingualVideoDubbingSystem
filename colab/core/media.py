"""ffmpeg and ffprobe helpers."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence

from dubflow_core.mixing import MIX_SAMPLE_RATE, atempo_chain, atempo_filter

from .errors import ProviderFailure

ASR_SAMPLE_RATE = 16000
SPEECH_SAMPLE_RATE = 24000

__all__ = [
    "ASR_SAMPLE_RATE", "SPEECH_SAMPLE_RATE", "MIX_SAMPLE_RATE", "Scratch",
    "concatenate", "duration", "extract_window", "ffmpeg", "normalise_speech",
    "retime", "run", "sample_rate", "write_waveform",
]


def run(command: Sequence[str]) -> str:
    result = subprocess.run(list(command), capture_output=True, text=True)
    if result.returncode:
        detail = (result.stderr or result.stdout or "command failed").strip()[-2000:]
        raise ProviderFailure(f"{command[0]} failed: {detail}")
    return result.stdout


def ffmpeg(arguments: Sequence[str]) -> None:
    run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *arguments])


def duration(path: Path) -> float:
    output = run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ])
    try:
        return float(output.strip())
    except ValueError as exc:
        raise ProviderFailure(f"ffprobe could not measure {path.name}") from exc


def sample_rate(path: Path) -> int:
    output = run([
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ])
    try:
        return int(output.strip().splitlines()[0])
    except (ValueError, IndexError) as exc:
        raise ProviderFailure(f"ffprobe found no audio stream in {path.name}") from exc


def normalise_speech(
    source: Path,
    target: Path,
    speed: float = 1.0,
    rate: int = SPEECH_SAMPLE_RATE,
) -> None:
    filters: List[str] = []
    if abs(speed - 1.0) > 1e-3:
        filters.extend(atempo_chain(speed))
    filters.extend([f"aresample={rate}", "aformat=channel_layouts=mono"])
    ffmpeg([
        "-i", str(source), "-vn", "-af", ",".join(filters),
        "-c:a", "pcm_s16le", str(target),
    ])


def retime(source: Path, target: Path, speed: float) -> None:
    ffmpeg([
        "-i", str(source), "-af", atempo_filter(speed),
        "-c:a", "pcm_s16le", str(target),
    ])


def extract_window(
    source: Path,
    target: Path,
    start: float,
    length: float,
    rate: int = SPEECH_SAMPLE_RATE,
) -> None:
    ffmpeg([
        "-ss", f"{max(0.0, start):.3f}", "-t", f"{max(0.05, length):.3f}",
        "-i", str(source), "-vn", "-ar", str(rate), "-ac", "1",
        "-c:a", "pcm_s16le", str(target),
    ])


def concatenate(pieces: Sequence[Path], target: Path, rate: int = SPEECH_SAMPLE_RATE) -> None:
    if not pieces:
        raise ProviderFailure("Nothing to concatenate")
    if len(pieces) == 1:
        normalise_speech(pieces[0], target, rate=rate)
        return
    inputs: List[str] = []
    for piece in pieces:
        inputs.extend(["-i", str(piece)])
    labels = "".join(f"[{index}:a]" for index in range(len(pieces)))
    ffmpeg([
        *inputs, "-filter_complex",
        f"{labels}concat=n={len(pieces)}:v=0:a=1,aresample={rate},"
        f"aformat=channel_layouts=mono[out]",
        "-map", "[out]", "-c:a", "pcm_s16le", str(target),
    ])


def write_waveform(
    waveform,  # noqa: ANN001 - a numpy array or a torch tensor
    rate: int,
    target: Path,
    speed: float = 1.0,
) -> None:
    import soundfile

    if abs(speed - 1.0) <= 1e-3:
        soundfile.write(str(target), waveform, rate, format="WAV", subtype="PCM_16")
        return
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        raw = Path(handle.name)
    try:
        soundfile.write(str(raw), waveform, rate, format="WAV", subtype="PCM_16")
        normalise_speech(raw, target, speed, rate=rate)
    finally:
        raw.unlink(missing_ok=True)


class Scratch:
    def __init__(self, suffix: str = ".wav") -> None:
        self.suffix = suffix
        self.path: Optional[Path] = None

    def __enter__(self) -> Path:
        with tempfile.NamedTemporaryFile(suffix=self.suffix, delete=False) as handle:
            self.path = Path(handle.name)
        return self.path

    def __exit__(self, *_: object) -> None:
        if self.path is not None:
            self.path.unlink(missing_ok=True)
