"""Edge-TTS - the naturalness ceiling, for free, with no cloning.

This is Microsoft's production neural TTS reached through the same websocket
the Edge browser's Read Aloud feature uses. No API key, no GPU, no cost. The
Vietnamese voices (HoaiMy, NamMinh) are studio-grade and will almost certainly
beat every open model in this set on a listening test.

It is in the comparison as a *reference line*, not as a candidate, and the
distinction is worth stating in the write-up:

* it cannot clone, so it cannot give each character their own voice - the exact
  defect that makes MMS unusable for multi-speaker dubbing;
* it is a remote closed service, so it is not reproducible, not offline, and
  its terms do not obviously cover redubbing third-party video.

What it gives is a ceiling: "how far is the best open cloning model from
commercial quality" is a much sharper conclusion than "viXTTS sounded fine".

The service returns MP3; the file is transcoded to wav so every engine in the
benchmark is measured on identical material.
"""
from __future__ import annotations

import asyncio
import subprocess
import time
from pathlib import Path

from .tts_base import Spoken, TTSEngine

_VOICES = {
    "vi": "vi-VN-HoaiMyNeural",
    "en": "en-US-AriaNeural",
    "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural",
    "es": "es-ES-ElviraNeural",
    "it": "it-IT-ElsaNeural",
    "pt": "pt-BR-FranciscaNeural",
    "ru": "ru-RU-SvetlanaNeural",
    "ja": "ja-JP-NanamiNeural",
    "ko": "ko-KR-SunHiNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
    "th": "th-TH-PremwadeeNeural",
    "id": "id-ID-GadisNeural",
    "hi": "hi-IN-SwaraNeural",
    "ar": "ar-SA-ZariyahNeural",
    "tr": "tr-TR-EmelNeural",
    "nl": "nl-NL-ColetteNeural",
    "pl": "pl-PL-ZofiaNeural",
}


class EdgeTTS(TTSEngine):
    name = "edge"
    title = "Edge-TTS (Microsoft neural, cloud reference)"
    voice_cloning = False
    languages = tuple(_VOICES)
    vram_gb = 0.0

    def __init__(self) -> None:
        super().__init__()
        self._sr = 24000

    def load(self) -> None:
        import edge_tts  # noqa: F401 - fail here rather than mid-benchmark

    def speak(self, text: str, language: str, out_path: Path,
              reference_wav: Path | None = None, speed: float = 1.0) -> Spoken:
        import edge_tts
        import soundfile as sf

        self.ensure_loaded()
        code = (language or "").lower().split("-")[0]
        if code not in _VOICES:
            raise RuntimeError(f"edge-tts has no voice mapped for '{language}'")

        # The service takes a percentage delta, not a multiplier.
        rate = f"{round((float(speed or 1.0) - 1.0) * 100):+d}%"
        mp3 = out_path.with_suffix(".mp3")

        started = time.perf_counter()
        with self._lock:
            asyncio.run(_stream(edge_tts, text, _VOICES[code], rate, mp3))
        compute = time.perf_counter() - started

        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3),
             "-ar", str(self._sr), "-ac", "1", str(out_path)],
            check=True,
        )
        mp3.unlink(missing_ok=True)

        info = sf.info(str(out_path))
        return Spoken(
            wav_path=out_path, sample_rate=info.samplerate, seconds=info.duration,
            # compute_seconds here is network latency, not GPU time. Reported
            # for completeness, but do NOT put it in an RTF column next to the
            # local engines - it measures the user's connection.
            compute_seconds=compute, voice_cloned=False, engine=self.name,
            notes={"voice": _VOICES[code], "measures": "network latency, not compute"},
        )


async def _stream(edge_tts, text: str, voice: str, rate: str, dest: Path) -> None:  # noqa: ANN001
    communicate = edge_tts.Communicate(text, voice, rate=rate)
    with dest.open("wb") as fh:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                fh.write(chunk["data"])
