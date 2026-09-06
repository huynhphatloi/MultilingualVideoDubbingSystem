"""Microsoft Edge speech generation."""
from __future__ import annotations

import asyncio
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional

from core.errors import ProviderFailure, UnsupportedLanguage
from core.media import Scratch, normalise_speech
from providers.base import ModelSpec
from providers.tts.base import SpeechRequest, TTSProvider

LOCALE_OVERRIDES: Dict[str, str] = {"no": "nb", "tl": "fil"}


def locale_prefix(language: str) -> str:
    return LOCALE_OVERRIDES.get(language, language)


class EdgeProvider(TTSProvider):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(spec)
        import edge_tts

        self.voices: List[dict] = asyncio.run(edge_tts.list_voices())
        self._chosen: Optional[str] = None

    @property
    def name(self) -> str:
        return f"edge-tts/{self._chosen}" if self._chosen else "edge-tts"

    @property
    def selected_voice(self) -> Optional[str]:
        return self._chosen

    def voices_for(self, language: str) -> List[dict]:
        prefix = locale_prefix(language) + "-"
        matches = [
            voice for voice in self.voices
            if str(voice.get("ShortName", "")).lower().startswith(prefix)
        ]
        if not matches:
            raise UnsupportedLanguage(
                f"Edge TTS publishes no voice for '{language}'. Pick another engine."
            )
        return matches

    def voice_for(
        self,
        language: str,
        speaker_id: Optional[str] = None,
    ) -> str:
        matches = self.voices_for(language)
        if speaker_id:
            ordered = sorted(matches, key=lambda voice: str(voice.get("ShortName", "")))
            suffix = re.search(r"(\d+)$", speaker_id)
            if suffix:
                index = int(suffix.group(1))
            else:
                digest = hashlib.sha256(speaker_id.encode("utf-8")).digest()
                index = int.from_bytes(digest[:8], "big")
            return str(ordered[index % len(ordered)]["ShortName"])
        female = [v for v in matches if str(v.get("Gender", "")).lower() == "female"]
        return str((female or matches)[0]["ShortName"])

    def synthesize(self, request: SpeechRequest, out: Path) -> None:
        import edge_tts

        voice = self.voice_for(
            request.language,
            speaker_id=request.speaker_id,
        )
        self._chosen = voice
        percent = int(round((request.speed - 1.0) * 100))
        with Scratch(".mp3") as raw:
            asyncio.run(
                edge_tts.Communicate(request.text, voice, rate=f"{percent:+d}%").save(str(raw))
            )
            if not raw.exists() or raw.stat().st_size == 0:
                raise ProviderFailure("Edge TTS returned no audio")
            normalise_speech(raw, out)


def cache_key(spec: ModelSpec, language: Optional[str] = None) -> str:
    return f"tts:{spec.id}"


def build(spec: ModelSpec, language: Optional[str] = None, **_: object) -> TTSProvider:
    return EdgeProvider(spec)
