"""What every speech recogniser has to provide.

Two shapes exist. A model that emits its own timestamps (Whisper, Parakeet)
implements `transcribe`. A model that only turns one clip into one string
(SeamlessM4T, MMS, SenseVoice) implements `transcribe_window`, and the base
class here does the windowing - from the diarization turns when they exist, and
from a voice-activity pass otherwise. That is how a model without timestamps
still produces the canonical segment schema instead of one wall of text.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence

from core.errors import InvalidRequest, ProviderFailure
from dubflow_core.segments import DEFAULT_SPEAKER, make_segment
from providers.base import ModelSpec


class Window(NamedTuple):
    start: float
    end: float
    speaker_id: str = DEFAULT_SPEAKER


class Transcript(NamedTuple):
    language: str
    segments: List[Dict]


class ASRProvider(ABC):
    """One loaded recogniser."""

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec

    @property
    def name(self) -> str:
        return self.spec.repo_id or self.spec.id

    def detect_language(self, audio: Path) -> str:
        raise InvalidRequest(
            f"'{self.spec.display_name}' cannot detect the spoken language. "
            f"Set source_language explicitly, or pick a recogniser whose "
            f"supports_language_detection is true."
        )

    @abstractmethod
    def transcribe(
        self,
        audio: Path,
        language: Optional[str],
        windows: Optional[Sequence[Window]] = None,
    ) -> Transcript:
        """Return the detected/confirmed language and canonical segments."""


class WindowedASR(ASRProvider):
    """Base for recognisers that transcribe one clip at a time."""

    #: Windows longer than this are split; most of these models were trained on
    #: utterance-length audio and degrade on long inputs.
    max_window_seconds = 30.0
    min_window_seconds = 0.25

    @abstractmethod
    def transcribe_window(self, clip: Path, language: str) -> str:
        """Text for one already-cut mono 16 kHz clip."""

    def transcribe(
        self,
        audio: Path,
        language: Optional[str],
        windows: Optional[Sequence[Window]] = None,
    ) -> Transcript:
        from core.media import Scratch, extract_window
        from providers.asr.windows import speech_windows

        if language is None:
            language = self.detect_language(audio)
        chosen = list(windows) if windows else speech_windows(audio)
        chosen = split_windows(chosen, self.max_window_seconds, self.min_window_seconds)
        if not chosen:
            raise ProviderFailure("No speech was found in the audio")

        segments: List[Dict] = []
        for index, window in enumerate(chosen):
            with Scratch(".wav") as clip:
                extract_window(audio, clip, window.start, window.end - window.start, rate=16000)
                text = (self.transcribe_window(clip, language) or "").strip()
            if text:
                segments.append(
                    make_segment(index, window.start, window.end, text, window.speaker_id)
                )
        return Transcript(language, segments)


def split_windows(
    windows: Sequence[Window],
    maximum: float,
    minimum: float,
) -> List[Window]:
    """Drop windows too short to hold speech and cut over-long ones."""
    result: List[Window] = []
    for window in sorted(windows, key=lambda item: item.start):
        span = window.end - window.start
        if span < minimum:
            continue
        if span <= maximum:
            result.append(window)
            continue
        pieces = int(span // maximum) + (1 if span % maximum else 0)
        step = span / pieces
        for index in range(pieces):
            start = window.start + index * step
            result.append(Window(round(start, 3), round(min(start + step, window.end), 3),
                                 window.speaker_id))
    return result


def windows_from_turns(turns: Sequence[Dict]) -> List[Window]:
    return [
        Window(float(turn["start"]), float(turn["end"]), turn["speaker_id"])
        for turn in turns
    ]
