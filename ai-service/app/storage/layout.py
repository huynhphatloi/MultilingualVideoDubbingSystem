"""Canonical object-key layout. Nothing outside this module hard-codes a path."""
from __future__ import annotations

from dataclasses import dataclass

ROOT = "jobs"


@dataclass(frozen=True)
class JobLayout:
    """All artifact keys for one job.

    jobs/{job_id}/
        source/input.mp4
        audio/{original,speech,background}.wav
        transcript/source.json
        translation/{lang}.json
        speakers/speaker_01.wav
        generated/segment_001.wav
        subtitles/translated.srt
        output/dubbed_{lang}.mp4
    """

    job_id: str

    # -------------------------------------------------------------- roots ---
    @property
    def prefix(self) -> str:
        return f"{ROOT}/{self.job_id}"

    # ------------------------------------------------------------- source ---
    def source_video(self, ext: str = "mp4") -> str:
        return f"{self.prefix}/source/input.{ext.lstrip('.')}"

    # -------------------------------------------------------------- audio ---
    @property
    def original_audio(self) -> str:
        return f"{self.prefix}/audio/original.wav"

    @property
    def speech_audio(self) -> str:
        return f"{self.prefix}/audio/speech.wav"

    @property
    def background_audio(self) -> str:
        return f"{self.prefix}/audio/background.wav"

    @property
    def dubbed_track(self) -> str:
        return f"{self.prefix}/audio/dubbed_speech.wav"

    @property
    def final_audio(self) -> str:
        return f"{self.prefix}/audio/final_mix.wav"

    # --------------------------------------------------------- transcript ---
    @property
    def transcript(self) -> str:
        return f"{self.prefix}/transcript/source.json"

    @property
    def diarization(self) -> str:
        return f"{self.prefix}/transcript/diarization.json"

    @property
    def segments(self) -> str:
        """ASR + diarization merged: the pipeline's working document."""
        return f"{self.prefix}/transcript/segments.json"

    # -------------------------------------------------------- translation ---
    def translation(self, target_lang: str) -> str:
        return f"{self.prefix}/translation/{target_lang}.json"

    # ----------------------------------------------------------- speakers ---
    def speaker_reference(self, speaker_id: str) -> str:
        return f"{self.prefix}/speakers/{_slug(speaker_id)}.wav"

    @property
    def speakers_prefix(self) -> str:
        return f"{self.prefix}/speakers"

    # ---------------------------------------------------------- generated ---
    def generated_segment(self, index: int, attempt: int = 0) -> str:
        suffix = "" if attempt == 0 else f"_v{attempt}"
        return f"{self.prefix}/generated/segment_{index:04d}{suffix}.wav"

    @property
    def generated_prefix(self) -> str:
        return f"{self.prefix}/generated"

    @property
    def synthesis_manifest(self) -> str:
        return f"{self.prefix}/generated/manifest.json"

    # ---------------------------------------------------------- subtitles ---
    def subtitle(self, target_lang: str, fmt: str = "srt") -> str:
        return f"{self.prefix}/subtitles/{target_lang}.{fmt}"

    def source_subtitle(self, source_lang: str, fmt: str = "srt") -> str:
        return f"{self.prefix}/subtitles/source_{source_lang}.{fmt}"

    # ------------------------------------------------------------- output ---
    def output_video(self, target_lang: str, ext: str = "mp4") -> str:
        return f"{self.prefix}/output/dubbed_{target_lang}.{ext}"

    # --------------------------------------------------------------- misc ---
    @property
    def report(self) -> str:
        return f"{self.prefix}/output/report.json"


def _slug(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in value).lower()
