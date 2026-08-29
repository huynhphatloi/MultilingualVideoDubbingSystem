"""Request/response contracts. These are exactly what the n8n nodes send."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# =========================================================== shared types ===
class Segment(BaseModel):
    """The unit of work that travels through the whole pipeline."""

    model_config = ConfigDict(extra="allow")

    segment_id: int
    start: float = Field(..., description="Start time in the ORIGINAL timeline (s)")
    end: float
    duration: float = 0.0

    speaker_id: str = "SPEAKER_00"
    voice_reference: str | None = None

    source_language: str | None = None
    target_language: str | None = None
    source_text: str = ""
    translated_text: str | None = None

    generated_audio: str | None = None
    generated_duration: float | None = None
    duration_ratio: float | None = None
    sync_action: Literal["accept", "stretch", "retranslate", "pending", "failed"] = "pending"
    attempts: int = 0
    tts_model: str | None = None
    words: list[dict[str, Any]] | None = None


class JobRef(BaseModel):
    job_id: str


class StageResponse(BaseModel):
    """Every stage returns metadata + object keys - never binary payloads."""

    model_config = ConfigDict(extra="allow")

    job_id: str
    stage: str
    status: Literal["completed", "failed", "skipped"] = "completed"
    duration_ms: int | None = None


# ================================================================== jobs ====
class CreateJobRequest(BaseModel):
    source_filename: str | None = None
    source_language: str | None = Field(
        None, description="ISO-639-1 code, or null/'auto' to auto-detect"
    )
    target_language: str = Field(..., description="ISO-639-1 code, e.g. 'vi'")
    options: dict[str, Any] = Field(default_factory=dict)


class CreateJobResponse(BaseModel):
    job_id: str
    status: str
    prefix: str
    target_language: str
    source_language: str | None = None


class RegisterVideoRequest(BaseModel):
    """Used when the file already lives in storage (e.g. uploaded by the UI)."""

    video_key: str


# ================================================================= media ====
class ExtractAudioRequest(JobRef):
    video_key: str | None = None
    sample_rate: int = 16000
    channels: int = 1


class ExtractAudioResponse(StageResponse):
    audio_key: str
    duration_seconds: float
    sample_rate: int
    channels: int
    has_audio_stream: bool
    peak_dbfs: float | None = None


class SeparateRequest(JobRef):
    audio_key: str | None = None
    model: str | None = None


class SeparateResponse(StageResponse):
    speech_key: str
    background_key: str
    model: str
    separated: bool = True
    #: "demucs" | "voiceover". Voice-over means the original dialogue is still
    #: audible under the dub because separation was unavailable.
    mode: str = "demucs"
    original_dialogue_present: bool = False


# ================================================================ speech ====
class TranscribeRequest(JobRef):
    audio_key: str | None = None
    language: str | None = Field(None, description="null/'auto' triggers detection")
    model: str | None = None
    word_timestamps: bool = True


class TranscribeResponse(StageResponse):
    transcript_key: str
    language: str
    language_confidence: float
    segment_count: int
    total_speech_seconds: float
    preview: list[dict[str, Any]] = Field(default_factory=list)


class DiarizeRequest(JobRef):
    audio_key: str | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    num_speakers: int | None = Field(
        None, description="Exact speaker count, when known. Overrides min/max."
    )


class DiarizeResponse(StageResponse):
    diarization_key: str
    speaker_count: int
    speakers: list[str]
    enabled: bool = True
    has_exclusive: bool = False
    model: str | None = None


class MergeRequest(JobRef):
    transcript_key: str | None = None
    diarization_key: str | None = None
    extract_voice_references: bool = True


class MergeResponse(StageResponse):
    segments_key: str
    segment_count: int
    speaker_count: int
    speakers: list[dict[str, Any]]
    turn_source: str = "regular"


# =========================================================== translation ====
class TranslateRequest(JobRef):
    segments_key: str | None = None
    source_language: str | None = None
    target_language: str | None = None
    engine: Literal["nllb", "seamless", "auto"] = "auto"
    duration_aware: bool = True


class TranslateResponse(StageResponse):
    translation_key: str
    segments_key: str
    engine: str
    segment_count: int
    source_language: str
    target_language: str


class AdaptTranslationRequest(JobRef):
    """Re-translate only the segments whose synthesised audio missed the target."""

    segment_ids: list[int] = Field(default_factory=list)
    segments_key: str | None = None
    engine: Literal["nllb", "seamless", "auto"] = "auto"


class AdaptTranslationResponse(StageResponse):
    segments_key: str
    adapted: list[int]
    unchanged: list[int]
    #: Segments the translator returned verbatim - no point re-voicing them.
    identical: list[int] = Field(default_factory=list)
    engine: str | None = None


class ReviewTranslationRequest(JobRef):
    """Human-in-the-loop edits coming from the web UI."""

    edits: list[dict[str, Any]] = Field(
        ..., description="[{segment_id, translated_text}] - overrides machine output"
    )


# ================================================================== tts =====
class SynthesizeRequest(JobRef):
    segments_key: str | None = None
    target_language: str | None = None
    segment_ids: list[int] | None = None
    prefer_voice_clone: bool | None = None
    force_model: str | None = None


class SynthesizeResponse(StageResponse):
    segments_key: str
    manifest_key: str
    synthesized: int
    failed: int
    models_used: dict[str, int]
    needs_adaptation: list[int]
    within_tolerance: int
    ratio_stats: dict[str, float]


# ================================================================= audio ====
class SynchronizeRequest(JobRef):
    segments_key: str | None = None
    total_duration: float | None = None


class SynchronizeResponse(StageResponse):
    dubbed_track_key: str
    stretched_segments: int
    placed_segments: int
    dropped_segments: list[int]
    #: Takes that still run over the next one - the adapt loop should have
    #: shortened these. Non-empty means the dub has two voices somewhere.
    overlapping_segments: list[dict[str, Any]] = Field(default_factory=list)
    truncated_at_end: list[int] = Field(default_factory=list)
    timeline_duration: float


class MixRequest(JobRef):
    dubbed_track_key: str | None = None
    background_key: str | None = None
    background_gain_db: float | None = None
    speech_gain_db: float | None = None
    loudnorm: bool | None = None
    #: Side-chain duck the background while the dub speaks. null = use MIX_DUCK_ENABLED.
    duck: bool | None = None


class MixResponse(StageResponse):
    final_audio_key: str
    background_used: bool
    #: Why the background was skipped, when it was.
    background_reason: str | None = None
    loudnorm_applied: bool
    ducking_applied: bool = False
    channel_layout: str = "stereo"
    #: Level correction applied to the dubbed track before mixing.
    speech_auto_gain_db: float = 0.0
    speech_peak_dbfs: float | None = None


# ============================================================== subtitle ====
class SubtitleRequest(JobRef):
    segments_key: str | None = None
    formats: list[Literal["srt", "vtt"]] = Field(default_factory=lambda: ["srt", "vtt"])
    include_source: bool = True
    max_chars_per_line: int = 42


class SubtitleResponse(StageResponse):
    subtitle_keys: dict[str, str]
    source_subtitle_keys: dict[str, str] = Field(default_factory=dict)
    cue_count: int


# ================================================================= video ====
class RenderRequest(JobRef):
    video_key: str | None = None
    audio_key: str | None = None
    burn_subtitles: bool = False
    subtitle_key: str | None = None
    target_language: str | None = None


class RenderResponse(StageResponse):
    output_key: str
    output_url: str
    duration_seconds: float
    size_bytes: int
    burned_subtitles: bool
    #: False means the video had to be re-encoded (slow + lossy). Expected only
    #: when burning subtitles or with an exotic source codec.
    video_stream_copied: bool = True
