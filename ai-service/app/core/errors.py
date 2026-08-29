"""Typed pipeline errors -> deterministic HTTP payloads that n8n can branch on."""
from __future__ import annotations

from typing import Any


class PipelineError(Exception):
    """Base class. `code` is stable and safe for n8n IF-nodes to switch on."""

    code: str = "pipeline_error"
    http_status: int = 500
    retryable: bool = False

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_payload(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "retryable": self.retryable,
                "details": self.details,
            }
        }


# ------------------------------------------------------------------ 4xx ------
class JobNotFound(PipelineError):
    code = "job_not_found"
    http_status = 404


class ArtifactNotFound(PipelineError):
    code = "artifact_not_found"
    http_status = 404


class InvalidInput(PipelineError):
    code = "invalid_input"
    http_status = 422


class UnsupportedLanguage(PipelineError):
    code = "unsupported_language"
    http_status = 422


class NoSpeechDetected(PipelineError):
    code = "no_speech_detected"
    http_status = 422


class LanguageDetectionFailed(PipelineError):
    code = "language_detection_failed"
    http_status = 422


# ------------------------------------------------------------------ 5xx ------
class ModelLoadError(PipelineError):
    code = "model_load_failed"
    http_status = 503
    retryable = True


class ModelUnavailable(PipelineError):
    code = "model_unavailable"
    http_status = 503
    retryable = True


class FFmpegError(PipelineError):
    code = "ffmpeg_failed"
    http_status = 500
    retryable = True


class SeparationError(PipelineError):
    code = "separation_failed"
    http_status = 500
    retryable = True


class DiarizationError(PipelineError):
    code = "diarization_failed"
    http_status = 500
    retryable = True


class TranslationError(PipelineError):
    code = "translation_failed"
    http_status = 500
    retryable = True


class SynthesisError(PipelineError):
    code = "synthesis_failed"
    http_status = 500
    retryable = True


class StorageError(PipelineError):
    code = "storage_failed"
    http_status = 500
    retryable = True
