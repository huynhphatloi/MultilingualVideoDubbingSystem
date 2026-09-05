"""Turning request parameters into a validated job configuration.

Both the end-to-end `POST /jobs` route and the per-stage routes build one of
these, so the compatibility rules - can this engine speak that language, can it
clone a voice, is its package installed - are written once and enforced before
any model is downloaded rather than three stages later.

The old flat parameters (`model`, `translation_engine`, `tts_engine`) still work
and map onto the new ones.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

import providers
from core.errors import InvalidRequest
from dubflow_core import alignment as align
from dubflow_core import languages as L
from providers.base import ModelSpec

TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}


def as_bool(value: Any, default: bool) -> bool:
    """Parse a flag, treating an empty field as absent rather than false.

    Multipart forms cannot omit a field: n8n sends every parameter it declares,
    empty when the operator left it alone. Reading "" as false turned the
    form's "Voice Cloning: Automatic" into "Voice Cloning: off", which then
    refused every engine that only speaks in a cloned voice.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text:
        return default
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    raise InvalidRequest(
        f"Expected a true/false value, got '{value}'. Use true/false, 1/0, "
        f"yes/no or on/off, or leave the field empty for the default."
    )


def as_language(value: Any, field_name: str, allow_auto: bool = False) -> Optional[str]:
    code = L.normalise(None if value is None else str(value))
    if code is None:
        if allow_auto:
            return None
        raise InvalidRequest(f"{field_name} is required")
    if not L.is_supported(code):
        raise InvalidRequest(
            f"Unsupported {field_name} '{code}'. "
            f"Supported codes: {', '.join(L.CODES)}."
        )
    return code


@dataclass
class Features:
    diarization: bool = False
    voice_cloning: bool = False
    alignment: bool = True
    source_separation: bool = False
    lip_sync: bool = False

    def public(self) -> Dict[str, bool]:
        return {
            "diarization": self.diarization,
            "voice_cloning": self.voice_cloning,
            "alignment": self.alignment,
            "source_separation": self.source_separation,
            "lip_sync": self.lip_sync,
        }


@dataclass
class JobConfig:
    source_language: Optional[str]
    target_language: str
    asr: ModelSpec
    translation: ModelSpec
    tts: ModelSpec
    features: Features = field(default_factory=Features)
    diarization: Optional[ModelSpec] = None
    separation: Optional[ModelSpec] = None
    lipsync: Optional[ModelSpec] = None
    limits: align.Limits = align.Limits()
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None

    def public(self) -> Dict[str, Any]:
        def choice(spec: Optional[ModelSpec]) -> Optional[Dict[str, Any]]:
            if spec is None:
                return None
            return {
                "provider": spec.provider,
                "model": spec.id,
                "display_name": spec.display_name,
                "repo_id": spec.repo_id,
                "license": spec.license,
            }

        payload: Dict[str, Any] = {
            "source_language": self.source_language or "auto",
            "target_language": self.target_language,
            "asr": choice(self.asr),
            "translation": choice(self.translation),
            "tts": dict(choice(self.tts) or {}, voice_cloning=self.features.voice_cloning),
            "features": self.features.public(),
        }
        if self.diarization is not None:
            payload["diarization"] = choice(self.diarization)
        if self.separation is not None:
            payload["separation"] = choice(self.separation)
        if self.lipsync is not None:
            payload["lipsync"] = choice(self.lipsync)
        payload["alignment_limits"] = {
            "min_speed": self.limits.min_speed,
            "max_speed": self.limits.max_speed,
            "tolerance": self.limits.tolerance,
            "allow_stretch": self.limits.allow_stretch,
        }
        return payload

    def confirm_source_language(self, detected: str) -> None:
        """Check the detected language against the models that will use it."""
        providers.registry("asr").require_language(self.asr, detected, "source")
        if detected != self.target_language:
            providers.registry("translation").require_language(
                self.translation, detected, "source"
            )


def _first(values: Mapping[str, Any], *names: str) -> Optional[str]:
    for name in names:
        value = values.get(name)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _speakers(values: Mapping[str, Any], name: str) -> Optional[int]:
    raw = _first(values, name)
    if raw is None:
        return None
    try:
        count = int(raw)
    except ValueError as exc:
        raise InvalidRequest(f"{name} must be a whole number") from exc
    if not 1 <= count <= 20:
        raise InvalidRequest(f"{name} must be between 1 and 20")
    return count


def rebuild(job: Mapping[str, Any]) -> JobConfig:
    """The configuration of a stored job, with the detected language filled in.

    Stages rebuild rather than deserialise so that a job resumed after an edit
    is validated against the registry it will actually run on.
    """
    config = build(job.get("request") or {})
    detected = job.get("source_language")
    if detected:
        config.source_language = detected
    return config


def build(values: Mapping[str, Any]) -> JobConfig:
    """Validate a request and resolve every model it selects.

    `values` is the flat form/JSON body. Both the new names
    (`asr_provider`/`asr_model`) and the old ones (`model`) are accepted.
    """
    target = as_language(values.get("target_language", "vi"), "target_language")
    source = as_language(values.get("source_language"), "source_language", allow_auto=True)

    asr_spec = providers.find(
        "asr",
        _first(values, "asr_provider"),
        _first(values, "asr_model", "model") or providers.DEFAULTS["asr"],
    )
    translation_spec = providers.find(
        "translation",
        _first(values, "translation_provider"),
        _first(values, "translation_model", "translation_engine")
        or providers.DEFAULTS["translation"],
    )
    tts_spec = providers.find(
        "tts",
        _first(values, "tts_provider"),
        _first(values, "tts_model", "tts_engine") or providers.DEFAULTS["tts"],
    )

    # Language compatibility comes first: "this engine does not speak that
    # language" is a more useful answer than "install its package" when the
    # combination was never going to work.
    # ----------------------------------------------------------------------
    if source is None and not asr_spec.supports_language_detection:
        raise InvalidRequest(
            f"'{asr_spec.display_name}' cannot detect the spoken language, so "
            f"source_language cannot be 'auto'. Name the source language, or "
            f"choose a recogniser whose supports_language_detection is true."
        )
    if source is not None:
        providers.registry("asr").require_language(asr_spec, source, "source")
        if source != target:
            providers.registry("translation").require_language(
                translation_spec, source, "source"
            )
    if source != target:
        providers.registry("translation").require_language(
            translation_spec, target, "target"
        )
    providers.registry("tts").require_language(tts_spec, target, "target")

    for task, spec in (("asr", asr_spec), ("translation", translation_spec), ("tts", tts_spec)):
        providers.registry(task).require_available(spec)

    # Features --------------------------------------------------------------
    features = Features(
        diarization=as_bool(values.get("enable_diarization"), False),
        # A cloning engine that cannot speak without a reference keeps its old
        # behaviour: the pipeline cuts one for it unless asked not to.
        voice_cloning=as_bool(
            values.get("enable_voice_cloning"), tts_spec.reference_required
        ),
        alignment=as_bool(values.get("enable_alignment"), True),
        source_separation=as_bool(values.get("enable_source_separation"), False),
        lip_sync=as_bool(values.get("enable_lip_sync"), False),
    )

    if features.voice_cloning and not tts_spec.supports_voice_cloning:
        raise InvalidRequest(
            f"'{tts_spec.display_name}' does not clone voices, so voice cloning "
            f"cannot be enabled for it. Turn it off, or pick an engine whose "
            f"supports_voice_cloning is true."
        )
    if tts_spec.reference_required and not features.voice_cloning:
        raise InvalidRequest(
            f"'{tts_spec.display_name}' only speaks in a cloned voice, so voice "
            f"cloning cannot be turned off for it."
        )

    diarization_spec = None
    if features.diarization:
        diarization_spec = providers.find(
            "diarization",
            _first(values, "diarization_provider"),
            _first(values, "diarization_model") or providers.DEFAULTS["diarization"],
        )
        providers.registry("diarization").require_available(diarization_spec)

    separation_spec = None
    if features.source_separation:
        separation_spec = providers.find(
            "separation",
            _first(values, "separation_provider"),
            _first(values, "separation_model") or providers.DEFAULTS["separation"],
        )
        providers.registry("separation").require_available(separation_spec)

    lipsync_spec = None
    if features.lip_sync:
        if not providers.registry("lipsync").ids():
            raise InvalidRequest(
                "Lip sync is switched on but this build registers no lip-sync "
                "provider. See colab/providers/lipsync/registry.py for what "
                "adding one involves."
            )
        lipsync_spec = providers.find(
            "lipsync",
            _first(values, "lipsync_provider"),
            _first(values, "lipsync_model") or providers.DEFAULTS["lipsync"],
        )
        providers.registry("lipsync").require_available(lipsync_spec)

    return JobConfig(
        source_language=source,
        target_language=target,
        asr=asr_spec,
        translation=translation_spec,
        tts=tts_spec,
        features=features,
        diarization=diarization_spec,
        separation=separation_spec,
        lipsync=lipsync_spec,
        limits=_limits(values),
        min_speakers=_speakers(values, "min_speakers"),
        max_speakers=_speakers(values, "max_speakers"),
    )


def _limits(values: Mapping[str, Any]) -> align.Limits:
    def number(name: str, fallback: float) -> float:
        raw = _first(values, name)
        if raw is None:
            return fallback
        try:
            return float(raw)
        except ValueError as exc:
            raise InvalidRequest(f"{name} must be a number") from exc

    minimum = number("min_speed", align.DEFAULT_MIN_SPEED)
    maximum = number("max_speed", align.DEFAULT_MAX_SPEED)
    if not 0.25 <= minimum <= 1.0:
        raise InvalidRequest("min_speed must be between 0.25 and 1.0")
    if not 1.0 <= maximum <= 3.0:
        raise InvalidRequest("max_speed must be between 1.0 and 3.0")
    return align.Limits(
        min_speed=minimum,
        max_speed=maximum,
        tolerance=number("alignment_tolerance", align.DEFAULT_TOLERANCE),
        allow_stretch=as_bool(values.get("allow_stretch"), False),
    )
