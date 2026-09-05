"""Model metadata and the registry that holds it.

A `ModelSpec` is data only: it never imports the library it describes, so the
capabilities endpoint, the request validator and the tests all work in a session
where none of the optional engines are installed. The implementation module is
imported the first time a model is actually asked to run.

Every field here is meant to be checkable against the model's own card. Nothing
in this file is a guess: a language belongs in `languages` only if the official
model card or the model's own source lists it.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from core.errors import InvalidRequest, MissingCredential, UnsupportedLanguage
from core.runtime import installed

TASKS = ("asr", "translation", "tts", "diarization", "separation", "lipsync")


@dataclass(frozen=True)
class ModelSpec:
    """Everything the frontend, the validator and the loader need to know."""

    id: str
    task: str
    provider: str
    display_name: str
    #: Hugging Face repository, or the upstream project when weights are not on
    #: the Hub. None for services that host the model themselves, such as Edge.
    repo_id: Optional[str] = None
    #: Application language codes (ISO-639-1) this model officially covers.
    #: Empty means the model is language-agnostic, not that it speaks nothing.
    languages: Tuple[str, ...] = ()
    multilingual: bool = False
    license: str = "unknown"
    #: pip name and import name of the optional dependency, None when the base
    #: install already covers it.
    optional_package: Optional[str] = None
    import_name: Optional[str] = None
    #: A second dependency that has to be installed by hand, such as MeloTTS.
    extra_requirements: Tuple[str, ...] = ()
    #: An environment variable that must hold a token, for gated weights.
    required_env: Optional[str] = None
    experimental: bool = False
    notes: str = ""

    # ASR ----------------------------------------------------------------
    supports_language_detection: bool = False
    supports_timestamps: bool = False

    # TTS ----------------------------------------------------------------
    supports_voice_cloning: bool = False
    supports_multispeaker: bool = False
    reference_required: bool = False
    #: Some engines want the transcript of the reference clip as well.
    reference_text_required: bool = False
    streaming: bool = False
    recommended_sample_rate: Optional[int] = None
    #: "native" when the engine varies its own speaking rate, "postprocess"
    #: when ffmpeg has to do it afterwards.
    speed_control: str = "postprocess"

    # Runtime -------------------------------------------------------------
    supports_gpu: bool = True
    supports_cpu: bool = True
    #: Import path of the module implementing this model.
    module: str = ""

    def installed(self) -> bool:
        return installed(self.import_name)

    def credential_present(self) -> bool:
        if not self.required_env:
            return True
        import os

        return bool(os.getenv(self.required_env, "").strip())

    def available(self) -> bool:
        return self.installed() and self.credential_present()

    def unavailable_reason(self) -> Optional[str]:
        if not self.installed():
            return (
                f"'{self.optional_package}' is not installed in this session. "
                f"Run `pip install {self.optional_package}` in a Colab cell and "
                f"restart the API cell."
            )
        if not self.credential_present():
            return (
                f"{self.required_env} is not set. These weights are gated: accept "
                f"the conditions on the model page, then put a Hugging Face token "
                f"in that environment variable before starting the API cell."
            )
        return None

    def speaks(self, language: Optional[str]) -> bool:
        if language is None or not self.languages:
            return True
        return language in self.languages

    def public(self) -> Dict:
        """The capabilities payload for one model."""
        payload: Dict = {
            "id": self.id,
            "task": self.task,
            "provider": self.provider,
            "display_name": self.display_name,
            "repo_id": self.repo_id,
            "languages": list(self.languages),
            "multilingual": self.multilingual,
            "license": self.license,
            "optional_package": self.optional_package,
            "extra_requirements": list(self.extra_requirements),
            "experimental": self.experimental,
            "notes": self.notes,
            "installed": self.installed(),
            "available": self.available(),
            "unavailable_reason": self.unavailable_reason(),
            "supports_gpu": self.supports_gpu,
            "supports_cpu": self.supports_cpu,
        }
        if self.task == "asr":
            payload.update({
                "supports_language_detection": self.supports_language_detection,
                "supports_timestamps": self.supports_timestamps,
            })
        if self.task == "tts":
            payload.update({
                "supports_voice_cloning": self.supports_voice_cloning,
                "supports_multispeaker": self.supports_multispeaker,
                "reference_required": self.reference_required,
                "reference_text_required": self.reference_text_required,
                "streaming": self.streaming,
                "recommended_sample_rate": self.recommended_sample_rate,
                "speed_control": self.speed_control,
            })
        return payload


@dataclass(frozen=True)
class ProviderSpec:
    """A family of models sharing one implementation."""

    id: str
    task: str
    display_name: str
    homepage: str = ""
    notes: str = ""


class Registry:
    """The models one task offers, plus the lazy import of their code."""

    def __init__(self, task: str) -> None:
        self.task = task
        self._providers: Dict[str, ProviderSpec] = {}
        self._models: Dict[str, ModelSpec] = {}
        self._order: List[str] = []

    # Registration --------------------------------------------------------
    def add_provider(self, spec: ProviderSpec) -> ProviderSpec:
        if spec.task != self.task:
            raise ValueError(f"{spec.id} is a {spec.task} provider, not {self.task}")
        self._providers[spec.id] = spec
        return spec

    def add(self, spec: ModelSpec) -> ModelSpec:
        if spec.task != self.task:
            raise ValueError(f"{spec.id} is a {spec.task} model, not {self.task}")
        if spec.id in self._models:
            raise ValueError(f"Duplicate {self.task} model id '{spec.id}'")
        if spec.provider not in self._providers:
            raise ValueError(f"'{spec.id}' names an unregistered provider '{spec.provider}'")
        self._models[spec.id] = spec
        self._order.append(spec.id)
        return spec

    def extend(self, specs: Iterable[ModelSpec]) -> None:
        for spec in specs:
            self.add(spec)

    # Lookup --------------------------------------------------------------
    def ids(self) -> List[str]:
        return list(self._order)

    def models(self) -> List[ModelSpec]:
        return [self._models[name] for name in self._order]

    def providers(self) -> List[ProviderSpec]:
        return list(self._providers.values())

    def has(self, model_id: str) -> bool:
        return model_id in self._models

    def get(self, model_id: Optional[str]) -> ModelSpec:
        name = (model_id or "").strip()
        if name not in self._models:
            raise InvalidRequest(
                f"Unknown {self.task} model '{model_id}'. "
                f"This session offers: {', '.join(self._order)}."
            )
        return self._models[name]

    def for_provider(self, provider_id: str) -> List[ModelSpec]:
        return [spec for spec in self.models() if spec.provider == provider_id]

    def resolve(
        self,
        provider_id: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> ModelSpec:
        """Pick a model from a provider and/or a model id.

        A model id alone is enough; a provider alone selects that provider's
        first model. Naming both and having them disagree is an error rather
        than a silent preference for one of them.
        """
        provider = (provider_id or "").strip().lower() or None
        model = (model_id or "").strip() or None
        if provider and provider not in self._providers:
            raise InvalidRequest(
                f"Unknown {self.task} provider '{provider_id}'. "
                f"This session offers: {', '.join(self._providers)}."
            )
        if model:
            spec = self.get(model)
            if provider and spec.provider != provider:
                raise InvalidRequest(
                    f"{self.task} model '{spec.id}' belongs to provider "
                    f"'{spec.provider}', not '{provider}'."
                )
            return spec
        if provider:
            candidates = self.for_provider(provider)
            if not candidates:
                raise InvalidRequest(f"Provider '{provider}' registers no {self.task} model")
            return candidates[0]
        raise InvalidRequest(f"No {self.task} model was requested")

    # Validation ----------------------------------------------------------
    def require_available(self, spec: ModelSpec) -> ModelSpec:
        if spec.available():
            return spec
        reason = spec.unavailable_reason() or "it is unavailable in this session"
        if spec.required_env and not spec.credential_present():
            raise MissingCredential(f"'{spec.id}' cannot run: {reason}")
        from core.errors import MissingDependency

        raise MissingDependency(spec.optional_package or spec.id, f"'{spec.id}'")

    def require_language(self, spec: ModelSpec, language: Optional[str], role: str) -> None:
        if spec.speaks(language):
            return
        others = [
            other.id for other in self.models()
            if other.speaks(language) and other.available()
        ]
        alternatives = (
            f" Models in this session that do: {', '.join(others)}."
            if others else
            " No model in this session covers it."
        )
        raise UnsupportedLanguage(
            f"'{spec.display_name}' ({spec.id}) does not support {role} language "
            f"'{language}'. Its official coverage is: "
            f"{', '.join(spec.languages) if spec.languages else 'unspecified'}."
            + alternatives
        )

    # Loading -------------------------------------------------------------
    def implementation(self, spec: ModelSpec):  # noqa: ANN201
        """Import the module that runs this model, only when it is needed."""
        if not spec.module:
            raise InvalidRequest(
                f"'{spec.id}' is registered but has no implementation in this build."
            )
        self.require_available(spec)
        return importlib.import_module(spec.module)

    # Reporting -----------------------------------------------------------
    def public(self) -> List[Dict]:
        """Providers with their models, the shape the frontend reads."""
        grouped: List[Dict] = []
        for provider in self.providers():
            models = [spec.public() for spec in self.for_provider(provider.id)]
            grouped.append({
                "id": provider.id,
                "task": provider.task,
                "display_name": provider.display_name,
                "homepage": provider.homepage,
                "notes": provider.notes,
                "available": any(model["available"] for model in models),
                "models": models,
            })
        return grouped


def check_language_lists(registries: Sequence[Registry], known: Sequence[str]) -> List[str]:
    """Report any model claiming a language the application cannot name."""
    problems: List[str] = []
    for registry in registries:
        for spec in registry.models():
            unknown = [code for code in spec.languages if code not in known]
            if unknown:
                problems.append(f"{spec.id}: unknown language codes {sorted(unknown)}")
    return problems
