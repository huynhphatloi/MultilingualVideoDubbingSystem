"""Provider registries and capability reporting."""
from __future__ import annotations

from typing import Dict, List, Optional

from core.runtime import device, loaded
from dubflow_core import languages as L
from providers import asr, diarization, separation, translation, tts
from providers.base import ModelSpec, Registry, TASKS, check_language_lists

REGISTRIES: Dict[str, Registry] = {
    "asr": asr.REGISTRY,
    "translation": translation.REGISTRY,
    "tts": tts.REGISTRY,
    "diarization": diarization.REGISTRY,
    "separation": separation.REGISTRY,
}

DEFAULTS: Dict[str, Optional[str]] = {
    "asr": asr.DEFAULT_MODEL,
    "translation": translation.DEFAULT_MODEL,
    "tts": tts.DEFAULT_MODEL,
    "diarization": diarization.DEFAULT_MODEL,
    "separation": separation.DEFAULT_MODEL,
}


def registry(task: str) -> Registry:
    if task not in REGISTRIES:
        raise KeyError(f"Unknown task '{task}' (expected one of {', '.join(TASKS)})")
    return REGISTRIES[task]


def find(task: str, provider: Optional[str], model: Optional[str]) -> ModelSpec:
    return registry(task).resolve(provider, model)


def every_model() -> List[ModelSpec]:
    return [spec for entry in REGISTRIES.values() for spec in entry.models()]


def capabilities() -> Dict:
    return {
        "device": device(),
        "languages": L.listing(),
        "tasks": list(REGISTRIES),
        "defaults": dict(DEFAULTS),
        "providers": {task: entry.public() for task, entry in REGISTRIES.items()},
        "loaded": loaded(),
    }


def consistency_problems() -> List[str]:
    problems = check_language_lists(list(REGISTRIES.values()), L.CODES)
    seen: Dict[str, str] = {}
    for task, entry in REGISTRIES.items():
        for spec in entry.models():
            if spec.id in seen and seen[spec.id] != task:
                problems.append(
                    f"model id '{spec.id}' is used by both {seen[spec.id]} and {task}"
                )
            seen[spec.id] = task
            if spec.optional_package and not spec.import_name:
                problems.append(f"{spec.id}: names a pip package but no import name")
            if spec.module and not spec.module.startswith("providers."):
                problems.append(f"{spec.id}: module '{spec.module}' is outside providers/")
    for task, default in DEFAULTS.items():
        if default is not None and not REGISTRIES[task].has(default):
            problems.append(f"default {task} model '{default}' is not registered")
    return problems
