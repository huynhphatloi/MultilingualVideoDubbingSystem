"""Engine registry - the single place that knows which engines exist.

Everything else (the server, the benchmark, the notebook) asks this module.
Adding an engine is one import plus one line in ``_CLASSES``; no other file
changes, because the client discovers what is available over ``GET /health``.

Engines are constructed lazily and cached. Construction is cheap - every heavy
import (torch, TTS, f5_tts, edge_tts) happens inside ``load()``, so a Colab
runtime that only installed the deps for one engine can still import this
module and answer ``/health`` for all of them.

The set below is chosen to span the axes a comparison actually has to argue
about, not to be exhaustive:

    engine     clones?  languages    where it runs   role in the comparison
    -------    -------  -----------  --------------  ----------------------------
    mms        no       ~1100        GPU (tiny)      the floor: what the pipeline
                                                     falls back to today
    xtts_v2    yes      17, no vi    GPU 2.5 GB      multilingual cloning baseline
    vixtts     yes      vi           GPU 2.5 GB      same arch, vi fine-tune
    f5_vi      yes      vi           GPU 3 GB        vi flow-matching, 1000h data
    f5_base    yes      en, zh       GPU 3 GB        same arch on its native data,
                                                     isolates "arch vs fine-tune"
    piper      no       50+          CPU             the speed ceiling - shows what
                                                     quality costs
    edge       no       100+         cloud, free     commercial naturalness ceiling
                                                     (no cloning, so it also shows
                                                     what cloning costs)
"""
from __future__ import annotations

from .tts_base import Spoken, TTSEngine
from .tts_edge import EdgeTTS
from .tts_f5 import F5Base, F5Vietnamese
from .tts_mms import MmsTTS
from .tts_piper import PiperTTS
from .tts_xtts import ViXtts, XttsV2

#: Declaration order is display order in every report and in /health.
_CLASSES: tuple[type[TTSEngine], ...] = (
    MmsTTS, XttsV2, ViXtts, F5Vietnamese, F5Base, PiperTTS, EdgeTTS,
)

_INSTANCES: dict[str, TTSEngine] = {}


def names() -> list[str]:
    return [cls.name for cls in _CLASSES]


def get(name: str) -> TTSEngine:
    """Return the singleton for ``name``; raise KeyError with the valid set."""
    if name not in _INSTANCES:
        for cls in _CLASSES:
            if cls.name == name:
                _INSTANCES[name] = cls()
                break
        else:
            raise KeyError(f"unknown engine {name!r}; have {names()}")
    return _INSTANCES[name]


def describe_all() -> list[dict]:
    """Catalogue for ``GET /health`` - does NOT load any weights."""
    return [get(name).describe() for name in names()]


def for_language(language: str, *, cloning: bool | None = None) -> list[str]:
    """Engines that can speak ``language``, optionally filtered by cloning."""
    out = []
    for name in names():
        engine = get(name)
        if not engine.supports(language):
            continue
        if cloning is not None and engine.voice_cloning is not cloning:
            continue
        out.append(name)
    return out


def unload_all(keep: str | None = None) -> list[str]:
    """Free VRAM for every loaded engine except ``keep``.

    A 16 GB T4 holds exactly one cloning model comfortably. The benchmark walks
    engines one at a time and calls this between them; without it the third
    engine OOMs and the run looks like a model defect rather than a memory one.
    """
    freed = []
    for name, engine in _INSTANCES.items():
        if name != keep and engine.loaded:
            engine.unload()
            freed.append(name)
    return freed


__all__ = ["Spoken", "TTSEngine", "describe_all", "for_language", "get",
           "names", "unload_all"]
