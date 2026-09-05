"""Translate every segment, and write the subtitle file.

When the source and target languages match, the text passes through untouched
and no model is loaded - the same skip the /translate endpoint has always had.
"""
from __future__ import annotations

from pathlib import Path

import providers
from core.config import rebuild
from core.errors import ProviderFailure
from dubflow_core.segments import subtitle

NAME = "translate"


def run(job: dict, folder: Path) -> None:
    config = rebuild(job)
    segments = job["segments"]
    texts = [segment["source_text"] for segment in segments]
    source = job["source_language"]

    if source == config.target_language:
        translations = list(texts)
        job["translation_model"] = None
        job["translation_skipped"] = True
    else:
        provider = providers.translation.load(config.translation)
        translations = provider.translate(texts, source, config.target_language)
        if len(translations) != len(texts):
            raise ProviderFailure("The translation model returned the wrong number of rows")
        job["translation_model"] = provider.name
        job["translation_skipped"] = False

    for segment, translated in zip(segments, translations):
        segment["translated_text"] = (translated or "").strip()

    path = folder / "translated.srt"
    path.write_text(subtitle(segments), encoding="utf-8")
    job["files"]["subtitle"] = path.name
