"""Generate speech for each translated segment."""
from __future__ import annotations

from pathlib import Path

import providers
from core.config import rebuild
from core.media import duration
from providers.tts import SpeechRequest

NAME = "synthesize"


def run(job: dict, folder: Path) -> None:
    config = rebuild(job)
    provider = providers.tts.load(config.tts, config.target_language)
    output_dir = folder / "tts"
    output_dir.mkdir(exist_ok=True)

    counts: dict = {}
    for segment in job["segments"]:
        text = (segment.get("translated_text") or "").strip()
        if not text:
            continue
        speaker = segment.get("speaker_id")
        output = output_dir / f"{segment['id']:04d}.wav"
        provider.synthesize(
            SpeechRequest(
                text=text,
                language=config.target_language,
                speed=1.0,
                speaker_id=speaker,
            ),
            output,
        )
        segment["tts_file"] = str(output.relative_to(folder))
        segment["tts_duration_raw"] = round(duration(output), 3)
        segment["tts_duration"] = segment["tts_duration_raw"]
        segment["tts_model"] = provider.name
        counts[provider.name] = counts.get(provider.name, 0) + 1

    job["models_used"] = counts
