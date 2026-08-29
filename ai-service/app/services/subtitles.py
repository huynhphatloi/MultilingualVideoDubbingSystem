"""Stage 10: subtitle generation (.srt / .vtt) from the segment document."""
from __future__ import annotations

import logging
import textwrap

from app.services.workspace import JobWorkspace

log = logging.getLogger(__name__)

_MIN_CUE = 0.6


def generate(job_id: str, segments_key: str, formats: list[str] | None = None,
             include_source: bool = True, max_chars_per_line: int = 42) -> dict:
    ws = JobWorkspace(job_id)
    document = ws.pull_json(segments_key)
    segments = sorted(document.get("segments", []), key=lambda s: s["start"])
    target = document.get("target_language") or "target"
    source = document.get("source_language") or "source"

    formats = formats or ["srt", "vtt"]
    translated_cues = _cues(segments, "translated_text", max_chars_per_line)
    source_cues = _cues(segments, "source_text", max_chars_per_line)

    subtitle_keys: dict[str, str] = {}
    source_keys: dict[str, str] = {}

    for fmt in formats:
        body = _render(translated_cues, fmt)
        path = ws.path("subtitles", f"{target}.{fmt}")
        path.write_text(body, encoding="utf-8")
        subtitle_keys[fmt] = ws.push(path, ws.layout.subtitle(target, fmt))

        if include_source and source_cues:
            src_body = _render(source_cues, fmt)
            src_path = ws.path("subtitles", f"source_{source}.{fmt}")
            src_path.write_text(src_body, encoding="utf-8")
            source_keys[fmt] = ws.push(src_path, ws.layout.source_subtitle(source, fmt))

    log.info("generated %d cue(s) in %s", len(translated_cues), ", ".join(formats))
    return {
        "subtitle_keys": subtitle_keys,
        "source_subtitle_keys": source_keys,
        "cue_count": len(translated_cues),
    }


# ---------------------------------------------------------------- internals --
def _cues(segments: list[dict], field: str, max_chars: int) -> list[dict]:
    cues: list[dict] = []
    for seg in segments:
        text = (seg.get(field) or "").strip()
        if not text:
            continue
        start = float(seg["start"])
        end = max(float(seg["end"]), start + _MIN_CUE)
        if cues and start < cues[-1]["end"]:
            cues[-1]["end"] = max(cues[-1]["start"] + _MIN_CUE, start - 0.02)
        cues.append({"start": start, "end": end, "text": _wrap(text, max_chars)})
    return cues


def _wrap(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    lines = textwrap.wrap(text, width=max_chars, break_long_words=False)
    return "\n".join(lines[:3])  # broadcast convention: at most 3 lines


def _render(cues: list[dict], fmt: str) -> str:
    if fmt == "vtt":
        blocks = ["WEBVTT", ""]
        for index, cue in enumerate(cues, start=1):
            blocks.append(str(index))
            blocks.append(f"{_ts(cue['start'], '.')} --> {_ts(cue['end'], '.')}")
            blocks.append(cue["text"])
            blocks.append("")
        return "\n".join(blocks)

    blocks = []
    for index, cue in enumerate(cues, start=1):
        blocks.append(str(index))
        blocks.append(f"{_ts(cue['start'], ',')} --> {_ts(cue['end'], ',')}")
        blocks.append(cue["text"])
        blocks.append("")
    return "\n".join(blocks)


def _ts(seconds: float, decimal: str) -> str:
    seconds = max(0.0, seconds)
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis == 1000:
        millis, secs = 0, secs + 1
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal}{millis:03d}"
