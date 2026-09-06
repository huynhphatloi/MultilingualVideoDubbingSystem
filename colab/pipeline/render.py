"""Mux the dubbed audio and the subtitles back onto the video."""
from __future__ import annotations

from pathlib import Path

from core.errors import ProviderFailure
from core.media import ffmpeg

NAME = "render"


def run(job: dict, folder: Path) -> None:
    files = job["files"]
    video = folder / files["input"]
    output = folder / f"dubbed_{job['config']['target_language']}.mp4"
    total = float(job["duration_seconds"])
    base = [
        "-i", str(video),
        "-i", str(folder / files["final_audio"]),
        "-i", str(folder / files["subtitle"]),
        "-map", "0:v:0", "-map", "1:a:0", "-map", "2:0",
    ]
    finish = [
        "-c:a", "aac", "-b:a", "192k", "-c:s", "mov_text",
        "-metadata:s:s:0", f"language={job['config']['target_language']}",
        "-t", f"{total:.3f}", "-movflags", "+faststart", str(output),
    ]
    try:
        ffmpeg(base + ["-c:v", "copy"] + finish)
    except ProviderFailure:
        # Stream copy refuses containers the MP4 muxer cannot take verbatim.
        ffmpeg(base + ["-c:v", "libx264", "-preset", "fast", "-crf", "22"] + finish)
    job["files"]["output"] = output.name
