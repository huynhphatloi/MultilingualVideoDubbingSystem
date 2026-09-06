"""Pipeline stage registration and execution."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from pipeline import (
    align,
    diarize,
    extract,
    merge_segments,
    mix,
    render,
    separate,
    synthesize,
    transcribe,
    translate,
)

# Stage order is part of the pipeline contract.
PIPELINE: List[Tuple[str, object]] = [
    ("extract", extract),
    ("diarize", diarize),
    ("transcribe", transcribe),
    ("merge_segments", merge_segments),
    ("translate", translate),
    ("synthesize", synthesize),
    ("align", align),
    ("separate", separate),
    ("mix", mix),
    ("render", render),
]

STAGE_NAMES = [name for name, _ in PIPELINE]
OPTIONAL_STAGES = [name for name, module in PIPELINE if hasattr(module, "enabled")]

STAGE_SLOTS = {
    "diarize": "diarization",
    "transcribe": "asr",
    "translate": "translation",
    "synthesize": "tts",
    "separate": "separation",
}


def stage(name: str):  # noqa: ANN201
    for registered, module in PIPELINE:
        if registered == name:
            return module
    raise KeyError(f"Unknown stage '{name}' (expected one of {', '.join(STAGE_NAMES)})")


def will_run(name: str, job: dict) -> bool:
    module = stage(name)
    check: Optional[Callable[[dict], bool]] = getattr(module, "enabled", None)
    return True if check is None else bool(check(job))


def planned_stages(job: dict) -> List[str]:
    return [name for name in STAGE_NAMES if will_run(name, job)]


def released_after(name: str, job: dict) -> List[str]:
    """Return model slots no later stage needs."""
    planned = planned_stages(job)
    if name not in planned:
        return []
    last_use: Dict[str, str] = {}
    for stage in planned:
        if stage in STAGE_SLOTS:
            last_use[STAGE_SLOTS[stage]] = stage
    return sorted(name_ for name_, stage in last_use.items() if stage == name)


def run_stage(name: str, job: dict, folder: Path) -> bool:
    if not will_run(name, job):
        return False
    stage(name).run(job, folder)
    return True
