"""The stage list and the runner.

Each stage is a module with a `run(job, folder)` and, when it is optional, an
`enabled(job)`. A stage that is switched off records itself as skipped and the
next one carries on, so a pipeline without diarization, alignment, separation or
lip sync is the same code path as one with all four.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

from pipeline import (
    align,
    diarize,
    extract,
    lipsync,
    merge_segments,
    mix,
    render,
    separate,
    synthesize,
    transcribe,
    translate,
    voice_reference,
)

#: Order matters: diarization before recognition so a windowed recogniser can
#: use the turns, references before synthesis, alignment before the mix.
PIPELINE: List[Tuple[str, object]] = [
    ("extract", extract),
    ("diarize", diarize),
    ("transcribe", transcribe),
    ("merge_segments", merge_segments),
    ("translate", translate),
    ("voice_references", voice_reference),
    ("synthesize", synthesize),
    ("align", align),
    ("separate", separate),
    ("mix", mix),
    ("lipsync", lipsync),
    ("render", render),
]

STAGE_NAMES = [name for name, _ in PIPELINE]
#: Stages that decide for themselves whether to run.
OPTIONAL_STAGES = [name for name, module in PIPELINE if hasattr(module, "enabled")]


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
    """The stages this job will actually execute, for progress reporting."""
    return [name for name in STAGE_NAMES if will_run(name, job)]


def run_stage(name: str, job: dict, folder: Path) -> bool:
    """Run one stage. Returns False when the stage skipped itself."""
    if not will_run(name, job):
        return False
    stage(name).run(job, folder)
    return True
