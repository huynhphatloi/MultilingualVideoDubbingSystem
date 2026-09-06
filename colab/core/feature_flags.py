"""Development features selected when the AI service starts."""
from __future__ import annotations

import os
from typing import Dict

TRUE_VALUES = {"1", "true", "yes", "on"}


def multi_voice_enabled() -> bool:
    return os.getenv("DUBFLOW_MULTI_VOICE", "").strip().lower() in TRUE_VALUES


def public() -> Dict[str, bool]:
    return {"multi_voice": multi_voice_enabled()}
