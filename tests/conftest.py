from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for entry in (ROOT, ROOT / "colab"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

os.environ.setdefault("DUBFLOW_DEVICE", "cpu")
os.environ.setdefault("DUBFLOW_MULTI_VOICE", "false")

import pytest  # noqa: E402


@pytest.fixture
def registry():  # noqa: ANN201
    import providers

    return providers


@pytest.fixture
def all_installed(monkeypatch):  # noqa: ANN201
    from providers.base import ModelSpec

    monkeypatch.setattr(ModelSpec, "installed", lambda self: True)
    monkeypatch.setattr(ModelSpec, "credential_present", lambda self: True)
