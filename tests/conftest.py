"""Test setup.

The tests import the registry, the shared core and the FastAPI app, none of
which need torch, transformers or any model package - that is the point of the
provider split, and these tests are what keeps it true.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for entry in (ROOT, ROOT / "colab"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

#: Keep every test's job directory out of /content and out of the user's tree.
os.environ.setdefault("JOBS_ROOT", str(Path(os.environ.get("TMPDIR", "/tmp")) / "dubflow-tests"))
os.environ.setdefault("DUBFLOW_DEVICE", "cpu")

import pytest  # noqa: E402


@pytest.fixture
def registry():  # noqa: ANN201
    import providers

    return providers


@pytest.fixture
def all_installed(monkeypatch):  # noqa: ANN201
    """Pretend every optional package is present.

    The compatibility rules - which language a model speaks, whether it clones
    a voice - are independent of what happens to be installed on the machine
    running the tests, and testing them should be too. Availability itself is
    covered by test_registry.
    """
    from providers.base import ModelSpec

    monkeypatch.setattr(ModelSpec, "installed", lambda self: True)
    monkeypatch.setattr(ModelSpec, "credential_present", lambda self: True)
