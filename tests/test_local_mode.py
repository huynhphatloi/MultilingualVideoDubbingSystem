from __future__ import annotations

import sys
from pathlib import Path

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ai-service"))


def test_local_backend_uses_the_existing_dynamic_backend_contract(monkeypatch):
    import app

    monkeypatch.setattr(app, "AI_BACKENDS", ["local"])
    monkeypatch.setenv("LOCAL_API_URL", "http://local-ai:8000")
    monkeypatch.setenv("LOCAL_API_TOKEN", "local-secret")

    assert app._configured_backends() == [
        ("local", "http://local-ai:8000", "local-secret")
    ]


def test_local_backend_failure_names_the_local_restart_command(monkeypatch):
    import app

    monkeypatch.setattr(app, "AI_BACKENDS", ["local"])
    monkeypatch.setenv("LOCAL_API_URL", "http://local-ai:8000")

    message = app._backend_problem()
    assert "make local-restart" in message
    assert "notebook" not in message


def test_health_reports_when_models_are_local(monkeypatch):
    import app

    monkeypatch.setattr(
        app, "_resolve_backend", lambda force=False: ("local", "http://local-ai:8000", "")
    )
    payload = fastapi_testclient.TestClient(app.app).get("/health").json()

    assert payload["ai_backend"] == "local"
    assert payload["local_models"] is True


def test_an_outdated_local_backend_names_the_local_rebuild(monkeypatch):
    import app

    monkeypatch.setattr(app, "_backend_version", lambda url, token: "3.0")
    message = app._outdated_message("local", "http://local-ai:8000", "", "/validate")

    assert "make local-restart" in message
    assert "Runtime > Restart session" not in message


def test_local_compose_overlay_wires_the_model_service():
    compose = (ROOT / "docker-compose.local.yml").read_text(encoding="utf-8")
    dockerfile = (ROOT / "local-ai-service/Dockerfile").read_text(encoding="utf-8")

    assert "LOCAL_API_URL: http://local-ai:8000" in compose
    assert "condition: service_healthy" in compose
    assert "uvicorn\", \"colab.server:app" in dockerfile
    assert "torch==${TORCH_VERSION}" in dockerfile
