from __future__ import annotations


import sys
import time
from pathlib import Path

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ai-service"))


def wait_for(predicate, timeout: float = 5.0) -> bool:  # noqa: ANN001
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def store():  # noqa: ANN201
    import tasks

    tasks.reset()
    yield tasks
    tasks.reset()


def test_a_task_reports_its_result_once_it_finishes(store):
    state = store.submit(lambda: {"segments": 3}, "transcribe:small")
    assert state["status"] in (store.QUEUED, store.RUNNING, store.DONE)
    assert wait_for(lambda: store.public(state["task_id"])["done"])
    finished = store.public(state["task_id"])
    assert finished["status"] == store.DONE
    assert finished["result"] == {"segments": 3}
    assert finished["label"] == "transcribe:small"


def test_a_failure_keeps_the_status_code_it_would_have_had(store):
    from core.errors import InvalidRequest

    def refuse():
        raise InvalidRequest("mms does not speak Japanese")

    state = store.submit(refuse, "translate:nllb")
    assert wait_for(lambda: store.public(state["task_id"])["done"])
    failed = store.public(state["task_id"])
    assert failed["status"] == store.FAILED
    assert failed["status_code"] == 400
    assert "Japanese" in failed["error"]


def test_an_unexpected_exception_is_reported_rather_than_swallowed(store):
    state = store.submit(lambda: 1 / 0, "boom")
    assert wait_for(lambda: store.public(state["task_id"])["done"])
    failed = store.public(state["task_id"])
    assert failed["status"] == store.FAILED
    assert failed["status_code"] == 500
    assert "ZeroDivisionError" in failed["error"]


def test_polling_an_unknown_task_is_a_404(store):
    from core.errors import ServiceError

    with pytest.raises(ServiceError) as failure:
        store.public("nope")
    assert failure.value.status_code == 404


def test_evicting_a_task_deletes_the_file_it_produced(store, tmp_path):
    stem = tmp_path / "background.wav"
    stem.write_bytes(b"audio")
    store.LIMIT = 1
    try:
        state = store.submit(lambda: {"file": str(stem)}, "separate:htdemucs")
        assert wait_for(lambda: store.public(state["task_id"])["done"])
        assert stem.exists()
        for _ in range(3):
            filler = store.submit(lambda: {"ok": True}, "filler")
            assert wait_for(lambda: store.public(filler["task_id"])["done"])
        assert not stem.exists()
    finally:
        store.LIMIT = 64


@pytest.fixture
def colab():  # noqa: ANN201
    import server
    import tasks

    tasks.reset()
    with fastapi_testclient.TestClient(server.app) as running:
        yield running
    tasks.reset()


@pytest.fixture
def stub_translation(monkeypatch):  # noqa: ANN201
    import providers

    class Engine:
        name = "stub/translation"

        def translate(self, texts, source, target):  # noqa: ANN001, ANN201
            return [f"[{target}] {text}" for text in texts]

    monkeypatch.setattr(providers.translation, "load", lambda spec: Engine())
    monkeypatch.setattr(
        providers.registry("translation"), "require_available", lambda spec: spec
    )
    return Engine


def test_translate_returns_a_task_id_instead_of_waiting(colab, stub_translation):
    response = colab.post("/translate", json={
        "source_language": "en", "target_language": "vi",
        "texts": ["Hello"], "async_mode": True,
    })
    assert response.status_code == 202
    state = response.json()
    assert state["task_id"]

    assert wait_for(lambda: colab.get(f"/tasks/{state['task_id']}").json()["done"])
    finished = colab.get(f"/tasks/{state['task_id']}").json()
    assert finished["status"] == "done"
    assert finished["result"]["translations"] == ["[vi] Hello"]


def test_work_that_is_already_done_needs_no_task(colab):
    response = colab.post("/translate", json={
        "source_language": "en", "target_language": "en",
        "texts": ["Hello"], "async_mode": True,
    })
    assert response.status_code == 200
    assert response.json()["skipped"] is True


def test_the_same_call_without_async_mode_still_answers_directly(colab):
    response = colab.post("/translate", json={
        "source_language": "en", "target_language": "en", "texts": ["Hello"],
    })
    assert response.status_code == 200
    assert response.json()["translations"] == ["Hello"]


def test_a_rejected_request_fails_before_a_task_is_created(colab):
    response = colab.post("/translate", json={
        "source_language": "en", "target_language": "vi",
        "texts": [], "async_mode": True,
    })
    assert response.status_code == 400
    assert "1 to 500" in response.json()["detail"]


def test_downloading_a_task_that_produced_no_file_is_a_409(colab, stub_translation):
    response = colab.post("/translate", json={
        "source_language": "en", "target_language": "vi",
        "texts": ["Hello"], "async_mode": True,
    })
    task_id = response.json()["task_id"]
    assert wait_for(lambda: colab.get(f"/tasks/{task_id}").json()["done"])
    assert colab.get(f"/tasks/{task_id}/download").status_code == 409


class Reply:
    def __init__(self, payload, status_code=200):  # noqa: ANN001
        self._payload = payload
        self.status_code = status_code
        self.content = b""
        self.headers = {}

    def json(self):  # noqa: ANN201
        return self._payload


def test_the_local_service_submits_and_then_polls(monkeypatch):
    import app

    monkeypatch.setattr(app, "ASYNC_STAGES", True)
    monkeypatch.setattr(app, "TASK_POLL_INTERVAL", 0)
    seen = []
    polls = {"count": 0}

    def fake_request(path, method="POST", **kwargs):  # noqa: ANN001, ANN202
        seen.append((path, kwargs.get("data") or kwargs.get("json") or {}))
        if path == "/transcribe":
            return Reply({"task_id": "abc", "status": "queued", "done": False}, 202)
        if path == "/tasks/abc":
            polls["count"] += 1
            if polls["count"] < 2:
                return Reply({"task_id": "abc", "status": "running", "done": False})
            return Reply({"task_id": "abc", "status": "done", "done": True,
                          "result": {"source_language": "en", "segments": []}})
        raise AssertionError(path)

    monkeypatch.setattr(app, "_colab_request", fake_request)
    result = app._stage_call("/transcribe", data={"language": ""})

    assert result == {"source_language": "en", "segments": []}
    assert seen[0][0] == "/transcribe"
    assert seen[0][1]["async_mode"] == "true", "the submit must ask for a task"
    assert polls["count"] == 2, "a task still running must be polled again"


def test_a_backend_without_tasks_is_answered_synchronously(monkeypatch):
    import app

    monkeypatch.setattr(app, "ASYNC_STAGES", True)

    def fake_request(path, method="POST", **kwargs):  # noqa: ANN001, ANN202
        assert path == "/translate"
        return Reply({"translations": ["xin chao"]}, 200)

    monkeypatch.setattr(app, "_colab_request", fake_request)
    assert app._stage_call("/translate", json={"texts": ["hello"]}) == {
        "translations": ["xin chao"]
    }


def test_a_failed_task_becomes_a_502_naming_the_reason(monkeypatch):
    import app

    monkeypatch.setattr(app, "ASYNC_STAGES", True)
    monkeypatch.setattr(app, "TASK_POLL_INTERVAL", 0)

    def fake_request(path, method="POST", **kwargs):  # noqa: ANN001, ANN202
        if path == "/transcribe":
            return Reply({"task_id": "z", "done": False}, 202)
        return Reply({"task_id": "z", "status": "failed", "done": True,
                      "error": "CUDA out of memory", "status_code": 500})

    monkeypatch.setattr(app, "_colab_request", fake_request)
    with pytest.raises(app.HTTPException) as failure:
        app._stage_call("/transcribe", data={})
    assert failure.value.status_code == 502
    assert "CUDA out of memory" in failure.value.detail


def test_a_task_that_never_finishes_times_out_with_a_usable_message(monkeypatch):
    import app

    monkeypatch.setattr(app, "ASYNC_STAGES", True)
    monkeypatch.setattr(app, "TASK_POLL_INTERVAL", 0)
    monkeypatch.setattr(app, "TASK_POLL_TIMEOUT", 0.05)

    def fake_request(path, method="POST", **kwargs):  # noqa: ANN001, ANN202
        if path == "/transcribe":
            return Reply({"task_id": "slow", "done": False}, 202)
        return Reply({"task_id": "slow", "status": "running", "done": False})

    monkeypatch.setattr(app, "_colab_request", fake_request)
    with pytest.raises(app.HTTPException) as failure:
        app._stage_call("/transcribe", data={})
    assert failure.value.status_code == 504
    assert "slow" in failure.value.detail


def test_turning_the_feature_off_restores_the_single_request(monkeypatch):
    import app

    monkeypatch.setattr(app, "ASYNC_STAGES", False)
    sent = {}

    def fake_request(path, method="POST", **kwargs):  # noqa: ANN001, ANN202
        sent.update(kwargs.get("data") or {})
        return Reply({"translations": []}, 200)

    monkeypatch.setattr(app, "_colab_request", fake_request)
    app._stage_call("/translate", data={"model": "nllb"})
    assert "async_mode" not in sent
