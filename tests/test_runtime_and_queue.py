from __future__ import annotations

import threading
import time

import pytest

from core import runtime


def test_a_slot_holds_one_model_and_evicts_on_a_new_key():
    slot = runtime.Slot("test")
    released = []

    class Model:
        def __init__(self, name):  # noqa: ANN001
            self.name = name

        def __del__(self):
            released.append(self.name)

    first = slot.get("a", lambda: Model("a"))
    assert slot.get("a", lambda: pytest.fail("rebuilt an identical key")) is first
    slot.get("b", lambda: Model("b"))
    assert slot.key == "b"
    slot.release()
    assert slot.key is None


def test_every_task_has_its_own_slot():
    assert set(runtime.SLOTS) == {
        "asr", "translation", "tts", "diarization", "separation"
    }
    runtime.slot("asr").get("x", lambda: object())
    assert runtime.loaded()["asr"] == "x"
    assert runtime.loaded()["tts"] is None
    runtime.release_all()
    assert set(runtime.loaded().values()) == {None}


def test_a_slot_is_safe_to_share_between_threads():
    slot = runtime.Slot("test")
    built = []

    def build():  # noqa: ANN202
        time.sleep(0.05)
        built.append(1)
        return object()

    threads = [threading.Thread(target=lambda: slot.get("same", build)) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(built) == 1


def test_device_falls_back_to_cpu_without_torch(monkeypatch):
    monkeypatch.setattr(runtime, "_device", None)
    monkeypatch.setenv("DUBFLOW_DEVICE", "cpu")
    assert runtime.device() == "cpu"


def test_the_queue_runs_jobs_one_at_a_time_in_order(monkeypatch, tmp_path):
    import jobs

    monkeypatch.setattr(jobs, "ROOT", tmp_path)
    running = []
    order = []
    overlapped = []

    def fake_process(job_id):  # noqa: ANN001
        running.append(job_id)
        if len(running) > 1:
            overlapped.append(tuple(running))
        time.sleep(0.05)
        order.append(job_id)
        running.remove(job_id)

    monkeypatch.setattr(jobs, "process", fake_process)
    for name in ("aaaaaaaaaaaa", "bbbbbbbbbbbb", "cccccccccccc"):
        jobs.enqueue(name)

    deadline = time.time() + 5
    while len(order) < 3 and time.time() < deadline:
        time.sleep(0.02)

    assert order == ["aaaaaaaaaaaa", "bbbbbbbbbbbb", "cccccccccccc"]
    assert overlapped == [], "two jobs ran at once; the GPU would be shared"


def test_a_job_deleted_while_queued_does_not_crash_the_worker(monkeypatch, tmp_path):
    import jobs

    monkeypatch.setattr(jobs, "ROOT", tmp_path)
    jobs.process("dddddddddddd")  # never created; must return quietly
