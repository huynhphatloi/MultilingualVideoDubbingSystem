from __future__ import annotations

import pytest

import pipeline
from core import config
from core.config import as_bool
from core.errors import InvalidRequest


class TestEmptyFormFields:
    def test_empty_string_falls_back_to_the_default(self):
        assert as_bool("", True) is True
        assert as_bool("   ", True) is True
        assert as_bool("", False) is False

    def test_explicit_values_still_win(self):
        assert as_bool("false", True) is False
        assert as_bool("true", False) is True
        assert as_bool("on", False) is True
        assert as_bool(None, True) is True

    def test_a_nonsense_value_is_rejected_rather_than_guessed(self):
        with pytest.raises(InvalidRequest):
            as_bool("maybe", True)

    def test_the_n8n_empty_flag_path_works(self, all_installed):
        """n8n cannot omit a form field: it sends every parameter it declares,
        empty when the operator left it alone. An empty flag must read as
        absent, not as false."""
        built = config.build({
            "target_language": "vi",
            "tts_model": "edge",
            "enable_diarization": "false",
            "enable_alignment": "",
            "enable_source_separation": "false",
        })
        assert built.features.alignment is True
        assert built.features.diarization is False

    def test_an_empty_alignment_flag_keeps_the_default(self, all_installed):
        built = config.build({"target_language": "vi", "enable_alignment": ""})
        assert built.features.alignment is True


class TestStageRerun:
    def test_completed_stages_holds_each_name_once(self, tmp_path, monkeypatch):
        import jobs

        monkeypatch.setattr(jobs, "ROOT", tmp_path)
        job = {"completed_stages": ["extract"], "skipped_stages": ["diarize"]}
        jobs._mark_completed(job, "extract")
        jobs._mark_completed(job, "transcribe")
        assert job["completed_stages"] == ["extract", "transcribe"]

    def test_a_stage_that_used_to_skip_and_now_runs_stops_counting_as_skipped(self):
        import jobs

        job = {"completed_stages": [], "skipped_stages": ["align"]}
        jobs._mark_completed(job, "align")
        assert job["completed_stages"] == ["align"]
        assert job["skipped_stages"] == []


class TestModelResidency:
    def default_job(self, **features):  # noqa: ANN201
        base = {"diarization": False, "alignment": True,
                "source_separation": False}
        base.update(features)
        return {"config": {"features": base}}

    def test_each_slot_is_released_exactly_once_at_its_last_use(self):
        job = self.default_job(diarization=True, source_separation=True)
        released = {
            name: pipeline.released_after(name, job)
            for name in pipeline.planned_stages(job)
        }
        assert released["diarize"] == ["diarization"]
        assert released["transcribe"] == ["asr"]
        assert released["translate"] == ["translation"]
        assert released["synthesize"] == ["tts"]
        assert released["separate"] == ["separation"]
        assert released["mix"] == [] and released["render"] == []
        flat = [slot for names in released.values() for slot in names]
        assert len(flat) == len(set(flat)), "a slot was released twice"

    def test_a_skipped_stage_never_holds_a_slot(self):
        job = self.default_job()
        assert "diarize" not in pipeline.planned_stages(job)
        assert pipeline.released_after("diarize", job) == []
        assert pipeline.released_after("separate", job) == []

    def test_the_pipeline_frees_the_recogniser_before_the_voice_loads(
        self, tmp_path, monkeypatch
    ):
        import jobs
        from core import runtime

        monkeypatch.setattr(jobs, "ROOT", tmp_path)
        monkeypatch.setattr(jobs, "KEEP_MODELS", False)
        runtime.release_all()
        resident = []

        def fake_run_stage(name, job, folder):  # noqa: ANN001
            slot_name = pipeline.STAGE_SLOTS.get(name)
            if slot_name:
                runtime.slot(slot_name).get(f"{name}-model", lambda: object())
            resident.append(
                (name, sorted(k for k, v in runtime.loaded().items() if v))
            )
            return True

        monkeypatch.setattr(pipeline, "run_stage", fake_run_stage)
        job = {
            "job_id": "eeeeeeeeeeee", "status": "queued", "completed_stages": [],
            "skipped_stages": [], "files": {}, "segments": [],
            "config": {"target_language": "vi", "features": {
                "diarization": False, "alignment": True,
                "source_separation": False}},
            "request": {"target_language": "vi"},
        }
        (tmp_path / "eeeeeeeeeeee").mkdir()
        jobs.write(job)
        jobs.process("eeeeeeeeeeee")

        assert jobs.read("eeeeeeeeeeee")["status"] == "completed"
        held = dict(resident)
        assert held["transcribe"] == ["asr"]
        assert held["translate"] == ["translation"], "the recogniser was still resident"
        assert held["synthesize"] == ["tts"], "the translator was still resident"
        assert all(len(slots) <= 1 for slots in held.values())
        runtime.release_all()


class TestAlignmentPlanLength:
    def test_one_plan_per_segment_even_when_nothing_was_measured(self):
        from dubflow_core import alignment

        segments = [
            {"start": 0.0, "end": 2.0},
            {"start": 2.0, "end": 4.0, "tts_duration_raw": 4.0},
            {"start": 4.0, "end": 6.0},
        ]
        plans = alignment.plan_segments(segments, media_duration=10.0)
        assert len(plans) == len(segments), "plans would zip onto the wrong lines"
        assert plans[0].status == "unmeasured"
        assert plans[1].status in {"aligned", "clamped"}


class TestSharedMixing:
    def test_the_two_services_use_the_same_gains(self):
        from dubflow_core import mixing

        voiceover = mixing.blend_filtergraph(separated=False)
        separated = mixing.blend_filtergraph(separated=True)
        assert "volume=0.25" in voiceover and "volume=1.5" in voiceover
        assert "volume=1.0" in separated
        assert voiceover.count("aresample=48000") == 2

    def test_atempo_is_chained_past_the_filter_limit(self):
        from dubflow_core import mixing

        assert mixing.atempo_filter(1.25) == "atempo=1.2500"
        assert mixing.atempo_filter(3.0) == "atempo=2.0,atempo=1.5000"
        assert mixing.atempo_filter(0.3) == "atempo=0.5,atempo=0.6000"

    def test_neither_service_still_builds_its_own_graph(self):
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        for path in (root / "ai-service/app.py", root / "colab/pipeline/mix.py"):
            source = path.read_text(encoding="utf-8")
            assert "loudnorm" not in source, f"{path.name} builds its own blend"
            assert "adelay=" not in source, f"{path.name} builds its own dub track"
            assert "atempo=" not in source, f"{path.name} chains its own atempo"


class TestOutdatedBackend:
    def service(self, monkeypatch, status=404, health=None):  # noqa: ANN201
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ai-service"))
        import app

        class Response:
            status_code = status
            text = "Not Found"

            def json(self):  # noqa: ANN201
                return {"detail": "Not Found"}

        monkeypatch.setattr(app, "_resolve_backend",
                            lambda force=False: ("colab", "https://x.example", ""))
        monkeypatch.setattr(app.httpx, "request", lambda *a, **k: Response())
        monkeypatch.setattr(app, "_backend_version", lambda url, token: health)
        monkeypatch.setattr(app, "_capabilities_cache", None)
        return app

    def test_every_new_endpoint_reports_the_version_and_the_restart(self, monkeypatch):
        app = self.service(monkeypatch, health="3.0")
        for path in app.MODERN_ENDPOINTS:
            with pytest.raises(app.OutdatedBackend) as failure:
                app._colab_request(path, json={})
            message = str(failure.value.detail)
            assert "running AI service 3.0" in message
            assert path in message
            assert "Restart session" in message
            assert "make colab" in message

    def test_a_session_reporting_no_version_is_described_honestly(self, monkeypatch):
        app = self.service(monkeypatch, health=None)
        with pytest.raises(app.OutdatedBackend) as failure:
            app._colab_request("/validate", json={})
        assert "a build older than 4.0" in str(failure.value.detail)

    def test_capabilities_labels_it_outdated_rather_than_offline(self, monkeypatch):
        app = self.service(monkeypatch, health="3.0")
        payload = app.capabilities()
        assert payload["available"] is False
        assert payload["reason"] == "outdated"
        assert "Restart session" in payload["hint"]
        assert payload["languages"] and payload["stages"]

    def test_an_ordinary_error_is_not_blamed_on_the_version(self, monkeypatch):
        app = self.service(monkeypatch, status=500)
        with pytest.raises(Exception) as failure:
            app._colab_request("/validate", json={})
        assert not isinstance(failure.value, app.OutdatedBackend)
