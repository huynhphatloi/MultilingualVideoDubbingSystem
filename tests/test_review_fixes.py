"""Regressions for the defects found in review.

Each test names the failure it prevents, because every one of these shipped
looking correct.
"""
from __future__ import annotations

import pytest

import pipeline
from core import config
from core.config import as_bool
from core.errors import InvalidRequest


class TestEmptyFormFields:
    """A multipart form cannot omit a field; empty must mean "not provided"."""

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

    def test_the_n8n_automatic_cloning_path_works(self, all_installed):
        """The form sends every flag it declares, empty when untouched.

        Reading that empty string as "off" turned "Voice Cloning: Automatic"
        into "off", which then refused every clone-only engine - which is every
        Vietnamese voice this project ships.
        """
        built = config.build({
            "target_language": "vi",
            "tts_model": "f5_vi",
            "enable_diarization": "false",
            "enable_voice_cloning": "",      # "Automatic"
            "enable_alignment": "true",
            "enable_source_separation": "false",
            "enable_lip_sync": "false",
        })
        assert built.features.voice_cloning is True
        assert built.features.alignment is True
        assert built.features.diarization is False

    def test_an_empty_alignment_flag_keeps_the_default(self, all_installed):
        built = config.build({"target_language": "vi", "enable_alignment": ""})
        assert built.features.alignment is True


class TestStageRerun:
    """A retried stage must not be counted twice."""

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
    """A slot is freed as soon as no later stage of the job needs it."""

    def default_job(self, **features):  # noqa: ANN201
        base = {"diarization": False, "alignment": True, "voice_cloning": False,
                "source_separation": False, "lip_sync": False}
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
        """Peak memory is one model, not three - the point of the whole scheme."""
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
                "diarization": False, "alignment": True, "voice_cloning": False,
                "source_separation": False, "lip_sync": False}},
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
            {"start": 0.0, "end": 2.0},                            # no clip
            {"start": 2.0, "end": 4.0, "tts_duration_raw": 4.0},
            {"start": 4.0, "end": 6.0},                            # no clip
        ]
        plans = alignment.plan_segments(segments, media_duration=10.0)
        assert len(plans) == len(segments), "plans would zip onto the wrong lines"
        assert plans[0].status == "unmeasured"
        assert plans[1].status in {"aligned", "clamped"}


class TestSharedMixing:
    """Both routes build the same filter graphs from one module."""

    def test_the_two_services_use_the_same_gains(self):
        from dubflow_core import mixing

        voiceover = mixing.blend_filtergraph(separated=False)
        separated = mixing.blend_filtergraph(separated=True)
        assert "volume=0.25" in voiceover and "volume=1.5" in voiceover
        assert "volume=1.0" in separated
        # A separation stem comes back at the model's rate, not the mix rate.
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
