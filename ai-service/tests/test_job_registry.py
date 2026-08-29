"""Regression tests for the job registry.

These run against a real file-backed SQLite database with foreign keys ON, so
each session gets its OWN connection and therefore its OWN transaction - the
same isolation Postgres gives us. That matters: the bug these tests were
written for was a job row that had only been *flushed*, never *committed*,
while a second session tried to insert its child stage rows and hit

    ForeignKeyViolation: Key (job_id)=(...) is not present in table "jobs"
"""
from __future__ import annotations

import pytest
from app.jobs import db as jobs_db
from app.jobs import service
from app.jobs.models import PIPELINE_STAGES, Job, JobStage, JobStatus, StageStatus
from sqlalchemy import create_engine, event, select


@pytest.fixture
def registry(tmp_path):
    """A throwaway database wired into app.jobs.db."""
    engine = create_engine(f"sqlite:///{tmp_path/'jobs.db'}", future=True)

    @event.listens_for(engine, "connect")
    def _fk_on(dbapi_connection, _record):  # noqa: ANN202
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    jobs_db.Base.metadata.create_all(engine)
    jobs_db.configure(engine)
    yield jobs_db.get_session_factory()
    jobs_db.configure(jobs_db._build_engine("sqlite://"))   # leave no state behind


def _make_job(session_factory, **kw):
    with session_factory() as session:
        job = service.create_job(
            session,
            source_filename=kw.get("source_filename", "clip.mp4"),
            source_language=kw.get("source_language"),
            target_language=kw.get("target_language", "vi"),
            options=kw.get("options", {}),
        )
        return job.id


# ------------------------------------------------------------------ create ---
def test_create_job_is_committed_before_it_returns(registry):
    """The regression: a later, INDEPENDENT session must see the job."""
    job_id = _make_job(registry)

    with registry() as fresh:                      # brand new connection
        job = fresh.get(Job, job_id)
        assert job is not None, "job was not committed - child inserts will hit the FK"
        assert job.status == JobStatus.CREATED


def test_all_pipeline_stages_are_seeded(registry):
    job_id = _make_job(registry)
    with registry() as fresh:
        stages = fresh.execute(
            select(JobStage).where(JobStage.job_id == job_id).order_by(JobStage.sequence)
        ).scalars().all()
    assert [s.name for s in stages] == list(PIPELINE_STAGES)


def test_create_job_stage_is_closed_in_the_same_transaction(registry):
    job_id = _make_job(registry, target_language="ja")
    with registry() as fresh:
        stage = fresh.execute(
            select(JobStage).where(JobStage.job_id == job_id, JobStage.name == "create_job")
        ).scalar_one()
    assert stage.status == StageStatus.COMPLETED
    assert stage.output["target_language"] == "ja"


# ------------------------------------------------- cross-session stage flow --
def test_stage_tracking_works_from_independent_sessions(registry):
    """jobs.track() opens its own session - exactly what used to explode."""
    job_id = _make_job(registry)

    with service.track(job_id, "extract_audio") as out:
        out["audio_key"] = "jobs/x/audio/original.wav"

    with registry() as fresh:
        job = service.get_job(fresh, job_id)
        stage = next(s for s in job.stages if s.name == "extract_audio")
        assert stage.status == StageStatus.COMPLETED
        assert stage.output["audio_key"].endswith("original.wav")
        assert stage.duration_ms is not None
        assert job.status == JobStatus.RUNNING


def test_failure_inside_track_marks_both_stage_and_job(registry):
    from app.core.errors import NoSpeechDetected

    job_id = _make_job(registry)
    with pytest.raises(NoSpeechDetected), service.track(job_id, "transcribe"):
        raise NoSpeechDetected("nothing to dub")

    with registry() as fresh:
        job = service.get_job(fresh, job_id)
        stage = next(s for s in job.stages if s.name == "transcribe")
        assert stage.status == StageStatus.FAILED
        assert stage.error_code == "no_speech_detected"
        assert job.status == JobStatus.FAILED
        assert job.error_code == "no_speech_detected"


def test_rerunning_a_stage_increments_the_attempt_counter(registry):
    job_id = _make_job(registry)
    for _ in range(2):
        with service.track(job_id, "synthesize"):
            pass

    with registry() as fresh:
        stage = next(s for s in service.get_job(fresh, job_id).stages
                     if s.name == "synthesize")
    assert stage.attempt == 2


# ------------------------------------------------------------------ update ---
def test_update_job_merges_dicts_instead_of_replacing_them(registry):
    job_id = _make_job(registry)
    service.update_job(job_id, metrics={"ratio_stats": {"mean": 1.1}})
    service.update_job(job_id, metrics={"tts_models": {"xtts_v2": 12}})

    with registry() as fresh:
        job = service.get_job(fresh, job_id)
    assert set(job.metrics) == {"ratio_stats", "tts_models"}


def test_progress_reflects_completed_stages(registry):
    job_id = _make_job(registry)
    with registry() as fresh:
        assert service.get_job(fresh, job_id).progress()["completed"] == 1  # create_job

    for name in ("store_video", "extract_audio"):
        with service.track(job_id, name):
            pass

    with registry() as fresh:
        progress = service.get_job(fresh, job_id).progress()
    assert progress["completed"] == 3
    assert progress["total"] == len(PIPELINE_STAGES)
    assert progress["current_stage"] is None


def test_mark_completed_records_the_outputs(registry):
    job_id = _make_job(registry)
    service.mark_completed(job_id, output_key="jobs/x/output/dubbed_vi.mp4",
                           subtitle_key="jobs/x/subtitles/vi.srt")
    with registry() as fresh:
        job = service.get_job(fresh, job_id)
    assert job.status == JobStatus.COMPLETED
    assert job.output_key.endswith("dubbed_vi.mp4")
    assert job.error is None if hasattr(job, "error") else True


def test_unknown_job_raises_a_typed_error(registry):
    from app.core.errors import JobNotFound

    with registry() as fresh, pytest.raises(JobNotFound):
        service.get_job(fresh, "does-not-exist")


# ------------------------------------------------------------- liveness ------
def test_progress_flags_a_job_that_nobody_is_driving(registry):
    """2/13 + RUNNING used to look identical to 'a model is working'."""
    job_id = _make_job(registry)
    with service.track(job_id, "store_video"):
        pass

    with registry() as fresh:
        p = service.get_job(fresh, job_id).progress()

    assert p["current_stage"] is None
    assert p["waiting_for_trigger"] is True, "UI cannot tell stuck from busy without this"
    assert p["running_seconds"] is None
    assert p["idle_seconds"] is not None and p["idle_seconds"] >= 0


def test_progress_reports_the_stage_that_is_actually_running(registry):
    job_id = _make_job(registry)
    service.start_stage(job_id, "separate_sources")

    with registry() as fresh:
        p = service.get_job(fresh, job_id).progress()

    assert p["current_stage"] == "separate_sources"
    assert p["waiting_for_trigger"] is False
    assert p["running_seconds"] is not None


def test_finished_job_is_never_reported_as_waiting(registry):
    job_id = _make_job(registry)
    service.mark_completed(job_id, output_key="jobs/x/output/dubbed_vi.mp4")

    with registry() as fresh:
        p = service.get_job(fresh, job_id).progress()
    assert p["waiting_for_trigger"] is False


def test_failed_job_is_never_reported_as_waiting(registry):
    job_id = _make_job(registry)
    service.mark_failed(job_id, "ffmpeg_failed", "boom")

    with registry() as fresh:
        p = service.get_job(fresh, job_id).progress()
    assert p["waiting_for_trigger"] is False
