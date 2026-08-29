"""The model picker: what the UI is offered, and how the choice reaches stage 7.

Two things are under test and they fail in different ways.

**The catalogue** is what the dropdown renders. Its job is not to list the
engines that work - filtering to those would be easy and would also be the bug.
It must list every engine WITH the reason each one is or is not usable, because
"XTTS-v2: its 17 languages do not include 'vi'" is the single most useful thing
this system tells a new operator, and a dropdown containing only MMS-TTS makes
the pipeline look like it never had another option.

**The routing of the choice** goes UI -> job.options -> synthesize. It does NOT
go through the request, and that is deliberate: n8n drives synthesis with a
generic call and would otherwise have to thread the parameter through thirteen
stages. Putting it on the job means every driver honours it - n8n, the smoke
test, a hand-rolled curl - without any of them knowing it exists.
"""
from __future__ import annotations

import pytest
from app.services.tts import router


@pytest.fixture(autouse=True)
def _clean_router():
    """The adapter set is cached for the process; tests here change settings."""
    router._adapters.cache_clear()
    yield
    router._adapters.cache_clear()


@pytest.fixture()
def isolated_db(tmp_path):
    """Point the whole app at a throwaway sqlite file.

    jobs_db.configure() is the supported way in; reaching for the module
    globals instead is how a test silently writes into the real dev database
    when one of them gets renamed.
    """
    from app.jobs import db as jobs_db
    from app.jobs import models  # noqa: F401 - registers the tables on Base.metadata

    engine = jobs_db._build_engine(f"sqlite:///{tmp_path}/jobs.db")
    jobs_db.Base.metadata.create_all(engine)
    jobs_db.configure(engine)
    yield jobs_db
    jobs_db.configure(jobs_db._build_engine("sqlite://"))   # leave no state behind


def _by_id(entries: list[dict]) -> dict[str, dict]:
    return {e["id"]: e for e in entries}


# ---------------------------------------------------------------- catalogue --
def test_unusable_engines_are_listed_with_a_reason_not_filtered_out():
    entries = _by_id(router.catalog("vi"))

    # Present, unusable, and explained - all three matter.
    assert "xtts_v2" in entries
    assert entries["xtts_v2"]["available"] is False
    assert "vi" in entries["xtts_v2"]["reason"]

    assert entries["mms_tts"]["available"] is True
    assert entries["mms_tts"]["reason"] is None


def test_every_entry_carries_what_the_dropdown_needs_to_render():
    for entry in router.catalog("vi"):
        assert entry["label"] and entry["label"] != entry["id"], entry
        assert set(entry) == {"id", "label", "blurb", "voice_cloning",
                              "available", "reason"}


def test_auto_leads_the_list_and_is_always_available():
    """It is not an engine - it means "no force_model", the only mode with a
    fallback chain. It must never be the thing that is unavailable."""
    entries = router.catalog("vi")
    assert entries[0]["id"] == "auto"
    assert entries[0]["available"] is True
    assert entries[0]["voice_cloning"] is None


def test_the_catalogue_changes_with_the_language():
    """The whole reason it is an endpoint and not a constant in the frontend."""
    vietnamese = _by_id(router.catalog("vi"))
    english = _by_id(router.catalog("en"))

    assert vietnamese["xtts_v2"]["available"] is False
    assert english["xtts_v2"]["available"] is True


def test_a_disabled_engine_says_it_is_disabled_not_that_it_lacks_the_language():
    """F5-TTS speaks English. Reporting "does not speak 'en'" when it is merely
    switched off sends the reader hunting for the wrong fix."""
    from app.core.config import settings

    saved = settings.f5_tts_enabled
    settings.f5_tts_enabled = False
    try:
        reason = _by_id(router.catalog("en"))["f5_tts"]["reason"]
        assert "F5_TTS_ENABLED" in reason
        assert "does not speak" not in reason
    finally:
        settings.f5_tts_enabled = saved


def test_an_unconfigured_remote_points_at_the_setting_to_change():
    reason = _by_id(router.catalog("vi"))["remote"]["reason"]
    assert "REMOTE_TTS_URL" in reason


def test_configured_remote_engines_each_get_their_own_entry():
    """This is what makes the picker a comparison tool rather than a toggle."""
    from app.core.config import settings

    saved = (settings.remote_tts_url, settings.remote_tts_engines,
             settings.remote_tts_languages)
    settings.remote_tts_url = "http://gpu.example"
    settings.remote_tts_engines = "vixtts,f5_vi"
    settings.remote_tts_languages = "vi"
    router._adapters.cache_clear()
    try:
        entries = _by_id(router.catalog("vi"))
        assert entries["remote:vixtts"]["available"] is True
        assert entries["remote:f5_vi"]["available"] is True
        # Labels must be words, not slugs - the dropdown is read by a person.
        assert "viXTTS" in entries["remote:vixtts"]["label"]
        assert entries["remote:f5_vi"]["blurb"]
    finally:
        (settings.remote_tts_url, settings.remote_tts_engines,
         settings.remote_tts_languages) = saved


def test_cloning_engines_are_unavailable_without_a_voice_reference():
    """A cloning engine with nothing to clone from fails at stage 7. Saying so
    up front beats a job that dies two minutes in."""
    entries = _by_id(router.catalog("en", has_voice_reference=False))
    assert entries["xtts_v2"]["available"] is False
    assert "reference" in entries["xtts_v2"]["reason"]
    assert entries["mms_tts"]["available"] is True


# ------------------------------------------------------- choice -> stage 7 --
def test_the_choice_is_read_back_off_the_job(isolated_db):
    """job.options survives the whole run, so n8n never has to carry it."""
    from app.jobs import service as jobs

    with isolated_db.session_scope() as session:
        job = jobs.create_job(session, target_language="vi",
                              options={"tts_model": "remote:f5_vi"})
        job_id = job.id

    assert jobs.job_option(job_id, "tts_model") == "remote:f5_vi"
    assert jobs.job_option(job_id, "never_set") is None
    assert jobs.job_option(job_id, "never_set", "fallback") == "fallback"


def test_a_missing_job_yields_the_default_rather_than_failing_the_stage(isolated_db):
    """An option is never worth failing synthesis over."""
    from app.jobs import service as jobs

    assert jobs.job_option("does-not-exist", "tts_model", "auto") == "auto"


def test_auto_means_no_forcing_so_the_fallback_chain_stays_alive():
    """The route maps the picker's "auto" to force_model=None. Passing the
    literal string through would raise UnsupportedLanguage("Unknown TTS model
    'auto'") on every job created from the UI."""
    from app.core.errors import UnsupportedLanguage

    with pytest.raises(UnsupportedLanguage):
        router.resolve("vi", has_voice_reference=True, force_model="auto")

    chain = router.resolve("vi", has_voice_reference=True, force_model=None)
    assert chain, "auto must resolve to a real chain"
