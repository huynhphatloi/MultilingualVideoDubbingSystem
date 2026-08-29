"""The two sides of the remote contract, talking to each other for real.

``test_remote_tts.py`` drives the client against a hand-written fake server.
That proves the CLIENT is right about the contract - it cannot prove the server
in ``colab/server.py`` implements the same one. The failure this file exists to
catch is a field renamed on one side only: the client keeps sending
``reference_id`` while the server started reading ``ref_id``, every call 409s,
and the first sign of it is a job that silently spent an hour on MMS-TTS.

All three stages are covered - transcribe, translate, synthesize - because
all three now run on the notebook: whisper-medium (1.4 GB) and NLLB-200
(2.3 GB) were the largest things this machine used to cache, and moving them
is the whole reason the local `models/` directory is empty.

The models are stubbed. What is under test is the HTTP shape - form field
names, status codes, headers, the reference cache - not audio quality, and
nobody should need 4 GB of weights to check a field name.

Skipped when ``colab/`` or FastAPI is not importable, so the suite still runs
on a machine set up only for the client.
"""
from __future__ import annotations

import math
import struct
import sys
import threading
import time
import wave
from pathlib import Path

import pytest

_COLAB = Path(__file__).resolve().parents[2] / "colab"
pytest.importorskip("fastapi", reason="colab server needs fastapi")
pytest.importorskip("uvicorn", reason="colab server needs uvicorn")
if not (_COLAB / "server.py").exists():  # pragma: no cover
    pytest.skip("colab/ is not present", allow_module_level=True)


def _sine(path: Path, seconds: float = 0.6, rate: int = 24000) -> Path:
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes(b"".join(
            struct.pack("<h", int(12000 * math.sin(2 * math.pi * 180 * i / rate)))
            for i in range(int(rate * seconds))))
    return path


@pytest.fixture(scope="module")
def colab_server(tmp_path_factory):  # noqa: ANN001, ANN201
    """The real colab/server.py, serving two stub engines on a real port."""
    import os

    tmp = tmp_path_factory.mktemp("colab")
    os.environ.update(
        AUTH_TOKEN="tok", DEFAULT_ENGINE="vixtts", SAMPLE_RATE="24000",
        REF_DIR=str(tmp / "refs"), OUT_DIR=str(tmp / "out"),
    )
    sys.path.insert(0, str(_COLAB))
    try:
        import engines
        from engines.tts_base import Spoken, TTSEngine
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"colab package not importable: {exc}")

    class _Stub(TTSEngine):
        voice_cloning = True
        languages = ("vi",)

        def load(self) -> None:
            pass

        def speak(self, text, language, out_path,  # noqa: ANN001
                  reference_wav=None, speed=1.0):  # noqa: ANN001, ANN202
            # The server is responsible for materialising the reference before
            # the engine sees it; assert that here rather than in a test, so a
            # regression shows up as a 500 on every call instead of silence.
            assert reference_wav is not None and Path(reference_wav).exists()
            _sine(out_path)
            return Spoken(wav_path=out_path, sample_rate=24000, seconds=0.6,
                          compute_seconds=0.01, voice_cloned=True, engine=self.name)

    class _Vi(_Stub):
        name, title = "vixtts", "stub vixtts"

    class _F5(_Stub):
        name, title = "f5_vi", "stub f5"

    engines._CLASSES = (_Vi, _F5)
    engines._INSTANCES.clear()

    # Stub the stage models: loading Whisper and NLLB for a contract test would
    # download the 3.7 GB this whole change exists to avoid.
    import stages

    def _fake_transcribe(audio_path, *, language, model, word_timestamps,  # noqa: ANN001
                         beam_size, vad_filter, condition_on_previous_text):  # noqa: ANN001
        assert Path(audio_path).exists(), "server did not save the upload"
        return {
            "language": language or "en",
            "language_confidence": 0.97,
            "duration": 3.0,
            "model": model,
            "windows": [
                {"start": 0.0, "end": 1.4, "source_text": "Call it, Captain.",
                 "avg_logprob": -0.2, "no_speech_prob": 0.01},
                {"start": 1.8, "end": 3.0, "source_text": "All right, listen up.",
                 "avg_logprob": -0.3, "no_speech_prob": 0.02},
            ],
        }

    def _fake_translate(items, *, source_language, target_language, engine,  # noqa: ANN001
                        num_beams, max_new_tokens):  # noqa: ANN001
        if not stages.flores(target_language):
            raise ValueError(f"no FLORES-200 code for {target_language!r}")
        return {
            "engine": engine,
            "results": [
                {"text": f"[{target_language}] {i['text']}",
                 "candidates": [f"[{target_language}] {i['text']}", "shorter"]}
                for i in items
            ],
        }

    stages.transcribe = _fake_transcribe
    stages.translate = _fake_translate
    stages.free_vram = lambda: []
    stages.loaded = lambda: {"whisper": [], "translation": []}

    import server
    import uvicorn

    config = uvicorn.Config(server.app, host="127.0.0.1", port=0, log_level="error")
    running = uvicorn.Server(config)
    threading.Thread(target=running.run, daemon=True).start()
    for _ in range(100):
        if running.started and running.servers:
            break
        time.sleep(0.05)
    else:  # pragma: no cover
        pytest.skip("colab server did not start")

    port = running.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}", engines, tmp / "refs"
    running.should_exit = True
    sys.path.remove(str(_COLAB))


@pytest.fixture()
def client(colab_server):  # noqa: ANN001, ANN201
    """The real RemoteAdapter set, pointed at the real server."""
    from app.core.config import settings
    from app.services.tts import remote as remote_mod
    from app.services.tts import router

    url = colab_server[0]
    saved = {k: getattr(settings, k) for k in
             ("remote_url", "remote_token", "remote_timeout", "remote_retries",
              "remote_asr_enabled", "remote_translation_enabled",
              "remote_tts_url", "remote_tts_token", "remote_tts_languages",
              "remote_tts_voice_clone", "remote_tts_retries", "remote_tts_timeout",
              "remote_tts_engines")}
    # One notebook serves all three stages, so the shared base URL is what the
    # ASR and translation paths read; the TTS one is set too because the
    # adapter still accepts a per-stage override.
    settings.remote_url = url
    settings.remote_token = "tok"
    settings.remote_timeout = 30.0
    settings.remote_retries = 1
    settings.remote_tts_url = url
    settings.remote_tts_token = "tok"
    settings.remote_tts_languages = "vi"
    settings.remote_tts_voice_clone = True
    settings.remote_tts_retries = 1
    settings.remote_tts_timeout = 30.0
    settings.remote_tts_engines = "vixtts,f5_vi"
    remote_mod._UPLOADED.clear()
    router._adapters.cache_clear()
    yield url, router._adapters()
    remote_mod._UPLOADED.clear()
    for key, value in saved.items():
        setattr(settings, key, value)
    router._adapters.cache_clear()


def _request(tmp_path: Path, name: str):  # noqa: ANN202
    from app.services.tts.base import SynthesisRequest

    return SynthesisRequest(
        text="Chúng ta không còn nhiều thời gian.", language="vi",
        output_path=tmp_path / name,
        voice_reference=_sine(tmp_path / "ref.wav", seconds=3.0),
        speaker_id="SPEAKER_00",
    )


# ------------------------------------------------------------------ tests ----
def test_health_answers_before_anything_is_loaded(colab_server):
    """Discovery must not cost a model load, or /health times out on a cold box."""
    import httpx

    body = httpx.get(f"{colab_server[0]}/health").json()

    assert body["engine"] == "vixtts"
    assert [e["name"] for e in body["engines"]] == ["vixtts", "f5_vi"]
    assert body["resident"] is None


@pytest.mark.parametrize("engine", ["vixtts", "f5_vi"])
def test_each_configured_engine_round_trips(client, tmp_path, engine):
    """The whole point: one parameter picks the model, and the answer says which."""
    _, adapters = client
    result = adapters[f"remote:{engine}"].synthesize(_request(tmp_path, f"{engine}.wav"))

    assert result.model == f"remote:{engine}"
    assert result.path.exists()
    assert result.duration > 0
    assert result.voice_cloned is True


def test_the_reference_goes_up_once_across_many_calls(client, colab_server, tmp_path):
    """Three calls for one character, one upload. This is most of the bandwidth."""
    _, adapters = client
    ref_dir = colab_server[2]
    for existing in ref_dir.glob("*.wav"):
        existing.unlink()

    for i in range(3):
        adapters["remote:vixtts"].synthesize(_request(tmp_path, f"call_{i}.wav"))

    assert len(list(ref_dir.glob("*.wav"))) == 1


def test_a_restarted_runtime_answers_409_and_the_segment_survives(client, colab_server,
                                                                  tmp_path):
    """The real 409 path, not the fake one - this is where the two sides drift."""
    _, adapters = client
    adapter = adapters["remote:vixtts"]
    adapter.synthesize(_request(tmp_path, "before.wav"))

    ref_dir = colab_server[2]
    for stale in ref_dir.glob("*.wav"):
        stale.unlink()          # the notebook died; the client still holds the hash

    result = adapter.synthesize(_request(tmp_path, "after.wav"))

    assert result.path.exists()
    assert len(list(ref_dir.glob("*.wav"))) == 1


@pytest.mark.parametrize(("payload", "status", "why"), [
    ({"text": "hi", "language": "ja", "engine": "vixtts"}, 400,
     "an unsupported language is a considered no - the client must not retry it"),
    ({"text": "hi", "language": "vi", "engine": "nope"}, 400,
     "an unknown engine id is a typo in .env, not a transient failure"),
])
def test_the_server_says_no_with_a_4xx(colab_server, payload, status, why):
    import httpx

    response = httpx.post(f"{colab_server[0]}/synthesize",
                          headers={"Authorization": "Bearer tok"}, data=payload)
    assert response.status_code == status, why


def test_a_missing_token_is_rejected(colab_server):
    import httpx

    response = httpx.post(f"{colab_server[0]}/synthesize",
                          data={"text": "hi", "language": "vi"})
    assert response.status_code == 401


def test_switching_engines_leaves_only_one_resident(client, tmp_path, colab_server):
    """A T4 holds one cloning model. Two residents is an OOM twenty segments in."""
    _, adapters = client
    engines = colab_server[1]

    adapters["remote:vixtts"].synthesize(_request(tmp_path, "a.wav"))
    adapters["remote:f5_vi"].synthesize(_request(tmp_path, "b.wav"))

    assert [n for n, e in engines._INSTANCES.items() if e.loaded] == ["f5_vi"]


# ------------------------------------------------------- ASR on the notebook --
def test_the_client_uploads_compressed_audio_and_gets_windows_back(client, tmp_path):
    """The client must send Opus, not wav.

    A ten-minute film is 19 MB of wav and 1.7 MB of Opus. Over a home upstream
    that is the difference between the stage taking seconds and taking minutes,
    so "it still works with wav" is not good enough - the compression IS the
    feature, and a regression to wav would be invisible on a 3-second fixture.
    """
    from app.core.config import settings
    from app.services import asr, ffmpeg

    settings.remote_asr_enabled = True
    try:
        source = tmp_path / "speech.wav"
        ffmpeg.silence(3.0, source, sample_rate=16000)

        class _Ws:
            def path(self, *parts):  # noqa: ANN002, ANN202
                target = tmp_path.joinpath(*parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                return target

        windows, language, confidence, duration = asr._decode_remote(
            _Ws(), source, None, "medium", word_timestamps=True)

        assert language == "en"
        assert confidence == pytest.approx(0.97)
        assert duration == pytest.approx(3.0)
        assert [w["source_text"] for w in windows] == [
            "Call it, Captain.", "All right, listen up."]

        opus = tmp_path / "scratch" / "asr_upload.opus"
        assert opus.exists(), "the client sent uncompressed audio"
        assert opus.stat().st_size < source.stat().st_size
    finally:
        settings.remote_asr_enabled = False


def test_remote_asr_is_off_unless_explicitly_enabled(client):
    """It is the one stage that uploads audio rather than text, so it must not
    switch itself on just because an endpoint happens to be configured."""
    from app.core.config import settings
    from app.services import asr

    settings.remote_asr_enabled = False
    assert asr._use_remote() is False


# ----------------------------------------------- translation on the notebook --
def test_translation_sends_the_whole_job_in_one_call(client):
    """Batched on purpose: per-line round-trips turn a 2-second stage into a
    40-second one over a home connection."""
    from app.core.config import settings
    from app.services.translation.base import TranslationRequest
    from app.services.translation.remote import RemoteTranslationEngine

    settings.remote_translation_enabled = True
    try:
        requests = [
            TranslationRequest(text=t, source_language="en", target_language="vi",
                               char_budget=budget)
            for t, budget in [("Call it, Captain.", 20), ("All right, listen up.", 24)]
        ]
        results = RemoteTranslationEngine().translate_batch(requests)

        assert len(results) == 2
        assert results[0].text == "[vi] Call it, Captain."
        assert results[0].engine.startswith("remote:")
        # Candidates come back so duration-aware selection stays client-side.
        assert len(results[0].candidates) == 2
        assert results[0].char_budget == 20
    finally:
        settings.remote_translation_enabled = False


def test_a_length_mismatch_is_caught_rather_than_shifting_every_subtitle(
        client, monkeypatch):
    """Returning n-1 results would move every line one slot up: a dub that
    renders perfectly and is wrong from the first word to the last."""
    from app.core.config import settings
    from app.core.errors import TranslationError
    from app.services import remote_gpu
    from app.services.translation.base import TranslationRequest
    from app.services.translation.remote import RemoteTranslationEngine

    settings.remote_translation_enabled = True
    monkeypatch.setattr(remote_gpu, "post",
                        lambda *a, **k: {"engine": "nllb", "results": [{"text": "one"}]})
    try:
        with pytest.raises(TranslationError, match="1 result"):
            RemoteTranslationEngine().translate_batch([
                TranslationRequest(text=t, source_language="en", target_language="vi")
                for t in ("a", "b")
            ])
    finally:
        settings.remote_translation_enabled = False


def test_an_unknown_language_pair_is_a_4xx_the_client_does_not_retry(colab_server):
    """Retrying an unsupported pair just burns the tunnel."""
    import httpx

    response = httpx.post(
        f"{colab_server[0]}/translate",
        headers={"Authorization": "Bearer tok"},
        json={"source_language": "en", "target_language": "xx",
              "items": [{"text": "hello"}]},
    )
    assert response.status_code == 400


def test_an_empty_batch_does_not_reach_the_gpu(client):
    from app.services.translation.remote import RemoteTranslationEngine

    assert RemoteTranslationEngine().translate_batch([]) == []
