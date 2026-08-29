"""The remote TTS contract, exercised against a real socket.

The interesting behaviour is not "does httpx post" - it is the reference cache.
The pipeline calls synthesize once or twice per segment and every call for one
character carries the same multi-second reference wav. Uploading it hundreds of
times over a home connection is most of the bandwidth for none of the benefit,
so the file goes up once and is addressed by its sha1 afterwards. The failure
mode that matters is a restarted notebook: its cache is empty, it answers 409,
and the client must notice and re-upload instead of failing the segment.
"""
from __future__ import annotations

import json
import threading
import urllib.parse
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from app.services.tts import remote as remote_mod


def _wav(path: Path, seconds: float = 0.5, rate: int = 24000) -> Path:
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes(b"\x00\x00" * int(rate * seconds))
    return path


class _State:
    def __init__(self) -> None:
        self.known: set[str] = set()
        self.calls: list[dict] = []
        self.forget_after_first = False


def _handler(state: _State):  # noqa: ANN202
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # noqa: ANN002, ANN201 - silence the test run
            pass

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("content-length", 0))
            body = self.rfile.read(length)
            ctype = self.headers.get("content-type", "")

            # httpx only switches to multipart when there is a file to send;
            # hash-only calls arrive urlencoded. Both shapes are valid for the
            # real server (FastAPI's Form accepts either), so the fake has to
            # read both or it silently never sees a reference_id.
            def part(name: str):  # noqa: ANN202 - one multipart field, by name
                marker = f'name="{name}"\r\n\r\n'.encode()
                if marker not in body:
                    return None
                return body.split(marker, 1)[1].split(b"\r\n", 1)[0].decode()

            if ctype.startswith("multipart/"):
                has_file = b'name="reference"' in body
                ref_id = part("reference_id")
                engine = part("engine")
            else:
                fields = urllib.parse.parse_qs(body.decode())
                has_file = False
                ref_id = (fields.get("reference_id") or [None])[0]
                engine = (fields.get("engine") or [None])[0]

            state.calls.append({"has_file": has_file, "reference_id": ref_id,
                                "engine": engine})

            if has_file and ref_id:
                state.known.add(ref_id)
            elif ref_id and ref_id not in state.known:
                payload = json.dumps({"error": "missing_reference"}).encode()
                self.send_response(409)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return

            if state.forget_after_first:
                state.known.clear()

            audio = Path(self.server.tmp) / "reply.wav"  # type: ignore[attr-defined]
            _wav(audio)
            data = audio.read_bytes()
            self.send_response(200)
            self.send_header("content-type", "audio/wav")
            self.send_header("content-length", str(len(data)))
            self.send_header("X-TTS-Model", engine or "vixtts")
            self.send_header("X-TTS-Voice-Cloned", "true")
            self.send_header("X-TTS-Sample-Rate", "24000")
            self.end_headers()
            self.wfile.write(data)

    return Handler


@pytest.fixture()
def server(tmp_path):  # noqa: ANN001, ANN201
    state = _State()
    httpd = HTTPServer(("127.0.0.1", 0), _handler(state))
    httpd.tmp = str(tmp_path)  # type: ignore[attr-defined]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}", state
    httpd.shutdown()


_TUNABLES = ("remote_tts_url", "remote_tts_token", "remote_tts_languages",
             "remote_tts_voice_clone", "remote_tts_retries", "remote_tts_timeout")


@pytest.fixture(autouse=True)
def _isolate():
    """Reset the upload cache and restore settings AFTER each test.

    Restoring inside the builder would be wrong: retries and timeout are read
    per call inside _post, not captured in __init__, so an early restore would
    quietly hand every test the production values.
    """
    from app.core.config import settings

    saved = {k: getattr(settings, k) for k in _TUNABLES}
    remote_mod._UPLOADED.clear()
    yield
    remote_mod._UPLOADED.clear()
    for k, v in saved.items():
        setattr(settings, k, v)


def _adapter(url: str, **overrides):  # noqa: ANN202
    from app.core.config import settings

    settings.remote_tts_url = url
    settings.remote_tts_languages = overrides.get("languages", "vi")
    settings.remote_tts_voice_clone = overrides.get("voice_clone", True)
    settings.remote_tts_retries = overrides.get("retries", 2)
    settings.remote_tts_timeout = 10.0
    settings.remote_tts_token = overrides.get("token", "")
    return remote_mod.RemoteAdapter(overrides.get("engine", ""))


def _request(tmp_path: Path, name: str = "out.wav"):  # noqa: ANN202
    from app.services.tts.base import SynthesisRequest

    return SynthesisRequest(
        text="Chúng ta không còn nhiều thời gian.",
        language="vi",
        output_path=tmp_path / name,
        voice_reference=_wav(tmp_path / "ref.wav", seconds=3.0),
        speaker_id="SPEAKER_00",
    )


# ------------------------------------------------------------- the contract --
def test_first_call_uploads_reference_and_the_rest_send_only_the_hash(server, tmp_path):
    url, state = server
    adapter = _adapter(url)

    for i in range(4):
        result = adapter.synthesize(_request(tmp_path, f"out_{i}.wav"))
        assert result.path.exists()

    assert [c["has_file"] for c in state.calls] == [True, False, False, False]
    # One reference, one upload - not four.
    assert len({c["reference_id"] for c in state.calls}) == 1


def test_result_reports_the_engine_that_spoke_not_the_transport(server, tmp_path):
    url, _ = server
    result = _adapter(url).synthesize(_request(tmp_path))
    # A manifest saying `remote: 32` tells you nothing when the point of the run
    # was to compare viXTTS against F5.
    assert result.model == "remote:vixtts"
    assert result.voice_cloned is True
    assert result.duration > 0


def test_a_restarted_server_answers_409_and_the_client_re_uploads(server, tmp_path):
    url, state = server
    adapter = _adapter(url)

    adapter.synthesize(_request(tmp_path, "first.wav"))
    assert state.calls[-1]["has_file"] is True

    # The notebook died and came back with an empty reference cache.
    state.known.clear()
    result = adapter.synthesize(_request(tmp_path, "second.wav"))

    assert result.path.exists()
    # Sent the hash, got 409, resent the file - all inside one synthesize call.
    assert [c["has_file"] for c in state.calls[-2:]] == [False, True]


def test_unreachable_endpoint_raises_model_unavailable_so_the_router_can_fall_back(tmp_path):
    from app.core.errors import ModelUnavailable

    # Port 1 is reserved and refuses instantly - no waiting on a timeout.
    adapter = _adapter("http://127.0.0.1:1", retries=1)
    with pytest.raises(ModelUnavailable) as exc:
        adapter.synthesize(_request(tmp_path))
    assert "Colab tunnel URL changes" in str(exc.value.details["hint"])


# ---------------------------------------------------------------- routing ---
def test_unconfigured_adapter_supports_nothing(tmp_path):
    adapter = _adapter("")
    assert adapter.supports("vi") is False
    assert adapter.describe()["configured"] is False


def test_language_allowlist_keeps_english_on_xtts(server):
    url, _ = server
    adapter = _adapter(url, languages="vi")
    assert adapter.supports("vi") is True
    assert adapter.supports("en") is False


def test_empty_allowlist_trusts_the_endpoint(server):
    url, _ = server
    adapter = _adapter(url, languages="")
    assert adapter.supports("ja") is True


def test_cloning_endpoint_refuses_a_segment_with_no_reference(server, tmp_path):
    from app.core.errors import SynthesisError
    from app.services.tts.base import SynthesisRequest

    url, _ = server
    adapter = _adapter(url, voice_clone=True)
    request = SynthesisRequest(text="Đi thôi.", language="vi",
                               output_path=tmp_path / "x.wav", voice_reference=None)
    with pytest.raises(SynthesisError, match="reference"):
        adapter.synthesize(request)


# ------------------------------------------------- one endpoint, many engines --
# The comparison this project has to make ("is F5 worth it over viXTTS?") is only
# credible if switching engines changes ONE thing. These tests pin that down: the
# choice travels as a form field, the name is a stable force_model key, and the
# manifest records the engine that spoke rather than the one we asked for.
def test_named_adapter_pins_its_engine_on_every_request(server, tmp_path):
    url, state = server
    adapter = _adapter(url, engine="f5_vi")

    result = adapter.synthesize(_request(tmp_path))

    assert state.calls[-1]["engine"] == "f5_vi"
    assert adapter.name == "remote:f5_vi"
    assert result.model == "remote:f5_vi"


def test_unnamed_adapter_lets_the_server_choose(server, tmp_path):
    url, state = server
    adapter = _adapter(url)

    result = adapter.synthesize(_request(tmp_path))

    # No engine field at all - the endpoint applies its own DEFAULT_ENGINE.
    assert state.calls[-1]["engine"] is None
    assert adapter.name == "remote"
    assert result.model == "remote:vixtts"


def test_the_manifest_records_what_spoke_not_what_was_asked_for(server, tmp_path):
    """A server that serves a different engine than requested must be visible.

    Silently accepting the substitution is how a comparison ends up measuring
    the wrong model for half its rows and nobody notices until the write-up.
    """
    url, _ = server
    adapter = _adapter(url, engine="f5_vi")
    original = adapter._request

    def swap(client, u, headers, request, reference, digest, send_file):  # noqa: ANN001, ANN202
        response = original(client, u, headers, request, reference, digest, send_file)
        response.headers["X-TTS-Model"] = "vixtts"   # the endpoint fell back
        return response

    adapter._request = swap
    assert adapter.synthesize(_request(tmp_path)).model == "remote:vixtts"


def test_router_registers_one_adapter_per_configured_engine(server):
    from app.core.config import settings
    from app.services.tts import router

    url, _ = server
    settings.remote_tts_url = url
    settings.remote_tts_engines = "vixtts,f5_vi"
    router._adapters.cache_clear()
    try:
        names = list(router._adapters())
        assert names[:2] == ["remote:vixtts", "remote:f5_vi"]

        # force_model is the whole point: one parameter re-runs a real job on a
        # different engine, with NO fallback chain to contaminate the result.
        chain = router.resolve("vi", has_voice_reference=True,
                               force_model="remote:f5_vi")
        assert [a.name for a in chain] == ["remote:f5_vi"]
    finally:
        settings.remote_tts_engines = ""
        router._adapters.cache_clear()


def test_an_empty_engine_list_keeps_the_single_remote_adapter():
    from app.core.config import settings
    from app.services.tts import router

    settings.remote_tts_engines = ""
    router._adapters.cache_clear()
    try:
        assert "remote" in router._adapters()
    finally:
        router._adapters.cache_clear()


def test_engine_list_is_deduplicated_and_order_preserving():
    from app.core.config import settings

    saved = settings.remote_tts_engines
    settings.remote_tts_engines = " vixtts , f5_vi ,vixtts,, "
    try:
        assert settings.remote_tts_engine_list == ["vixtts", "f5_vi"]
    finally:
        settings.remote_tts_engines = saved
