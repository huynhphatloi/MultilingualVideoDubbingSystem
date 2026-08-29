"""Remote TTS adapter - synthesis over HTTP on somebody else's GPU.

Why this exists: F5-TTS and the CosyVoice family have kernels with no Metal
implementation, so on this Mac they silently fall back to CPU and a single line
takes longer than the clip it belongs to. The models themselves are fine; the
GPU is the problem. This adapter keeps every other stage (Demucs, Whisper,
NLLB, pyannote) on local MPS and sends ONLY the synthesis step to a CUDA box -
a Colab notebook, a Modal endpoint, an HF Space, a rented pod. What crosses the
wire is a line of text plus a few seconds of reference audio, never the video.

The contract is deliberately tiny so the server side fits in one notebook cell:

    GET  /health      -> {"engine", "languages", "voice_cloning", "sample_rate"}
    POST /synthesize  -> audio/wav bytes
         multipart: text, language, speed, reference_id?, reference?
         409 {"error": "missing_reference"} asks the client to resend the file

`reference_id` is what makes this usable over a home connection. The pipeline
calls synthesize once (sometimes twice) per segment, and every call for the
same character carries the same reference wav. Uploading it a few hundred times
is most of the bandwidth for none of the benefit, so the file goes up once and
is addressed by its sha1 afterwards. A restarted notebook forgets its cache and
answers 409, which re-uploads exactly the references still in use.

One endpoint, several engines. ``REMOTE_TTS_ENGINES=vixtts,f5_vi`` registers one
adapter per id - ``remote:vixtts``, ``remote:f5_vi`` - each pinning the ``engine``
field on its requests. That is what makes a comparison a one-parameter change:
``force_model="remote:f5_vi"`` re-runs a real job on a different engine, and the
per-segment ``tts_model`` column records which one spoke. Leaving the list empty
keeps the single unnamed ``remote`` adapter, which lets the server pick.
"""
from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

from app.core import languages
from app.core.config import settings
from app.core.errors import ModelUnavailable, SynthesisError
from app.services import ffmpeg
from app.services.tts.base import SpeechGenerator, SynthesisRequest, SynthesisResult

log = logging.getLogger(__name__)

#: sha1 of every reference the server has confirmed it holds. Process-wide:
#: two jobs dubbing the same source share the upload.
_UPLOADED: set[str] = set()

#: Labels for the engine ids `colab/engines` serves, so the model picker reads
#: as words rather than slugs. The endpoint is the authority (GET /engines
#: carries its own titles); this is a local convenience that must never be a
#: gate - an id missing here still works and simply shows as itself.
_ENGINE_LABELS: dict[str, tuple[str, str]] = {
    "vixtts": ("viXTTS (remote GPU, clones)",
               "XTTS-v2 fine-tuned on Vietnamese. Documented to struggle on "
               "sentences under 10 words - which is most subtitle lines."),
    "xtts_v2": ("XTTS-v2 (remote GPU, clones)",
                "The official 17-language checkpoint. No Vietnamese."),
    "f5_vi": ("F5-TTS Vietnamese (remote GPU, clones)",
              "Flow-matching, fine-tuned on ~1000h of ViVoice. Flat cost per "
              "line, so it fares better than viXTTS on short ones."),
    "f5_base": ("F5-TTS Base (remote GPU, clones)",
                "The official English/Chinese release."),
    "mms": ("MMS-TTS (remote GPU, one voice)",
            "The same single-voice model as local MMS, just on the GPU."),
    "piper": ("Piper (remote, CPU, one voice)",
              "Tiny and faster than real time. No cloning."),
    "edge": ("Edge-TTS (cloud reference, one voice)",
             "Commercial-grade naturalness, but it cannot clone and it is a "
             "closed service - a yardstick, not a shipping choice."),
}


def _digest(path: Path) -> str:
    h = hashlib.sha1()  # noqa: S324 - a cache key, not a security boundary
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class RemoteAdapter(SpeechGenerator):
    name = "remote"

    def __init__(self, engine: str = "") -> None:
        # Read once at construction: the router caches adapters for the process
        # lifetime, so re-reading settings per call would never see a change
        # anyway and only costs a lookup.
        self.engine = engine.strip()
        #: `remote` when the server picks, `remote:<id>` when this adapter pins
        #: one. The name is also the `force_model` key, so it has to be stable.
        self.name = f"remote:{self.engine}" if self.engine else "remote"
        label, blurb = _ENGINE_LABELS.get(
            self.engine, (f"{self.engine or 'Remote endpoint'} (remote GPU)", ""))
        self.title = label
        self.blurb = blurb or (
            "Whatever engine the configured endpoint serves by default."
            if not self.engine else "")
        # REMOTE_TTS_URL when set, otherwise the shared REMOTE_URL - one
        # notebook normally serves ASR, translation and synthesis alike.
        self.base_url = settings.tts_endpoint
        self.token = settings.tts_endpoint_token
        self.supports_cloning = settings.remote_tts_voice_clone
        self._allowed = {
            code.strip().lower()
            for code in (settings.remote_tts_languages or "").split(",")
            if code.strip()
        }

    # ------------------------------------------------------------ routing --
    def supports(self, language: str) -> bool:
        if not self.base_url:
            return False
        if not self._allowed:
            # Unset means "whatever the endpoint is serving". The operator
            # pointed us at it on purpose; trust them and let a 4xx say no.
            return True
        return (languages.normalize(language) or "") in self._allowed

    def unavailable_reason(self, language: str) -> str:
        if not self.base_url:
            return "no GPU endpoint — set REMOTE_TTS_URL (see colab/README.md)"
        return f"REMOTE_TTS_LANGUAGES does not list '{language}'"

    def describe(self) -> dict:
        return {
            "name": self.name,
            "title": self.title,
            "blurb": self.blurb,
            "voice_cloning": self.supports_cloning,
            "configured": bool(self.base_url),
            "endpoint": self.base_url or None,
            "engine": self.engine or None,
        }

    # ---------------------------------------------------------- synthesis --
    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        if not self.base_url:
            raise SynthesisError("Remote TTS is not configured (set REMOTE_TTS_URL)")

        reference = Path(request.voice_reference) if request.voice_reference else None
        if reference is not None and not reference.exists():
            reference = None
        if self.supports_cloning and reference is None:
            raise SynthesisError("Remote TTS is configured to clone but got no reference wav")

        out = Path(request.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        audio, headers = self._post(request, reference)
        out.write_bytes(audio)
        if out.stat().st_size == 0:
            raise SynthesisError("Remote TTS returned an empty file")

        sample_rate = _int_header(headers, "x-tts-sample-rate", settings.tts_sample_rate)
        if sample_rate != settings.tts_sample_rate:
            tmp = out.with_name(out.stem + "_rs.wav")
            ffmpeg.resample(out, tmp, settings.tts_sample_rate, channels=1)
            tmp.replace(out)
            sample_rate = settings.tts_sample_rate

        # Report the engine that actually SPOKE, not the transport and not
        # what we asked for. A manifest saying `remote: 32` tells you nothing
        # when the point of the run was to compare viXTTS against F5 - and a
        # server that silently served a different engine than requested has to
        # be visible here, or the comparison quietly measures the wrong model.
        spoke = headers.get("x-tts-model") or self.engine
        cloned = (headers.get("x-tts-voice-cloned", "").lower() == "true"
                  if "x-tts-voice-cloned" in headers else self.supports_cloning)

        return SynthesisResult(
            path=out, duration=ffmpeg.duration_of(out),
            model=f"remote:{spoke}" if spoke else "remote",
            voice_cloned=cloned, sample_rate=sample_rate,
        )

    # ---------------------------------------------------------- transport --
    def _post(self, request: SynthesisRequest, reference: Path | None) -> tuple[bytes, dict]:
        import httpx

        digest = _digest(reference) if reference is not None else None
        url = f"{self.base_url}/synthesize"
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}

        last: Exception | None = None
        for attempt in range(1, max(1, settings.remote_tts_retries) + 1):
            # Re-send the file on the first attempt of a new reference, and on
            # every retry after a 409 or a transport failure - a reconnected
            # tunnel usually means a restarted server with an empty cache.
            send_file = reference is not None and digest not in _UPLOADED
            try:
                with httpx.Client(timeout=settings.remote_tts_timeout) as client:
                    response = self._request(client, url, headers, request, reference,
                                             digest, send_file)
                    if response.status_code == 409 and reference is not None:
                        _UPLOADED.discard(digest)
                        response = self._request(client, url, headers, request, reference,
                                                 digest, send_file=True)
                    if response.status_code >= 400:
                        raise SynthesisError(
                            f"Remote TTS returned {response.status_code}",
                            details={"body": response.text[:400], "url": url},
                        )
                    if digest:
                        _UPLOADED.add(digest)
                    return response.content, {k.lower(): v for k, v in response.headers.items()}
            except SynthesisError as exc:
                # A 4xx is the server's considered answer. Retrying an
                # unsupported language just burns the tunnel.
                if "returned 5" not in str(exc):
                    raise
                last = exc
            except Exception as exc:  # noqa: BLE001 - httpx timeouts, dropped tunnels
                last = exc

            if attempt < settings.remote_tts_retries:
                backoff = min(8.0, 1.5 ** attempt)
                log.warning("remote TTS attempt %d/%d failed (%s); retrying in %.1fs",
                            attempt, settings.remote_tts_retries, last, backoff)
                time.sleep(backoff)

        raise ModelUnavailable(
            f"Remote TTS unreachable after {settings.remote_tts_retries} attempt(s): {last}",
            details={
                "endpoint": self.base_url,
                "hint": "A Colab tunnel URL changes every restart - re-run the serve cell "
                        "and update REMOTE_TTS_URL.",
            },
        )

    def _request(self, client, url: str, headers: dict, request: SynthesisRequest,  # noqa: ANN001
                 reference: Path | None, digest: str | None, send_file: bool):  # noqa: ANN202
        data = {
            "text": request.text,
            "language": languages.normalize(request.language) or request.language,
            "speed": str(request.speed or 1.0),
        }
        if self.engine:
            data["engine"] = self.engine
        if request.speaker_id:
            data["speaker_id"] = request.speaker_id
        if digest:
            data["reference_id"] = digest

        if send_file and reference is not None:
            with reference.open("rb") as fh:
                return client.post(url, headers=headers, data=data,
                                   files={"reference": (reference.name, fh, "audio/wav")})
        return client.post(url, headers=headers, data=data)


def _int_header(headers: dict, key: str, fallback: int) -> int:
    try:
        return int(headers[key])
    except (KeyError, TypeError, ValueError):
        return fallback
