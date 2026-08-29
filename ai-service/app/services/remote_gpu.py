"""Shared transport for every stage that runs on the Colab GPU.

One notebook serves ASR, translation and synthesis, so one base URL and one
token drive all three. This module holds what they have in common - retries,
auth, error shaping - so `asr.py` and the translation engine do not each grow
their own half-correct copy of it.

**Why the models are not here at all.** Whisper-medium is 1.4 GB and
NLLB-200-distilled is 2.3 GB. Caching them locally costs ~4 GB of disk for
weights this machine cannot even run quickly - Whisper falls back to CPU int8
here, and coqui/F5 kernels have no Metal build. Sending the work to a free
Colab T4 costs a few MB of upload per job instead.

**What now leaves the machine.** This is a real change and it should be stated
plainly rather than buried: with remote ASR the SPEECH TRACK of the video is
uploaded (compressed), not just text. Previously only a line of text and a few
seconds of reference audio ever left. The video itself still never does, and
the audio is transcoded to ~24 kbps Opus first - about 1.8 MB for ten minutes
of film rather than 19 MB of wav - but "nothing but text leaves" is no longer
true, and anyone dubbing material they cannot share should keep
REMOTE_ASR_ENABLED off.

**Retries.** A Colab tunnel dies on session timeout and trycloudflare hands out
a new URL each restart, so a dropped connection is the normal case, not the
exceptional one. Retries cover the blip; a 4xx is the server's considered
answer and is never retried.
"""
from __future__ import annotations

import logging
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.errors import ModelUnavailable, PipelineError

log = logging.getLogger(__name__)

#: What a dead tunnel looks like, and what to do about it. Every remote stage
#: raises this same hint because the remedy is always the same.
TUNNEL_HINT = (
    "A Colab trycloudflare URL changes every time the serve cell is re-run. "
    "Re-run it and update REMOTE_URL in .env, then restart the AI service."
)


def endpoint() -> str:
    """The Colab base URL, or "" when nothing is configured."""
    return (settings.remote_url or "").rstrip("/")


def token() -> str:
    return settings.remote_token or ""


def configured() -> bool:
    return bool(endpoint())


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {token()}"} if token() else {}


def post(path: str, *, data: dict[str, Any] | None = None,
         json_body: Any = None, files: dict | None = None,
         timeout: float | None = None, error: type[PipelineError] = ModelUnavailable,
         stage: str = "remote") -> Any:
    """POST to the Colab server and return the decoded JSON body.

    `files` is a mapping of field name -> (filename, path, content_type); the
    file is opened per attempt rather than once, because a retried request
    needs the stream rewound and re-opening is simpler than seeking.
    """
    import httpx

    base = endpoint()
    if not base:
        raise error(f"{stage}: no GPU endpoint configured (set REMOTE_URL)",
                    details={"hint": TUNNEL_HINT})

    url = f"{base}{path}"
    attempts = max(1, settings.remote_retries)
    wait = timeout if timeout is not None else settings.remote_timeout
    last: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            # ExitStack because a retry needs the upload stream rewound, and
            # re-opening per attempt is simpler than seeking it back.
            with ExitStack() as stack:
                client = stack.enter_context(httpx.Client(timeout=wait))
                kwargs: dict[str, Any] = {"headers": _headers()}
                if json_body is not None:
                    kwargs["json"] = json_body
                if data:
                    kwargs["data"] = data
                if files:
                    kwargs["files"] = {
                        field: (name, stack.enter_context(Path(path_).open("rb")), ctype)
                        for field, (name, path_, ctype) in files.items()
                    }

                response = client.post(url, **kwargs)
                if response.status_code >= 400:
                    detail = response.text[:400]
                    # A 4xx is the server's considered answer - an unsupported
                    # language, a bad model name. Retrying only burns the
                    # tunnel and delays a message the caller needs now.
                    if response.status_code < 500:
                        raise error(
                            f"{stage}: the GPU endpoint returned "
                            f"{response.status_code}",
                            details={"body": detail, "url": url},
                        )
                    raise _Transient(f"{response.status_code}: {detail}")
                return response.json()
        except _Transient as exc:
            last = exc
        except PipelineError:
            raise
        except Exception as exc:  # noqa: BLE001 - timeouts, dropped tunnels, DNS
            last = exc

        if attempt < attempts:
            backoff = min(8.0, 1.5 ** attempt)
            log.warning("%s attempt %d/%d failed (%s); retrying in %.1fs",
                        stage, attempt, attempts, last, backoff)
            time.sleep(backoff)

    raise ModelUnavailable(
        f"{stage}: the GPU endpoint is unreachable after {attempts} attempt(s): {last}",
        details={"endpoint": base, "hint": TUNNEL_HINT},
    )


def health() -> dict:
    """What the endpoint says it can do. Used by /health/models and the UI."""
    import httpx

    base = endpoint()
    if not base:
        return {"configured": False}
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{base}/health", headers=_headers())
            response.raise_for_status()
            return {"configured": True, **response.json()}
    except Exception as exc:  # noqa: BLE001 - a dead tunnel is information
        return {"configured": True, "reachable": False, "error": str(exc)[:200]}


class _Transient(Exception):
    """A 5xx. Worth retrying, unlike everything the server says deliberately."""
