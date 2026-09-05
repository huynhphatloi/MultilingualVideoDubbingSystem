"""Errors the providers and pipeline raise.

Providers never import FastAPI: a provider module has to be importable in a
plain script, in a test, and inside the notebook. `ServiceError` carries the
status code the HTTP layer should use, and `server.py` is the only place that
turns one into an HTTPException.
"""
from __future__ import annotations

from typing import Optional


class ServiceError(Exception):
    """An error with a status code and a message meant for the caller."""

    status_code = 500

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code


class InvalidRequest(ServiceError):
    """The caller asked for something the registry does not offer."""

    status_code = 400


class UnsupportedLanguage(InvalidRequest):
    """The chosen model does not speak or understand that language."""


class MissingDependency(ServiceError):
    """An optional package is not installed in this session."""

    status_code = 503

    def __init__(self, pip_name: str, purpose: str) -> None:
        super().__init__(
            f"{purpose} needs the '{pip_name}' package, which this session does not "
            f"have. Run `pip install {pip_name}` in a Colab cell (or turn on its "
            f"flag in the first notebook cell) and restart the API cell."
        )
        self.pip_name = pip_name


class MissingCredential(ServiceError):
    """A gated model needs a token the session does not have."""

    status_code = 503


class ProviderFailure(ServiceError):
    """The model ran and failed."""

    status_code = 500
