"""Errors shared by providers, pipeline stages, and the HTTP layer."""
from __future__ import annotations

from typing import Optional


class ServiceError(Exception):
    status_code = 500

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code


class InvalidRequest(ServiceError):
    status_code = 400


class UnsupportedLanguage(InvalidRequest):
    pass


class MissingDependency(ServiceError):
    status_code = 503

    def __init__(self, pip_name: str, purpose: str) -> None:
        super().__init__(
            f"{purpose} needs the '{pip_name}' package, which this session does not "
            f"have. Run `pip install {pip_name}` in a Colab cell (or turn on its "
            f"flag in the first notebook cell) and restart the API cell."
        )
        self.pip_name = pip_name


class MissingCredential(ServiceError):
    status_code = 503


class ProviderFailure(ServiceError):
    status_code = 500
