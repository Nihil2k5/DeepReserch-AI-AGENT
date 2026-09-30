"""Typed exception hierarchy mapped to HTTP status codes by the API."""

from __future__ import annotations


class ResearchError(Exception):
    status_code = 500


class ConfigError(ResearchError):
    status_code = 500


class ProviderError(ResearchError):
    status_code = 502


class LLMError(ProviderError):
    pass


class SearchError(ProviderError):
    pass


class StorageError(ResearchError):
    status_code = 500


class NotFoundError(ResearchError):
    status_code = 404


class ConflictError(ResearchError):
    status_code = 409


class BudgetExceeded(ResearchError):
    """A hard run budget (cost, calls, wall clock) was hit."""

    status_code = 429


class RunCancelled(ResearchError):
    status_code = 409


def error_payload(exc: ResearchError, request_id: str | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "error": exc.__class__.__name__,
        "message": str(exc),
    }
    if request_id:
        payload["request_id"] = request_id
    return payload
