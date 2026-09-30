"""Shared HTTP helper with retry/backoff.

Both paid upstreams (the LLM provider and the search API) return 429s and
transient 5xx responses under load. Retrying with jittered exponential backoff
turns most of those into a slightly slower success instead of a failed run.
"""

from __future__ import annotations

import random
import time

import httpx

from .errors import ProviderError
from .logging_config import get_logger

logger = get_logger(__name__)

RETRY_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


def request_with_retries(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    retries: int = 3,
    backoff: float = 0.8,
    max_backoff: float = 8.0,
    error_type: type[ProviderError] = ProviderError,
    **kwargs,
) -> httpx.Response:
    """Perform a request, retrying transient failures with jittered backoff."""
    last_error: str = "unknown error"

    for attempt in range(retries + 1):
        try:
            response = client.request(method, url, **kwargs)
        except httpx.HTTPError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            if response.status_code < 400:
                return response
            if response.status_code not in RETRY_STATUSES:
                raise error_type(
                    f"{method} {url} failed with {response.status_code}: {response.text[:300]}"
                )
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            # Honour Retry-After when the server tells us how long to wait.
            retry_after = response.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                time.sleep(min(float(retry_after), max_backoff))

        if attempt < retries:
            delay = min(backoff * (2**attempt), max_backoff)
            delay *= 0.5 + random.random()  # full jitter
            logger.warning("Retrying %s %s in %.2fs (%s)", method, url, delay, last_error)
            time.sleep(delay)

    raise error_type(f"{method} {url} failed after {retries + 1} attempts: {last_error}")
