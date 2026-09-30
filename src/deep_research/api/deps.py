"""FastAPI dependencies."""

from __future__ import annotations

import threading

from fastapi import Depends, Header, HTTPException, status

from ..agent import ResearchAgent
from ..config import Settings, get_settings
from ..logging_config import configure_logging

_lock = threading.Lock()
_agent: ResearchAgent | None = None


def get_agent() -> ResearchAgent:
    """Process-wide singleton so pools and clients are shared across requests."""
    global _agent
    if _agent is None:
        with _lock:
            if _agent is None:
                settings = get_settings()
                configure_logging(settings.log_level)
                _agent = ResearchAgent.from_settings(settings)
    return _agent


def reset_agent() -> None:
    global _agent
    with _lock:
        if _agent is not None:
            _agent.close()
        _agent = None


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> None:
    if settings.api_key is None:
        return
    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header.",
        )
