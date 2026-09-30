"""Search tool construction from settings."""

from __future__ import annotations

from ..config import Settings
from ..errors import ConfigError
from .base import SearchTool
from .browser_search import BrowserSearchTool
from .exa_search import ExaSearchTool
from .fake_search import FakeSearchTool


def build_search_tool(settings: Settings) -> SearchTool:
    if settings.search_provider == "fake":
        return FakeSearchTool()

    if settings.search_provider == "exa":
        if not settings.exa_api_key:
            raise ConfigError("DR_EXA_API_KEY is required for the exa provider")
        return ExaSearchTool(
            api_key=settings.exa_api_key,
            timeout=max(settings.request_timeout, 45.0),
        )

    if settings.search_provider == "browser":
        return BrowserSearchTool(
            num_results=settings.search_results,
            timeout=settings.browser_page_timeout_s,
            max_page_chars=settings.browser_max_page_chars,
        )

    raise ConfigError(f"Unknown search provider: {settings.search_provider}")
