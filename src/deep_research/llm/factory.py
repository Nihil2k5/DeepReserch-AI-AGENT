"""LLM construction from settings."""

from __future__ import annotations

from ..config import Settings
from ..errors import ConfigError
from .base import LLM
from .fake import FakeLLM
from .openai_client import OpenAIChatLLM


def build_llm(settings: Settings) -> LLM:
    if settings.llm_provider == "fake":
        return FakeLLM()

    if settings.llm_provider == "openai":
        if not settings.llm_api_key:
            raise ConfigError("DR_LLM_API_KEY is required for the openai provider")
        return OpenAIChatLLM(
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            timeout=settings.request_timeout,
            price_prompt_per_mtok=settings.price_prompt_per_mtok,
            price_completion_per_mtok=settings.price_completion_per_mtok,
        )

    if settings.llm_provider == "ollama":
        # Ollama serves an OpenAI-compatible API at /v1 and ignores the key.
        return OpenAIChatLLM(
            api_key="ollama",
            model=settings.llm_model,
            base_url=settings.llm_base_url or "http://localhost:11434/v1",
            timeout=max(settings.request_timeout, 120.0),
            price_prompt_per_mtok=0.0,
            price_completion_per_mtok=0.0,
        )

    raise ConfigError(f"Unknown LLM provider: {settings.llm_provider}")
