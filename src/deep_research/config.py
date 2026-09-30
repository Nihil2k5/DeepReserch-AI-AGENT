"""Environment-driven settings.

The defaults target the production path (real LLM + real search). ``fake``
providers exist so the orchestration logic is unit-testable without network
access or credentials, but they are never selected by default.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["openai", "ollama", "fake"]
SearchProvider = Literal["exa", "browser", "fake"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="DR_",
        extra="ignore",
        case_sensitive=False,
    )

    # --- App ---------------------------------------------------------------
    app_name: str = "Deep Research Agent"
    environment: str = "development"
    log_level: str = "INFO"
    api_key: str | None = None

    # --- Required infrastructure -------------------------------------------
    database_url: str = "postgresql://research:research@localhost:5432/research"
    redis_url: str = "redis://localhost:6379/0"

    # --- LLM ---------------------------------------------------------------
    llm_provider: LLMProvider = "openai"
    llm_api_key: str | None = None
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_reasoning_model: str = "gpt-4o"
    request_timeout: float = 90.0

    # --- Search ------------------------------------------------------------
    search_provider: SearchProvider = "exa"
    exa_api_key: str | None = None
    search_results: int = 5
    max_search_concurrency: int = 4

    # --- Browser search provider -------------------------------------------
    browser_headless: bool = True
    browser_page_timeout_s: float = 15.0
    browser_max_page_chars: int = 6000

    # --- Budgets -----------------------------------------------------------
    max_llm_calls: int = 60
    max_searches: int = 40
    max_cost_usd: float = 1.50
    max_wall_clock_s: int = 900
    max_subquestions: int = 4
    max_research_rounds: int = 2

    # --- Observability -----------------------------------------------------
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # --- Worker ------------------------------------------------------------
    worker_poll_interval_s: float = 2.0
    worker_concurrency: int = 2
    worker_id: str | None = None

    # --- Pricing (USD per 1M tokens) ---------------------------------------
    price_prompt_per_mtok: float = 0.15
    price_completion_per_mtok: float = 0.60

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    def validate_runtime(self, *, require_infra: bool = True) -> None:
        """Fail fast with an actionable message instead of a confusing stack trace."""
        if not require_infra:
            return

        missing: list[str] = []
        if self.llm_provider == "openai" and not self.llm_api_key:
            missing.append("DR_LLM_API_KEY")
        if self.search_provider == "exa" and not self.exa_api_key:
            missing.append("DR_EXA_API_KEY")
        if not self.database_url:
            missing.append("DR_DATABASE_URL")
        if not self.redis_url:
            missing.append("DR_REDIS_URL")

        if missing:
            raise ValueError(
                "Missing required configuration: "
                + ", ".join(missing)
                + ". See .env.example — this service requires real infrastructure."
            )

        if self.max_subquestions < 1:
            raise ValueError("DR_MAX_SUBQUESTIONS must be >= 1")
        if self.max_research_rounds < 1:
            raise ValueError("DR_MAX_RESEARCH_ROUNDS must be >= 1")


@lru_cache
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
