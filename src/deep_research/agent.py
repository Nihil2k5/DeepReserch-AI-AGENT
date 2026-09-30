"""Agent wiring: assemble the engine from settings.

This is the single place that decides which concrete LLM, search tool, store and
observability backend a run uses. Everything above it depends on interfaces only.
"""

from __future__ import annotations

import uuid

from .config import Settings, get_settings
from .db import Database
from .errors import NotFoundError
from .graph.engine import ResearchEngine
from .llm.base import LLM
from .llm.factory import build_llm
from .locking import RedisCoordinator
from .logging_config import get_logger
from .models import Report, Run, RunStatus
from .repository import PostgresRunRepository, RunRepository
from .tools.base import SearchTool
from .tools.factory import build_search_tool
from .tracing import build_exporter

logger = get_logger(__name__)


class ResearchAgent:
    """Owns the long-lived resources and exposes run creation/execution."""

    def __init__(
        self,
        settings: Settings,
        repo: RunRepository,
        llm: LLM,
        search: SearchTool,
        coordinator: RedisCoordinator | None = None,
        exporter=None,
    ) -> None:
        self.settings = settings
        self.repo = repo
        self.llm = llm
        self.search = search
        self.coordinator = coordinator
        self.exporter = exporter
        #: Cached result of the one-shot Langfuse credential check.
        self.langfuse_ok: bool | None = None
        self.engine = ResearchEngine(
            settings=settings,
            repo=repo,
            llm=llm,
            search=search,
            exporter=exporter,
        )

    # -- construction ----------------------------------------------------
    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> ResearchAgent:
        settings = settings or get_settings()
        settings.validate_runtime()

        db = Database(settings.database_url)
        repo = PostgresRunRepository(db)
        applied = repo.migrate()
        if applied:
            logger.info("Applied migrations: %s", ", ".join(applied))

        coordinator = RedisCoordinator(settings.redis_url)

        agent = cls(
            settings=settings,
            repo=repo,
            llm=build_llm(settings),
            search=build_search_tool(settings),
            coordinator=coordinator,
            exporter=build_exporter(settings),
        )
        # Validate observability credentials once, loudly, at startup. A silently
        # disabled exporter is the worst outcome: you think you have traces.
        agent.check_observability()
        return agent

    # -- runs ------------------------------------------------------------
    def create_run(self, question: str, focus: str | None = None) -> Run:
        run = Run(id=uuid.uuid4().hex[:20], question=question, focus=focus)
        created = self.repo.create_run(run)
        self.repo.append_event(created.id, "run.queued", "Run queued", {"question": question})
        return created

    def get_run(self, run_id: str) -> Run:
        run = self.repo.get_run(run_id)
        if run is None:
            raise NotFoundError(f"Run not found: {run_id}")
        return run

    def execute(self, run_id: str) -> Report:
        """Execute (or resume) a run to completion. Used by the worker and CLI."""
        run = self.get_run(run_id)
        if run.status in {RunStatus.SUCCEEDED, RunStatus.CANCELLED}:
            if run.report is not None:
                return run.report
            raise NotFoundError(f"Run {run_id} is {run.status.value} with no report")

        self.repo.mark_running(run.id, self.settings.worker_id or "inline")
        try:
            report = self.engine.execute(run)
        except Exception as exc:
            logger.exception("Run %s failed", run.id)
            # Close the trace so aborted runs don't leave an open root span.
            if self.exporter is not None:
                self.exporter.finish_run(run.id)
                self.exporter.flush()
            self.repo.append_event(run.id, "run.failed", str(exc), level="error")
            self.repo.mark_finished(run.id, RunStatus.FAILED, error=str(exc))
            raise

        self.repo.mark_finished(run.id, RunStatus.SUCCEEDED, report=report)
        self.repo.append_event(run.id, "run.succeeded", "Report ready")
        return report

    def cancel(self, run_id: str) -> bool:
        self.get_run(run_id)
        cancelled = self.repo.request_cancel(run_id)
        if cancelled:
            self.repo.append_event(run_id, "run.cancel_requested", "Cancellation requested")
        return cancelled

    # -- ops -------------------------------------------------------------
    def check_observability(self) -> bool | None:
        """One-shot Langfuse credential check. ``None`` when not configured."""
        if self.exporter is None:
            return None
        if self.langfuse_ok is None:
            self.langfuse_ok = self.exporter.auth_check()
            if not self.langfuse_ok:
                logger.error(
                    "Langfuse is configured but the credentials were rejected, so traces "
                    "will only be stored in Postgres. Check DR_LANGFUSE_PUBLIC_KEY, "
                    "DR_LANGFUSE_SECRET_KEY and DR_LANGFUSE_HOST "
                    "(EU: cloud.langfuse.com, US: us.cloud.langfuse.com, JP: jp.cloud.langfuse.com)."
                )
            else:
                logger.info("Langfuse tracing enabled at %s", self.settings.langfuse_host)
        return self.langfuse_ok

    def health(self) -> dict[str, object]:
        database_ok = False
        try:
            database_ok = bool(getattr(self.repo, "db", None) and self.repo.db.ping())
        except Exception:
            database_ok = False

        redis_ok = bool(self.coordinator and self.coordinator.ping())
        return {
            "status": "ok" if database_ok else "degraded",
            "database": database_ok,
            "redis": redis_ok,
            "llm": self.llm.name,
            "search": self.search.name,
            "langfuse": {
                "configured": self.exporter is not None,
                "authenticated": self.langfuse_ok,
            },
        }

    def close(self) -> None:
        self.llm.close()
        self.search.close()
        if self.coordinator:
            self.coordinator.close()
        self.repo.close()
