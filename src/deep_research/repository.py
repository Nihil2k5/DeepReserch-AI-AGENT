"""Run persistence: checkpoints, spans, events and usage accounting.

Two implementations behind one interface:

* :class:`PostgresRunRepository` — production. The job queue uses
  ``SELECT ... FOR UPDATE SKIP LOCKED`` so multiple workers can claim runs
  safely with no extra coordination service.
* :class:`InMemoryRunRepository` — used by unit tests so the orchestration logic
  can be verified without a database.
"""

from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone

from .db import Database
from .errors import StorageError
from .logging_config import get_logger
from .models import (
    Event,
    Report,
    Run,
    RunMetrics,
    RunStatus,
    Span,
    SpanKind,
    SpanStatus,
)

logger = get_logger(__name__)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RunRepository(ABC):
    """Persistence boundary for the whole agent."""

    @abstractmethod
    def migrate(self) -> list[str]: ...

    @abstractmethod
    def create_run(self, run: Run) -> Run: ...

    @abstractmethod
    def get_run(self, run_id: str) -> Run | None: ...

    @abstractmethod
    def list_runs(self, limit: int = 50, status: RunStatus | None = None) -> list[Run]: ...

    @abstractmethod
    def claim_next_run(self, worker_id: str) -> Run | None: ...

    @abstractmethod
    def mark_running(self, run_id: str, worker_id: str) -> None: ...

    @abstractmethod
    def mark_finished(
        self,
        run_id: str,
        status: RunStatus,
        *,
        report: Report | None = None,
        error: str | None = None,
        metrics: RunMetrics | None = None,
    ) -> None: ...

    @abstractmethod
    def save_checkpoint(self, run_id: str, node_key: str, output: dict) -> None: ...

    @abstractmethod
    def get_checkpoints(self, run_id: str) -> dict[str, dict]: ...

    @abstractmethod
    def save_span(self, span: Span) -> None: ...

    @abstractmethod
    def list_spans(self, run_id: str) -> list[Span]: ...

    @abstractmethod
    def append_event(
        self,
        run_id: str,
        event_type: str,
        message: str = "",
        data: dict | None = None,
        level: str = "info",
    ) -> Event: ...

    @abstractmethod
    def list_events(self, run_id: str, after_id: int = 0, limit: int = 500) -> list[Event]: ...

    @abstractmethod
    def add_usage(
        self,
        run_id: str,
        *,
        llm_calls: int = 0,
        searches: int = 0,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost_usd: float = 0.0,
    ) -> RunMetrics: ...

    @abstractmethod
    def request_cancel(self, run_id: str) -> bool: ...

    @abstractmethod
    def is_cancel_requested(self, run_id: str) -> bool: ...

    @abstractmethod
    def reset_stale_runs(self, older_than_minutes: int = 30) -> int: ...

    def close(self) -> None:
        """Release resources."""


# ---------------------------------------------------------------------------
# Postgres
# ---------------------------------------------------------------------------
_RUN_COLUMNS = (
    "id, question, focus, status, created_at, updated_at, started_at, finished_at, "
    "worker_id, attempt, error, cancel_requested, report, "
    "llm_calls, searches, tokens_in, tokens_out, cost_usd, "
    "duration_ms, nodes_executed, nodes_resumed"
)


def _row_to_run(row: dict) -> Run:
    report = row.get("report")
    if isinstance(report, str):
        report = json.loads(report)
    return Run(
        id=row["id"],
        question=row["question"],
        focus=row.get("focus"),
        status=RunStatus(row["status"]),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
        started_at=row.get("started_at"),
        finished_at=row.get("finished_at"),
        worker_id=row.get("worker_id"),
        attempt=row.get("attempt") or 0,
        error=row.get("error"),
        cancel_requested=bool(row.get("cancel_requested")),
        report=Report.model_validate(report) if report else None,
        metrics=RunMetrics(
            llm_calls=row.get("llm_calls") or 0,
            searches=row.get("searches") or 0,
            tokens_in=row.get("tokens_in") or 0,
            tokens_out=row.get("tokens_out") or 0,
            cost_usd=float(row.get("cost_usd") or 0.0),
            duration_ms=float(row.get("duration_ms") or 0.0),
            nodes_executed=row.get("nodes_executed") or 0,
            nodes_resumed=row.get("nodes_resumed") or 0,
        ),
    )


def _row_to_span(row: dict) -> Span:
    attributes = row.get("attributes") or {}
    if isinstance(attributes, str):
        attributes = json.loads(attributes)
    return Span(
        span_id=row["span_id"],
        run_id=row["run_id"],
        parent_id=row.get("parent_id"),
        name=row["name"],
        kind=SpanKind(row["kind"]),
        started_at=row["started_at"],
        ended_at=row.get("ended_at"),
        duration_ms=row.get("duration_ms"),
        status=SpanStatus(row["status"]) if row.get("status") else SpanStatus.OK,
        attributes=attributes,
        error=row.get("error"),
    )


class PostgresRunRepository(RunRepository):
    def __init__(self, db: Database) -> None:
        self.db = db

    def migrate(self) -> list[str]:
        return self.db.migrate()

    # -- runs ------------------------------------------------------------
    def create_run(self, run: Run) -> Run:
        row = self.db.execute_one(
            """
            INSERT INTO runs (id, question, focus, status)
            VALUES (%s, %s, %s, %s)
            RETURNING *
            """,
            (run.id, run.question, run.focus, run.status.value),
        )
        if row is None:  # pragma: no cover
            raise StorageError("Failed to create run")
        return _row_to_run(row)

    def get_run(self, run_id: str) -> Run | None:
        row = self.db.execute_one(f"SELECT {_RUN_COLUMNS} FROM runs WHERE id = %s", (run_id,))
        return _row_to_run(row) if row else None

    def list_runs(self, limit: int = 50, status: RunStatus | None = None) -> list[Run]:
        if status is None:
            rows = self.db.execute(
                f"SELECT {_RUN_COLUMNS} FROM runs ORDER BY created_at DESC LIMIT %s", (limit,)
            )
        else:
            rows = self.db.execute(
                f"SELECT {_RUN_COLUMNS} FROM runs WHERE status = %s "
                "ORDER BY created_at DESC LIMIT %s",
                (status.value, limit),
            )
        return [_row_to_run(row) for row in rows]

    def claim_next_run(self, worker_id: str) -> Run | None:
        """Atomically claim the oldest queued run (safe across workers)."""
        row = self.db.execute_one(
            """
            UPDATE runs
               SET status = 'running',
                   worker_id = %s,
                   started_at = COALESCE(started_at, now()),
                   attempt = attempt + 1,
                   updated_at = now()
             WHERE id = (
                   SELECT id FROM runs
                    WHERE status = 'queued'
                    ORDER BY created_at
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
             )
            RETURNING *
            """,
            (worker_id,),
        )
        return _row_to_run(row) if row else None

    def mark_running(self, run_id: str, worker_id: str) -> None:
        self.db.execute(
            """
            UPDATE runs
               SET status = 'running', worker_id = %s,
                   started_at = COALESCE(started_at, now()), updated_at = now()
             WHERE id = %s
            """,
            (worker_id, run_id),
        )

    def mark_finished(
        self,
        run_id: str,
        status: RunStatus,
        *,
        report: Report | None = None,
        error: str | None = None,
        metrics: RunMetrics | None = None,
    ) -> None:
        self.db.execute(
            """
            UPDATE runs
               SET status = %s, finished_at = now(), updated_at = now(),
                   error = %s,
                   report = COALESCE(%s::jsonb, report),
                   duration_ms = COALESCE(%s, duration_ms)
             WHERE id = %s
            """,
            (
                status.value,
                error,
                json.dumps(report.model_dump(mode="json")) if report else None,
                metrics.duration_ms if metrics else None,
                run_id,
            ),
        )

    def save_metrics(self, run_id: str, metrics: RunMetrics) -> None:
        self.db.execute(
            "UPDATE runs SET duration_ms = %s, nodes_executed = %s, nodes_resumed = %s, "
            "updated_at = now() WHERE id = %s",
            (metrics.duration_ms, metrics.nodes_executed, metrics.nodes_resumed, run_id),
        )

    # -- usage -----------------------------------------------------------
    def add_usage(
        self,
        run_id: str,
        *,
        llm_calls: int = 0,
        searches: int = 0,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost_usd: float = 0.0,
    ) -> RunMetrics:
        row = self.db.execute_one(
            """
            UPDATE runs
               SET llm_calls = llm_calls + %s,
                   searches = searches + %s,
                   tokens_in = tokens_in + %s,
                   tokens_out = tokens_out + %s,
                   cost_usd = cost_usd + %s,
                   updated_at = now()
             WHERE id = %s
            RETURNING llm_calls, searches, tokens_in, tokens_out, cost_usd
            """,
            (llm_calls, searches, tokens_in, tokens_out, cost_usd, run_id),
        )
        if row is None:
            raise StorageError(f"Unknown run: {run_id}")
        return RunMetrics(
            llm_calls=row["llm_calls"],
            searches=row["searches"],
            tokens_in=row["tokens_in"],
            tokens_out=row["tokens_out"],
            cost_usd=float(row["cost_usd"]),
        )

    # -- checkpoints -----------------------------------------------------
    def save_checkpoint(self, run_id: str, node_key: str, output: dict) -> None:
        self.db.execute(
            """
            INSERT INTO checkpoints (run_id, node_key, status, output)
            VALUES (%s, %s, 'done', %s::jsonb)
            ON CONFLICT (run_id, node_key)
            DO UPDATE SET output = EXCLUDED.output, status = 'done', created_at = now()
            """,
            (run_id, node_key, json.dumps(output, default=str)),
        )

    def get_checkpoints(self, run_id: str) -> dict[str, dict]:
        rows = self.db.execute(
            "SELECT node_key, output FROM checkpoints WHERE run_id = %s AND status = 'done'",
            (run_id,),
        )
        result: dict[str, dict] = {}
        for row in rows:
            output = row["output"]
            result[row["node_key"]] = json.loads(output) if isinstance(output, str) else output
        return result

    # -- spans -----------------------------------------------------------
    def save_span(self, span: Span) -> None:
        self.db.execute(
            """
            INSERT INTO spans (span_id, run_id, parent_id, name, kind, started_at, ended_at,
                               duration_ms, status, attributes, error)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
            """,
            (
                span.span_id,
                span.run_id,
                span.parent_id,
                span.name,
                span.kind.value,
                span.started_at,
                span.ended_at,
                span.duration_ms,
                span.status.value,
                json.dumps(span.attributes, default=str),
                span.error,
            ),
        )

    def list_spans(self, run_id: str) -> list[Span]:
        rows = self.db.execute(
            "SELECT * FROM spans WHERE run_id = %s ORDER BY started_at", (run_id,)
        )
        return [_row_to_span(row) for row in rows]

    # -- events ----------------------------------------------------------
    def append_event(
        self,
        run_id: str,
        event_type: str,
        message: str = "",
        data: dict | None = None,
        level: str = "info",
    ) -> Event:
        row = self.db.execute_one(
            """
            INSERT INTO events (run_id, level, type, message, data)
            VALUES (%s, %s, %s, %s, %s::jsonb)
            RETURNING id, run_id, level, type, message, data, created_at
            """,
            (run_id, level, event_type, message, json.dumps(data or {}, default=str)),
        )
        if row is None:  # pragma: no cover
            raise StorageError("Failed to append event")
        return Event.model_validate(row)

    def list_events(self, run_id: str, after_id: int = 0, limit: int = 500) -> list[Event]:
        rows = self.db.execute(
            "SELECT id, run_id, level, type, message, data, created_at FROM events "
            "WHERE run_id = %s AND id > %s ORDER BY id LIMIT %s",
            (run_id, after_id, limit),
        )
        return [Event.model_validate(row) for row in rows]

    # -- control ---------------------------------------------------------
    def request_cancel(self, run_id: str) -> bool:
        row = self.db.execute_one(
            "UPDATE runs SET cancel_requested = TRUE, updated_at = now() "
            "WHERE id = %s AND status IN ('queued', 'running') RETURNING id",
            (run_id,),
        )
        return row is not None

    def is_cancel_requested(self, run_id: str) -> bool:
        row = self.db.execute_one("SELECT cancel_requested FROM runs WHERE id = %s", (run_id,))
        return bool(row and row["cancel_requested"])

    def reset_stale_runs(self, older_than_minutes: int = 30) -> int:
        rows = self.db.execute(
            """
            UPDATE runs
               SET status = 'queued', worker_id = NULL, updated_at = now()
             WHERE status = 'running'
               AND updated_at < now() - make_interval(mins => %s)
            """,
            (older_than_minutes,),
        )
        return len(rows)

    def close(self) -> None:
        self.db.close()


# ---------------------------------------------------------------------------
# In-memory (tests)
# ---------------------------------------------------------------------------
class InMemoryRunRepository(RunRepository):
    def __init__(self) -> None:
        self._runs: dict[str, Run] = {}
        self._order: list[str] = []
        self._checkpoints: dict[str, dict[str, dict]] = {}
        self._spans: dict[str, list[Span]] = {}
        self._events: dict[str, list[Event]] = {}
        self._event_ids: dict[str, int] = {}
        self._lock = threading.RLock()

    def migrate(self) -> list[str]:
        return []

    def create_run(self, run: Run) -> Run:
        with self._lock:
            run = run.model_copy(update={"created_at": run.created_at or utcnow()})
            self._runs[run.id] = run
            self._order.append(run.id)
            return run.model_copy(deep=True)

    def get_run(self, run_id: str) -> Run | None:
        with self._lock:
            run = self._runs.get(run_id)
            return run.model_copy(deep=True) if run else None

    def list_runs(self, limit: int = 50, status: RunStatus | None = None) -> list[Run]:
        with self._lock:
            runs = [self._runs[i] for i in reversed(self._order)]
            if status is not None:
                runs = [r for r in runs if r.status == status]
            return [r.model_copy(deep=True) for r in runs[:limit]]

    def claim_next_run(self, worker_id: str) -> Run | None:
        with self._lock:
            for run_id in self._order:
                run = self._runs[run_id]
                if run.status == RunStatus.QUEUED:
                    run.status = RunStatus.RUNNING
                    run.worker_id = worker_id
                    run.attempt += 1
                    run.started_at = run.started_at or utcnow()
                    return run.model_copy(deep=True)
            return None

    def mark_running(self, run_id: str, worker_id: str) -> None:
        with self._lock:
            run = self._require(run_id)
            run.status = RunStatus.RUNNING
            run.worker_id = worker_id
            run.started_at = run.started_at or utcnow()

    def mark_finished(
        self,
        run_id: str,
        status: RunStatus,
        *,
        report: Report | None = None,
        error: str | None = None,
        metrics: RunMetrics | None = None,
    ) -> None:
        with self._lock:
            run = self._require(run_id)
            run.status = status
            run.finished_at = utcnow()
            run.error = error
            if report is not None:
                run.report = report
            if metrics is not None and metrics.duration_ms:
                run.metrics.duration_ms = metrics.duration_ms
                run.metrics.nodes_executed = metrics.nodes_executed
                run.metrics.nodes_resumed = metrics.nodes_resumed

    def save_metrics(self, run_id: str, metrics: RunMetrics) -> None:
        with self._lock:
            run = self._require(run_id)
            run.metrics.duration_ms = metrics.duration_ms
            run.metrics.nodes_executed = metrics.nodes_executed
            run.metrics.nodes_resumed = metrics.nodes_resumed

    def add_usage(
        self,
        run_id: str,
        *,
        llm_calls: int = 0,
        searches: int = 0,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost_usd: float = 0.0,
    ) -> RunMetrics:
        with self._lock:
            run = self._require(run_id)
            run.metrics.llm_calls += llm_calls
            run.metrics.searches += searches
            run.metrics.tokens_in += tokens_in
            run.metrics.tokens_out += tokens_out
            run.metrics.cost_usd = round(run.metrics.cost_usd + cost_usd, 8)
            return run.metrics.model_copy(deep=True)

    def save_checkpoint(self, run_id: str, node_key: str, output: dict) -> None:
        with self._lock:
            self._checkpoints.setdefault(run_id, {})[node_key] = output

    def get_checkpoints(self, run_id: str) -> dict[str, dict]:
        with self._lock:
            return dict(self._checkpoints.get(run_id, {}))

    def save_span(self, span: Span) -> None:
        with self._lock:
            self._spans.setdefault(span.run_id, []).append(span)

    def list_spans(self, run_id: str) -> list[Span]:
        with self._lock:
            return list(self._spans.get(run_id, []))

    def append_event(
        self,
        run_id: str,
        event_type: str,
        message: str = "",
        data: dict | None = None,
        level: str = "info",
    ) -> Event:
        with self._lock:
            next_id = self._event_ids.get(run_id, 0) + 1
            self._event_ids[run_id] = next_id
            event = Event(
                id=next_id,
                run_id=run_id,
                level=level,
                type=event_type,
                message=message,
                data=data or {},
                created_at=utcnow(),
            )
            self._events.setdefault(run_id, []).append(event)
            return event

    def list_events(self, run_id: str, after_id: int = 0, limit: int = 500) -> list[Event]:
        with self._lock:
            events = [e for e in self._events.get(run_id, []) if (e.id or 0) > after_id]
            return events[:limit]

    def request_cancel(self, run_id: str) -> bool:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None or run.status in {
                RunStatus.SUCCEEDED,
                RunStatus.FAILED,
                RunStatus.CANCELLED,
            }:
                return False
            run.cancel_requested = True
            return True

    def is_cancel_requested(self, run_id: str) -> bool:
        with self._lock:
            run = self._runs.get(run_id)
            return bool(run and run.cancel_requested)

    def reset_stale_runs(self, older_than_minutes: int = 30) -> int:
        with self._lock:
            count = 0
            for run in self._runs.values():
                if run.status == RunStatus.RUNNING:
                    run.status = RunStatus.QUEUED
                    run.worker_id = None
                    count += 1
            return count

    def _require(self, run_id: str) -> Run:
        run = self._runs.get(run_id)
        if run is None:
            raise StorageError(f"Unknown run: {run_id}")
        return run
