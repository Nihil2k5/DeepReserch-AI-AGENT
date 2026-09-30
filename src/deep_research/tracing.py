"""Span-based tracing.

Every node, LLM call and tool call opens a span. Spans nest via a thread-local
parent stack (sub-question research runs in a thread pool) and are persisted to
Postgres, so the full tree is reconstructible after the fact.

Spans are also mirrored into Langfuse, live, as they start and finish.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from .config import Settings
from .logging_config import get_logger
from .models import Span, SpanKind, SpanStatus
from .repository import RunRepository, utcnow

logger = get_logger(__name__)

#: Maps our span kinds onto Langfuse observation types.
_AS_TYPE: dict[SpanKind, str] = {
    SpanKind.NODE: "span",
    SpanKind.LLM: "generation",
    SpanKind.TOOL: "tool",
}


class SpanRecorder:
    """Handle yielded inside a span block, for attaching attributes mid-flight."""

    def __init__(self) -> None:
        self.attributes: dict[str, Any] = {}

    def set(self, **attributes: Any) -> None:
        self.attributes.update(attributes)

    def output(self, value: Any) -> None:
        self.attributes["output"] = value


class LangfuseExporter:
    """Mirrors spans into Langfuse (OpenTelemetry-based API, ``langfuse>=3``).

    Nesting is passed **explicitly**. Langfuse propagates parent context through
    OpenTelemetry contextvars, which do not cross a thread boundary — so a span
    opened inside a research worker thread would otherwise land as a new root and
    the tree would flatten. Instead we record each observation's id and hand it
    to its children as ``parent_span_id``.

    Failures are contained: the first error disables the exporter with a single
    warning. Observability must never be able to break a research run, and our
    own span table is the authoritative record either way.
    """

    def __init__(self, public_key: str, secret_key: str, host: str) -> None:
        from langfuse import Langfuse  # imported lazily; optional dependency

        self._client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
        self._trace_ids: dict[str, str] = {}
        self._observation_ids: dict[tuple[str, str], str] = {}
        self._open: dict[tuple[str, str], Any] = {}
        #: One root observation per run, so a run renders as a single tree.
        self._roots: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._disabled = False

    # -- run-level root --------------------------------------------------
    def ensure_trace(self, run_id: str, name: str) -> None:
        """Open the run's root observation; later top-level spans nest under it.

        Without this, the SDK parents each top-level span to its own synthetic
        root and a run shows up as several disconnected trees.
        """
        if self._disabled:
            return
        with self._lock:
            if run_id in self._roots:
                return
        try:
            root = self._client.start_observation(
                trace_context={"trace_id": self._trace_id(run_id)},
                name=name,
                as_type="span",
                metadata={"run_id": run_id},
            )
            with self._lock:
                self._roots[run_id] = root
                self._observation_ids[(run_id, "__root__")] = root.id
        except Exception as exc:
            self._disable(exc)

    def finish_run(self, run_id: str) -> None:
        """Close the run's root observation and release per-run state."""
        with self._lock:
            root = self._roots.pop(run_id, None)
            self._observation_ids.pop((run_id, "__root__"), None)
            self._trace_ids.pop(run_id, None)
            for key in [k for k in self._open if k[0] == run_id]:
                self._open.pop(key, None)
        if root is None or self._disabled:
            return
        try:
            root.end()
        except Exception as exc:  # pragma: no cover
            self._disable(exc)

    # -- observation lifecycle -------------------------------------------
    def _trace_id(self, run_id: str) -> str:
        """One deterministic trace per run, derived from the run id."""
        with self._lock:
            cached = self._trace_ids.get(run_id)
        if cached:
            return cached

        trace_id = self._client.create_trace_id(seed=run_id)
        with self._lock:
            self._trace_ids[run_id] = trace_id
        return trace_id

    def start(
        self,
        *,
        run_id: str,
        span_id: str,
        parent_span_id: str | None,
        name: str,
        kind: SpanKind,
        attributes: dict[str, Any],
    ) -> Any | None:
        if self._disabled:
            return None
        try:
            trace_context: dict[str, Any] = {"trace_id": self._trace_id(run_id)}
            with self._lock:
                parent = (
                    self._observation_ids.get((run_id, parent_span_id))
                    if parent_span_id
                    else self._observation_ids.get((run_id, "__root__"))
                )
            if parent:
                trace_context["parent_span_id"] = parent

            observation = self._client.start_observation(
                trace_context=trace_context,
                name=name,
                as_type=_AS_TYPE.get(kind, "span"),
                input=attributes.get("input"),
                metadata={key: value for key, value in attributes.items() if key != "input"},
            )
            with self._lock:
                self._open[(run_id, span_id)] = observation
                self._observation_ids[(run_id, span_id)] = observation.id
            return observation
        except Exception as exc:
            self._disable(exc)
            return None

    def finish(self, handle: Any, span: Span) -> None:
        if self._disabled or handle is None:
            return
        try:
            attributes = span.attributes
            kwargs: dict[str, Any] = {
                "output": attributes.get("output"),
                "metadata": {
                    key: value for key, value in attributes.items() if key != "output"
                },
                "level": "ERROR" if span.status == SpanStatus.ERROR else "DEFAULT",
                "status_message": span.error,
            }

            # Generation-specific fields only make sense for LLM spans.
            if span.kind == SpanKind.LLM:
                usage: dict[str, int] = {}
                if attributes.get("tokens_in") is not None:
                    usage["input"] = int(attributes["tokens_in"])
                if attributes.get("tokens_out") is not None:
                    usage["output"] = int(attributes["tokens_out"])
                if usage:
                    kwargs["usage_details"] = usage
                if attributes.get("cost_usd"):
                    kwargs["cost_details"] = {"total": float(attributes["cost_usd"])}
                if attributes.get("model"):
                    kwargs["model"] = str(attributes["model"])

            handle.update(**kwargs)
            handle.end()
        except Exception as exc:
            self._disable(exc)
        finally:
            with self._lock:
                self._open.pop((span.run_id, span.span_id), None)
                self._observation_ids.pop((span.run_id, span.span_id), None)

    # -- diagnostics -----------------------------------------------------
    def auth_check(self) -> bool:
        """Verify the keys/host actually authenticate. Worth doing at startup."""
        try:
            return bool(self._client.auth_check())
        except Exception as exc:
            logger.warning("Langfuse credentials rejected: %s", exc)
            return False

    def flush(self) -> None:
        try:
            self._client.flush()
        except Exception:  # pragma: no cover
            pass

    def _disable(self, exc: Exception) -> None:
        logger.warning(
            "Langfuse export disabled after error: %s. Traces remain available in "
            "Postgres; check DR_LANGFUSE_* credentials and host.",
            exc,
        )
        self._disabled = True


def build_exporter(settings: Settings) -> LangfuseExporter | None:
    """Return a Langfuse exporter, or ``None`` when it is not configured."""
    if not settings.langfuse_enabled:
        return None
    try:
        return LangfuseExporter(
            public_key=settings.langfuse_public_key or "",
            secret_key=settings.langfuse_secret_key or "",
            host=settings.langfuse_host,
        )
    except ImportError:
        logger.warning(
            "Langfuse keys are set but the SDK is missing. "
            "Install with: pip install '.[observability]'"
        )
        return None
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not initialise Langfuse: %s", exc)
        return None


class Tracer:
    def __init__(
        self,
        repo: RunRepository,
        run_id: str,
        exporter: LangfuseExporter | None = None,
        trace_name: str | None = None,
    ) -> None:
        self.repo = repo
        self.run_id = run_id
        self.exporter = exporter
        self._local = threading.local()
        if exporter is not None:
            exporter.ensure_trace(run_id, trace_name or f"run {run_id}")

    def _stack(self) -> list[str]:
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = []
            self._local.stack = stack
        return stack

    @property
    def current_span_id(self) -> str | None:
        stack = self._stack()
        return stack[-1] if stack else None

    @contextmanager
    def span(
        self,
        name: str,
        kind: SpanKind,
        parent_id: str | None = None,
        **attributes: Any,
    ) -> Iterator[SpanRecorder]:
        """Open a span. Nests under the current span unless ``parent_id`` is given.

        The explicit ``parent_id`` matters for fan-out: worker threads start with
        an empty trace stack, so they must be told which span they belong under
        or the tree flattens into disconnected roots.
        """
        stack = self._stack()
        effective_parent = parent_id if parent_id is not None else (stack[-1] if stack else None)
        span_id = uuid.uuid4().hex[:16]
        recorder = SpanRecorder()
        started = utcnow()
        status = SpanStatus.OK
        error: str | None = None

        handle = None
        if self.exporter is not None:
            handle = self.exporter.start(
                run_id=self.run_id,
                span_id=span_id,
                parent_span_id=effective_parent,
                name=name,
                kind=kind,
                attributes=dict(attributes),
            )

        stack.append(span_id)
        try:
            yield recorder
        except BaseException as exc:
            status = SpanStatus.ERROR
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            stack.pop()
            ended = utcnow()
            span = Span(
                span_id=span_id,
                run_id=self.run_id,
                parent_id=effective_parent,
                name=name,
                kind=kind,
                started_at=started,
                ended_at=ended,
                duration_ms=round((ended - started).total_seconds() * 1000, 2),
                status=status,
                attributes={**attributes, **recorder.attributes},
                error=error,
            )
            try:
                self.repo.save_span(span)
            except Exception as exc:  # pragma: no cover
                logger.warning("Could not persist span %s: %s", name, exc)
            if self.exporter is not None:
                self.exporter.finish(handle, span)
