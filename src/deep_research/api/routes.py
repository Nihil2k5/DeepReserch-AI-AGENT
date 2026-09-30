"""HTTP routes."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, Query
from fastapi.responses import PlainTextResponse, StreamingResponse

from .. import __version__
from ..agent import ResearchAgent
from ..models import (
    TERMINAL_STATUSES,
    CreateRunRequest,
    CreateRunResponse,
    Run,
    RunDetail,
    RunStatus,
)
from .deps import get_agent, require_api_key

router = APIRouter()


# --- ops -------------------------------------------------------------------
@router.get("/health", tags=["ops"])
def health(agent: ResearchAgent = Depends(get_agent)) -> dict:
    payload = agent.health()
    payload["version"] = __version__
    return payload


@router.get("/metrics", response_class=PlainTextResponse, tags=["ops"])
def metrics(agent: ResearchAgent = Depends(get_agent)) -> str:
    """Run-level metrics in Prometheus text format."""
    runs = agent.repo.list_runs(limit=1000)
    by_status: dict[str, int] = {}
    total_cost = 0.0
    durations: list[float] = []
    for run in runs:
        by_status[run.status.value] = by_status.get(run.status.value, 0) + 1
        total_cost += run.metrics.cost_usd
        if run.metrics.duration_ms:
            durations.append(run.metrics.duration_ms)

    lines = [
        "# TYPE research_runs gauge",
        f"research_runs {len(runs)}",
        "# TYPE research_runs_by_status gauge",
    ]
    lines += [
        f'research_runs_by_status{{status="{name}"}} {count}'
        for name, count in sorted(by_status.items())
    ]
    lines += [
        "# TYPE research_cost_usd_total counter",
        f"research_cost_usd_total {round(total_cost, 6)}",
        "# TYPE research_run_duration_ms summary",
        f"research_run_duration_ms_count {len(durations)}",
        f"research_run_duration_ms_sum {round(sum(durations), 2)}",
    ]
    return "\n".join(lines) + "\n"


# --- runs ------------------------------------------------------------------
@router.post(
    "/runs",
    response_model=CreateRunResponse,
    status_code=202,
    tags=["runs"],
    dependencies=[Depends(require_api_key)],
)
def create_run(
    request: CreateRunRequest,
    agent: ResearchAgent = Depends(get_agent),
) -> CreateRunResponse:
    """Queue a research run. A worker picks it up; nothing blocks here."""
    run = agent.create_run(request.question, request.focus)
    return CreateRunResponse(run_id=run.id, status=run.status)


@router.get("/runs", response_model=list[Run], tags=["runs"])
def list_runs(
    status: RunStatus | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    agent: ResearchAgent = Depends(get_agent),
) -> list[Run]:
    return agent.repo.list_runs(limit=limit, status=status)


@router.get("/runs/{run_id}", response_model=RunDetail, tags=["runs"])
def get_run(run_id: str, agent: ResearchAgent = Depends(get_agent)) -> RunDetail:
    run = agent.get_run(run_id)
    checkpoints = sorted(agent.repo.get_checkpoints(run_id).keys())
    return RunDetail(
        run=run,
        checkpoints=checkpoints,
        span_count=len(agent.repo.list_spans(run_id)),
    )


@router.get("/runs/{run_id}/report", tags=["runs"])
def get_report(
    run_id: str,
    format: str = Query(default="json", pattern="^(json|markdown)$"),
    agent: ResearchAgent = Depends(get_agent),
):
    """The finished report, as structured JSON or rendered markdown."""
    run = agent.get_run(run_id)
    # Always the same envelope so clients (e.g. the /ui dashboard) can rely on
    # {status, report} whether or not the run has finished with a report.
    if run.report is None:
        return {"status": run.status.value, "report": None}
    if format == "markdown":
        return PlainTextResponse(run.report.markdown or "", media_type="text/markdown")
    return {"status": run.status.value, "report": run.report}


@router.get("/runs/{run_id}/spans", tags=["runs"])
def get_spans(run_id: str, agent: ResearchAgent = Depends(get_agent)) -> list[dict]:
    agent.get_run(run_id)
    return [span.model_dump(mode="json") for span in agent.repo.list_spans(run_id)]


@router.get("/runs/{run_id}/events", tags=["runs"])
def get_events(
    run_id: str,
    after_id: int = Query(default=0, ge=0),
    limit: int = Query(default=200, ge=1, le=1000),
    agent: ResearchAgent = Depends(get_agent),
) -> list[dict]:
    agent.get_run(run_id)
    events = agent.repo.list_events(run_id, after_id=after_id, limit=limit)
    return [event.model_dump(mode="json") for event in events]


@router.get("/runs/{run_id}/events/stream", tags=["runs"])
async def stream_events(
    run_id: str,
    after: int = 0,
    agent: ResearchAgent = Depends(get_agent),
) -> StreamingResponse:
    """Server-Sent Events tail of the run's progress log.

    Polls the durable event log rather than holding in-process state, so it works
    no matter which worker is executing the run.
    """
    agent.get_run(run_id)

    async def event_stream():
        cursor = after
        idle = 0
        while True:
            events = await asyncio.to_thread(
                agent.repo.list_events, run_id, cursor, 200
            )
            for event in events:
                cursor = event.id or cursor
                yield f"event: progress\ndata: {json.dumps(event.model_dump(mode='json'))}\n\n"
                idle = 0

            if not events:
                idle += 1
                if idle % 10 == 0:
                    yield ": keep-alive\n\n"

            run = await asyncio.to_thread(agent.repo.get_run, run_id)
            if run is None or run.status in TERMINAL_STATUSES:
                payload = {
                    "status": run.status.value if run else "unknown",
                    "report_ready": bool(run and run.report),
                }
                yield f"event: done\ndata: {json.dumps(payload)}\n\n"
                return

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/runs/{run_id}/cancel", tags=["runs"], dependencies=[Depends(require_api_key)])
def cancel_run(run_id: str, agent: ResearchAgent = Depends(get_agent)) -> dict:
    cancelled = agent.cancel(run_id)
    return {"run_id": run_id, "cancel_requested": cancelled}


@router.post("/runs/{run_id}/execute", tags=["runs"], dependencies=[Depends(require_api_key)])
def execute_run(run_id: str, agent: ResearchAgent = Depends(get_agent)) -> dict:
    """Execute a run synchronously. Handy for demos and tests; not for production
    traffic, where you want the worker so a slow run doesn't hold a request."""
    report = agent.execute(run_id)
    return {
        "run_id": run_id,
        "status": RunStatus.SUCCEEDED.value,
        "sections": len(report.sections),
        "citations": len(report.citations),
        "coverage_score": report.critique.coverage_score if report.critique else None,
        "markdown": report.markdown,
    }
