"""Domain schemas shared by the graph, repositories, API and eval harness."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# --- enums -----------------------------------------------------------------
class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATUSES = {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}


class SpanKind(str, Enum):
    NODE = "node"
    LLM = "llm"
    TOOL = "tool"


class SpanStatus(str, Enum):
    OK = "ok"
    ERROR = "error"


# --- planning / research ---------------------------------------------------
class SubQuestion(BaseModel):
    id: str
    question: str
    rationale: str = ""


class SearchHit(BaseModel):
    url: str
    title: str = ""
    published: str | None = None
    snippet: str = ""
    text: str = ""
    score: float = 0.0


class Evidence(BaseModel):
    """One verifiable claim with the exact quote and URL that support it."""

    id: str
    sub_question_id: str
    claim: str
    quote: str
    url: str
    title: str = ""
    confidence: float = 0.5


class GapAnalysis(BaseModel):
    sub_question_id: str
    supported_ids: list[str] = Field(default_factory=list)
    unsupported_ids: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    follow_up_queries: list[str] = Field(default_factory=list)

    @property
    def needs_more_research(self) -> bool:
        return bool(self.gaps or self.unsupported_ids)


# --- report ----------------------------------------------------------------
class Citation(BaseModel):
    index: int
    url: str
    title: str = ""


class ReportSection(BaseModel):
    heading: str
    body: str


class Critique(BaseModel):
    coverage_score: float = 0.0
    unsupported_claims: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    verdict: str = ""


class Report(BaseModel):
    question: str
    executive_summary: str = ""
    sub_questions: list[str] = Field(default_factory=list)
    sections: list[ReportSection] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    critique: Critique | None = None
    markdown: str = ""


# --- run -------------------------------------------------------------------
class RunMetrics(BaseModel):
    llm_calls: int = 0
    searches: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    duration_ms: float = 0.0
    nodes_executed: int = 0
    nodes_resumed: int = 0


class Run(BaseModel):
    id: str
    question: str
    focus: str | None = None
    status: RunStatus = RunStatus.QUEUED
    created_at: datetime | None = None
    updated_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    worker_id: str | None = None
    attempt: int = 0
    error: str | None = None
    cancel_requested: bool = False
    report: Report | None = None
    metrics: RunMetrics = Field(default_factory=RunMetrics)


class Span(BaseModel):
    span_id: str
    run_id: str
    parent_id: str | None = None
    name: str
    kind: SpanKind
    started_at: datetime
    ended_at: datetime | None = None
    duration_ms: float | None = None
    status: SpanStatus = SpanStatus.OK
    attributes: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class Event(BaseModel):
    id: int | None = None
    run_id: str
    level: str = "info"
    type: str
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None


# --- API -------------------------------------------------------------------
class CreateRunRequest(BaseModel):
    question: str = Field(min_length=8, max_length=500)
    focus: str | None = Field(
        default=None,
        max_length=300,
        description="Optional steer, e.g. 'focus on cost benchmarks from 2025'.",
    )


class CreateRunResponse(BaseModel):
    run_id: str
    status: RunStatus


class RunDetail(BaseModel):
    run: Run
    checkpoints: list[str] = Field(
        default_factory=list, description="Completed node keys (resume points)."
    )
    span_count: int = 0
