"""Node execution context.

Every model call and web search goes through here, which is what guarantees the
three invariants the rest of the system relies on:

1. **Budgets are enforced before the call**, never after.
2. **Every call is traced** as a span, nested under the current node.
3. **Usage is persisted** to the run row as it is spent, so a crashed run still
   reports what it cost.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..budget import Budget
from ..config import Settings
from ..llm.base import LLM, LLMResponse
from ..models import Run, SearchHit, SpanKind
from ..repository import RunRepository
from ..tools.base import SearchTool
from ..tracing import Tracer
from .state import ResearchState


@dataclass
class NodeContext:
    run: Run
    state: ResearchState
    llm: LLM
    search: SearchTool
    tracer: Tracer
    budget: Budget
    settings: Settings
    repo: RunRepository

    # -- model calls -----------------------------------------------------
    def call_llm(
        self,
        *,
        name: str,
        system: str,
        user: str,
        model: str | None = None,
        json_mode: bool = True,
        max_tokens: int = 1800,
        temperature: float = 0.0,
    ) -> LLMResponse:
        self.budget.check(llm=True)
        with self.tracer.span(
            name,
            SpanKind.LLM,
            model=model or self.settings.llm_model,
            input=user[:2000],
        ) as recorder:
            response = self.llm.complete(
                system=system,
                user=user,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
            )
            recorder.set(
                tokens_in=response.prompt_tokens,
                tokens_out=response.completion_tokens,
                cost_usd=response.cost_usd,
                output=response.text[:2000],
            )

        self.budget.record_llm(
            tokens_in=response.prompt_tokens,
            tokens_out=response.completion_tokens,
            cost_usd=response.cost_usd,
        )
        self.repo.add_usage(
            self.run.id,
            llm_calls=1,
            tokens_in=response.prompt_tokens,
            tokens_out=response.completion_tokens,
            cost_usd=response.cost_usd,
        )
        return response

    # -- tool calls ------------------------------------------------------
    def search_web(self, query: str, num_results: int | None = None) -> list[SearchHit]:
        self.budget.check(search=True)
        limit = num_results or self.settings.search_results

        with self.tracer.span(f"search:{query[:70]}", SpanKind.TOOL, input=query) as recorder:
            hits = self.search.search(query, limit)
            recorder.set(result_count=len(hits), output=[hit.url for hit in hits])

        self.budget.record_search()
        self.repo.add_usage(self.run.id, searches=1)
        return hits

    # -- progress --------------------------------------------------------
    def emit(
        self,
        event_type: str,
        message: str = "",
        data: dict | None = None,
        level: str = "info",
    ) -> None:
        try:
            self.repo.append_event(self.run.id, event_type, message, data, level=level)
        except Exception:  # pragma: no cover - progress must never break a run
            pass

    def cancelled(self) -> bool:
        return self.repo.is_cancel_requested(self.run.id)
