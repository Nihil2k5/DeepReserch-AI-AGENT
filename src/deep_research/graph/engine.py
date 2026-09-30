"""Durable execution engine.

The contract: **a run is a set of uniquely-keyed nodes, and a completed node is
never recomputed.**

Each node writes its output to the ``checkpoints`` table under a deterministic
key. Executing a run means walking the plan, skipping any key already present,
and writing new results as it goes. That gives crash recovery for free — kill
the worker mid-run and a second worker resumes from the last checkpoint instead
of paying for the whole research again — and it makes the graph inspectable:
``SELECT node_key FROM checkpoints`` is a progress bar.

Sub-question research fans out across a thread pool, so independent searches
overlap. Spans are parented explicitly into the round's stage span because the
worker threads start with an empty trace stack.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from ..budget import Budget
from ..config import Settings
from ..errors import BudgetExceeded, RunCancelled
from ..llm.base import LLM
from ..logging_config import get_logger
from ..models import Report, Run, SpanKind
from ..reporting import render_markdown
from ..repository import RunRepository
from ..tools.base import SearchTool
from ..tracing import Tracer
from .context import NodeContext
from .nodes import CriticNode, PlannerNode, ResearcherNode, SynthesizerNode, VerifierNode
from .nodes.base import Node
from .state import CRITIQUE_KEY, PLAN_KEY, SYNTHESIZE_KEY, ResearchState, research_key, verify_key

logger = get_logger(__name__)


class ResearchEngine:
    def __init__(
        self,
        *,
        settings: Settings,
        repo: RunRepository,
        llm: LLM,
        search: SearchTool,
        exporter=None,
        max_workers: int | None = None,
    ) -> None:
        self.settings = settings
        self.repo = repo
        self.llm = llm
        self.search = search
        self.exporter = exporter
        self.max_workers = max_workers or max(1, settings.max_search_concurrency)

    # -- public API ------------------------------------------------------
    def execute(self, run: Run) -> Report:
        """Run (or resume) a research run and return the final report."""
        started = time.perf_counter()
        tracer = Tracer(self.repo, run.id, self.exporter, trace_name=run.question)
        budget = Budget(self.settings, initial=run.metrics)

        checkpoints = self.repo.get_checkpoints(run.id)
        state = ResearchState(run.question, run.focus, checkpoints)
        ctx = NodeContext(
            run=run,
            state=state,
            llm=self.llm,
            search=self.search,
            tracer=tracer,
            budget=budget,
            settings=self.settings,
            repo=self.repo,
        )
        counters = {"executed": 0, "resumed": 0}

        if state.has_plan or research_key("", 0) in checkpoints:
            logger.info("Resuming run %s with %d checkpoints", run.id, len(checkpoints))
            ctx.emit(
                "run.resumed",
                f"Resuming from {len(checkpoints)} completed nodes",
                {"resumed_nodes": len(checkpoints)},
            )
        else:
            ctx.emit("run.started", "Research run started")

        # --- 1. plan -----------------------------------------------------
        # Every stage goes through _execute_node, which is what makes resumption
        # a property of the engine rather than something each node must remember.
        self._maybe_cancel(ctx)
        self._execute_node(
            ctx,
            PlannerNode(
                max_sub_questions=self.settings.max_subquestions,
                model=self.settings.llm_reasoning_model,
            ),
            PLAN_KEY,
            counters,
        )
        plan = state.plan or []
        if not plan:  # pragma: no cover - planner always falls back to one question
            raise RunCancelled("Planner produced no usable sub-questions")

        # --- 2. research + verify rounds ---------------------------------
        self._maybe_cancel(ctx)
        budget_exhausted: str | None = None
        try:
            self._run_research_rounds(ctx, plan, counters)
        except BudgetExceeded as exc:
            # Don't throw away research: grant synthesis headroom, write up what we
            # have, and record the shortfall as a limitation.
            budget_exhausted = str(exc)
            logger.warning("Budget exhausted mid-research: %s", exc)
            budget.reserve_for_finalization()
            ctx.emit("budget.exhausted", str(exc), level="warning")

        self._maybe_cancel(ctx)

        # --- 3. synthesize -----------------------------------------------
        extra_limitations: list[str] = []
        if budget_exhausted:
            extra_limitations.append(
                "Research was cut short by a run budget limit, so coverage is partial."
            )

        self._execute_node(
            ctx,
            SynthesizerNode(
                question=run.question,
                evidence=state.supported_evidence(),
                focus=run.focus,
                sub_questions=[sq.question for sq in plan],
                extra_limitations=extra_limitations,
                model=self.settings.llm_reasoning_model,
            ),
            SYNTHESIZE_KEY,
            counters,
        )

        report = state.report
        if report is None:  # pragma: no cover - synthesizer always checkpoints
            raise RunCancelled("Synthesizer produced no report")

        # --- 4. critique (best effort) -----------------------------------
        self._maybe_cancel(ctx)
        try:
            self._execute_node(
                ctx,
                CriticNode(
                    report=report,
                    evidence=state.supported_evidence(),
                    model=self.settings.llm_model,
                ),
                CRITIQUE_KEY,
                counters,
            )
        except BudgetExceeded as exc:
            logger.warning("Skipping critique, budget exhausted: %s", exc)
        except Exception as exc:  # critique is optional; never fail a run for it
            logger.warning("Critique failed: %s", exc)

        if state.critique is not None:
            report = report.model_copy(update={"critique": state.critique})

        report = report.model_copy(update={"markdown": render_markdown(report)})

        metrics = budget.snapshot()
        metrics.duration_ms = round((time.perf_counter() - started) * 1000, 2)
        metrics.nodes_executed = counters["executed"]
        metrics.nodes_resumed = counters["resumed"]
        self.repo.save_metrics(run.id, metrics)

        if self.exporter is not None:
            self.exporter.finish_run(run.id)
            self.exporter.flush()

        logger.info(
            "Run %s finished: %d nodes executed, %d resumed, %d evidence items, %.1fs",
            run.id,
            counters["executed"],
            counters["resumed"],
            len(state.supported_evidence()),
            metrics.duration_ms / 1000,
        )
        return report

    # -- stages ----------------------------------------------------------
    def _run_research_rounds(self, ctx: NodeContext, plan, counters: dict) -> None:
        for round_index in range(self.settings.max_research_rounds):
            # Count everything already on disk for this round as replayed work.
            for sub_question in plan:
                if research_key(sub_question.id, round_index) in ctx.state.checkpoints:
                    counters["resumed"] += 1

            pending = [
                sub_question
                for sub_question in plan
                if self._needs_research(ctx.state, sub_question.id, round_index)
            ]
            if not pending:
                break

            self._research_parallel(ctx, pending, round_index, counters)

            for sub_question in pending:
                self._maybe_cancel(ctx)
                self._execute_node(
                    ctx,
                    VerifierNode(
                        sub_question=sub_question,
                        evidence=ctx.state.evidence(sub_question.id, round_index),
                        round_index=round_index,
                        model=self.settings.llm_model,
                    ),
                    verify_key(sub_question.id, round_index),
                    counters,
                )

    def _needs_research(self, state: ResearchState, sub_question_id: str, round_index: int) -> bool:
        if research_key(sub_question_id, round_index) in state.checkpoints:
            return False
        if round_index == 0:
            return True
        previous = state.verification(sub_question_id, round_index - 1)
        return bool(previous and previous.needs_more_research)

    def _research_parallel(self, ctx: NodeContext, pending, round_index: int, counters: dict) -> None:
        stage_key = f"research.round{round_index}"
        # Stage-level events so dashboard clients see the fan-out as one stage.
        ctx.emit(
            "node.started",
            f"Researching {len(pending)} sub-questions (round {round_index})",
            {"node_key": "research", "round_index": round_index},
        )

        with ctx.tracer.span(
            stage_key,
            SpanKind.NODE,
            round_index=round_index,
            sub_question_ids=[sq.id for sq in pending],
        ) as stage:
            # Read inside the block: this is the span the workers nest under.
            parent_span_id = ctx.tracer.current_span_id

            def run_one(sub_question) -> dict:
                node = ResearcherNode(
                    sub_question=sub_question,
                    round_index=round_index,
                    follow_up_queries=ctx.state.follow_up_queries(sub_question.id, round_index),
                    model=self.settings.llm_model,
                )
                key = research_key(sub_question.id, round_index)
                with ctx.tracer.span(
                    node.name, SpanKind.NODE, parent_id=parent_span_id, node_key=key
                ) as recorder:
                    output = node.run(ctx)
                    recorder.set(evidence_count=len(output.get("evidence", [])))
                ctx.repo.save_checkpoint(ctx.run.id, key, output)
                # Keep the live state in step so later nodes (synthesize, verify)
                # see this evidence within the same execution.
                ctx.state.checkpoints[key] = output
                counters["executed"] += 1
                return output

            workers = min(self.max_workers, len(pending))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(run_one, pending))

            stage.set(total_evidence=sum(len(r.get("evidence", [])) for r in results))

        ctx.emit(
            "node.completed",
            f"Research round {round_index} complete",
            {"node_key": "research", "round_index": round_index},
        )

    # -- helpers ---------------------------------------------------------
    def _execute_node(self, ctx: NodeContext, node: Node, key: str, counters: dict) -> dict:
        existing = ctx.state.checkpoints.get(key)
        if existing is not None:
            counters["resumed"] += 1
            logger.debug("Resuming node %s from checkpoint", key)
            return existing

        ctx.emit("node.started", f"Running {node.name}", {"node_key": key})
        with ctx.tracer.span(node.name, SpanKind.NODE, node_key=key) as recorder:
            output = node.run(ctx)
            recorder.set(output_key=key)

        ctx.repo.save_checkpoint(ctx.run.id, key, output)
        ctx.state.checkpoints[key] = output
        counters["executed"] += 1
        ctx.emit("node.completed", f"{node.name} complete", {"node_key": key})
        return output

    @staticmethod
    def _maybe_cancel(ctx: NodeContext) -> None:
        if ctx.cancelled():
            raise RunCancelled("Run was cancelled")
