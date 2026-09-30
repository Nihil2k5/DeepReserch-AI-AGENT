from __future__ import annotations

import pytest

from deep_research.errors import RunCancelled
from deep_research.graph.engine import ResearchEngine
from deep_research.graph.state import research_key, verify_key
from deep_research.models import Run

from .conftest import SYNTHESIZE_MARKER, scripted_llm


# --- happy path ------------------------------------------------------------
def test_full_run_produces_cited_report(engine, repo, run):
    report = engine.execute(run)

    assert report.question == run.question
    assert report.sections, "report should have sections"
    assert report.citations, "report should cite sources"
    assert report.executive_summary

    # citations must point at real retrieved URLs
    urls = {citation.url for citation in report.citations}
    assert urls <= {"https://example.org/alpha", "https://example.com/beta"}

    assert report.markdown.startswith(f"# {run.question}")
    assert "## Sources" in report.markdown


def test_run_records_every_node_as_a_checkpoint(engine, repo, run):
    engine.execute(run)
    keys = set(repo.get_checkpoints(run.id).keys())

    assert "plan" in keys
    assert "synthesize" in keys
    assert "critique" in keys
    assert research_key("q1", 0) in keys
    assert verify_key("q2", 0) in keys


def test_critique_is_attached_to_the_report(engine, run):
    report = engine.execute(run)
    assert report.critique is not None
    assert report.critique.coverage_score == pytest.approx(0.75)
    assert "Quality review" in report.markdown


def test_usage_and_metrics_are_persisted(engine, repo, run):
    engine.execute(run)
    stored = repo.get_run(run.id)

    assert stored.metrics.llm_calls > 0
    assert stored.metrics.searches > 0
    assert stored.metrics.tokens_in > 0
    assert stored.metrics.nodes_executed >= 6  # plan + 2x(research,verify) + synth + critique
    assert stored.metrics.duration_ms > 0


def test_progress_events_are_emitted(engine, repo, run):
    engine.execute(run)
    event_types = {event.type for event in repo.list_events(run.id)}
    assert "run.started" in event_types
    assert "node.completed" in event_types
    assert "synthesize.done" in event_types


# --- durability ------------------------------------------------------------
def test_second_execution_resumes_and_recomputes_nothing(engine, repo, run, llm):
    first = engine.execute(run)
    calls_after_first = llm.call_count

    second = engine.execute(run)

    assert llm.call_count == calls_after_first, "resumed run must not re-call the model"
    assert second.model_dump(exclude={"markdown"}) == first.model_dump(exclude={"markdown"})

    stored = repo.get_run(run.id)
    assert stored.metrics.nodes_executed == 0
    assert stored.metrics.nodes_resumed >= 5  # plan + synth + critique + 2 verifies


def test_partial_resume_only_recomputes_missing_nodes(engine, repo, run, llm, search):
    engine.execute(run)
    # Simulate a crash after synthesis: drop the critique checkpoint.
    repo._checkpoints[run.id].pop("critique")
    calls_before = llm.call_count

    report = engine.execute(run)

    assert report.critique is not None
    # Only the critique call should have been re-issued.
    assert llm.call_count == calls_before + 1
    assert repo.get_run(run.id).metrics.nodes_executed == 1


def test_resume_emits_a_resumed_event(engine, repo, run):
    engine.execute(run)
    repo._events[run.id] = []
    engine.execute(run)
    types = {event.type for event in repo.list_events(run.id)}
    assert "run.resumed" in types


# --- cancellation ----------------------------------------------------------
def test_cancelled_run_stops_before_planning(engine, repo, run):
    repo.request_cancel(run.id)
    with pytest.raises(RunCancelled):
        engine.execute(run)


def test_cancel_mid_run_is_honoured(engine, repo, run, llm):
    # Cancel as soon as the plan is written, i.e. before synthesis.
    original = llm.complete

    def cancel_after_plan(**kwargs):
        response = original(**kwargs)
        if "research planner" in kwargs.get("system", ""):
            repo.request_cancel(run.id)
        return response

    llm.complete = cancel_after_plan  # type: ignore[method-assign]

    with pytest.raises(RunCancelled):
        engine.execute(run)

    assert "synthesize" not in repo.get_checkpoints(run.id)


# --- budgets ---------------------------------------------------------------
def test_exhausted_budget_still_yields_a_partial_report(settings, repo, search):
    tight = settings.model_copy(update={"max_llm_calls": 3})
    llm = scripted_llm()
    engine = ResearchEngine(settings=tight, repo=repo, llm=llm, search=search)
    run = repo.create_run(Run(id="tight", question="Compare alpha and beta systems."))

    report = engine.execute(run)

    # Research was cut short but a report was still written.
    assert report.sections
    assert any("budget" in limitation.lower() for limitation in report.limitations)
    assert any(event.type == "budget.exhausted" for event in repo.list_events(run.id))


def test_search_budget_is_enforced(settings, repo, llm):
    from deep_research.models import SearchHit
    from deep_research.tools.fake_search import FakeSearchTool

    tight = settings.model_copy(update={"max_searches": 0})
    tool = FakeSearchTool(
        hits=[SearchHit(url="https://example.org/a", title="A", text="Some text.", score=1.0)]
    )
    engine = ResearchEngine(settings=tight, repo=repo, llm=llm, search=tool)
    run = repo.create_run(Run(id="nosearch", question="Anything at all?"))

    report = engine.execute(run)

    assert tool.queries == [], "no search may run once the budget is exhausted"
    assert report.sections  # still synthesized, with no evidence
    assert report.citations == []


# --- verification ----------------------------------------------------------
def test_unsupported_evidence_is_excluded_from_the_report(settings, repo, search):
    # Both sub-questions surface the same two claims, so rejecting the beta claim
    # means rejecting it in every sub-question it appeared in.
    llm = scripted_llm(
        verify_response={
            "supported": ["q1-r0-e1", "q2-r0-e1"],
            "unsupported": ["q1-r0-e2", "q2-r0-e2"],
            "gaps": ["need longevity data"],
            "follow_up_queries": [],
        }
    )
    engine = ResearchEngine(settings=settings, repo=repo, llm=llm, search=search)
    run = repo.create_run(Run(id="verify", question="Compare alpha and beta systems."))

    report = engine.execute(run)

    synth_prompt = next(
        call["user"] for call in llm.calls if SYNTHESIZE_MARKER in call["system"]
    )
    # Rejected claims must not reach the synthesizer; supported ones must.
    assert "Beta deployments last about 50 years" not in synth_prompt
    assert "Alpha systems reach 92 percent round-trip efficiency" in synth_prompt

    checkpoints = repo.get_checkpoints(run.id)
    rejected: set[str] = set()
    for key, payload in checkpoints.items():
        if key.startswith("verify:"):
            rejected.update(payload.get("unsupported_ids") or [])
    assert rejected == {"q1-r0-e2", "q2-r0-e2"}

    # All four claims were collected, but only the two that survived verification
    # can appear as sources in the report.
    collected = [
        item
        for key, payload in checkpoints.items()
        if key.startswith("research:")
        for item in payload.get("evidence", [])
    ]
    assert len(collected) == 4

    assert {citation.url for citation in report.citations} == {"https://example.org/alpha"}
    # The beta section's [2] marker no longer resolves, so it must be stripped.
    assert all("[2]" not in section.body for section in report.sections)


def test_gap_analysis_is_checkpointed(settings, repo, search):
    llm = scripted_llm(
        verify_response={
            "supported": ["q1-r0-e1"],
            "unsupported": ["q1-r0-e2"],
            "gaps": ["missing cost data"],
            "follow_up_queries": ["alpha systems capital cost"],
        }
    )
    engine = ResearchEngine(settings=settings, repo=repo, llm=llm, search=search)
    run = repo.create_run(Run(id="gaps", question="Compare alpha and beta systems."))
    engine.execute(run)

    analysis = repo.get_checkpoints(run.id)[verify_key("q1", 0)]
    assert analysis["gaps"] == ["missing cost data"]
    assert analysis["unsupported_ids"] == ["q1-r0-e2"]


def test_follow_up_round_uses_verifier_queries(settings, repo, search):
    llm = scripted_llm(
        verify_response={
            "supported": ["q1-r0-e1"],
            "unsupported": [],
            "gaps": ["still missing cost data"],
            "follow_up_queries": ["alpha capital cost per kwh"],
        }
    )
    two_rounds = settings.model_copy(update={"max_research_rounds": 2})
    engine = ResearchEngine(settings=two_rounds, repo=repo, llm=llm, search=search)
    run = repo.create_run(Run(id="rounds", question="Compare alpha and beta systems."))

    engine.execute(run)

    assert research_key("q1", 1) in repo.get_checkpoints(run.id)
    assert "alpha capital cost per kwh" in search.queries


# --- tracing ---------------------------------------------------------------
def test_spans_form_a_tree(engine, repo, run):
    engine.execute(run)
    spans = repo.list_spans(run.id)
    by_id = {span.span_id: span for span in spans}

    stage = next((s for s in spans if s.name == "research.round0"), None)
    assert stage is not None, "the parallel stage should open its own span"

    children = [s for s in spans if s.parent_id == stage.span_id]
    assert children, "research node spans must nest under the stage span"
    assert all(s.kind.value == "node" for s in children)

    llm_spans = [s for s in spans if s.kind.value == "llm"]
    assert llm_spans
    assert all(
        s.parent_id is None or s.parent_id in by_id for s in spans
    ), "every parent reference must resolve"
    assert all(s.duration_ms is not None and s.duration_ms >= 0 for s in spans)


def test_search_and_llm_calls_are_traced_separately(engine, repo, run):
    engine.execute(run)
    kinds = {span.kind.value for span in repo.list_spans(run.id)}
    assert {"node", "llm", "tool"} <= kinds


def test_span_attributes_capture_usage(engine, repo, run):
    engine.execute(run)
    llm_spans = [s for s in repo.list_spans(run.id) if s.kind.value == "llm"]
    assert all("tokens_in" in span.attributes for span in llm_spans)
    assert all("cost_usd" in span.attributes for span in llm_spans)
