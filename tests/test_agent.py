from __future__ import annotations

import pytest

from deep_research.agent import ResearchAgent
from deep_research.errors import NotFoundError, RunCancelled
from deep_research.models import RunStatus


def test_create_run_queues_and_logs_an_event(agent: ResearchAgent, repo):
    run = agent.create_run("What is the efficiency of alpha systems?")

    assert run.status == RunStatus.QUEUED
    assert run.id
    event_types = [event.type for event in repo.list_events(run.id)]
    assert "run.queued" in event_types


def test_execute_marks_the_run_succeeded_with_a_report(agent: ResearchAgent):
    run = agent.create_run("Compare alpha and beta systems.")
    report = agent.execute(run.id)

    stored = agent.get_run(run.id)
    assert stored.status == RunStatus.SUCCEEDED
    assert stored.report is not None
    assert stored.report.executive_summary == report.executive_summary
    assert stored.finished_at is not None
    assert stored.worker_id  # inline execution still records an owner


def test_execute_twice_returns_the_stored_report(agent: ResearchAgent):
    run = agent.create_run("Compare alpha and beta systems.")
    first = agent.execute(run.id)
    second = agent.execute(run.id)
    assert second.markdown == first.markdown


def test_get_run_raises_for_unknown_id(agent: ResearchAgent):
    with pytest.raises(NotFoundError):
        agent.get_run("does-not-exist")


def test_failed_run_is_recorded_with_the_error(agent: ResearchAgent, llm):
    run = agent.create_run("Compare alpha and beta systems.")

    def boom(**_kwargs):
        raise RuntimeError("provider exploded")

    llm.complete = boom  # type: ignore[method-assign]

    with pytest.raises(RuntimeError):
        agent.execute(run.id)

    stored = agent.get_run(run.id)
    assert stored.status == RunStatus.FAILED
    assert "provider exploded" in (stored.error or "")


def test_cancelled_run_fails_with_a_clear_error(agent: ResearchAgent, repo):
    run = agent.create_run("Compare alpha and beta systems.")
    assert agent.cancel(run.id) is True
    assert repo.is_cancel_requested(run.id)

    with pytest.raises(RunCancelled):
        agent.execute(run.id)

    stored = agent.get_run(run.id)
    assert stored.status == RunStatus.FAILED
    assert "cancel" in (stored.error or "").lower()


def test_get_run_after_cancel_still_returns_the_row(agent: ResearchAgent):
    run = agent.create_run("Compare alpha and beta systems.")
    agent.cancel(run.id)
    assert agent.get_run(run.id).cancel_requested is True


def test_health_reports_component_names(agent: ResearchAgent):
    health = agent.health()
    assert health["llm"] == "fake"
    assert health["search"] == "fake"
    assert "database" in health
    assert "redis" in health
