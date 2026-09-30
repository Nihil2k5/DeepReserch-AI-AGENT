from __future__ import annotations

import pytest

from deep_research.errors import StorageError
from deep_research.models import Report, Run, RunStatus, Span, SpanKind
from deep_research.repository import InMemoryRunRepository, utcnow


@pytest.fixture
def repo() -> InMemoryRunRepository:
    return InMemoryRunRepository()


def test_create_and_get_roundtrip(repo):
    created = repo.create_run(Run(id="r1", question="Q?"))
    assert created.created_at is not None
    assert repo.get_run("r1").question == "Q?"
    assert repo.get_run("missing") is None


def test_get_run_returns_a_copy(repo):
    repo.create_run(Run(id="r1", question="Q?"))
    fetched = repo.get_run("r1")
    fetched.question = "mutated"
    assert repo.get_run("r1").question == "Q?"


def test_claim_next_run_is_first_in_first_out(repo):
    repo.create_run(Run(id="r1", question="first?"))
    repo.create_run(Run(id="r2", question="second?"))

    claimed = repo.claim_next_run("worker-a")
    assert claimed.id == "r1"
    assert claimed.status == RunStatus.RUNNING
    assert claimed.attempt == 1
    assert claimed.worker_id == "worker-a"

    assert repo.claim_next_run("worker-b").id == "r2"
    assert repo.claim_next_run("worker-b") is None


def test_claimed_runs_are_not_handed_out_twice(repo):
    repo.create_run(Run(id="r1", question="Q?"))
    assert repo.claim_next_run("w1") is not None
    assert repo.claim_next_run("w2") is None


def test_mark_finished_records_report_and_error(repo):
    repo.create_run(Run(id="r1", question="Q?"))
    report = Report(question="Q?", executive_summary="A.")
    repo.mark_finished("r1", RunStatus.SUCCEEDED, report=report)

    stored = repo.get_run("r1")
    assert stored.status == RunStatus.SUCCEEDED
    assert stored.finished_at is not None
    assert stored.report.executive_summary == "A."


def test_mark_finished_unknown_run_raises(repo):
    with pytest.raises(StorageError):
        repo.mark_finished("nope", RunStatus.FAILED)


def test_add_usage_accumulates(repo):
    repo.create_run(Run(id="r1", question="Q?"))
    repo.add_usage("r1", llm_calls=1, tokens_in=10, tokens_out=5, cost_usd=0.01)
    metrics = repo.add_usage("r1", searches=2, cost_usd=0.02)

    assert metrics.llm_calls == 1
    assert metrics.searches == 2
    assert metrics.tokens_in == 10
    assert metrics.cost_usd == pytest.approx(0.03)


def test_checkpoints_roundtrip_and_overwrite(repo):
    repo.create_run(Run(id="r1", question="Q?"))
    repo.save_checkpoint("r1", "plan", {"a": 1})
    repo.save_checkpoint("r1", "plan", {"a": 2})
    repo.save_checkpoint("r1", "synthesize", {"b": 1})

    checkpoints = repo.get_checkpoints("r1")
    assert checkpoints["plan"] == {"a": 2}
    assert set(checkpoints) == {"plan", "synthesize"}


def test_events_are_ordered_and_filterable(repo):
    repo.create_run(Run(id="r1", question="Q?"))
    first = repo.append_event("r1", "run.started", "go")
    repo.append_event("r1", "node.completed", "done")

    assert first.id == 1
    assert [e.id for e in repo.list_events("r1")] == [1, 2]
    assert [e.id for e in repo.list_events("r1", after_id=1)] == [2]


def test_spans_are_stored_per_run(repo):
    repo.create_run(Run(id="r1", question="Q?"))
    span = Span(
        span_id="s1",
        run_id="r1",
        name="plan",
        kind=SpanKind.NODE,
        started_at=utcnow(),
    )
    repo.save_span(span)
    assert [s.span_id for s in repo.list_spans("r1")] == ["s1"]
    assert repo.list_spans("other") == []


def test_cancel_only_applies_to_active_runs(repo):
    repo.create_run(Run(id="r1", question="Q?"))
    assert repo.request_cancel("r1") is True
    assert repo.is_cancel_requested("r1") is True

    repo.mark_finished("r1", RunStatus.SUCCEEDED)
    assert repo.request_cancel("r1") is False


def test_reset_stale_runs_requeues_running_ones(repo):
    repo.create_run(Run(id="r1", question="Q?"))
    repo.claim_next_run("w1")

    assert repo.reset_stale_runs() == 1
    requeued = repo.get_run("r1")
    assert requeued.status == RunStatus.QUEUED
    assert requeued.worker_id is None
    assert requeued.attempt == 1  # attempt counter is preserved for observability


def test_list_runs_filters_and_orders(repo):
    repo.create_run(Run(id="r1", question="first?"))
    repo.create_run(Run(id="r2", question="second?"))
    repo.claim_next_run("w1")

    assert [r.id for r in repo.list_runs()] == ["r2", "r1"]
    assert [r.id for r in repo.list_runs(status=RunStatus.RUNNING)] == ["r1"]
