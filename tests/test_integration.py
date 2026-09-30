"""Integration tests against real infrastructure.

Skipped unless ``DR_TEST_DATABASE_URL`` is set. These verify the parts that the
in-memory repository deliberately fakes: the Postgres schema, the
``FOR UPDATE SKIP LOCKED`` claim under concurrency, and the JSON round-trips.

    createdb research_test
    DR_TEST_DATABASE_URL=postgresql://localhost/research_test pytest -m integration
"""

from __future__ import annotations

import os
import threading

import pytest

from deep_research.db import Database
from deep_research.models import Run, RunStatus
from deep_research.repository import PostgresRunRepository

pytestmark = pytest.mark.integration

TEST_DSN = os.environ.get("DR_TEST_DATABASE_URL")

requires_db = pytest.mark.skipif(
    not TEST_DSN, reason="set DR_TEST_DATABASE_URL to run integration tests"
)


@pytest.fixture
def pg_repo() -> PostgresRunRepository:
    db = Database(TEST_DSN, size=4)
    repo = PostgresRunRepository(db)
    repo.migrate()
    for run in repo.list_runs(limit=1000):
        repo.db.execute("DELETE FROM runs WHERE id = %s", (run.id,))
    yield repo
    repo.close()


@requires_db
def test_migrations_are_idempotent(pg_repo):
    assert pg_repo.migrate() == []  # already applied


@requires_db
def test_run_roundtrip_preserves_all_fields(pg_repo):
    from deep_research.models import Report, ReportSection

    created = pg_repo.create_run(Run(id="itest-1", question="Integration question?", focus="cost"))
    assert created.id == "itest-1"

    report = Report(
        question="Integration question?",
        executive_summary="Summary.",
        sections=[ReportSection(heading="H", body="B [1].")],
    )
    pg_repo.mark_finished("itest-1", RunStatus.SUCCEEDED, report=report)

    fetched = pg_repo.get_run("itest-1")
    assert fetched.status == RunStatus.SUCCEEDED
    assert fetched.focus == "cost"
    assert fetched.report.sections[0].heading == "H"
    assert fetched.finished_at is not None


@requires_db
def test_usage_accumulates_with_sql_updates(pg_repo):
    pg_repo.create_run(Run(id="itest-2", question="Usage question?"))
    pg_repo.add_usage("itest-2", llm_calls=1, tokens_in=10, cost_usd=0.01)
    metrics = pg_repo.add_usage("itest-2", searches=3, tokens_out=5, cost_usd=0.02)

    assert metrics.llm_calls == 1
    assert metrics.searches == 3
    assert metrics.tokens_out == 5
    assert float(metrics.cost_usd) == pytest.approx(0.03)


@requires_db
def test_checkpoints_and_spans_survive_a_new_connection(pg_repo):
    from deep_research.models import Span, SpanKind
    from deep_research.repository import utcnow

    pg_repo.create_run(Run(id="itest-3", question="Checkpoint question?"))
    pg_repo.save_checkpoint("itest-3", "plan", {"sub_questions": [{"id": "q1"}]})
    pg_repo.save_span(
        Span(span_id="s1", run_id="itest-3", name="plan", kind=SpanKind.NODE, started_at=utcnow())
    )

    fresh = PostgresRunRepository(Database(TEST_DSN, size=1))
    try:
        assert fresh.get_checkpoints("itest-3")["plan"]["sub_questions"][0]["id"] == "q1"
        assert [s.span_id for s in fresh.list_spans("itest-3")] == ["s1"]
    finally:
        fresh.close()


@requires_db
def test_skip_locked_claim_hands_each_run_to_exactly_one_worker(pg_repo):
    for index in range(10):
        pg_repo.create_run(Run(id=f"itest-race-{index}", question="Race question number?"))

    claimed: list[str] = []
    lock = threading.Lock()

    def worker(name: str) -> None:
        repo = PostgresRunRepository(Database(TEST_DSN, size=2))
        try:
            while True:
                run = repo.claim_next_run(name)
                if run is None:
                    return
                with lock:
                    claimed.append(run.id)
        finally:
            repo.close()

    threads = [threading.Thread(target=worker, args=(f"w{i}",)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(claimed) == 10
    assert len(set(claimed)) == 10, "a run was handed to two workers"


@requires_db
def test_cancel_and_stale_sweep(pg_repo):
    pg_repo.create_run(Run(id="itest-4", question="Cancel question?"))
    assert pg_repo.request_cancel("itest-4") is True
    assert pg_repo.is_cancel_requested("itest-4") is True

    pg_repo.create_run(Run(id="itest-5", question="Stale question?"))
    pg_repo.claim_next_run("worker-x")
    assert pg_repo.reset_stale_runs(older_than_minutes=0) >= 1
    assert pg_repo.get_run("itest-5").status == RunStatus.QUEUED
