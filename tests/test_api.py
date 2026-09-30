from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from deep_research.agent import ResearchAgent
from deep_research.api.app import create_app
from deep_research.api.deps import get_agent
from deep_research.config import Settings, get_settings


def _client(agent: ResearchAgent, settings: Settings) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_agent] = lambda: agent
    app.dependency_overrides[get_settings] = lambda: settings
    # Not used as a context manager: we don't want the lifespan to build the
    # real singleton agent (which would try to reach Postgres).
    return TestClient(app)


@pytest.fixture
def client(agent: ResearchAgent, settings: Settings) -> TestClient:
    test_client = _client(agent, settings)
    yield test_client
    test_client.close()


def _create(client: TestClient, question: str = "Compare alpha and beta systems.") -> str:
    response = client.post("/runs", json={"question": question})
    assert response.status_code == 202, response.text
    return response.json()["run_id"]


# --- ops -------------------------------------------------------------------
def test_health_reports_dependencies(client: TestClient):
    payload = client.get("/health").json()
    assert payload["llm"] == "fake"
    assert payload["search"] == "fake"
    assert payload["database"] is False  # in-memory repo in tests
    assert payload["version"]


def test_metrics_are_prometheus_shaped(client: TestClient):
    run_id = _create(client)
    client.post(f"/runs/{run_id}/execute")

    text = client.get("/metrics").text
    assert "research_runs " in text
    assert 'research_runs_by_status{status="succeeded"} 1' in text
    assert "research_cost_usd_total" in text


# --- runs ------------------------------------------------------------------
def test_create_run_returns_202_and_queues_it(client: TestClient):
    run_id = _create(client)
    detail = client.get(f"/runs/{run_id}").json()
    assert detail["run"]["status"] == "queued"
    assert detail["checkpoints"] == []
    assert detail["span_count"] == 0


def test_create_run_validates_question_length(client: TestClient):
    assert client.post("/runs", json={"question": "short"}).status_code == 422


def test_execute_run_returns_report_summary(client: TestClient):
    run_id = _create(client)
    payload = client.post(f"/runs/{run_id}/execute").json()

    assert payload["status"] == "succeeded"
    assert payload["sections"] >= 1
    assert payload["citations"] >= 1
    assert payload["coverage_score"] == pytest.approx(0.75)
    assert payload["markdown"].startswith("# ")


def test_run_detail_exposes_checkpoints_and_spans(client: TestClient):
    run_id = _create(client)
    client.post(f"/runs/{run_id}/execute")

    detail = client.get(f"/runs/{run_id}").json()
    assert "plan" in detail["checkpoints"]
    assert "synthesize" in detail["checkpoints"]
    assert detail["span_count"] > 0


def test_report_endpoint_json_and_markdown(client: TestClient):
    run_id = _create(client)
    client.post(f"/runs/{run_id}/execute")

    structured = client.get(f"/runs/{run_id}/report").json()
    # Envelope shape: {status, report} in every state, so clients can rely on it.
    assert structured["status"] == "succeeded"
    assert structured["report"]["sections"]

    rendered = client.get(f"/runs/{run_id}/report", params={"format": "markdown"})
    assert rendered.status_code == 200
    assert rendered.text.startswith("# ")


def test_report_is_null_until_the_run_finishes(client: TestClient):
    run_id = _create(client)
    payload = client.get(f"/runs/{run_id}/report").json()
    assert payload["report"] is None
    assert payload["status"] == "queued"


def test_spans_endpoint_returns_the_tree(client: TestClient):
    run_id = _create(client)
    client.post(f"/runs/{run_id}/execute")

    spans = client.get(f"/runs/{run_id}/spans").json()
    assert spans
    kinds = {span["kind"] for span in spans}
    assert {"node", "llm", "tool"} <= kinds
    assert any(span["parent_id"] for span in spans)


def test_events_endpoint_supports_cursor(client: TestClient):
    run_id = _create(client)
    client.post(f"/runs/{run_id}/execute")

    events = client.get(f"/runs/{run_id}/events").json()
    assert events
    first_id = events[0]["id"]

    after = client.get(f"/runs/{run_id}/events", params={"after_id": first_id}).json()
    assert all(event["id"] > first_id for event in after)


def test_list_runs_filters_by_status(client: TestClient):
    run_id = _create(client)
    client.post(f"/runs/{run_id}/execute")

    succeeded = client.get("/runs", params={"status": "succeeded"}).json()
    assert [run["id"] for run in succeeded] == [run_id]

    queued = client.get("/runs", params={"status": "queued"}).json()
    assert queued == []


def test_cancel_endpoint(client: TestClient):
    run_id = _create(client)
    assert client.post(f"/runs/{run_id}/cancel").json()["cancel_requested"] is True
    assert client.get(f"/runs/{run_id}").json()["run"]["cancel_requested"] is True


def test_unknown_run_returns_404(client: TestClient):
    response = client.get("/runs/nope")
    assert response.status_code == 404
    assert response.json()["error"] == "NotFoundError"


# --- auth ------------------------------------------------------------------
def test_api_key_is_enforced_when_configured(agent: ResearchAgent, settings: Settings):
    secured = settings.model_copy(update={"api_key": "secret-key"})
    test_client = _client(agent, secured)
    try:
        payload = {"question": "Compare alpha and beta systems."}
        assert test_client.post("/runs", json=payload).status_code == 401
        assert (
            test_client.post("/runs", json=payload, headers={"X-API-Key": "wrong"}).status_code
            == 401
        )
        allowed = test_client.post(
            "/runs", json=payload, headers={"X-API-Key": "secret-key"}
        )
        assert allowed.status_code == 202

        # Read-only routes stay open.
        assert test_client.get("/runs").status_code == 200
    finally:
        test_client.close()
