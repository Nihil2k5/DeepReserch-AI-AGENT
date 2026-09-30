"""Verify the Langfuse export path without needing real credentials.

A local HTTP server stands in for Langfuse Cloud and captures the OTLP payload
the SDK actually sends. That lets us assert the things that matter and are easy
to get silently wrong:

* spans are exported at all (a broken exporter disables itself quietly);
* they land under **one trace with one root** per run;
* the fan-out research spans nest under their round's stage span even though
  they are opened on worker threads;
* LLM and tool spans carry the right observation type and usage details.

Without this test an SDK API change would simply stop producing traces while
every run still reported success.

Note: langfuse installs a **process-global** OpenTelemetry exporter, so the first
client created in this process fixes the export target. Exactly one test here can
therefore assert on captured payloads; the others deliberately do not need to.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("langfuse")

from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (  # noqa: E402
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)

from deep_research.graph.engine import ResearchEngine  # noqa: E402
from deep_research.models import Run  # noqa: E402
from deep_research.tracing import LangfuseExporter  # noqa: E402


class _Capture:
    def __init__(self) -> None:
        self.bodies: list[bytes] = []
        self.paths: list[str] = []
        self.lock = threading.Lock()

    def record(self, path: str, body: bytes) -> None:
        with self.lock:
            self.paths.append(path)
            self.bodies.append(body)


@pytest.fixture
def otlp_capture():
    capture = _Capture()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            capture.record(self.path, self.rfile.read(length) if length else b"")
            payload = ExportTraceServiceResponse().SerializeToString()
            self.send_response(200)
            self.send_header("Content-Type", "application/x-protobuf")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}", capture
    finally:
        server.shutdown()
        server.server_close()


def _decode(capture: _Capture) -> list[dict]:
    """Flatten every captured OTLP span into a simple dict."""
    spans: list[dict] = []
    for body in capture.bodies:
        request = ExportTraceServiceRequest()
        request.ParseFromString(body)
        for resource_spans in request.resource_spans:
            for scope_spans in resource_spans.scope_spans:
                for span in scope_spans.spans:
                    spans.append(
                        {
                            "trace_id": span.trace_id.hex(),
                            "span_id": span.span_id.hex(),
                            "parent_id": span.parent_span_id.hex(),
                            "name": span.name,
                            "attrs": {kv.key: str(kv.value) for kv in span.attributes},
                        }
                    )
    return spans


def _type_of(span: dict) -> str:
    return span["attrs"].get("langfuse.observation.type", "")


def test_full_run_exports_one_nested_trace(otlp_capture, settings, repo, llm, search):
    host, capture = otlp_capture
    exporter = LangfuseExporter(public_key="pk-lf-test", secret_key="sk-lf-test", host=host)
    engine = ResearchEngine(
        settings=settings, repo=repo, llm=llm, search=search, exporter=exporter
    )
    question = "Compare alpha and beta systems."
    run = repo.create_run(Run(id="lf-run-1", question=question))

    engine.execute(run)
    exporter.flush()

    spans = _decode(capture)
    assert spans, f"no spans exported; POSTed to {capture.paths}"
    ids = {span["span_id"] for span in spans}

    # --- every stage reached Langfuse ---------------------------------
    names = {span["name"] for span in spans}
    for expected in (
        "plan",
        "plan.decompose",
        "research.round0",
        "research",
        "research.queries",
        "research.extract",
        "verify",
        "verify.check",
        "synthesize",
        "synthesize.report",
        "critique",
        "critique.review",
    ):
        assert expected in names, f"{expected!r} missing; saw {sorted(names)}"
    assert any(name.startswith("search:") for name in names)

    # --- one trace, one root, named after the question ------------------
    assert len({span["trace_id"] for span in spans}) == 1

    roots = [span for span in spans if span["parent_id"] not in ids]
    assert len(roots) == 1, f"expected one root, got {[r['name'] for r in roots]}"
    assert roots[0]["name"] == question
    root_id = roots[0]["span_id"]

    # --- top-level stages hang off the run root -------------------------
    for stage in ("plan", "research.round0", "verify", "synthesize", "critique"):
        matching = [span for span in spans if span["name"] == stage]
        assert matching
        assert all(span["parent_id"] == root_id for span in matching), (
            f"{stage} is not parented to the run root"
        )

    # --- fan-out nests under the round stage span (across threads) ------
    stage = next(span for span in spans if span["name"] == "research.round0")
    children = [span for span in spans if span["parent_id"] == stage["span_id"]]
    assert len(children) == 2, "one research span per sub-question"
    assert {child["name"] for child in children} == {"research"}

    child_ids = {child["span_id"] for child in children}
    grandchildren = [span for span in spans if span["parent_id"] in child_ids]
    assert len(grandchildren) == 6  # 2 sub-questions x (queries + search + extract)
    assert {span["name"] for span in grandchildren} == {
        "research.queries",
        "research.extract",
        "search:alpha systems efficiency",
    }

    # --- LLM spans are generations with usage; searches are tools -------
    generations = [span for span in spans if "generation" in _type_of(span)]
    assert generations, "no span was exported as a generation"
    assert all(
        "langfuse.observation.usage_details" in span["attrs"] for span in generations
    )
    assert any("langfuse.observation.model.name" in span["attrs"] for span in generations)

    tools = [span for span in spans if "tool" in _type_of(span)]
    assert tools, "search spans should be exported as tool observations"

    # Our own span table remains the authoritative record.
    assert len(repo.list_spans(run.id)) == len(spans) - 1  # minus the run root


def test_run_failure_still_closes_the_root(settings, repo, llm, search):
    """A failing run must not leave an open root span behind."""
    from deep_research.agent import ResearchAgent

    exporter = LangfuseExporter(
        public_key="pk-lf-test", secret_key="sk-lf-test", host="http://127.0.0.1:1"
    )

    def boom(**_kwargs):
        raise RuntimeError("provider exploded")

    llm.complete = boom  # type: ignore[method-assign]

    agent = ResearchAgent(
        settings=settings, repo=repo, llm=llm, search=search, coordinator=None, exporter=exporter
    )
    run = agent.create_run("Compare alpha and beta systems.")

    with pytest.raises(RuntimeError):
        agent.execute(run.id)

    assert not exporter._roots, "root observation was left open after a failure"


def test_export_failure_does_not_break_the_run(settings, repo, llm, search):
    """Point the exporter at a dead host: the run must still succeed."""
    exporter = LangfuseExporter(
        public_key="pk-lf-test", secret_key="sk-lf-test", host="http://127.0.0.1:1"
    )
    engine = ResearchEngine(
        settings=settings, repo=repo, llm=llm, search=search, exporter=exporter
    )
    run = repo.create_run(Run(id="lf-run-3", question="Compare alpha and beta systems."))

    report = engine.execute(run)

    assert report.sections
    assert repo.list_spans(run.id)


def test_auth_check_reports_rejected_credentials_without_raising():
    exporter = LangfuseExporter(
        public_key="pk-lf-test", secret_key="sk-lf-test", host="http://127.0.0.1:1"
    )

    def explode():
        raise RuntimeError("401 Unauthorized")

    exporter._client.auth_check = explode  # type: ignore[method-assign]
    assert exporter.auth_check() is False


def test_auth_check_reports_missing_sdk(monkeypatch):
    """Keys set without the SDK installed must warn, not crash startup."""
    import builtins

    from deep_research import tracing

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "langfuse":
            raise ImportError("no langfuse")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from deep_research.config import Settings

    settings = Settings(langfuse_public_key="pk", langfuse_secret_key="sk")
    assert tracing.build_exporter(settings) is None
