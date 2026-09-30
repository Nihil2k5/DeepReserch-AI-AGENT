"""Shared fixtures.

Every fixture builds a fully in-memory graph: fake LLM, fake search, in-memory
repository, no Redis, no Langfuse. That keeps the orchestration logic — the part
that is actually hard to get right — verifiable without any infrastructure.
"""

from __future__ import annotations

import pytest

from deep_research.agent import ResearchAgent
from deep_research.config import Settings
from deep_research.graph.engine import ResearchEngine
from deep_research.llm.fake import FakeLLM
from deep_research.models import Run, SearchHit
from deep_research.repository import InMemoryRunRepository
from deep_research.tools.fake_search import FakeSearchTool

# --- prompt markers used to route fake responses ---------------------------
PLANNER_MARKER = "research planner"
QUERIES_MARKER = "web search queries"
EXTRACT_MARKER = "extract verifiable evidence"
VERIFY_MARKER = "strict fact-checker"
SYNTHESIZE_MARKER = "research analyst writing a cited report"
CRITIQUE_MARKER = "review a research report"


@pytest.fixture
def settings() -> Settings:
    return Settings(
        llm_provider="fake",
        search_provider="fake",
        max_subquestions=2,
        max_research_rounds=1,
        max_llm_calls=50,
        max_searches=20,
        max_cost_usd=5.0,
        max_wall_clock_s=300,
        search_results=2,
        langfuse_public_key=None,
        langfuse_secret_key=None,
    )


@pytest.fixture
def repo() -> InMemoryRunRepository:
    return InMemoryRunRepository()


@pytest.fixture
def search() -> FakeSearchTool:
    return FakeSearchTool(
        hits=[
            SearchHit(
                url="https://example.org/alpha",
                title="Alpha Source",
                text="Alpha systems achieve 92 percent round-trip efficiency in trials.",
                score=0.9,
            ),
            SearchHit(
                url="https://example.com/beta",
                title="Beta Source",
                text="Beta deployments typically last about 50 years before refurbishment.",
                score=0.8,
            ),
        ]
    )


def scripted_llm(
    sub_questions: list[tuple[str, str]] | None = None,
    verify_response: dict | None = None,
    critique_score: float = 0.75,
) -> FakeLLM:
    """Build a FakeLLM with routes for every node in the graph."""
    sub_questions = sub_questions or [
        ("q1", "What is the efficiency of alpha systems?"),
        ("q2", "How long do beta deployments last?"),
    ]

    plan = {
        "sub_questions": [
            {"id": sub_id, "question": question, "rationale": "because"}
            for sub_id, question in sub_questions
        ]
    }
    verification = verify_response or {
        "supported": [],
        "unsupported": [],
        "gaps": [],
        "follow_up_queries": [],
    }

    llm = FakeLLM(default="{}")
    llm.route_json(PLANNER_MARKER, plan)
    llm.route_json(QUERIES_MARKER, {"queries": ["alpha systems efficiency"]})
    llm.route_json(
        EXTRACT_MARKER,
        {
            "evidence": [
                {
                    "claim": "Alpha systems reach 92 percent round-trip efficiency.",
                    "quote": "Alpha systems achieve 92 percent round-trip efficiency in trials.",
                    "source": 1,
                    "confidence": 0.9,
                },
                {
                    "claim": "Beta deployments last about 50 years.",
                    "quote": "Beta deployments typically last about 50 years before refurbishment.",
                    "source": 2,
                    "confidence": 0.8,
                },
            ]
        },
    )
    llm.route_json(VERIFY_MARKER, verification)
    llm.route_json(
        SYNTHESIZE_MARKER,
        {
            "executive_summary": "Alpha is efficient [1] while beta is long-lived [2].",
            "sections": [
                {
                    "heading": "Efficiency",
                    "body": "Alpha systems reach 92 percent round-trip efficiency [1].",
                },
                {
                    "heading": "Longevity",
                    "body": "Beta deployments last around 50 years [2].",
                },
            ],
            "limitations": ["No cost data was found."],
        },
    )
    llm.route_json(
        CRITIQUE_MARKER,
        {
            "coverage_score": critique_score,
            "unsupported_claims": [],
            "improvements": ["Look for cost benchmarks."],
            "verdict": "Reasonable coverage.",
        },
    )
    return llm


@pytest.fixture
def llm() -> FakeLLM:
    return scripted_llm()


@pytest.fixture
def engine(settings, repo, llm, search) -> ResearchEngine:
    return ResearchEngine(settings=settings, repo=repo, llm=llm, search=search)


@pytest.fixture
def run(repo) -> Run:
    created = repo.create_run(Run(id="run1", question="Compare alpha and beta systems."))
    return created


@pytest.fixture
def agent(settings, repo, llm, search) -> ResearchAgent:
    instance = ResearchAgent(
        settings=settings, repo=repo, llm=llm, search=search, coordinator=None, exporter=None
    )
    yield instance
    instance.close()
