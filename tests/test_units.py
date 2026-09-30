from __future__ import annotations

import pytest

from deep_research.budget import Budget
from deep_research.config import Settings
from deep_research.errors import BudgetExceeded
from deep_research.graph.state import (
    PLAN_KEY,
    SYNTHESIZE_KEY,
    ResearchState,
    research_key,
    verify_key,
)
from deep_research.models import Critique, Evidence, Report, ReportSection
from deep_research.reporting import build_source_index, render_markdown, sanitize_citations
from deep_research.structured import extract_json


# --- budget ----------------------------------------------------------------
def test_budget_rejects_before_the_breaching_call():
    settings = Settings(max_llm_calls=2, max_searches=1, max_cost_usd=0.10)
    budget = Budget(settings)

    budget.check(llm=True)  # allowed
    budget.record_llm(tokens_in=10, tokens_out=10, cost_usd=0.01)
    budget.check(llm=True)  # still allowed (1 < 2)
    budget.record_llm(tokens_in=10, tokens_out=10, cost_usd=0.01)

    with pytest.raises(BudgetExceeded, match="LLM call budget"):
        budget.check(llm=True)


def test_budget_enforces_cost_limit():
    settings = Settings(max_llm_calls=100, max_cost_usd=0.01)
    budget = Budget(settings)
    budget.record_llm(tokens_in=1, tokens_out=1, cost_usd=0.02)
    with pytest.raises(BudgetExceeded, match="Cost budget"):
        budget.check(llm=True)


def test_budget_enforces_wall_clock():
    now = {"value": 0.0}
    settings = Settings(max_wall_clock_s=10)
    budget = Budget(settings, clock=lambda: now["value"])
    budget.check(llm=True)

    now["value"] = 11.0
    with pytest.raises(BudgetExceeded, match="Wall-clock"):
        budget.check(llm=True)


def test_reserve_for_finalization_grants_headroom():
    settings = Settings(max_llm_calls=1, max_cost_usd=0.001)
    budget = Budget(settings)
    budget.record_llm(tokens_in=1, tokens_out=1, cost_usd=0.002)
    with pytest.raises(BudgetExceeded):
        budget.check(llm=True)

    budget.reserve_for_finalization(llm_calls=3, cost_usd=1.0)
    budget.check(llm=True)  # must no longer raise


def test_budget_snapshot_tracks_spend():
    settings = Settings()
    budget = Budget(settings)
    budget.record_llm(tokens_in=100, tokens_out=50, cost_usd=0.25)
    budget.record_search(2)

    snapshot = budget.snapshot()
    assert snapshot.llm_calls == 1
    assert snapshot.searches == 2
    assert snapshot.tokens_in == 100
    assert snapshot.cost_usd == 0.25
    assert budget.remaining()["searches"] == settings.max_searches - 2


# --- structured parsing ----------------------------------------------------
def test_extract_json_handles_plain_object():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_handles_fenced_block():
    assert extract_json('Here you go:\n```json\n{"a": 2}\n```\n') == {"a": 2}


def test_extract_json_handles_prose_around_object():
    assert extract_json('Sure! {"a": {"b": 3}} hope that helps') == {"a": {"b": 3}}


def test_extract_json_handles_braces_inside_strings():
    assert extract_json('{"text": "a } brace"}') == {"text": "a } brace"}


def test_extract_json_returns_default_on_garbage():
    assert extract_json("not json at all", default=None) is None
    assert extract_json("", default={}) == {}


# --- reporting -------------------------------------------------------------
def _evidence() -> list[Evidence]:
    return [
        Evidence(
            id="e1",
            sub_question_id="q1",
            claim="Alpha is efficient.",
            quote="Alpha is efficient.",
            url="https://example.org/alpha",
            title="Alpha",
        ),
        Evidence(
            id="e2",
            sub_question_id="q2",
            claim="Beta is durable.",
            quote="Beta is durable.",
            url="https://example.com/beta",
            title="Beta",
        ),
        Evidence(
            id="e3",
            sub_question_id="q2",
            claim="Alpha again.",
            quote="Alpha again.",
            url="https://example.org/alpha",
            title="Alpha",
        ),
    ]


def test_build_source_index_dedupes_and_numbers_in_order():
    citations = build_source_index(_evidence())
    assert [c.index for c in citations] == [1, 2]
    assert citations[0].url == "https://example.org/alpha"
    assert citations[1].url == "https://example.com/beta"


def test_sanitize_strips_unknown_markers():
    text, used = sanitize_citations("Alpha is fast [1] but zeta is faster [99].", valid={1})
    assert "[1]" in text
    assert "[99]" not in text
    assert used == {1}


def test_sanitize_tidies_spacing_left_by_removed_markers():
    text, _ = sanitize_citations("Value is 42 [7] .", valid={1})
    assert "  " not in text
    assert " ." not in text


def test_render_markdown_contains_every_section():
    report = Report(
        question="Does alpha beat beta?",
        executive_summary="Alpha wins [1].",
        sub_questions=["Which is faster?"],
        sections=[ReportSection(heading="Speed", body="Alpha is faster [1].")],
        citations=build_source_index(_evidence())[:1],
        limitations=["No cost data."],
        critique=Critique(coverage_score=0.6, verdict="Partial."),
    )
    markdown = render_markdown(report)

    assert markdown.startswith("# Does alpha beat beta?")
    assert "## Summary" in markdown
    assert "## Research plan" in markdown
    assert "## Speed" in markdown
    assert "## Limitations" in markdown
    assert "Coverage score: 0.60" in markdown
    assert "1. [Alpha](https://example.org/alpha)" in markdown


# --- state -----------------------------------------------------------------
def test_state_plan_and_evidence_from_checkpoints():
    checkpoints = {
        PLAN_KEY: {"sub_questions": [{"id": "q1", "question": "Q1?"}]},
        research_key("q1", 0): {"evidence": [_evidence()[0].model_dump()]},
        verify_key("q1", 0): {"sub_question_id": "q1", "unsupported_ids": [], "gaps": []},
    }
    state = ResearchState("Question?", None, checkpoints)

    assert state.has_plan
    assert [sq.id for sq in state.plan] == ["q1"]
    assert len(state.evidence("q1", 0)) == 1
    assert state.rounds_completed("q1") == 1


def test_supported_evidence_excludes_rejected_ids():
    checkpoints = {
        research_key("q1", 0): {
            "evidence": [item.model_dump() for item in _evidence()[:2]]
        },
        verify_key("q1", 0): {"sub_question_id": "q1", "unsupported_ids": ["e1"]},
    }
    state = ResearchState("Q?", None, checkpoints)

    assert state.unsupported_ids() == {"e1"}
    assert [item.id for item in state.supported_evidence()] == ["e2"]


def test_all_evidence_is_sorted_across_rounds():
    checkpoints = {
        research_key("q2", 1): {"evidence": [_evidence()[1].model_dump()]},
        research_key("q1", 0): {"evidence": [_evidence()[0].model_dump()]},
    }
    state = ResearchState("Q?", None, checkpoints)
    assert [item.id for item in state.all_evidence()] == ["e1", "e2"]


def test_follow_up_queries_come_from_previous_round():
    checkpoints = {
        verify_key("q1", 0): {
            "sub_question_id": "q1",
            "follow_up_queries": ["alpha cost"],
        },
    }
    state = ResearchState("Q?", None, checkpoints)
    assert state.follow_up_queries("q1", 1) == ["alpha cost"]
    assert state.follow_up_queries("q1", 0) == []


def test_report_property_reads_synthesize_checkpoint():
    report = Report(question="Q?", executive_summary="A.")
    state = ResearchState("Q?", None, {SYNTHESIZE_KEY: report.model_dump()})
    assert state.has_report
    assert state.report is not None
    assert state.report.executive_summary == "A."


# --- config ----------------------------------------------------------------
def test_validate_runtime_reports_every_missing_value():
    settings = Settings(
        llm_provider="openai",
        search_provider="exa",
        llm_api_key=None,
        exa_api_key=None,
        database_url="",
        redis_url="",
    )
    with pytest.raises(ValueError) as excinfo:
        settings.validate_runtime()

    message = str(excinfo.value)
    for expected in ("DR_LLM_API_KEY", "DR_EXA_API_KEY", "DR_DATABASE_URL", "DR_REDIS_URL"):
        assert expected in message


def test_validate_runtime_skipped_for_fake_providers():
    Settings(
        llm_provider="fake",
        search_provider="fake",
        llm_api_key=None,
        exa_api_key=None,
    ).validate_runtime()


def test_validate_runtime_rejects_bad_round_settings():
    settings = Settings(
        llm_provider="fake", search_provider="fake", max_research_rounds=0
    )
    with pytest.raises(ValueError, match="MAX_RESEARCH_ROUNDS"):
        settings.validate_runtime()


def test_validate_runtime_rejects_zero_subquestions():
    settings = Settings(llm_provider="fake", search_provider="fake", max_subquestions=0)
    with pytest.raises(ValueError, match="MAX_SUBQUESTIONS"):
        settings.validate_runtime()


def test_langfuse_enabled_only_with_both_keys():
    # `_env_file=None` keeps this independent of a developer's local .env.
    base = Settings(_env_file=None, langfuse_public_key=None, langfuse_secret_key=None)
    assert not base.langfuse_enabled
    assert not base.model_copy(update={"langfuse_public_key": "pk"}).langfuse_enabled
    assert base.model_copy(
        update={"langfuse_public_key": "pk", "langfuse_secret_key": "sk"}
    ).langfuse_enabled
