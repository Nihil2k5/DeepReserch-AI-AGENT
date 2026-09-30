from __future__ import annotations

import json

import pytest

from deep_research.evaluation.dataset import load_dataset
from deep_research.evaluation.judge import FactJudge
from deep_research.evaluation.metrics import (
    aggregate,
    citation_density,
    citation_validity,
    domain,
    quality_score,
    source_diversity,
)
from deep_research.llm.fake import FakeLLM
from deep_research.models import Citation, Report, ReportSection


def _report(body: str = "Alpha is fast [1]. Beta is slow [2].") -> Report:
    return Report(
        question="Which is faster?",
        executive_summary="Alpha is faster [1].",
        sections=[ReportSection(heading="Speed", body=body)],
        citations=[
            Citation(index=1, url="https://www.example.org/alpha", title="Alpha"),
            Citation(index=2, url="https://news.example.com/beta", title="Beta"),
        ],
        limitations=["No cost data."],
    )


# --- metrics ---------------------------------------------------------------
def test_citation_validity_is_one_when_all_markers_resolve():
    assert citation_validity(_report()) == 1.0


def test_citation_validity_flags_dangling_markers():
    report = _report(body="Alpha is fast [1]. Something odd [9].")
    assert citation_validity(report) == pytest.approx(2 / 3)


def test_citation_validity_without_markers_counts_as_perfect():
    report = _report(body="No markers here at all.")
    assert citation_validity(report) == 1.0


def test_citation_density_is_sentence_level():
    report = _report(body="Alpha is fast [1]. Beta is slow [2]. An unsourced aside.")
    assert citation_density(report) == pytest.approx(3 / 4)


def test_domain_strips_www_and_handles_junk():
    assert domain("https://www.example.org/a") == "example.org"
    assert domain("https://news.example.com/b") == "news.example.com"
    assert domain("not a url") == ""


def test_source_diversity_counts_distinct_domains():
    assert source_diversity(_report()) == 2


def test_quality_score_is_bounded_and_penalises_missing_limitations():
    good = quality_score(_report())
    assert 0.0 <= good <= 1.0

    no_limitations = _report().model_copy(update={"limitations": []})
    assert quality_score(no_limitations) < good


def test_aggregate_averages_numeric_fields_only():
    records = [
        {"fact_coverage": 1.0, "cost_usd": 0.10, "note": "x"},
        {"fact_coverage": 0.0, "cost_usd": 0.30, "note": "y"},
    ]
    summary = aggregate(records)
    assert summary["fact_coverage"] == 0.5
    assert summary["cost_usd"] == pytest.approx(0.2)
    assert "note" not in summary


# --- judge -----------------------------------------------------------------
def test_judge_scores_full_coverage():
    llm = FakeLLM(
        routes={
            "You grade research reports": json.dumps(
                {
                    "established": [{"fact": "Alpha is efficient", "evidence": "..."}],
                    "missing": [],
                }
            )
        }
    )
    result = FactJudge(llm).grade("Q?", "report", ["Alpha is efficient"])

    assert result.coverage == 1.0
    assert result.missing == []
    assert not result.parse_failed


def test_judge_counts_unmentioned_facts_as_missing():
    llm = FakeLLM(
        routes={
            "You grade research reports": json.dumps(
                {"established": [{"fact": "Alpha is efficient"}], "missing": []}
            )
        }
    )
    result = FactJudge(llm).grade("Q?", "report", ["Alpha is efficient", "Beta is durable"])

    assert result.coverage == 0.5
    assert "Beta is durable" in result.missing


def test_judge_handles_unparseable_output_without_inflating_coverage():
    llm = FakeLLM(routes={"You grade research reports": "I cannot do that."})
    result = FactJudge(llm).grade("Q?", "report", ["A fact"])

    assert result.parse_failed
    assert result.coverage == 0.0
    assert result.missing == ["A fact"]


def test_judge_with_no_facts_is_a_noop():
    result = FactJudge(FakeLLM()).grade("Q?", "report", [])
    assert result.coverage == 1.0


# --- dataset ---------------------------------------------------------------
def test_load_dataset_assigns_ids_and_skips_comments(tmp_path):
    path = tmp_path / "ds.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"question": "Q1", "must_include": ["a"]}',
                "# comment",
                '{"question": "Q2"}',
                "",
            ]
        ),
        encoding="utf-8",
    )
    cases = load_dataset(path)
    assert [case.id for case in cases] == ["c1", "c2"]
    assert cases[0].must_include == ["a"]


def test_load_dataset_rejects_invalid_json(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_dataset(path)
