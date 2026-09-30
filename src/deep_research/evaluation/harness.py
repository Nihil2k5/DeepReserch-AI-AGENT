"""Run the dataset through the *real* agent and score the reports.

Two layers of scoring, deliberately separated:

* **Deterministic** (:mod:`.metrics`) — citation validity, density, source
  diversity. Always computable, cheap, gates CI.
* **Judged** (:mod:`.judge`) — whether the report actually establishes the
  required facts. Costs a model call per case and is the number that means
  something to a human.

Cost and latency are recorded per case so a quality gain can be weighed against
what it cost to get.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from ..agent import ResearchAgent
from ..logging_config import get_logger
from .dataset import EvalCase
from .judge import FactJudge
from .metrics import (
    citation_density,
    citation_validity,
    quality_score,
    source_diversity,
)

logger = get_logger(__name__)


class EvalHarness:
    def __init__(self, agent: ResearchAgent, judge: FactJudge | None = None) -> None:
        self.agent = agent
        self.judge = judge

    # -- one case --------------------------------------------------------
    def evaluate_case(self, case: EvalCase) -> dict:
        record: dict = {"id": case.id, "question": case.question}

        run = self.agent.create_run(case.question)
        record["run_id"] = run.id

        try:
            report = self.agent.execute(run.id)
        except Exception as exc:
            logger.warning("Case %s failed: %s", case.id, exc)
            record.update({"failed": True, "error": str(exc)})
            return record

        final = self.agent.get_run(run.id)
        record["status"] = final.status.value
        record["markdown"] = report.markdown
        record["sections"] = len(report.sections)
        record["sources"] = len(report.citations)
        record["citation_urls"] = [citation.url for citation in report.citations]
        record["coverage_score"] = (
            report.critique.coverage_score if report.critique else None
        )
        record["unsupported_claims"] = (
            len(report.critique.unsupported_claims) if report.critique else None
        )
        record["limitations"] = len(report.limitations)

        # deterministic metrics
        record["citation_validity"] = round(citation_validity(report), 4)
        record["citation_density"] = round(citation_density(report), 4)
        record["source_diversity"] = source_diversity(report)
        record["quality_score"] = quality_score(report)

        # cost / latency
        record["cost_usd"] = final.metrics.cost_usd
        record["duration_ms"] = final.metrics.duration_ms
        record["llm_calls"] = final.metrics.llm_calls
        record["searches"] = final.metrics.searches

        # judged content coverage
        if self.judge is not None and case.must_include:
            result = self.judge.grade(
                case.question, report.markdown or "", case.must_include
            )
            record["fact_coverage"] = round(result.coverage, 4)
            record["facts_missing"] = len(result.missing)
            record["judge_cost_usd"] = result.cost_usd
            record["missing_facts"] = result.missing
            record["judge_parse_failed"] = result.parse_failed

        logger.info(
            "Case %s: quality=%.2f coverage=%s cost=$%.4f",
            case.id,
            record["quality_score"],
            record.get("fact_coverage"),
            record["cost_usd"],
        )
        return record

    # -- whole dataset ---------------------------------------------------
    def run(self, cases: list[EvalCase]) -> dict:
        records = [self.evaluate_case(case) for case in cases]

        successful = [r for r in records if not r.get("failed")]
        numeric_keys = [
            "citation_validity",
            "citation_density",
            "source_diversity",
            "quality_score",
            "coverage_score",
            "fact_coverage",
            "cost_usd",
            "duration_ms",
            "llm_calls",
            "searches",
            "sections",
            "sources",
        ]
        summary: dict[str, float] = {}
        for key in numeric_keys:
            values = [
                r[key] for r in successful if isinstance(r.get(key), (int, float))
            ]
            if values:
                summary[key] = round(mean(values), 6)
        summary["cases"] = len(records)
        summary["failures"] = len(records) - len(successful)

        return {
            "summary": summary,
            "config": {
                "llm": self.agent.llm.name,
                "search": self.agent.search.name,
                "judge_enabled": self.judge is not None,
                "max_subquestions": self.agent.settings.max_subquestions,
                "max_research_rounds": self.agent.settings.max_research_rounds,
            },
            "results": records,
        }


def write_report(report: dict, out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return path


def render_markdown(report: dict) -> str:
    summary = report["summary"]
    lines = [
        "# Deep Research Evaluation",
        "",
        f"**Cases:** {summary.get('cases', 0)}  ",
        f"**Failures:** {summary.get('failures', 0)}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    for key, value in sorted(summary.items()):
        lines.append(f"| {key} | {value} |")

    lines += [
        "",
        "## Per case",
        "",
        "| Case | Quality | Fact coverage | Sources | Cost | Duration |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for record in report["results"]:
        if record.get("failed"):
            lines.append(f"| {record['id']} | FAILED | - | - | - | - |")
            continue
        coverage = record.get("fact_coverage")
        lines.append(
            f"| {record['id']} | {record['quality_score']:.2f} | "
            f"{'-' if coverage is None else f'{coverage:.2f}'} | "
            f"{record['sources']} | ${record['cost_usd']:.4f} | "
            f"{record['duration_ms'] / 1000:.1f}s |"
        )

    lines += ["", "## Configuration", ""]
    lines += [f"- **{key}**: {value}" for key, value in report["config"].items()]
    return "\n".join(lines) + "\n"
