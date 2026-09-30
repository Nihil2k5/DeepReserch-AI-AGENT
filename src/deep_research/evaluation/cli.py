"""``research-eval`` — run the dataset against the live agent and write a report.

This incurs real spend: every case runs a full research pipeline (LLM + search
calls). Use ``--limit`` while iterating.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..agent import ResearchAgent
from ..config import get_settings
from ..llm.factory import build_llm
from ..logging_config import configure_logging
from .dataset import load_dataset
from .harness import EvalHarness, render_markdown, write_report
from .judge import FactJudge


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="research-eval",
        description="Evaluate deep-research report quality over a labelled dataset.",
    )
    parser.add_argument("--dataset", default="eval/dataset.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N cases")
    parser.add_argument("--out", default="eval/reports/report.json")
    parser.add_argument("--markdown", default="eval/reports/report.md")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM fact grading")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    configure_logging("WARNING" if args.quiet else settings.log_level)

    try:
        settings.validate_runtime()
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    agent = ResearchAgent.from_settings(settings)
    try:
        cases = load_dataset(args.dataset)
        if args.limit:
            cases = cases[: args.limit]

        judge = None
        if not args.no_judge:
            # Judge with the cheaper model: it's grading, not writing.
            judge = FactJudge(build_llm(settings), model=settings.llm_model)

        print(
            f"Evaluating {len(cases)} case(s); this runs real research "
            f"(LLM + search) and costs real money."
        )
        report = EvalHarness(agent, judge=judge).run(cases)

        write_report(report, args.out)
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown).write_text(render_markdown(report), encoding="utf-8")

        print("\nSummary")
        print("-" * 44)
        for key, value in sorted(report["summary"].items()):
            print(f"  {key:<20} {value}")
        print(f"\nReports written to {args.out} and {args.markdown}")
        return 0
    finally:
        agent.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
