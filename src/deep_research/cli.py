"""``research`` — queue and/or execute a research question from the terminal."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .agent import ResearchAgent
from .config import get_settings
from .errors import ResearchError
from .logging_config import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="research",
        description="Run a deep research question and print the cited report.",
    )
    parser.add_argument("question", help="The research question")
    parser.add_argument("--focus", default=None, help="Optional steer for the researcher")
    parser.add_argument("--out", default=None, help="Write the markdown report to this file")
    parser.add_argument(
        "--queue",
        action="store_true",
        help="Only enqueue the run and print its id (let a worker execute it)",
    )
    parser.add_argument("--json", action="store_true", help="Print the raw report JSON")
    parser.add_argument("--verbose", action="store_true", help="Show debug logs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    configure_logging("DEBUG" if args.verbose else settings.log_level)

    try:
        settings.validate_runtime()
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    agent = ResearchAgent.from_settings(settings)
    try:
        run = agent.create_run(args.question, args.focus)
        print(f"run {run.id} queued")

        if args.queue:
            print("waiting for a worker to pick it up (research-worker)")
            return 0

        report = agent.execute(run.id)

        if args.json:
            print(report.model_dump_json(indent=2))
        else:
            print("\n" + (report.markdown or ""))

        if args.out:
            path = Path(args.out)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(report.markdown or "", encoding="utf-8")
            print(f"\nReport written to {path}")

        final = agent.get_run(run.id)
        print(
            f"\n{len(report.sections)} sections · {len(report.citations)} sources · "
            f"{report.critique.coverage_score:.2f} coverage · "
            f"${final.metrics.cost_usd:.4f} · {final.metrics.duration_ms / 1000:.1f}s · "
            f"{final.metrics.llm_calls} LLM calls · {final.metrics.searches} searches"
        )
        return 0
    except ResearchError as exc:
        print(f"{exc.__class__.__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        agent.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
