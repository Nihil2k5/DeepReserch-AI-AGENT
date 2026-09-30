"""Critic: score the finished report against the evidence.

A second, adversarial reading of the report — this time checking whether the
prose stayed inside the evidence. The score is what the evaluation harness
trends over time, and what makes "is this report good?" a measurable question
rather than a vibe.
"""

from __future__ import annotations

from ...models import Critique, Evidence, Report
from ...structured import extract_json
from ..context import NodeContext
from .base import Node

SYSTEM_PROMPT = """You review a research report against the evidence it was written from.

Rules:
- coverage_score: 0-1, how completely the report answers the original question
  using the available evidence. Judge coverage of the *question*, not prose quality.
- unsupported_claims: statements in the report that the evidence does not
  support, quoted briefly. Empty list if there are none.
- improvements: concrete, actionable next steps to strengthen the report.
- verdict: one sentence.

Respond with JSON only:
{"coverage_score": 0.75, "unsupported_claims": ["..."], "improvements": ["..."],
 "verdict": "..."}"""

MAX_QUOTE_CHARS = 260


class CriticNode(Node):
    name = "critique"

    def __init__(self, report: Report, evidence: list[Evidence], model: str | None = None) -> None:
        self.report = report
        self.evidence = list(evidence)
        self.model = model

    def run(self, ctx: NodeContext) -> dict:
        report_text = [self.report.executive_summary]
        for section in self.report.sections:
            report_text.append(f"## {section.heading}\n{section.body}")
        joined = "\n\n".join(part for part in report_text if part)

        evidence_block = "\n".join(
            f"- {item.claim} (source [x], quote: \"{item.quote[:MAX_QUOTE_CHARS]}\")"
            for item in self.evidence[:40]
        ) or "(no evidence survived verification)"

        user = (
            f"Original question:\n{self.report.question}\n\n"
            f"Report:\n{joined}\n\n"
            f"Evidence available:\n{evidence_block}"
        )

        response = ctx.call_llm(
            name="critique.review",
            system=SYSTEM_PROMPT,
            user=user,
            model=self.model,
            json_mode=True,
            max_tokens=1000,
        )

        payload = extract_json(response.text, default={}) or {}
        if not isinstance(payload, dict):
            payload = {}

        try:
            score = float(payload.get("coverage_score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0

        critique = Critique(
            coverage_score=min(1.0, max(0.0, score)),
            unsupported_claims=[
                str(item).strip()
                for item in payload.get("unsupported_claims") or []
                if str(item).strip()
            ],
            improvements=[
                str(item).strip()
                for item in payload.get("improvements") or []
                if str(item).strip()
            ],
            verdict=str(payload.get("verdict") or "").strip(),
        )

        ctx.emit(
            "critique.done",
            f"Coverage {critique.coverage_score:.2f}",
            critique.model_dump(),
        )
        return critique.model_dump()
