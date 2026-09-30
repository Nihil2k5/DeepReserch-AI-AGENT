"""Synthesizer: turn verified evidence into a cited report.

Only evidence that survived the verifier is passed in. Sources are numbered here
and the numbers handed to the model, then any marker referencing a number we did
not issue is stripped afterwards (:mod:`deep_research.reporting`), so the report
cannot cite a source that doesn't exist.
"""

from __future__ import annotations

import json

from ...logging_config import get_logger
from ...models import Evidence, Report, ReportSection
from ...reporting import build_source_index, sanitize_citations
from ...structured import extract_json
from ..context import NodeContext
from .base import Node

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a research analyst writing a cited report from verified evidence.

Rules:
- Use ONLY the evidence provided. Do not add outside knowledge, ever.
- Cite factual statements with inline source numbers like [1], using ONLY the
  numbers from the provided source list.
- Structure the report as:
  * an executive_summary of 2-4 sentences that answers the question directly
  * 3-5 thematic sections with short, specific headings
- Where evidence is thin, conflicting or missing, say so explicitly in that section.
- limitations must list the real gaps: what the evidence could not establish.

Respond with JSON only:
{"executive_summary": "...", "sections": [{"heading": "...", "body": "..."}],
 "limitations": ["..."]}"""

#: Evidence items and quote length handed to the model.
MAX_EVIDENCE_ITEMS = 40
MAX_QUOTE_CHARS = 300


class SynthesizerNode(Node):
    name = "synthesize"

    def __init__(
        self,
        question: str,
        evidence: list[Evidence],
        focus: str | None = None,
        sub_questions: list[str] | None = None,
        extra_limitations: list[str] | None = None,
        model: str | None = None,
    ) -> None:
        self.question = question
        self.evidence = list(evidence)
        self.focus = focus
        self.sub_questions = list(sub_questions or [])
        self.extra_limitations = list(extra_limitations or [])
        self.model = model

    def run(self, ctx: NodeContext) -> dict:
        citations = build_source_index(self.evidence)
        number_by_url = {citation.url: citation.index for citation in citations}

        if self.evidence:
            grouped: dict[str, list[str]] = {}
            for item in self.evidence[:MAX_EVIDENCE_ITEMS]:
                number = number_by_url.get(item.url, 0)
                quote = item.quote[:MAX_QUOTE_CHARS]
                grouped.setdefault(item.sub_question_id, []).append(
                    f"- (source [{number}], confidence {item.confidence:.2f}) "
                    f"{item.claim} — quote: \"{quote}\""
                )

            evidence_block = "\n\n".join(
                f"### {group_id}\n" + "\n".join(lines) for group_id, lines in grouped.items()
            )
            sources_block = "\n".join(
                f"[{c.index}] {c.title or c.url} — {c.url}" for c in citations
            )
        else:
            evidence_block = "(no evidence survived verification)"
            sources_block = "(no sources)"

        focus_line = f"\nAnalyst focus: {self.focus}" if self.focus else ""
        user = (
            f"Research question:\n{self.question}{focus_line}\n\n"
            f"Source list (cite these numbers only):\n{sources_block}\n\n"
            f"Verified evidence:\n{evidence_block}"
        )

        def draft() -> dict:
            response = ctx.call_llm(
                name="synthesize.report",
                system=SYSTEM_PROMPT,
                user=user,
                model=self.model,
                json_mode=True,
                max_tokens=3000,
            )
            payload = extract_json(response.text, default={}) or {}
            return payload if isinstance(payload, dict) else {}

        def parse(payload: dict) -> tuple[str, list[ReportSection], set[int]]:
            valid_numbers = {citation.index for citation in citations}

            summary, summary_used = sanitize_citations(
                str(payload.get("executive_summary") or "").strip(), valid_numbers
            )

            sections: list[ReportSection] = []
            used_numbers = set(summary_used)
            raw_sections = payload.get("sections")
            if isinstance(raw_sections, list):
                for item in raw_sections:
                    if not isinstance(item, dict):
                        continue
                    heading = str(item.get("heading") or "").strip()
                    body, body_used = sanitize_citations(
                        str(item.get("body") or "").strip(), valid_numbers
                    )
                    if not heading or not body:
                        continue
                    used_numbers |= body_used
                    sections.append(ReportSection(heading=heading, body=body))
            return summary, sections, used_numbers

        payload = draft()
        summary, sections, used_numbers = parse(payload)

        # Quality gate: a grounded report with a source list but zero inline
        # citations is a small-model failure mode (the claim text still comes
        # from evidence). One repair round that shows the model its own draft
        # and the rule it broke fixes it cheaply.
        if citations and not used_numbers:
            logger.warning("Synthesizer cited no sources; requesting one repair pass")
            repair_user = (
                f"Your draft violated the citation rule. Here is your draft:\n\n"
                f"{json.dumps(payload, ensure_ascii=False)[:4000]}\n\n"
                "Rewrite it with the SAME content and structure, but add an inline "
                "source number like [1] or [2] to every factual statement, using "
                "ONLY numbers from the source list. Respond with JSON only, same shape."
            )
            repair = ctx.call_llm(
                name="synthesize.repair",
                system=SYSTEM_PROMPT,
                user=repair_user,
                model=self.model,
                json_mode=True,
                max_tokens=3000,
            )
            payload2 = extract_json(repair.text, default={}) or {}
            if isinstance(payload2, dict):
                summary2, sections2, used2 = parse(payload2)
                if used2:
                    summary, sections, used_numbers = summary2, sections2, used2

        limitations = [
            str(item).strip() for item in payload.get("limitations") or [] if str(item).strip()
        ]
        limitations.extend(self.extra_limitations)

        if not sections:
            logger.warning("Synthesizer produced no sections; emitting summary-only report")
            sections = [
                ReportSection(
                    heading="Findings",
                    body=summary or "The available evidence was insufficient to answer this question.",
                )
            ]

        report = Report(
            question=self.question,
            executive_summary=summary,
            sub_questions=self.sub_questions,
            sections=sections,
            citations=[c for c in citations if c.index in used_numbers],
            limitations=limitations,
        )

        ctx.emit(
            "synthesize.done",
            f"Report drafted: {len(sections)} sections, "
            f"{len(report.citations)} of {len(citations)} sources cited",
            {"sections": len(sections), "cited": len(report.citations)},
        )
        return report.model_dump()
