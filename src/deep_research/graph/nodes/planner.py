"""Planner: decompose the question into independent sub-questions.

Decomposition is what lets the research stage run in parallel, and it is also
what makes the coverage of a report auditable: every sub-question is either
answered with evidence or explicitly reported as a gap.
"""

from __future__ import annotations

import re

from ...logging_config import get_logger
from ...models import SubQuestion
from ...structured import extract_json
from ..context import NodeContext
from .base import Node

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a research planner. You break a complex question into \
independent sub-questions that can each be researched separately on the web.

Rules:
- Produce between 3 and 6 sub-questions.
- Each must be self-contained: a reader seeing only that sub-question knows what to look up.
- Cover the question from complementary angles (definition/context, concrete data,
  comparison/tradeoffs, recent developments, counterpoints or risks).
- Prefer sub-questions that will surface verifiable facts, numbers and dates.
- Do not include meta questions like "what is the best way to research this".

Respond with JSON only, matching exactly this shape:
{"sub_questions": [{"id": "q1", "question": "...", "rationale": "..."}]}"""


class PlannerNode(Node):
    name = "plan"

    def __init__(self, max_sub_questions: int = 4, model: str | None = None) -> None:
        self.max_sub_questions = max_sub_questions
        self.model = model

    def run(self, ctx: NodeContext) -> dict:
        focus_line = f"\nAdditional focus: {ctx.state.focus}" if ctx.state.focus else ""
        user = (
            f"Research question:\n{ctx.state.question}{focus_line}\n\n"
            f"Produce at most {self.max_sub_questions} sub-questions."
        )

        response = ctx.call_llm(
            name="plan.decompose",
            system=SYSTEM_PROMPT,
            user=user,
            model=self.model,
            json_mode=True,
            max_tokens=1200,
        )

        payload = extract_json(response.text, default={}) or {}
        raw_items = payload.get("sub_questions") if isinstance(payload, dict) else None
        if not isinstance(raw_items, list):
            logger.warning("Planner returned no usable sub_questions; using fallback")
            raw_items = []

        sub_questions: list[SubQuestion] = []
        for index, item in enumerate(raw_items[: self.max_sub_questions], start=1):
            if isinstance(item, str):
                item = {"question": item}
            if not isinstance(item, dict):
                continue
            question = str(item.get("question") or "").strip()
            if not question:
                continue
            sub_questions.append(
                SubQuestion(
                    id=self._normalise_id(item.get("id"), index),
                    question=question,
                    rationale=str(item.get("rationale") or "").strip(),
                )
            )

        if not sub_questions:
            # Degrade gracefully rather than failing the whole run.
            sub_questions = [SubQuestion(id="q1", question=ctx.state.question)]

        ctx.emit(
            "plan.ready",
            f"Planned {len(sub_questions)} sub-questions",
            {"sub_questions": [sq.model_dump() for sq in sub_questions]},
        )
        return {"sub_questions": [sq.model_dump() for sq in sub_questions]}

    @staticmethod
    def _normalise_id(raw: object, index: int) -> str:
        value = str(raw or "").strip().lower()
        value = re.sub(r"[^a-z0-9_-]+", "-", value).strip("-")
        return value or f"q{index}"
