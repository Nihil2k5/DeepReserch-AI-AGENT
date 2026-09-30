"""Verifier: the adversarial step.

The synthesizer is asked to write from evidence; the verifier's job is to try to
knock that evidence down first. It checks each claim against its own quote (does
the quote actually state this?), flags the ones that fail, and names the gaps.

Unsupported evidence is dropped before synthesis, so a hallucinated claim can't
reach the report even if the extractor produced it. This is the node that turns
a summarizer into a research system.
"""

from __future__ import annotations

from ...logging_config import get_logger
from ...models import Evidence, GapAnalysis, SubQuestion
from ...structured import extract_json
from ..context import NodeContext
from .base import Node

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are a strict fact-checker auditing a research agent's evidence.

For each evidence item you get a `claim` and the `quote` it was extracted from.

Rules:
- Mark an item supported ONLY if the quote itself states the claim.
  Paraphrase is fine; a different or stronger claim is not.
- If the quote is off-topic, too vague, or does not support the claim, mark it unsupported.
- Be conservative: when in doubt, mark unsupported.
- Then list the real gaps: what a good answer to the sub-question still needs.
- Suggest 1-3 follow-up search queries that would close those gaps.
  Leave the list empty if the evidence is already sufficient.

Respond with JSON only:
{
  "supported": ["e1", "e2"],
  "unsupported": ["e3"],
  "gaps": ["..."],
  "follow_up_queries": ["..."]
}"""


class VerifierNode(Node):
    name = "verify"

    def __init__(
        self,
        sub_question: SubQuestion,
        evidence: list[Evidence],
        round_index: int = 0,
        model: str | None = None,
    ) -> None:
        self.sub_question = sub_question
        self.evidence = list(evidence)
        self.round_index = round_index
        self.model = model

    def run(self, ctx: NodeContext) -> dict:
        if not self.evidence:
            analysis = GapAnalysis(
                sub_question_id=self.sub_question.id,
                gaps=["No evidence was found for this sub-question."],
                follow_up_queries=[],
            )
            ctx.emit("verify.no_evidence", f"No evidence for {self.sub_question.id}")
            return analysis.model_dump()

        items = "\n\n".join(
            f"id: {item.id}\nclaim: {item.claim}\nquote: {item.quote}\nsource: {item.url}"
            for item in self.evidence
        )
        user = (
            f"Sub-question: {self.sub_question.question}\n\n"
            f"Evidence to audit:\n{items}"
        )

        response = ctx.call_llm(
            name="verify.check",
            system=SYSTEM_PROMPT,
            user=user,
            model=self.model,
            json_mode=True,
            max_tokens=1200,
        )

        payload = extract_json(response.text, default={}) or {}
        if not isinstance(payload, dict):
            payload = {}

        known_ids = {item.id for item in self.evidence}
        supported = self._ids(payload.get("supported"), known_ids)
        unsupported = self._ids(payload.get("unsupported"), known_ids)

        # Never lose evidence to a malformed verifier response: anything the
        # verifier didn't explicitly reject stays supported.
        if not supported and not unsupported:
            logger.warning("Verifier returned no verdicts; treating all evidence as supported")
            supported = set(known_ids)
        elif not supported:
            supported = known_ids - unsupported

        gaps = [str(g).strip() for g in payload.get("gaps") or [] if str(g).strip()]
        follow_ups = [
            str(q).strip() for q in payload.get("follow_up_queries") or [] if str(q).strip()
        ][:3]

        analysis = GapAnalysis(
            sub_question_id=self.sub_question.id,
            supported_ids=sorted(supported),
            unsupported_ids=sorted(unsupported),
            gaps=gaps,
            follow_up_queries=follow_ups,
        )

        ctx.emit(
            "verify.done",
            f"{self.sub_question.id}: {len(supported)} supported, "
            f"{len(unsupported)} rejected, {len(gaps)} gaps",
            analysis.model_dump(),
        )
        return analysis.model_dump()

    @staticmethod
    def _ids(raw: object, known: set[str]) -> set[str]:
        if not isinstance(raw, list):
            return set()
        return {str(value) for value in raw if str(value) in known}
