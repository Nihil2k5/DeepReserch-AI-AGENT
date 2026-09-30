"""LLM judge for content-level grading.

Deterministic metrics can't tell whether a report *answered* the question, only
whether it was structured like an answer. The judge closes that gap: for each
required fact it asks whether the report establishes it, citing the sentence.

The judge is deliberately a different prompt and a cheaper model than the
synthesizer — grading your own homework with the same prompt is how eval numbers
become meaningless.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..llm.base import LLM
from ..logging_config import get_logger
from ..structured import extract_json

logger = get_logger(__name__)

SYSTEM_PROMPT = """You grade research reports.

You are given a question, the report, and a list of facts that a good report
must establish.

For each fact, decide whether the report establishes it. Be strict:
- Mentioning a related word is not establishing a fact.
- The report must state the fact (approximately is fine) somewhere in its body.
- If the report explicitly says the evidence was insufficient, that fact is NOT established.

Respond with JSON only:
{"established": [{"fact": "...", "evidence": "the sentence that establishes it"}],
 "missing": ["..."]}"""

MAX_REPORT_CHARS = 12000


class JudgeResult(BaseModel):
    established: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    parse_failed: bool = False

    @property
    def coverage(self) -> float:
        total = len(self.established) + len(self.missing)
        return 1.0 if total == 0 else len(self.established) / total


class FactJudge:
    def __init__(self, llm: LLM, model: str | None = None) -> None:
        self.llm = llm
        self.model = model

    def grade(self, question: str, report_text: str, facts: list[str]) -> JudgeResult:
        if not facts:
            return JudgeResult()

        fact_list = "\n".join(f"- {fact}" for fact in facts)
        user = (
            f"Question:\n{question}\n\n"
            f"Required facts:\n{fact_list}\n\n"
            f"Report:\n{report_text[:MAX_REPORT_CHARS]}"
        )

        response = self.llm.complete(
            system=SYSTEM_PROMPT,
            user=user,
            model=self.model,
            temperature=0.0,
            max_tokens=1200,
            json_mode=True,
        )

        payload = extract_json(response.text, default=None)
        if not isinstance(payload, dict):
            logger.warning("Judge returned unparseable output; scoring zero coverage")
            return JudgeResult(
                missing=list(facts),
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                cost_usd=response.cost_usd,
                parse_failed=True,
            )

        raw_established = payload.get("established") or []
        established: list[str] = []
        for item in raw_established:
            if isinstance(item, dict):
                fact = str(item.get("fact") or "").strip()
            else:
                fact = str(item).strip()
            if fact:
                established.append(fact)

        declared_missing = [
            str(item).strip() for item in payload.get("missing") or [] if str(item).strip()
        ]

        # Reconcile: anything the judge never mentions is counted as missing, so a
        # partial response can only ever understate coverage, never inflate it.
        mentioned = {fact.lower() for fact in established} | {
            fact.lower() for fact in declared_missing
        }
        for fact in facts:
            if fact.lower() not in mentioned:
                declared_missing.append(fact)

        return JudgeResult(
            established=established,
            missing=declared_missing,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            cost_usd=response.cost_usd,
        )
