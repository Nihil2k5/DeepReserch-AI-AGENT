"""Research state derived from node checkpoints.

The durable store *is* the state: there is no separate in-memory graph object to
keep in sync. A resumed run reconstructs itself purely by reading checkpoints,
which is what makes crash recovery correct rather than best-effort.
"""

from __future__ import annotations

from ..models import Critique, Evidence, GapAnalysis, Report, SubQuestion

PLAN_KEY = "plan"
SYNTHESIZE_KEY = "synthesize"
CRITIQUE_KEY = "critique"

#: Upper bound used when scanning checkpoint keys for research rounds.
_MAX_ROUNDS_SCAN = 64


def research_key(sub_question_id: str, round_index: int) -> str:
    if round_index == 0:
        return f"research:{sub_question_id}"
    return f"research:{sub_question_id}:r{round_index}"


def verify_key(sub_question_id: str, round_index: int) -> str:
    if round_index == 0:
        return f"verify:{sub_question_id}"
    return f"verify:{sub_question_id}:r{round_index}"


class ResearchState:
    """A read view over the checkpoint table for one run."""

    def __init__(self, question: str, focus: str | None, checkpoints: dict[str, dict]) -> None:
        self.question = question
        self.focus = focus
        self.checkpoints = checkpoints

    # -- plan ------------------------------------------------------------
    @property
    def plan(self) -> list[SubQuestion] | None:
        raw = self.checkpoints.get(PLAN_KEY)
        if raw is None:
            return None
        return [SubQuestion.model_validate(item) for item in raw.get("sub_questions", [])]

    @property
    def has_plan(self) -> bool:
        return PLAN_KEY in self.checkpoints

    # -- evidence --------------------------------------------------------
    def evidence(self, sub_question_id: str, round_index: int = 0) -> list[Evidence]:
        raw = self.checkpoints.get(research_key(sub_question_id, round_index))
        if not raw:
            return []
        return [Evidence.model_validate(item) for item in raw.get("evidence", [])]

    def all_evidence(self) -> list[Evidence]:
        """Every evidence item across all sub-questions and rounds."""
        found: list[Evidence] = []
        for key, raw in self.checkpoints.items():
            if not key.startswith("research:") or not raw:
                continue
            found.extend(Evidence.model_validate(item) for item in raw.get("evidence", []))
        found.sort(key=lambda item: (item.sub_question_id, item.id))
        return found

    def unsupported_ids(self) -> set[str]:
        """Every evidence id any verifier round rejected."""
        rejected: set[str] = set()
        for key, raw in self.checkpoints.items():
            if not key.startswith("verify:") or not raw:
                continue
            rejected.update(raw.get("unsupported_ids") or [])
        return rejected

    def supported_evidence(self) -> list[Evidence]:
        """Evidence that survived verification — the only thing a report may use."""
        rejected = self.unsupported_ids()
        return [item for item in self.all_evidence() if item.id not in rejected]

    def rounds_completed(self, sub_question_id: str) -> int:
        count = 0
        for round_index in range(_MAX_ROUNDS_SCAN):
            if research_key(sub_question_id, round_index) in self.checkpoints:
                count += 1
            else:
                break
        return count

    # -- verification ----------------------------------------------------
    def verification(self, sub_question_id: str, round_index: int = 0) -> GapAnalysis | None:
        raw = self.checkpoints.get(verify_key(sub_question_id, round_index))
        if not raw:
            return None
        return GapAnalysis.model_validate(raw)

    def follow_up_queries(self, sub_question_id: str, round_index: int) -> list[str]:
        """Queries suggested by the verification of the *previous* round."""
        if round_index == 0:
            return []
        previous = self.verification(sub_question_id, round_index - 1)
        return previous.follow_up_queries if previous else []

    # -- output ----------------------------------------------------------
    @property
    def report(self) -> Report | None:
        raw = self.checkpoints.get(SYNTHESIZE_KEY)
        if not raw:
            return None
        return Report.model_validate(raw)

    @property
    def critique(self) -> Critique | None:
        raw = self.checkpoints.get(CRITIQUE_KEY)
        if not raw:
            return None
        return Critique.model_validate(raw)

    @property
    def has_report(self) -> bool:
        return SYNTHESIZE_KEY in self.checkpoints
