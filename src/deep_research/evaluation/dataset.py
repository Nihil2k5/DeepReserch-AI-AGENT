"""Evaluation dataset.

Each case lists the facts a good report must establish. Facts are graded by an
LLM judge (see :mod:`deep_research.evaluation.judge`) because "does this report
establish X" is a semantic question — keyword matching would score a report that
merely mentions the words as if it had answered.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    id: str = ""
    question: str
    must_include: list[str] = Field(
        default_factory=list,
        description="Facts a good report must establish (graded semantically).",
    )
    min_sources: int = 3
    notes: str = ""


def load_dataset(path: str | Path) -> list[EvalCase]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    cases: list[EvalCase] = []
    for line_number, line in enumerate(
        file_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{file_path}:{line_number} is not valid JSON: {exc}") from exc

        case = EvalCase.model_validate(payload)
        if not case.id:
            case.id = f"c{len(cases) + 1}"
        cases.append(case)

    if not cases:
        raise ValueError(f"Dataset {file_path} contains no cases")
    return cases
