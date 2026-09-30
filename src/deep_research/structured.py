"""Best-effort extraction of structured data from model output.

Models wrap JSON in prose or fences often enough that parsing needs a fallback
chain. Failing loudly here would waste an otherwise successful LLM call, so the
chain is: direct parse -> fenced block -> first balanced object/array.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _balanced_slice(text: str, opener: str, closer: str) -> str | None:
    start = text.find(opener)
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def extract_json(text: str, default: Any = None) -> Any:
    """Return parsed JSON from ``text``, or ``default`` when nothing parses."""
    if not text:
        return default

    candidates: list[str] = [text.strip()]

    for match in _FENCE_RE.findall(text):
        candidates.append(match.strip())

    for opener, closer in (("{", "}"), ("[", "]")):
        sliced = _balanced_slice(text, opener, closer)
        if sliced:
            candidates.append(sliced)

    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    return default
