"""Deterministic report-quality metrics.

These run without any model, which means they can gate CI. They measure the
mechanical properties a report must have before its content is even worth
grading: citations resolve, claims are cited, sources aren't all one domain.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

from ..models import Report

MARKER_RE = re.compile(r"\[(\d+)\]")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def citation_markers(text: str) -> list[int]:
    return [int(value) for value in MARKER_RE.findall(text)]


def sentences(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE_RE.split(text) if part.strip()]


def _body_texts(report: Report) -> list[str]:
    texts = [report.executive_summary]
    texts.extend(section.body for section in report.sections)
    return [text for text in texts if text]


def citation_validity(report: Report) -> float:
    """Fraction of inline markers that resolve to a listed source.

    By construction this should be 1.0; the metric exists to prove the invariant
    holds end to end rather than to trust that it does.
    """
    valid = {citation.index for citation in report.citations}
    total = 0
    resolved = 0
    for text in _body_texts(report):
        for marker in citation_markers(text):
            total += 1
            if marker in valid:
                resolved += 1
    return 1.0 if total == 0 else resolved / total


def citation_density(report: Report) -> float:
    """Fraction of sentences carrying at least one citation.

    Proxy for "how much of this is sourced" — an uncited research report is just
    an essay.
    """
    all_sentences: list[str] = []
    for text in _body_texts(report):
        all_sentences.extend(sentences(text))
    if not all_sentences:
        return 0.0
    cited = sum(1 for sentence in all_sentences if citation_markers(sentence))
    return cited / len(all_sentences)


def domain(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
    except ValueError:
        return ""
    return netloc[4:] if netloc.startswith("www.") else netloc


def source_diversity(report: Report) -> int:
    """Number of distinct domains cited."""
    return len({domain(citation.url) for citation in report.citations if citation.url})


def cited_source_count(report: Report) -> int:
    return len(report.citations)


def quality_score(report: Report) -> float:
    """A single 0-1 roll-up used for regression tracking in CI."""
    components = [
        citation_validity(report),
        citation_density(report),
        min(1.0, source_diversity(report) / 4.0),
        1.0 if report.limitations else 0.5,
    ]
    return round(sum(components) / len(components), 4)


def aggregate(records: list[dict]) -> dict:
    """Mean of every numeric per-case metric."""
    names = sorted(
        {key for record in records for key, value in record.items() if isinstance(value, (int, float))}
    )
    summary: dict[str, float] = {}
    for name in names:
        values = [record[name] for record in records if isinstance(record.get(name), (int, float))]
        if values:
            summary[name] = round(sum(values) / len(values), 6)
    return summary
