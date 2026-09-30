"""Citation handling and markdown rendering.

Citation numbers are assigned by *us*, never by the model: the source list is
numbered before synthesis and the model is told which numbers exist. Markers
that reference a number we never issued are stripped, so a report can never cite
a source that doesn't exist.
"""

from __future__ import annotations

import re

from .models import Citation, Evidence, Report

CITATION_RE = re.compile(r"\[(\d+)\]")


def build_source_index(evidence: list[Evidence]) -> list[Citation]:
    """Number sources in order of first appearance. Stable across runs."""
    seen: dict[str, int] = {}
    citations: list[Citation] = []
    for item in evidence:
        if item.url in seen:
            continue
        index = len(seen) + 1
        seen[item.url] = index
        citations.append(Citation(index=index, url=item.url, title=item.title))
    return citations


def sanitize_citations(text: str, valid: set[int]) -> tuple[str, set[int]]:
    """Strip unknown ``[n]`` markers; return cleaned text and the numbers used."""
    used: set[int] = set()

    def replace(match: re.Match[str]) -> str:
        number = int(match.group(1))
        if number in valid:
            used.add(number)
            return match.group(0)
        return ""

    cleaned = CITATION_RE.sub(replace, text)
    # Tidy the gaps a removed marker can leave behind.
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
    return cleaned.strip(), used


def render_markdown(report: Report) -> str:
    lines: list[str] = [f"# {report.question}", ""]

    if report.executive_summary:
        lines += ["## Summary", "", report.executive_summary, ""]

    if report.sub_questions:
        lines += ["## Research plan", ""]
        lines += [f"- {question}" for question in report.sub_questions]
        lines.append("")

    for section in report.sections:
        lines += [f"## {section.heading}", "", section.body, ""]

    if report.limitations:
        lines += ["## Limitations", ""]
        lines += [f"- {item}" for item in report.limitations]
        lines.append("")

    if report.critique:
        critique = report.critique
        lines += ["## Quality review", ""]
        lines.append(f"- Coverage score: {critique.coverage_score:.2f}")
        if critique.verdict:
            lines.append(f"- Verdict: {critique.verdict}")
        if critique.unsupported_claims:
            lines.append("- Claims needing more support:")
            lines += [f"  - {claim}" for claim in critique.unsupported_claims]
        if critique.improvements:
            lines.append("- Suggested improvements:")
            lines += [f"  - {item}" for item in critique.improvements]
        lines.append("")

    if report.citations:
        lines += ["## Sources", ""]
        for citation in sorted(report.citations, key=lambda c: c.index):
            label = citation.title or citation.url
            lines.append(f"{citation.index}. [{label}]({citation.url})")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
