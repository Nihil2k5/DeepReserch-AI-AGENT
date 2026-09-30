"""Researcher: search the web for one sub-question and extract claim-level evidence.

Two model calls per round:

1. **Query generation** — turn a sub-question into a few targeted search queries
   (skipped on follow-up rounds, where the verifier already supplied them).
2. **Extraction** — read the retrieved page text and emit atomic claims, each
   with the *exact quote* that supports it and the source it came from.

Quotes are mandatory rather than optional: a claim you can't quote is a claim
you can't verify, and the verifier node's entire job is checking claim against
quote.
"""

from __future__ import annotations

import re

from ...errors import BudgetExceeded
from ...logging_config import get_logger
from ...models import Evidence, SearchHit, SubQuestion
from ...structured import extract_json
from ..context import NodeContext
from .base import Node

logger = get_logger(__name__)

QUERY_SYSTEM = """You write web search queries for a research agent.

Rules:
- Produce 2-3 queries, each phrased as a real search someone would type.
- Vary them: one broad, one specific to data/numbers, one targeting authoritative sources.
- Do not use boolean operators, quotes or site: filters.
- Each query must stand alone (no pronouns referring to the original question).

Respond with JSON only: {"queries": ["...", "..."]}"""

EXTRACT_SYSTEM = """You extract verifiable evidence from web sources.

For each source, extract claims that help answer the sub-question.

Rules:
- Every claim MUST come with an exact quote copied verbatim from that source.
  Never paraphrase inside the quote field.
- One claim per distinct fact. Prefer numbers, dates, names and concrete statements.
- Set "source" to the bracketed number of the source the quote came from.
- "confidence" is 0-1 for how directly the quote supports the claim.
- Extract at most 6 claims. If a source has nothing relevant, skip it.
- Never invent facts that are not present in the sources.

Respond with JSON only:
{"evidence": [{"claim": "...", "quote": "...", "source": 1, "confidence": 0.9}]}"""

#: Per-source text budget when building the extraction prompt.
MAX_SOURCE_CHARS = 2200
#: Maximum sources handed to the extractor.
MAX_SOURCES = 8
#: Maximum claims kept per research node.
MAX_EVIDENCE = 6


class ResearcherNode(Node):
    name = "research"

    def __init__(
        self,
        sub_question: SubQuestion,
        round_index: int = 0,
        follow_up_queries: list[str] | None = None,
        model: str | None = None,
    ) -> None:
        self.sub_question = sub_question
        self.round_index = round_index
        self.follow_up_queries = list(follow_up_queries or [])
        self.model = model

    # -- queries ---------------------------------------------------------
    def _queries(self, ctx: NodeContext) -> list[str]:
        if self.follow_up_queries:
            queries = list(self.follow_up_queries)
        else:
            response = ctx.call_llm(
                name="research.queries",
                system=QUERY_SYSTEM,
                user=f"Sub-question: {self.sub_question.question}",
                model=self.model,
                json_mode=True,
                max_tokens=500,
            )
            payload = extract_json(response.text, default={}) or {}
            raw = payload.get("queries") if isinstance(payload, dict) else None
            queries = [str(q).strip() for q in raw] if isinstance(raw, list) else []

        queries = [q for q in queries if q][:3]
        return queries or [self.sub_question.question]

    # -- sources ---------------------------------------------------------
    def _gather(self, ctx: NodeContext, queries: list[str]) -> list[SearchHit]:
        seen: set[str] = set()
        hits: list[SearchHit] = []
        for query in queries:
            try:
                results = ctx.search_web(query)
            except BudgetExceeded:
                # A budget breach is a run-level decision, not a flaky query.
                raise
            except Exception as exc:
                # A single failed query shouldn't sink the sub-question.
                logger.warning("Search failed for %r: %s", query, exc)
                continue
            for hit in results:
                if hit.url in seen or not hit.text:
                    continue
                seen.add(hit.url)
                hits.append(hit)

        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:MAX_SOURCES]

    def run(self, ctx: NodeContext) -> dict:
        queries = self._queries(ctx)
        ctx.emit(
            "research.queries",
            f"{len(queries)} queries for {self.sub_question.id}",
            {"sub_question_id": self.sub_question.id, "queries": queries},
        )

        hits = self._gather(ctx, queries)
        if not hits:
            logger.info("No usable sources for %s", self.sub_question.id)
            ctx.emit("research.empty", f"No sources found for {self.sub_question.id}")
            return {
                "sub_question_id": self.sub_question.id,
                "round_index": self.round_index,
                "queries": queries,
                "urls": [],
                "evidence": [],
            }

        sources_block = "\n\n".join(
            f"[{index}] {hit.title}\nURL: {hit.url}\n{hit.text[:MAX_SOURCE_CHARS]}"
            for index, hit in enumerate(hits, start=1)
        )
        user = (
            f"Sub-question: {self.sub_question.question}\n\n"
            f"Sources:\n{sources_block}"
        )

        response = ctx.call_llm(
            name="research.extract",
            system=EXTRACT_SYSTEM,
            user=user,
            model=self.model,
            json_mode=True,
            max_tokens=1800,
        )

        payload = extract_json(response.text, default={}) or {}
        raw_items = payload.get("evidence") if isinstance(payload, dict) else None
        if not isinstance(raw_items, list):
            raw_items = []

        evidence: list[Evidence] = []
        for position, item in enumerate(raw_items[:MAX_EVIDENCE], start=1):
            if not isinstance(item, dict):
                continue
            claim = str(item.get("claim") or "").strip()
            quote = str(item.get("quote") or "").strip()
            if not claim or not quote:
                continue
            try:
                source_index = int(item.get("source") or 0)
            except (TypeError, ValueError):
                source_index = 0
            if not 1 <= source_index <= len(hits):
                # Model referenced a source we didn't send; drop rather than guess.
                logger.debug("Dropping evidence with invalid source index %s", source_index)
                continue

            hit = hits[source_index - 1]
            evidence.append(
                Evidence(
                    id=f"{self.sub_question.id}-r{self.round_index}-e{position}",
                    sub_question_id=self.sub_question.id,
                    claim=claim,
                    quote=self._clean_quote(quote),
                    url=hit.url,
                    title=hit.title,
                    confidence=self._clamp(item.get("confidence")),
                )
            )

        ctx.emit(
            "research.evidence",
            f"{len(evidence)} claims for {self.sub_question.id}",
            {
                "sub_question_id": self.sub_question.id,
                "round_index": self.round_index,
                "count": len(evidence),
            },
        )
        return {
            "sub_question_id": self.sub_question.id,
            "round_index": self.round_index,
            "queries": queries,
            "urls": [hit.url for hit in hits],
            "evidence": [item.model_dump() for item in evidence],
        }

    @staticmethod
    def _clean_quote(quote: str) -> str:
        return re.sub(r"\s+", " ", quote).strip()[:600]

    @staticmethod
    def _clamp(value: object) -> float:
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0.5
        return min(1.0, max(0.0, number))
