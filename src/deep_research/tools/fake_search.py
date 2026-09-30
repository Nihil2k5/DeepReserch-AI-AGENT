"""Deterministic search tool used by unit tests."""

from __future__ import annotations

from ..models import SearchHit
from .base import SearchTool


class FakeSearchTool(SearchTool):
    def __init__(
        self,
        hits: list[SearchHit] | None = None,
        hits_by_query: dict[str, list[SearchHit]] | None = None,
    ) -> None:
        self.name = "fake"
        self._default = list(hits or [])
        self._by_query = dict(hits_by_query or {})
        self.queries: list[str] = []

    def search(self, query: str, num_results: int = 5) -> list[SearchHit]:
        self.queries.append(query)

        results = self._by_query.get(query)
        if results is None:
            # Fall back to a substring match so tests don't need exact queries.
            for key, value in self._by_query.items():
                if key.lower() in query.lower() or query.lower() in key.lower():
                    results = value
                    break
        if results is None:
            results = self._default

        return list(results)[:num_results]
