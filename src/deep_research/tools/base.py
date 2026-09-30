"""Search tool interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import SearchHit


class SearchTool(ABC):
    """Web search that returns extractable page text, not just links.

    Clean page text is the whole point: an agent that only gets titles and URLs
    has to make a second hop to read anything, and most of them return HTML
    soup that wastes context window.
    """

    name: str = "search"

    @abstractmethod
    def search(self, query: str, num_results: int = 5) -> list[SearchHit]: ...

    def close(self) -> None:
        """Release network resources."""
