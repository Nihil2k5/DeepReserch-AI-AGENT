"""Exa search tool.

Uses Exa's REST API directly over httpx rather than the ``exa-py`` SDK: it keeps
the dependency surface to one HTTP client, makes the retry policy explicit, and
means a pinned SDK version can't break the agent. The SDK is a fine alternative
if you prefer it.

Exa is chosen over a keyword SERP API because it returns **extracted page text**
alongside each result, which the extractor node can quote directly.
"""

from __future__ import annotations

import httpx

from ..errors import SearchError
from ..http_client import request_with_retries
from ..logging_config import get_logger
from ..models import SearchHit
from .base import SearchTool

logger = get_logger(__name__)

DEFAULT_BASE_URL = "https://api.exa.ai"


class ExaSearchTool(SearchTool):
    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 45.0,
        max_characters: int = 4000,
        client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.max_characters = max_characters
        self.name = "exa"
        self._client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def search(self, query: str, num_results: int = 5) -> list[SearchHit]:
        payload = {
            "query": query,
            "numResults": num_results,
            "contents": {"text": {"maxCharacters": self.max_characters}},
        }
        response = request_with_retries(
            self._client,
            "POST",
            f"{self.base_url}/search",
            error_type=SearchError,
            headers={"x-api-key": self.api_key, "Content-Type": "application/json"},
            json=payload,
        )

        body = response.json()
        raw_results = body.get("results") or []
        hits: list[SearchHit] = []
        for item in raw_results:
            url = item.get("url")
            if not url:
                continue
            hits.append(
                SearchHit(
                    url=url,
                    title=item.get("title") or url,
                    published=item.get("publishedDate"),
                    snippet=(item.get("text") or "")[:300],
                    text=item.get("text") or "",
                    score=float(item.get("score") or 0.0),
                )
            )

        logger.debug("Exa returned %d results for %r", len(hits), query[:80])
        return hits
