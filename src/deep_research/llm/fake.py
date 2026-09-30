"""Deterministic LLM used by unit tests.

Research nodes for different sub-questions run concurrently, so a strict FIFO
queue would map responses to nodes nondeterministically. Instead, responses are
**routed by matching a substring of the prompt** (system + user). A queue is
still available as a fallback. Never selected in production.
"""

from __future__ import annotations

import json

from .base import LLM, LLMResponse


class FakeLLM(LLM):
    def __init__(
        self,
        responses: list[str] | None = None,
        default: str = "{}",
        routes: dict[str, str] | None = None,
    ) -> None:
        self.name = "fake"
        self._queue: list[str] = list(responses or [])
        self._default = default
        # Matched against `system + "\n" + user`; first hit wins (insertion order).
        self._routes: dict[str, str] = dict(routes or {})
        self.calls: list[dict] = []

    # -- scripting -------------------------------------------------------
    def route(self, marker: str, response: str) -> FakeLLM:
        self._routes[marker] = response
        return self

    def route_json(self, marker: str, payload: dict) -> FakeLLM:
        return self.route(marker, json.dumps(payload))

    def queue(self, *responses: str) -> FakeLLM:
        self._queue.extend(responses)
        return self

    @property
    def call_count(self) -> int:
        return len(self.calls)

    # -- matching --------------------------------------------------------
    def _select(self, system: str, user: str) -> str:
        haystack = f"{system}\n{user}"
        for marker, response in self._routes.items():
            if marker in haystack:
                return response
        if self._queue:
            return self._queue.pop(0)
        return self._default

    def complete(
        self,
        *,
        system: str,
        user: str,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1500,
        json_mode: bool = False,
    ) -> LLMResponse:
        self.calls.append(
            {
                "system": system,
                "user": user,
                "model": model,
                "json_mode": json_mode,
                "max_tokens": max_tokens,
            }
        )
        text = self._select(system, user)
        return LLMResponse(
            text=text,
            model="fake",
            prompt_tokens=max(1, (len(system) + len(user)) // 4),
            completion_tokens=max(1, len(text) // 4),
            cost_usd=0.0,
        )
