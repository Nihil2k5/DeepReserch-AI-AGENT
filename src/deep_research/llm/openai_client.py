"""Chat completions via any OpenAI-compatible endpoint."""

from __future__ import annotations

import httpx

from ..errors import LLMError
from ..http_client import request_with_retries
from .base import LLM, LLMResponse


class OpenAIChatLLM(LLM):
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 90.0,
        client: httpx.Client | None = None,
        price_prompt_per_mtok: float = 0.15,
        price_completion_per_mtok: float = 0.60,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.name = f"openai:{model}"
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None
        self.price_prompt = price_prompt_per_mtok
        self.price_completion = price_completion_per_mtok

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

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
        body: dict = {
            "model": model or self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        response = request_with_retries(
            self._client,
            "POST",
            f"{self.base_url}/chat/completions",
            error_type=LLMError,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=body,
        )

        payload = response.json()
        try:
            choice = payload["choices"][0]
            text = choice["message"]["content"] or ""
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Malformed completion response: {str(payload)[:300]}") from exc

        usage = payload.get("usage", {}) or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        cost = (
            prompt_tokens / 1_000_000 * self.price_prompt
            + completion_tokens / 1_000_000 * self.price_completion
        )

        return LLMResponse(
            text=text.strip(),
            model=payload.get("model", body["model"]),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=round(cost, 8),
            finish_reason=choice.get("finish_reason", "stop"),
        )
