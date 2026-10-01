from __future__ import annotations

from typing import AsyncIterator

import anthropic
from src.services.providers.base import LLMProvider


class AnthropicProvider(LLMProvider):
    name = "claude"

    def __init__(self, api_key: str, model: str = "claude-haiku-4-5-20251001"):
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model

    def _build_kwargs(self, message: str, system: str, history: list[dict[str, str]] | None) -> dict:
        messages: list[dict[str, str]] = []
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": message})
        kwargs = {"model": self.model, "messages": messages, "max_tokens": 4096}
        if system:
            kwargs["system"] = system
        return kwargs

    async def generate(
        self,
        message: str,
        system: str = "",
        history: list[dict[str, str]] | None = None,
        json: bool = False,
    ) -> str:
        kwargs = self._build_kwargs(message, system, history)
        response = await self._client.messages.create(**kwargs)
        return response.content[0].text

    async def generate_stream(
        self,
        message: str,
        system: str = "",
        history: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[str]:
        kwargs = self._build_kwargs(message, system, history)
        async with self._client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text

    async def list_models(self) -> list[str]:
        from src.services.providers import PROVIDER_MODELS
        return PROVIDER_MODELS.get("claude", [])
