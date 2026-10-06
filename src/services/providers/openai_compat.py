from __future__ import annotations

from typing import AsyncIterator

from openai import AsyncOpenAI
from src.services.providers.base import LLMProvider

OPENAI_COMPAT_PROVIDERS = {
    "groq": "https://api.groq.com/openai/v1",
    "cerebras": "https://api.cerebras.ai/v1",
    "openai": "https://api.openai.com/v1",
    "mistral": "https://api.mistral.ai/v1",
    "together": "https://api.together.xyz/v1",
    "dashscope": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",  # Alibaba Model Studio (international)
}

# Hybrid-reasoning models on DashScope think by default: slower, more tokens, and the reply can arrive
# as reasoning only. Engram wants direct JSON answers.
_EXTRA_BODY = {"dashscope": {"enable_thinking": False}}

# Together's json_object mode wants a schema; there we rely on the prompt + tolerant parsing.
_NO_JSON_MODE = {"together"}


class OpenAICompatProvider(LLMProvider):
    def __init__(self, name: str, base_url: str, api_key: str, model: str):
        self.name = name
        # fail fast: the caller has its own timeout and fallback chain (the SDK default is 10 min, 2 retries)
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=170, max_retries=1)
        self.model = model

    def _build_messages(self, message: str, system: str, history: list[dict[str, str]] | None) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": message})
        return messages

    async def generate(
        self,
        message: str,
        system: str = "",
        history: list[dict[str, str]] | None = None,
        json: bool = False,
    ) -> str:
        messages = self._build_messages(message, system, history)
        extra = {"response_format": {"type": "json_object"}} if json and self.name not in _NO_JSON_MODE else {}
        if self.name in _EXTRA_BODY:
            extra["extra_body"] = _EXTRA_BODY[self.name]
        response = await self._client.chat.completions.create(
            model=self.model, messages=messages, **extra,
        )
        return response.choices[0].message.content

    async def generate_stream(
        self,
        message: str,
        system: str = "",
        history: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[str]:
        messages = self._build_messages(message, system, history)
        stream = await self._client.chat.completions.create(
            model=self.model, messages=messages, stream=True,
            **({"extra_body": _EXTRA_BODY[self.name]} if self.name in _EXTRA_BODY else {}),
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content

    async def list_models(self) -> list[str]:
        from src.services.providers import PROVIDER_MODELS
        return PROVIDER_MODELS.get(self.name, [])
