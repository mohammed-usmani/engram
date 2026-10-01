from __future__ import annotations

from typing import AsyncIterator

from google import genai
from src.services.providers.base import LLMProvider


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self._client = genai.Client(api_key=api_key)
        self.model = model

    def _build_contents(self, message: str, history: list[dict[str, str]] | None):
        contents = []
        if history:
            for m in history:
                role = "model" if m["role"] == "assistant" else m["role"]
                contents.append(genai.types.Content(role=role, parts=[genai.types.Part(text=m["content"])]))
        contents.append(genai.types.Content(role="user", parts=[genai.types.Part(text=message)]))
        return contents

    def _build_config(self, system: str, json: bool = False):
        kw = {}
        if system:
            kw["system_instruction"] = system
        if json:
            kw["response_mime_type"] = "application/json"
        return genai.types.GenerateContentConfig(**kw) if kw else None

    async def generate(
        self,
        message: str,
        system: str = "",
        history: list[dict[str, str]] | None = None,
        json: bool = False,
    ) -> str:
        contents = self._build_contents(message, history)
        config = self._build_config(system, json)
        response = await self._client.aio.models.generate_content(
            model=self.model, contents=contents, config=config,
        )
        return response.text

    async def generate_stream(
        self,
        message: str,
        system: str = "",
        history: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[str]:
        contents = self._build_contents(message, history)
        config = self._build_config(system)
        stream = await self._client.aio.models.generate_content_stream(
            model=self.model, contents=contents, config=config,
        )
        async for chunk in stream:
            if chunk.text:
                yield chunk.text

    async def list_models(self) -> list[str]:
        from src.services.providers import PROVIDER_MODELS
        return PROVIDER_MODELS.get("gemini", [])
