from __future__ import annotations

import os
from typing import AsyncIterator

from ollama import AsyncClient
from src.config import settings
from src.services.providers.base import LLMProvider

def _client() -> AsyncClient:
    # per call: a module-level client binds to the first event loop and breaks on the next
    return AsyncClient(host=settings.ollama_base_url)


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, model: str | None = None):
        self.model = model or "mistral:latest"

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
        # JSON calls are extraction work: reasoning models (qwen3, …) would spend a minute "thinking" first
        # and Ollama's default 4096-token window would silently truncate a session transcript.
        extra = {"format": "json", "think": False,
                 "options": {"num_ctx": int(os.environ.get("OLLAMA_NUM_CTX", "16384"))}} if json else {}
        response = await _client().chat(model=self.model, messages=messages, **extra)
        return response["message"]["content"]

    async def generate_stream(
        self,
        message: str,
        system: str = "",
        history: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[str]:
        messages = self._build_messages(message, system, history)
        stream = await _client().chat(model=self.model, messages=messages, stream=True)
        async for chunk in stream:
            token = chunk.get("message", {}).get("content", "")
            if token:
                yield token

    async def list_models(self) -> list[str]:
        response = await _client().list()
        return sorted([m["model"] for m in response["models"]])
