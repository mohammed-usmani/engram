from __future__ import annotations

import logging
from ollama import AsyncClient
from src.config import settings

log = logging.getLogger(__name__)

def _client() -> AsyncClient:
    # per call: a module-level client binds to the first event loop and breaks on the next
    return AsyncClient(host=settings.ollama_base_url)


async def generate(
    message: str,
    system: str = "",
    history: list[dict[str, str]] | None = None,
    model: str = "mistral:latest",
) -> str:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": message})

    response = await _client().chat(model=model, messages=messages)
    return response["message"]["content"]


async def embed(text: str) -> list[float]:
    response = await _client().embed(model=settings.ollama_embed_model, input=text)
    return response["embeddings"][0]


async def embed_batch(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = await _client().embed(model=settings.ollama_embed_model, input=batch)
        all_embeddings.extend(response["embeddings"])
    return all_embeddings


async def list_models() -> list[str]:
    """Return names of all locally installed Ollama models."""
    response = await _client().list()
    return sorted([m["model"] for m in response["models"]])
