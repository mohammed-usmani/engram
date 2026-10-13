from __future__ import annotations

import logging
from ollama import AsyncClient
from src.config import settings

log = logging.getLogger(__name__)

def _client() -> AsyncClient:
    # per call: a module-level client binds to the first event loop and breaks on the next
    return AsyncClient(host=settings.ollama_base_url)


class LocalLLMDisabled(RuntimeError):
    """OLLAMA_LLM_ENABLED=0: Ollama serves embeddings only, never chat/completion calls."""


def llm_enabled() -> bool:
    """Running a local LLM can saturate a laptop; this switch keeps Ollama to embeddings."""
    import os
    return os.environ.get("OLLAMA_LLM_ENABLED", "1").strip().lower() not in ("0", "false", "off", "no")


def require_llm() -> None:
    if not llm_enabled():
        raise LocalLLMDisabled("Local Ollama LLM calls are turned off (OLLAMA_LLM_ENABLED=0); "
                               "Ollama is used for embeddings only. Pick a cloud model in Settings.")


async def generate(
    message: str,
    system: str = "",
    history: list[dict[str, str]] | None = None,
    model: str = "mistral:latest",
) -> str:
    require_llm()
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
