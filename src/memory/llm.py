"""JSON completions for memory tasks over a free-first provider fallback chain."""
from __future__ import annotations

import asyncio
import json
import logging
import os

from src.services.providers import get_provider
from src.services.providers.base import LLMProvider

log = logging.getLogger(__name__)

DEFAULT_CHAIN = "gemini,groq,mistral,ollama"

# Env var(s) holding each provider's key. Ollama is local and needs none.
_KEY_ENV = {
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "groq": ("GROQ_API_KEY",),
    "mistral": ("MISTRAL_API_KEY",),
    "together": ("TOGETHER_API_KEY",),
    "cerebras": ("CEREBRAS_API_KEY",),
    "claude": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
}


class LLMUnavailable(RuntimeError):
    """Every provider in the chain failed or returned unparseable output."""


# Set from the Settings page (src/settings_store.py): keys saved there win over .env, and per-task
# model assignments win over MEMORY_LLM_CHAIN. Empty means "use the environment".
_saved_keys: dict[str, str] = {}
_assign: dict[str, list[tuple[str, str | None]]] = {}
_models: dict[str, str] = {}      # provider -> default model chosen on the Settings page
_base_urls: dict[str, str] = {}   # provider -> custom OpenAI-compatible URL


def configure(keys: dict[str, str], assign: dict[str, list[tuple[str, str | None]]],
              models: dict[str, str] | None = None, base_urls: dict[str, str] | None = None) -> None:
    for d, new in ((_saved_keys, keys), (_assign, assign), (_models, models or {}), (_base_urls, base_urls or {})):
        d.clear()
        d.update({k: v for k, v in new.items() if v})


def env_key(name: str) -> str | None:
    return next((os.environ[k] for k in _KEY_ENV.get(name, ()) if os.environ.get(k)), None)


def key_source(name: str) -> str | None:
    """'saved' (Settings page), 'env' (.env), 'local' (Ollama needs none) or None."""
    if name == "ollama":
        return "local"
    return "saved" if _saved_keys.get(name) else "env" if env_key(name) else None


def _key(name: str) -> str | None:
    return _saved_keys.get(name) or env_key(name)


def chain(task: str = "default") -> list[str]:
    """Providers to try, in order. MEMORY_LLM_CHAIN_<TASK> overrides MEMORY_LLM_CHAIN."""
    raw = os.environ.get(f"MEMORY_LLM_CHAIN_{task.upper()}") or os.environ.get("MEMORY_LLM_CHAIN", DEFAULT_CHAIN)
    names = [n.strip() for n in raw.split(",") if n.strip()]
    return [n for n in names if n == "ollama" or _key(n)]


def steps(task: str = "default") -> list[tuple[str, str | None]]:
    """(provider, model) pairs to try in order. Settings-page assignments for this task, else the
    extraction assignment, else MEMORY_LLM_CHAIN; providers without a key are skipped."""
    chosen = _assign.get(task) or (_assign.get("extract") if task != "chat" else None)
    if chosen:
        return [(n, m) for n, m in chosen if n == "ollama" or _key(n)]
    if task == "chat":
        return [("ollama", None)]  # chat stays local and free unless the user picks otherwise
    return [(n, None) for n in chain(task)]


def default_model(name: str) -> str | None:
    """The model a provider uses when a task doesn't name one: Settings page, then .env."""
    if name == "ollama":
        return _models.get(name) or os.environ.get("MEMORY_MODEL_OLLAMA") or os.environ.get("OLLAMA_LLM_MODEL", "qwen2.5-coder:7b")
    return _models.get(name) or os.environ.get(f"MEMORY_MODEL_{name.upper()}")


def _make(name: str, model: str | None = None) -> LLMProvider:
    model = model or default_model(name)
    if name == "ollama":
        return get_provider("ollama", model=model)
    return get_provider(name, api_key=_key(name), model=model, base_url=_base_urls.get(name))


make = _make  # public: the chat page and the provider "Test" button build providers the same way


def parse_json(raw: str):
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # models without a JSON mode sometimes wrap the object in prose: take the outermost {...} / [...]
        starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
        if not starts:
            raise
        start = min(starts)
        end = text.rfind("}" if text[start] == "{" else "]")
        return json.loads(text[start:end + 1])


async def complete_json(prompt: str, system: str = "", task: str = "default") -> dict | list:
    errors = []
    attempts = int(os.environ.get("MEMORY_LLM_ATTEMPTS", "2"))
    for name, model in steps(task):
        # Together occasionally never answers one request (seen: 400 s) or returns empty content,
        # and the same request then succeeds in ~20 s, so retry a provider before falling back.
        for attempt in range(attempts):
            try:
                # one stuck call must not hold the queue for minutes (seen: 356s)
                raw = await asyncio.wait_for(
                    (_make(name, model) if model else _make(name)).generate(prompt, system=system + "\nRespond with valid JSON only.", json=True),
                    timeout=float(os.environ.get("MEMORY_LLM_TIMEOUT", "180")))  # a 12k-char chunk took ~110 s
                return parse_json(raw)
            except Exception as e:  # rate limit, timeout, bad key, invalid JSON — retry, then the next one
                msg = (str(e) or type(e).__name__)[:200]
                log.warning("memory llm %s attempt %d failed for task=%s: %s", name, attempt + 1, task, msg)
                errors.append(f"{name}: {msg[:120]}")  # a timeout has no message
    raise LLMUnavailable("; ".join(errors) or "no providers configured")
