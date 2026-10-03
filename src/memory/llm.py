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


def _key(name: str) -> str | None:
    return next((os.environ[k] for k in _KEY_ENV.get(name, ()) if os.environ.get(k)), None)


def chain(task: str = "default") -> list[str]:
    """Providers to try, in order. MEMORY_LLM_CHAIN_<TASK> overrides MEMORY_LLM_CHAIN."""
    raw = os.environ.get(f"MEMORY_LLM_CHAIN_{task.upper()}") or os.environ.get("MEMORY_LLM_CHAIN", DEFAULT_CHAIN)
    names = [n.strip() for n in raw.split(",") if n.strip()]
    return [n for n in names if n == "ollama" or _key(n)]


def _make(name: str) -> LLMProvider:
    model = os.environ.get(f"MEMORY_MODEL_{name.upper()}")
    if name == "ollama":
        return get_provider("ollama", model=model or os.environ.get("OLLAMA_LLM_MODEL", "qwen2.5-coder:7b"))
    return get_provider(name, api_key=_key(name), model=model)


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
    for name in chain(task):
        try:
            # one stuck call must not hold the single-worker queue for minutes (seen: 356s)
            raw = await asyncio.wait_for(
                _make(name).generate(prompt, system=system + "\nRespond with valid JSON only.", json=True),
                timeout=float(os.environ.get("MEMORY_LLM_TIMEOUT", "180")))  # a 12k-char chunk took ~110 s
            return parse_json(raw)
        except Exception as e:  # rate limit, timeout, bad key, invalid JSON — all mean "try the next one"
            log.warning("memory llm %s failed for task=%s: %s", name, task, (str(e) or type(e).__name__)[:200])
            errors.append(f"{name}: {(str(e) or type(e).__name__)[:120]}")  # a timeout has no message
    raise LLMUnavailable("; ".join(errors) or "no providers configured")
