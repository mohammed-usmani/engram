from __future__ import annotations

from src.services.providers.base import LLMProvider
from src.services.providers.ollama_provider import OllamaProvider
from src.services.providers.openai_compat import OpenAICompatProvider, OPENAI_COMPAT_PROVIDERS
from src.services.providers.google_provider import GeminiProvider
from src.services.providers.anthropic_provider import AnthropicProvider

PROVIDER_MODELS: dict[str, list[str]] = {
    "groq": ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile", "meta-llama/llama-4-scout-17b-16e-instruct"],
    "together": ["deepseek-ai/DeepSeek-V4-Flash-0731", "openai/gpt-oss-120b", "openai/gpt-oss-20b", "meta-llama/Llama-3.3-70B-Instruct-Turbo"],
    "mistral": ["mistral-small-latest", "mistral-medium-latest", "mistral-large-latest"],
    "cerebras": ["llama3.1-8b", "gpt-oss-120b", "qwen-3-235b-a22b-instruct-2507", "zai-glm-4.7"],
    "openai": ["gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o"],
    "gemini": ["gemini-flash-lite-latest", "gemini-flash-latest", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-3-flash-preview", "gemini-3.1-pro-preview"],
    "claude": ["claude-haiku-4-5-20251001", "claude-sonnet-5-5", "claude-opus-5-5"],
    "dashscope": ["deepseek-v4-flash-0731", "qwen-flash", "qwen-plus", "qwen3-235b-a22b-instruct-2507",
                  "qwen3-coder-flash", "kimi-k3", "deepseek-v4-pro"],
}

_DEFAULT_MODELS: dict[str, str] = {
    "groq": "openai/gpt-oss-120b",
    "mistral": "mistral-small-latest",
    "together": "openai/gpt-oss-120b",
    "cerebras": "llama3.1-8b",
    "openai": "gpt-4.1-mini",
    "gemini": "gemini-flash-lite-latest",
    "claude": "claude-haiku-4-5-20251001",
    "dashscope": "deepseek-v4-flash-0731",
}


def list_provider_names() -> list[str]:
    return ["ollama", "dashscope", "groq", "cerebras", "mistral", "together", "openai", "gemini", "claude"]


def get_provider(
    name: str,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> LLMProvider:
    if name == "ollama":
        return OllamaProvider(model=model)
    elif name in OPENAI_COMPAT_PROVIDERS:
        url = base_url or OPENAI_COMPAT_PROVIDERS[name]
        m = model or _DEFAULT_MODELS.get(name, "")
        return OpenAICompatProvider(name, url, api_key or "", m)
    elif name == "gemini":
        m = model or _DEFAULT_MODELS["gemini"]
        return GeminiProvider(api_key or "", m)
    elif name == "claude":
        m = model or _DEFAULT_MODELS["claude"]
        return AnthropicProvider(api_key or "", m)
    else:
        raise ValueError(f"Unknown provider: {name}")
