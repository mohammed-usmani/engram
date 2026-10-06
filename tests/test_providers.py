from unittest.mock import AsyncMock, patch, MagicMock
import pytest
from src.services.providers import get_provider, list_provider_names
from src.services.providers.ollama_provider import OllamaProvider
from src.services.providers.openai_compat import OpenAICompatProvider
from src.services.providers.google_provider import GeminiProvider
from src.services.providers.anthropic_provider import AnthropicProvider


def test_list_provider_names():
    names = list_provider_names()
    assert set(names) == {"ollama", "dashscope", "groq", "cerebras", "mistral", "together", "openai", "gemini", "claude"}


def test_get_ollama_provider():
    p = get_provider("ollama")
    assert isinstance(p, OllamaProvider)


def test_get_groq_provider():
    p = get_provider("groq", api_key="test", model="llama-3.3-70b-versatile")
    assert isinstance(p, OpenAICompatProvider)
    assert p.name == "groq"


def test_get_gemini_provider():
    p = get_provider("gemini", api_key="test")
    assert isinstance(p, GeminiProvider)


def test_get_anthropic_provider():
    p = get_provider("claude", api_key="test")
    assert isinstance(p, AnthropicProvider)


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider("unknown_provider")


@patch("src.services.providers.ollama_provider._client")
async def test_ollama_generate(mock_client):
    mock_client.return_value.chat = AsyncMock(return_value={"message": {"content": "Hello!"}})
    p = OllamaProvider(model="mistral:latest")
    result = await p.generate("Hi")
    assert result == "Hello!"


@patch("src.services.providers.ollama_provider._client")
async def test_ollama_list_models(mock_client):
    mock_client.return_value.list = AsyncMock(return_value={"models": [
        {"model": "mistral:latest"},
        {"model": "nomic-embed-text:latest"},
    ]})
    p = OllamaProvider()
    models = await p.list_models()
    assert "mistral:latest" in models


@patch("src.services.providers.openai_compat.AsyncOpenAI")
async def test_openai_compat_generate(MockClient):
    mock_instance = MagicMock()
    mock_instance.chat.completions.create = AsyncMock(return_value=MagicMock(
        choices=[MagicMock(message=MagicMock(content="Hello from Groq!"))]
    ))
    MockClient.return_value = mock_instance
    p = OpenAICompatProvider("groq", "https://api.groq.com/openai/v1", "test-key", "llama-3.3-70b")
    result = await p.generate("Hi")
    assert result == "Hello from Groq!"


@patch("src.services.providers.ollama_provider._client")
async def test_ollama_json_calls_skip_thinking(mock_client):
    mock_client.return_value.chat = AsyncMock(return_value={"message": {"content": "{}"}})
    await OllamaProvider(model="qwen3:8b").generate("x", json=True)
    kw = mock_client.return_value.chat.call_args.kwargs
    assert kw["format"] == "json" and kw["think"] is False
    assert kw["options"]["num_ctx"] >= 16384  # session transcripts are ~7k tokens; default 4096 truncates
