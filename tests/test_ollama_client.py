from unittest.mock import AsyncMock, patch
from src.services.ollama_client import generate, embed, embed_batch


@patch("src.services.ollama_client._client")
async def test_generate(mock_client):
    mock_client.return_value.chat = AsyncMock(return_value={"message": {"content": "Hello back!"}})
    result = await generate("Hi", system="You are helpful")
    assert result == "Hello back!"
    mock_client.return_value.chat.assert_called_once()
    call_kwargs = mock_client.return_value.chat.call_args[1]
    assert call_kwargs["model"] == "mistral:latest"
    assert any(m["role"] == "system" for m in call_kwargs["messages"])


@patch("src.services.ollama_client._client")
async def test_generate_with_history(mock_client):
    mock_client.return_value.chat = AsyncMock(return_value={"message": {"content": "Sure!"}})
    history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    result = await generate("How are you?", system="Be nice", history=history)
    assert result == "Sure!"
    call_kwargs = mock_client.return_value.chat.call_args[1]
    assert len(call_kwargs["messages"]) == 4


@patch("src.services.ollama_client._client")
async def test_embed(mock_client):
    mock_client.return_value.embed = AsyncMock(return_value={"embeddings": [[0.1] * 768]})
    result = await embed("some text")
    assert len(result) == 768


@patch("src.services.ollama_client._client")
async def test_embed_batch(mock_client):
    mock_client.return_value.embed = AsyncMock(
        side_effect=[
            {"embeddings": [[0.1] * 768, [0.2] * 768]},
            {"embeddings": [[0.3] * 768]},
        ]
    )
    result = await embed_batch(["a", "b", "c"], batch_size=2)
    assert len(result) == 3
    assert mock_client.return_value.embed.call_count == 2
