import pytest

from src.memory import llm


class _Fake:
    def __init__(self, name, reply=None, exc=None):
        self.name, self.reply, self.exc, self.calls = name, reply, exc, 0

    async def generate(self, message, system="", history=None, json=False):
        self.calls += 1
        assert json is True
        if self.exc:
            raise self.exc
        return self.reply


@pytest.fixture
def fakes(monkeypatch):
    table = {}
    monkeypatch.setattr(llm, "_make", lambda name: table[name])
    monkeypatch.setattr(llm, "chain", lambda task="default": list(table))
    return table


async def test_falls_through_rate_limit(fakes):
    fakes["gemini"] = _Fake("gemini", exc=RuntimeError("429 Too Many Requests"))
    fakes["groq"] = _Fake("groq", reply='{"a": 1}')
    assert await llm.complete_json("x") == {"a": 1}
    assert fakes["gemini"].calls == 1


async def test_invalid_json_moves_on_and_fenced_json_parses(fakes):
    fakes["gemini"] = _Fake("gemini", reply="sorry, no")
    fakes["groq"] = _Fake("groq", reply='```json\n[{"b": 2}]\n```')
    assert await llm.complete_json("x") == [{"b": 2}]


async def test_all_fail_raises(fakes):
    fakes["gemini"] = _Fake("gemini", exc=RuntimeError("down"))
    with pytest.raises(llm.LLMUnavailable):
        await llm.complete_json("x")


def test_chain_skips_providers_without_keys(monkeypatch):
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY", "MISTRAL_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "g")
    monkeypatch.setenv("MEMORY_LLM_CHAIN", "gemini,groq,mistral,ollama")
    assert llm.chain() == ["groq", "ollama"]


def test_together_in_chain_when_key_set(monkeypatch):
    monkeypatch.setenv("TOGETHER_API_KEY", "t")
    monkeypatch.setenv("MEMORY_LLM_CHAIN", "together,ollama")
    assert llm.chain() == ["together", "ollama"]


def test_together_provider_is_openai_compatible():
    from src.services.providers import get_provider
    p = get_provider("together", api_key="t")
    assert p.name == "together" and p.model == "openai/gpt-oss-120b"
    assert str(p._client.base_url).startswith("https://api.together.xyz/v1")


def test_parse_json_tolerates_preamble():
    assert llm.parse_json('Sure! Here is the JSON:\n{"a": [1, 2]}\nHope that helps.') == {"a": [1, 2]}
    assert llm.parse_json('Result: [{"b": 1}]') == [{"b": 1}]


async def test_slow_provider_times_out_and_falls_through(fakes, monkeypatch):
    import asyncio

    class Slow(_Fake):
        async def generate(self, message, system="", history=None, json=False):
            await asyncio.sleep(5)
    monkeypatch.setenv("MEMORY_LLM_TIMEOUT", "0.05")
    fakes["together"] = Slow("together")
    fakes["ollama"] = _Fake("ollama", reply='{"ok": 1}')
    t = asyncio.get_running_loop().time()
    assert await llm.complete_json("x") == {"ok": 1}
    assert asyncio.get_running_loop().time() - t < 1  # gave up on the slow provider, didn't wait it out
