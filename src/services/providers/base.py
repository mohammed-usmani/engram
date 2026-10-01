from __future__ import annotations
from abc import ABC, abstractmethod
from typing import AsyncIterator


class LLMProvider(ABC):
    name: str

    @abstractmethod
    async def generate(
        self,
        message: str,
        system: str = "",
        history: list[dict[str, str]] | None = None,
        json: bool = False,
    ) -> str:
        ...

    @abstractmethod
    async def generate_stream(
        self,
        message: str,
        system: str = "",
        history: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[str]:
        """Yield token chunks as they arrive."""
        ...

    @abstractmethod
    async def list_models(self) -> list[str]:
        ...
