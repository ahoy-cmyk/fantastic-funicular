"""LLM provider implementations."""

from abc import abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol


@dataclass
class Message:
    """Represents a chat message."""

    role: str  # 'user', 'assistant', 'system'
    content: str
    metadata: dict[str, Any] | None = None


@dataclass
class CompletionResponse:
    """Response from LLM completion."""

    content: str
    model: str
    usage: dict[str, int] | None = None
    metadata: dict[str, Any] | None = None


class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> CompletionResponse:
        """Generate a completion from messages."""
        ...

    @abstractmethod
    async def stream_complete(
        self,
        messages: list[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream a completion from messages."""
        ...

    @abstractmethod
    async def list_models(self) -> list[str]:
        """List available models."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the provider is healthy."""
        ...
