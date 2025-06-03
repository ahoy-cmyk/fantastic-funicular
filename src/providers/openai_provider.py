"""OpenAI API provider implementation."""

import os
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from src.providers import CompletionResponse, LLMProvider, Message
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class OpenAIProvider(LLMProvider):
    """Provider for OpenAI API models."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            base_url: Optional base URL for API (for compatibility with LM Studio)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=base_url)

    async def complete(
        self,
        messages: list[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> CompletionResponse:
        """Generate a completion from messages."""
        try:
            # Convert messages to OpenAI format
            openai_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

            # Make request
            response = await self.client.chat.completions.create(
                model=model,
                messages=openai_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            choice = response.choices[0]

            return CompletionResponse(
                content=choice.message.content,
                model=response.model,
                usage=(
                    {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }
                    if response.usage
                    else None
                ),
                metadata={"finish_reason": choice.finish_reason, "id": response.id},
            )

        except Exception as e:
            logger.error(f"OpenAI completion error: {e}")
            raise

    async def stream_complete(
        self,
        messages: list[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream a completion from messages."""
        try:
            # Convert messages to OpenAI format
            openai_messages = [{"role": msg.role, "content": msg.content} for msg in messages]

            # Stream response
            stream = await self.client.chat.completions.create(
                model=model,
                messages=openai_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs,
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"OpenAI stream error: {e}")
            raise

    async def list_models(self) -> list[str]:
        """List available models."""
        try:
            response = await self.client.models.list()
            return [model.id for model in response.data if model.id.startswith(("gpt-", "text-"))]
        except Exception as e:
            logger.error(f"Failed to list OpenAI models: {e}")
            return []

    async def health_check(self) -> bool:
        """Check if the provider is healthy."""
        try:
            models = await self.list_models()
            return len(models) > 0
        except Exception:
            return False
