"""Ollama LLM provider implementation."""

from collections.abc import AsyncIterator

import ollama

from src.providers import CompletionResponse, LLMProvider, Message
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class OllamaProvider(LLMProvider):
    """Provider for Ollama local models."""

    def __init__(self, host: str = "http://localhost:11434"):
        """Initialize Ollama provider.

        Args:
            host: Ollama server host URL
        """
        self.host = host
        # Don't store a persistent client - create fresh for each request

    def _create_client(self):
        """Create a fresh Ollama client."""
        client = ollama.AsyncClient(host=self.host)
        logger.info("Created fresh Ollama client")
        return client

    async def complete(
        self,
        messages: list[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> CompletionResponse:
        """Generate a completion from messages."""
        # Create fresh client for this request
        client = self._create_client()

        try:
            # Convert messages to Ollama format
            ollama_messages = []
            for msg in messages:
                if isinstance(msg, dict):
                    ollama_messages.append(msg)
                else:
                    ollama_messages.append({"role": msg.role, "content": msg.content})

            # Make request
            response = await client.chat(
                model=model,
                messages=ollama_messages,
                options={"temperature": temperature, "num_predict": max_tokens, **kwargs},
            )

            return CompletionResponse(
                content=response["message"]["content"],
                model=response["model"],
                usage={
                    "prompt_tokens": response.get("prompt_eval_count", 0),
                    "completion_tokens": response.get("eval_count", 0),
                    "total_tokens": response.get("prompt_eval_count", 0)
                    + response.get("eval_count", 0),
                },
                metadata={
                    "eval_duration": response.get("eval_duration"),
                    "total_duration": response.get("total_duration"),
                },
            )

        except Exception as e:
            logger.error(f"Ollama completion error: {e}")
            raise
        finally:
            # Clean up client resources
            del client

    async def stream_complete(
        self,
        messages: list[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream a completion from messages."""
        # Create fresh client for this streaming request
        client = self._create_client()

        try:
            # Convert messages to Ollama format
            ollama_messages = []
            for msg in messages:
                if isinstance(msg, dict):
                    ollama_messages.append(msg)
                else:
                    ollama_messages.append({"role": msg.role, "content": msg.content})

            logger.info(f"Ollama: Sending {len(ollama_messages)} messages to model {model}")
            for i, msg in enumerate(ollama_messages[-3:]):  # Log last 3 messages
                role = msg.get("role", "unknown")
                content_preview = (
                    msg.get("content", "")[:50] + "..."
                    if len(msg.get("content", "")) > 50
                    else msg.get("content", "")
                )
                logger.info(f"  Message[-{3-i}]: {role}: {content_preview}")

            # Stream response with fresh client
            stream = await client.chat(
                model=model,
                messages=ollama_messages,
                stream=True,
                options={"temperature": temperature, "num_predict": max_tokens, **kwargs},
            )

            chunk_count = 0
            async for chunk in stream:
                if chunk.get("message", {}).get("content"):
                    chunk_count += 1
                    content = chunk["message"]["content"]
                    yield content

            logger.info(f"Ollama: Stream completed with {chunk_count} chunks")

        except Exception as e:
            # Handle event loop closure gracefully during app shutdown
            if "Event loop is closed" in str(e) or "RuntimeError" in str(type(e).__name__):
                logger.info("Ollama stream stopped due to app shutdown")
                return
            logger.error(f"Ollama stream error: {e}")
            raise
        finally:
            # Clean up client resources
            del client

    async def list_models(self) -> list[str]:
        """List available models."""
        try:
            # Use the tags endpoint directly with httpx
            import httpx

            async with httpx.AsyncClient() as http_client:
                response = await http_client.get(f"{self.host}/api/tags")
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []

    async def health_check(self) -> bool:
        """Check if the provider is healthy."""
        try:
            models = await self.list_models()
            return len(models) > 0
        except Exception:
            return False

    async def pull_model(self, model: str) -> bool:
        """Pull a model from Ollama registry."""
        # Create fresh client for model pulling
        client = self._create_client()

        try:
            logger.info(f"Pulling model: {model}")
            await client.pull(model)
            return True
        except Exception as e:
            logger.error(f"Failed to pull model {model}: {e}")
            return False
        finally:
            # Clean up client resources
            del client
