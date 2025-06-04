"""Synchronous Ollama wrapper for use in threads."""

import requests

from src.providers import CompletionResponse, Message
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class OllamaSyncProvider:
    """Synchronous Ollama provider for use in non-async contexts."""

    def __init__(self, host: str = "http://localhost:11434"):
        self.host = host

    def complete(
        self,
        messages: list[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> CompletionResponse:
        """Generate a completion synchronously."""
        try:
            # Convert messages to Ollama format
            ollama_messages = []
            for msg in messages:
                if isinstance(msg, dict):
                    ollama_messages.append(msg)
                else:
                    ollama_messages.append({"role": msg.role, "content": msg.content})

            # Make request
            response = requests.post(
                f"{self.host}/api/chat",
                json={
                    "model": model,
                    "messages": ollama_messages,
                    "options": {"temperature": temperature, "num_predict": max_tokens, **kwargs},
                    "stream": False,
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            return CompletionResponse(
                content=data["message"]["content"],
                model=data["model"],
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                    "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                },
                metadata={
                    "eval_duration": data.get("eval_duration"),
                    "total_duration": data.get("total_duration"),
                },
            )

        except Exception as e:
            logger.error(f"Ollama sync completion error: {e}")
            raise

    def list_models(self) -> list[str]:
        """List available models synchronously."""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []
