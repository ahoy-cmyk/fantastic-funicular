"""LM Studio provider implementation."""

from src.providers.openai_provider import OpenAIProvider
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class LMStudioProvider(OpenAIProvider):
    """Provider for LM Studio local models.

    LM Studio exposes an OpenAI-compatible API, so we can reuse
    the OpenAI provider with a custom base URL.
    """

    def __init__(self, host: str = "http://localhost:1234", api_key: str = "lm-studio"):
        """Initialize LM Studio provider.

        Args:
            host: LM Studio server host URL
            api_key: Dummy API key (LM Studio doesn't require auth)
        """
        super().__init__(api_key=api_key, base_url=f"{host}/v1")
        self.host = host
        logger.info(f"Initialized LM Studio provider at {host}")

    async def list_models(self) -> list[str]:
        """List available models in LM Studio."""
        try:
            response = await self.client.models.list()
            # LM Studio returns local model paths
            return [model.id for model in response.data]
        except Exception as e:
            logger.error(f"Failed to list LM Studio models: {e}")
            return []
