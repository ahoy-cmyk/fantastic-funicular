"""Tests for LLM provider implementations."""

from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from src.providers import Message, Response
from src.providers.lmstudio_provider import LMStudioProvider
from src.providers.ollama_provider import OllamaProvider
from src.providers.openai_provider import OpenAIProvider


@pytest.mark.unit
class TestOllamaProvider:
    """Test Ollama provider functionality."""

    @pytest.fixture
    def ollama_provider(self):
        """Create Ollama provider for testing."""
        return OllamaProvider(host="http://localhost:11434")

    @pytest.mark.asyncio
    async def test_list_models(self, ollama_provider):
        """Test listing available models."""
        mock_response = {
            "models": [
                {"name": "llama2:latest", "size": 3826793677},
                {"name": "codellama:latest", "size": 3826793677},
            ]
        }

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_resp = AsyncMock()
            mock_resp.json = AsyncMock(return_value=mock_response)
            mock_resp.status = 200
            mock_get.return_value.__aenter__.return_value = mock_resp

            models = await ollama_provider.list_models()

            assert len(models) == 2
            assert "llama2:latest" in models
            assert "codellama:latest" in models

    @pytest.mark.asyncio
    async def test_complete(self, ollama_provider):
        """Test text completion."""
        messages = [Message(role="user", content="Hello")]

        mock_response = {
            "message": {"content": "Hello! How can I help you today?"},
            "model": "llama2:latest",
            "done": True,
        }

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_resp = AsyncMock()
            mock_resp.json = AsyncMock(return_value=mock_response)
            mock_resp.status = 200
            mock_post.return_value.__aenter__.return_value = mock_resp

            response = await ollama_provider.complete(messages, model="llama2:latest")

            assert isinstance(response, Response)
            assert response.content == "Hello! How can I help you today?"
            assert response.model == "llama2:latest"

    @pytest.mark.asyncio
    async def test_stream_complete(self, ollama_provider):
        """Test streaming completion."""
        messages = [Message(role="user", content="Hello")]

        # Mock streaming response
        mock_chunks = [
            '{"message":{"content":"Hello"},"done":false}\n',
            '{"message":{"content":" there"},"done":false}\n',
            '{"message":{"content":"!"},"done":true}\n',
        ]

        async def mock_stream():
            for chunk in mock_chunks:
                yield chunk.encode()

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_resp = AsyncMock()
            mock_resp.content.iter_chunked = mock_stream
            mock_resp.status = 200
            mock_post.return_value.__aenter__.return_value = mock_resp

            chunks = []
            async for chunk in ollama_provider.stream_complete(messages, model="llama2:latest"):
                chunks.append(chunk)

            assert len(chunks) == 3
            assert chunks == ["Hello", " there", "!"]

    @pytest.mark.asyncio
    async def test_health_check(self, ollama_provider):
        """Test health check."""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_get.return_value.__aenter__.return_value = mock_resp

            health = await ollama_provider.health_check()
            assert health is True

        # Test unhealthy response
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_get.side_effect = aiohttp.ClientError("Connection failed")

            health = await ollama_provider.health_check()
            assert health is False


@pytest.mark.unit
class TestOpenAIProvider:
    """Test OpenAI provider functionality."""

    @pytest.fixture
    def openai_provider(self):
        """Create OpenAI provider for testing."""
        return OpenAIProvider(api_key="test-key")

    @pytest.mark.asyncio
    async def test_list_models(self, openai_provider):
        """Test listing available models."""
        with patch.object(openai_provider, "_client") as mock_client:
            mock_models = AsyncMock()
            mock_models.data = [
                MagicMock(id="gpt-4", object="model"),
                MagicMock(id="gpt-3.5-turbo", object="model"),
            ]
            mock_client.models.list = AsyncMock(return_value=mock_models)

            models = await openai_provider.list_models()

            assert len(models) == 2
            assert "gpt-4" in models
            assert "gpt-3.5-turbo" in models

    @pytest.mark.asyncio
    async def test_complete(self, openai_provider):
        """Test text completion."""
        messages = [Message(role="user", content="Hello")]

        with patch.object(openai_provider, "_client") as mock_client:
            mock_completion = MagicMock()
            mock_completion.choices = [
                MagicMock(message=MagicMock(content="Hello! How can I help?"))
            ]
            mock_completion.model = "gpt-4"

            mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

            response = await openai_provider.complete(messages, model="gpt-4")

            assert isinstance(response, Response)
            assert response.content == "Hello! How can I help?"
            assert response.model == "gpt-4"

    @pytest.mark.asyncio
    async def test_stream_complete(self, openai_provider):
        """Test streaming completion."""
        messages = [Message(role="user", content="Hello")]

        # Mock streaming chunks
        mock_chunks = [
            MagicMock(choices=[MagicMock(delta=MagicMock(content="Hello"))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content=" there"))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content="!"))]),
        ]

        async def mock_stream():
            for chunk in mock_chunks:
                yield chunk

        with patch.object(openai_provider, "_client") as mock_client:
            mock_client.chat.completions.create = mock_stream

            chunks = []
            async for chunk in openai_provider.stream_complete(messages, model="gpt-4"):
                chunks.append(chunk)

            assert chunks == ["Hello", " there", "!"]


@pytest.mark.unit
class TestLMStudioProvider:
    """Test LM Studio provider functionality."""

    @pytest.fixture
    def lmstudio_provider(self):
        """Create LM Studio provider for testing."""
        return LMStudioProvider(host="http://localhost:1234")

    @pytest.mark.asyncio
    async def test_list_models(self, lmstudio_provider):
        """Test listing available models."""
        mock_response = {
            "data": [
                {"id": "local-model", "object": "model"},
                {"id": "another-model", "object": "model"},
            ]
        }

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_resp = AsyncMock()
            mock_resp.json = AsyncMock(return_value=mock_response)
            mock_resp.status = 200
            mock_get.return_value.__aenter__.return_value = mock_resp

            models = await lmstudio_provider.list_models()

            assert len(models) == 2
            assert "local-model" in models
            assert "another-model" in models

    @pytest.mark.asyncio
    async def test_complete(self, lmstudio_provider):
        """Test text completion."""
        messages = [Message(role="user", content="Hello")]

        mock_response = {
            "choices": [{"message": {"content": "Hello! How can I assist you?"}}],
            "model": "local-model",
        }

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_resp = AsyncMock()
            mock_resp.json = AsyncMock(return_value=mock_response)
            mock_resp.status = 200
            mock_post.return_value.__aenter__.return_value = mock_resp

            response = await lmstudio_provider.complete(messages, model="local-model")

            assert isinstance(response, Response)
            assert response.content == "Hello! How can I assist you?"
            assert response.model == "local-model"

    @pytest.mark.asyncio
    async def test_health_check(self, lmstudio_provider):
        """Test health check."""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_get.return_value.__aenter__.return_value = mock_resp

            health = await lmstudio_provider.health_check()
            assert health is True
