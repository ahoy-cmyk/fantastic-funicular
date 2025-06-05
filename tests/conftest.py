"""Pytest configuration and shared fixtures."""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.core.chat_manager import ChatManager
from src.core.config import _config_manager
from src.mcp.manager import MCPManager
from src.memory.manager import MemoryManager


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_config():
    """Mock configuration manager with test settings."""
    original_config = _config_manager._config.copy()

    # Set test configuration
    test_config = {
        "providers": {
            "ollama_enabled": True,
            "ollama_host": "http://localhost:11434",
            "openai_enabled": False,
            "lmstudio_enabled": False,
        },
        "memory": {
            "enabled": True,
            "provider": "chroma",
            "collection_name": "test_memories",
        },
        "mcp": {
            "enabled": True,
            "auto_connect": False,
            "servers": {},
        },
        "system_prompt": "You are a test assistant.",
        "system_prompt_memory_integration": True,
    }

    _config_manager._config.update(test_config)

    yield _config_manager

    # Restore original configuration
    _config_manager._config = original_config


@pytest.fixture
def mock_memory_manager():
    """Mock memory manager for testing."""
    mock = AsyncMock(spec=MemoryManager)
    mock.remember = AsyncMock(return_value=True)
    mock.recall = AsyncMock(return_value=[])
    mock.search = AsyncMock(return_value=[])
    mock.forget = AsyncMock(return_value=True)
    mock.clear_all = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def mock_mcp_manager():
    """Mock MCP manager for testing."""
    mock = AsyncMock(spec=MCPManager)
    mock.add_server = AsyncMock(return_value=True)
    mock.remove_server = AsyncMock(return_value=True)
    mock.list_servers = AsyncMock(return_value=[])
    mock.list_all_tools = AsyncMock(return_value=[])
    mock.execute_tool = AsyncMock()
    mock.health_check_all = AsyncMock(return_value={})
    return mock


@pytest.fixture
def mock_llm_provider():
    """Mock LLM provider for testing."""
    from src.providers import LLMProvider, Response

    mock = AsyncMock(spec=LLMProvider)
    mock.list_models = AsyncMock(return_value=["test-model"])
    mock.complete = AsyncMock(return_value=Response(content="Test response", model="test-model"))

    async def mock_stream_complete(*args, **kwargs):
        """Mock streaming response."""
        for chunk in ["Test ", "streaming ", "response"]:
            yield chunk

    mock.stream_complete = mock_stream_complete
    mock.health_check = AsyncMock(return_value=True)
    return mock


@pytest.fixture
async def chat_manager(mock_config, mock_memory_manager, mock_mcp_manager):
    """Create a chat manager instance for testing."""
    # Temporarily disable provider initialization
    original_init = ChatManager._initialize_providers
    ChatManager._initialize_providers = lambda self: None

    # Create chat manager
    manager = ChatManager()

    # Replace managers with mocks
    manager.memory_manager = mock_memory_manager
    manager.mcp_manager = mock_mcp_manager

    yield manager

    # Restore original method
    ChatManager._initialize_providers = original_init

    # Cleanup
    if manager.current_session:
        await manager.close_session()


@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    from src.providers import Message

    return [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Hello, how are you?"),
        Message(role="assistant", content="I'm doing well, thank you for asking!"),
        Message(role="user", content="What's the weather like?"),
    ]


@pytest.fixture
def sample_memory_data():
    """Sample memory data for testing."""
    from src.memory import MemoryType

    return [
        {
            "content": "User's name is John Doe",
            "memory_type": MemoryType.LONG_TERM,
            "importance": 0.9,
            "metadata": {"type": "personal_info"},
        },
        {
            "content": "User prefers Python programming",
            "memory_type": MemoryType.LONG_TERM,
            "importance": 0.7,
            "metadata": {"type": "preference"},
        },
        {
            "content": "Discussed machine learning concepts",
            "memory_type": MemoryType.SEMANTIC,
            "importance": 0.6,
            "metadata": {"type": "conversation"},
        },
    ]


@pytest.fixture
def sample_mcp_tools():
    """Sample MCP tools for testing."""
    from src.mcp import MCPTool, ToolType

    return [
        MCPTool(
            name="test_tool",
            description="A test tool for demonstrations",
            parameters={"input": {"type": "string", "description": "Test input"}},
            tool_type=ToolType.FUNCTION,
            server="test_server",
        ),
        MCPTool(
            name="search_tool",
            description="Search for information",
            parameters={
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "description": "Number of results"},
            },
            tool_type=ToolType.RETRIEVAL,
            server="search_server",
        ),
    ]


# Test environment setup
def pytest_configure(config):
    """Configure pytest environment."""
    # Set test environment variables
    os.environ["TESTING"] = "1"
    os.environ["LOG_LEVEL"] = "DEBUG"

    # Disable GUI components during testing
    os.environ["KIVY_NO_CONSOLELOG"] = "1"
    os.environ["KIVY_LOG_LEVEL"] = "error"


def pytest_unconfigure(config):
    """Cleanup after tests."""
    # Clean up environment variables
    if "TESTING" in os.environ:
        del os.environ["TESTING"]
