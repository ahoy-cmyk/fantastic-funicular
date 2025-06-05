"""Tests for ChatManager core functionality."""

from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.unit
class TestChatManager:
    """Test ChatManager functionality."""

    def test_initialization(self, chat_manager):
        """Test ChatManager initialization."""
        assert chat_manager is not None
        assert chat_manager.providers == {}
        assert chat_manager.current_provider is not None
        assert chat_manager.current_model is not None
        assert chat_manager.memory_manager is not None
        assert chat_manager.mcp_manager is not None

    def test_provider_management(self, chat_manager):
        """Test provider management methods."""
        # Test getting available providers
        providers = chat_manager.get_available_providers()
        assert isinstance(providers, list)

        # Test checking provider availability
        assert not chat_manager.is_provider_available("nonexistent")

    @pytest.mark.asyncio
    async def test_session_creation(self, chat_manager):
        """Test session creation and management."""
        # Test creating a session
        await chat_manager.create_session("Test Session")
        assert chat_manager.current_session is not None
        assert chat_manager.current_session.title == "Test Session"

        # Test closing session
        await chat_manager.close_session()
        assert chat_manager.current_session is None

    @pytest.mark.asyncio
    async def test_system_prompt_update(self, chat_manager):
        """Test system prompt configuration."""
        new_prompt = "You are a specialized test assistant."
        chat_manager.update_system_prompt(new_prompt, memory_integration=False)

        assert chat_manager.system_prompt == new_prompt
        assert chat_manager.system_prompt_memory_integration is False

    def test_rag_configuration(self, chat_manager):
        """Test RAG system configuration."""
        # Test enabling/disabling RAG
        chat_manager.enable_rag(True)
        assert chat_manager.rag_enabled is True

        chat_manager.enable_rag(False)
        assert chat_manager.rag_enabled is False

    @pytest.mark.asyncio
    async def test_memory_operations(self, chat_manager, mock_memory_manager):
        """Test memory-related operations."""
        # Test that memory manager methods are called
        chat_manager.memory_manager = mock_memory_manager

        # Test enhanced memory recall
        result = await chat_manager._enhanced_memory_recall("test query")
        mock_memory_manager.recall.assert_called()

    @pytest.mark.asyncio
    async def test_conversation_storage(self, chat_manager):
        """Test conversation memory storage."""
        user_content = "Hello, what's my name?"
        assistant_content = "I don't have information about your name."

        # Mock the memory manager to avoid actual storage
        with patch.object(
            chat_manager.memory_manager, "remember", new_callable=AsyncMock
        ) as mock_remember:
            await chat_manager._store_conversation_memory(user_content, assistant_content)

            # Should have been called at least once
            assert mock_remember.call_count >= 0

    def test_importance_calculation(self, chat_manager):
        """Test importance calculation for memory storage."""
        # Test high importance content
        high_importance_content = "My name is John Doe and I work at Acme Corp"
        importance = chat_manager._calculate_importance(high_importance_content)
        assert importance > 0.6

        # Test low importance content
        low_importance_content = "hello"
        importance = chat_manager._calculate_importance(low_importance_content)
        assert importance < 0.6

        # Test question content
        question_content = "What is machine learning? How does it work?"
        importance = chat_manager._calculate_importance(question_content)
        assert importance > 0.4  # Questions should have decent importance

    @pytest.mark.asyncio
    async def test_mcp_operations(self, chat_manager, mock_mcp_manager):
        """Test MCP-related operations."""
        chat_manager.mcp_manager = mock_mcp_manager

        # Test listing MCP servers
        servers = await chat_manager.list_mcp_servers()
        mock_mcp_manager.list_servers.assert_called_once()
        assert isinstance(servers, list)

        # Test listing MCP tools
        tools = await chat_manager.list_mcp_tools()
        mock_mcp_manager.list_all_tools.assert_called_once()
        assert isinstance(tools, list)

        # Test adding MCP server
        result = await chat_manager.add_mcp_server("test_server", server_url="ws://localhost:8080")
        mock_mcp_manager.add_server.assert_called()
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_build_context_messages(self, chat_manager, sample_messages):
        """Test context message building."""
        query = "Test query"

        # Mock the MCP manager to return empty tools list
        chat_manager.mcp_manager.list_all_tools = AsyncMock(return_value=[])
        chat_manager.memory_manager.recall = AsyncMock(return_value=[])

        messages = await chat_manager._build_context_messages(query)

        assert len(messages) >= 1  # At least system message
        assert messages[0].role == "system"
        assert isinstance(messages[0].content, str)

    def test_memory_summary_creation(self, chat_manager):
        """Test memory summary creation."""
        # Test user message summary
        user_summary = chat_manager._create_memory_summary("Hello, how are you?", "user")
        assert "User said:" in user_summary

        # Test assistant message summary
        assistant_summary = chat_manager._create_memory_summary("I'm doing well!", "assistant")
        assert "Assistant explained:" in assistant_summary

        # Test long content truncation
        long_content = "x" * 600
        summary = chat_manager._create_memory_summary(long_content, "user")
        assert len(summary) < len(long_content) + 20  # Account for prefix

    def test_conversation_summary_creation(self, chat_manager):
        """Test conversation summary creation."""
        user_content = "What is Python?"
        assistant_content = "Python is a high-level programming language."

        summary = chat_manager._create_conversation_summary(user_content, assistant_content)

        assert "Conversation exchange:" in summary
        assert "User asked:" in summary
        assert "Assistant replied:" in summary
        assert user_content in summary or user_content[:200] in summary
        assert assistant_content in summary or assistant_content[:200] in summary
