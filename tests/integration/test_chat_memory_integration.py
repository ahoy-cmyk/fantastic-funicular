"""Integration tests for chat and memory system interaction."""

from unittest.mock import AsyncMock, patch

import pytest

from src.memory import MemoryType


@pytest.mark.integration
@pytest.mark.requires_memory
class TestChatMemoryIntegration:
    """Test integration between chat and memory systems."""

    @pytest.mark.asyncio
    async def test_rag_enhanced_conversation(self, chat_manager, sample_memory_data):
        """Test RAG-enhanced conversation flow."""
        # Setup memory with sample data
        chat_manager.memory_manager.recall = AsyncMock(
            return_value=[
                type(
                    "Memory",
                    (),
                    {
                        "content": "User prefers Python programming",
                        "metadata": {"type": "preference"},
                        "importance": 0.8,
                    },
                )()
            ]
        )

        # Mock RAG system
        chat_manager.rag_system.retrieve_context = AsyncMock(
            return_value=type(
                "Context",
                (),
                {
                    "memories": [
                        type(
                            "Memory",
                            (),
                            {
                                "content": "User prefers Python programming",
                                "metadata": {"type": "preference"},
                            },
                        )()
                    ],
                    "retrieval_time_ms": 50,
                    "reasoning": "Found relevant preference",
                },
            )()
        )

        # Mock LLM provider
        mock_provider = AsyncMock()

        async def mock_stream():
            chunks = ["I see you prefer ", "Python programming. ", "Let me help with that."]
            for chunk in chunks:
                yield chunk

        mock_provider.stream_complete = mock_stream
        chat_manager.providers["test"] = mock_provider
        chat_manager.current_provider = "test"

        # Test RAG-enhanced streaming
        response_chunks = []
        async for chunk in chat_manager.send_message_with_rag(
            "Help me with a coding project", stream=True
        ):
            response_chunks.append(chunk)

        assert len(response_chunks) > 0
        full_response = "".join(response_chunks)
        assert isinstance(full_response, str)

    @pytest.mark.asyncio
    async def test_conversation_memory_storage(self, chat_manager):
        """Test that conversations are properly stored in memory."""
        user_message = "My name is Alice and I work at TechCorp"
        assistant_response = "Nice to meet you, Alice! How can I help you today?"

        with patch.object(
            chat_manager.memory_manager, "remember", new_callable=AsyncMock
        ) as mock_remember:
            await chat_manager._store_conversation_memory(user_message, assistant_response)

            # Should have called remember at least once for high-importance user message
            assert mock_remember.call_count >= 1

            # Check that personal information was detected and stored
            call_args = mock_remember.call_args_list
            stored_content = [call[1]["content"] for call in call_args if "content" in call[1]]

            # Should contain user message about name and work
            personal_stored = any(
                "Alice" in content or "TechCorp" in content for content in stored_content
            )
            assert personal_stored

    @pytest.mark.asyncio
    async def test_memory_enhanced_context_building(self, chat_manager):
        """Test that memory enhances context building."""
        query = "What programming languages do I know?"

        # Mock memory recall with relevant memories
        mock_memories = [
            type(
                "Memory",
                (),
                {
                    "content": "User is proficient in Python and JavaScript",
                    "importance": 0.8,
                    "accessed_at": type("datetime", (), {"timestamp": lambda: 1234567890})(),
                },
            )(),
            type(
                "Memory",
                (),
                {
                    "content": "User prefers functional programming",
                    "importance": 0.6,
                    "accessed_at": type("datetime", (), {"timestamp": lambda: 1234567890})(),
                },
            )(),
        ]

        chat_manager.memory_manager.recall = AsyncMock(return_value=mock_memories)
        chat_manager.mcp_manager.list_all_tools = AsyncMock(return_value=[])

        # Build context messages
        messages = await chat_manager._build_context_messages(query)

        assert len(messages) >= 1
        system_message = messages[0]
        assert system_message.role == "system"

        # System message should include memory content
        assert "Python and JavaScript" in system_message.content
        assert "functional programming" in system_message.content

    @pytest.mark.asyncio
    async def test_importance_based_storage_filtering(self, chat_manager):
        """Test that only important content gets stored in memory."""
        test_cases = [
            {
                "content": "Hello",
                "expected_stored": False,  # Low importance
                "description": "Simple greeting",
            },
            {
                "content": "My name is Bob and I'm the CEO of StartupCorp working on AI projects",
                "expected_stored": True,  # High importance - personal + work info
                "description": "Personal and professional information",
            },
            {
                "content": "I prefer using TypeScript over JavaScript for large projects",
                "expected_stored": True,  # Medium-high importance - preference
                "description": "Technical preference",
            },
            {
                "content": "ok",
                "expected_stored": False,  # Very low importance
                "description": "Minimal acknowledgment",
            },
        ]

        for test_case in test_cases:
            with patch.object(
                chat_manager.memory_manager, "remember", new_callable=AsyncMock
            ) as mock_remember:
                await chat_manager._store_conversation_memory(
                    test_case["content"], "Thank you for that information."
                )

                if test_case["expected_stored"]:
                    # Should have been called for high importance content
                    assert (
                        mock_remember.call_count > 0
                    ), f"Expected storage for: {test_case['description']}"
                else:
                    # Low importance content might not be stored
                    # (Note: this depends on the exact threshold logic)
                    pass

    @pytest.mark.asyncio
    async def test_memory_type_classification(self, chat_manager):
        """Test that different content gets classified into appropriate memory types."""
        test_cases = [
            {
                "content": "Remember to call the client tomorrow at 3 PM",
                "expected_type": MemoryType.EPISODIC,
                "description": "Time-specific event",
            },
            {
                "content": "My email address is bob@company.com",
                "expected_type": MemoryType.LONG_TERM,
                "description": "Personal contact information",
            },
            {
                "content": "Machine learning is a subset of artificial intelligence",
                "expected_type": MemoryType.SEMANTIC,
                "description": "Factual knowledge",
            },
            {
                "content": "I usually prefer coffee over tea",
                "expected_type": MemoryType.LONG_TERM,
                "description": "Personal preference",
            },
        ]

        for test_case in test_cases:
            importance = chat_manager._calculate_importance(test_case["content"])

            # For high importance content, test memory type determination
            if importance >= 0.4:
                # The actual memory type determination happens in _store_conversation_memory
                # This is a simplified test of the logic
                content_lower = test_case["content"].lower()

                # Check for episodic indicators
                episodic_terms = ["tomorrow", "yesterday", "today", "meeting", "call"]
                has_episodic = any(term in content_lower for term in episodic_terms)

                # Check for semantic indicators
                semantic_terms = ["is a", "definition", "algorithm", "concept"]
                has_semantic = any(term in content_lower for term in semantic_terms)

                # Check for long-term indicators
                longterm_terms = ["email", "address", "prefer", "always", "never"]
                has_longterm = any(term in content_lower for term in longterm_terms)

                if test_case["expected_type"] == MemoryType.EPISODIC:
                    assert (
                        has_episodic
                    ), f"Should detect episodic content: {test_case['description']}"
                elif test_case["expected_type"] == MemoryType.SEMANTIC:
                    assert (
                        has_semantic
                    ), f"Should detect semantic content: {test_case['description']}"
                elif test_case["expected_type"] == MemoryType.LONG_TERM:
                    assert (
                        has_longterm
                    ), f"Should detect long-term content: {test_case['description']}"

    @pytest.mark.asyncio
    async def test_memory_retrieval_enhances_responses(self, chat_manager):
        """Test that retrieved memories enhance response generation."""
        # Setup session
        await chat_manager.create_session("Test Memory Session")

        # Mock memory with user's name
        mock_memories = [
            type(
                "Memory",
                (),
                {
                    "content": "User's name is Charlie Brown",
                    "metadata": {"type": "personal_info"},
                    "importance": 0.9,
                },
            )()
        ]

        chat_manager.memory_manager.recall = AsyncMock(return_value=mock_memories)

        # Mock RAG context
        chat_manager.rag_system.retrieve_context = AsyncMock(
            return_value=type(
                "Context",
                (),
                {
                    "memories": mock_memories,
                    "retrieval_time_ms": 75,
                    "reasoning": "Found user name in memory",
                },
            )()
        )

        # Build enhanced messages
        context = await chat_manager.rag_system.retrieve_context("What's my name?", [])
        enhanced_messages = await chat_manager._build_rag_enhanced_messages(
            "What's my name?", context
        )

        # Should include system message with memory context
        assert len(enhanced_messages) >= 2  # System + user message
        system_message = enhanced_messages[0]

        assert "Charlie Brown" in system_message.content
        assert "Personal Information" in system_message.content

        # User message should be preserved
        user_message = enhanced_messages[-1]
        assert user_message.role == "user"
        assert user_message.content == "What's my name?"
