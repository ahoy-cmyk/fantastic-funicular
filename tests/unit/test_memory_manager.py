"""Tests for MemoryManager functionality."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.memory import MemoryType
from src.memory.manager import MemoryManager


@pytest.mark.unit
class TestMemoryManager:
    """Test MemoryManager functionality."""

    @pytest.fixture
    def memory_manager(self):
        """Create a memory manager for testing."""
        with patch("src.memory.manager.VectorStore") as mock_vector_store:
            mock_store = AsyncMock()
            mock_vector_store.return_value = mock_store

            manager = MemoryManager()
            manager.vector_store = mock_store
            return manager

    @pytest.mark.asyncio
    async def test_remember(self, memory_manager):
        """Test storing memories."""
        content = "User's name is Alice"
        memory_type = MemoryType.LONG_TERM
        importance = 0.8

        # Mock the vector store
        memory_manager.vector_store.add_memory = AsyncMock(return_value=True)

        result = await memory_manager.remember(
            content=content, memory_type=memory_type, importance=importance
        )

        assert result is True
        memory_manager.vector_store.add_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_recall(self, memory_manager):
        """Test retrieving memories."""
        query = "What is the user's name?"

        # Mock memory data
        mock_memories = [
            MagicMock(
                content="User's name is Alice",
                memory_type=MemoryType.LONG_TERM,
                importance=0.8,
                created_at=datetime.now(),
                accessed_at=datetime.now(),
            )
        ]

        memory_manager.vector_store.search_similar = AsyncMock(return_value=mock_memories)

        memories = await memory_manager.recall(query=query, limit=5)

        assert len(memories) == 1
        assert memories[0].content == "User's name is Alice"
        memory_manager.vector_store.search_similar.assert_called_once()

    @pytest.mark.asyncio
    async def test_recall_with_filters(self, memory_manager):
        """Test retrieving memories with filters."""
        query = "user preferences"

        mock_memories = []
        memory_manager.vector_store.search_similar = AsyncMock(return_value=mock_memories)

        memories = await memory_manager.recall(
            query=query, memory_types=[MemoryType.LONG_TERM], importance_threshold=0.7, limit=10
        )

        assert isinstance(memories, list)
        memory_manager.vector_store.search_similar.assert_called_once()

    @pytest.mark.asyncio
    async def test_forget(self, memory_manager):
        """Test forgetting memories."""
        memory_id = "test-memory-id"

        memory_manager.vector_store.delete_memory = AsyncMock(return_value=True)

        result = await memory_manager.forget(memory_id)

        assert result is True
        memory_manager.vector_store.delete_memory.assert_called_once_with(memory_id)

    @pytest.mark.asyncio
    async def test_search(self, memory_manager):
        """Test searching memories."""
        query = "machine learning"

        mock_memories = [
            MagicMock(
                content="Machine learning is a subset of AI",
                memory_type=MemoryType.SEMANTIC,
                importance=0.6,
            )
        ]

        memory_manager.vector_store.search_similar = AsyncMock(return_value=mock_memories)

        results = await memory_manager.search(query=query, limit=5)

        assert len(results) == 1
        assert "machine learning" in results[0].content.lower()

    @pytest.mark.asyncio
    async def test_clear_all(self, memory_manager):
        """Test clearing all memories."""
        memory_manager.vector_store.clear_all = AsyncMock(return_value=True)

        result = await memory_manager.clear_all()

        assert result is True
        memory_manager.vector_store.clear_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_stats(self, memory_manager):
        """Test getting memory statistics."""
        mock_stats = {
            "total_memories": 150,
            "memory_types": {"short_term": 50, "long_term": 75, "semantic": 20, "episodic": 5},
            "average_importance": 0.65,
            "storage_size_mb": 2.3,
        }

        memory_manager.vector_store.get_stats = AsyncMock(return_value=mock_stats)

        stats = await memory_manager.get_stats()

        assert stats["total_memories"] == 150
        assert "memory_types" in stats
        assert stats["average_importance"] == 0.65

    @pytest.mark.asyncio
    async def test_consolidate_memories(self, memory_manager):
        """Test memory consolidation."""
        memory_manager.vector_store.consolidate = AsyncMock(
            return_value={"consolidated": 10, "removed": 5}
        )

        result = await memory_manager.consolidate_memories()

        assert "consolidated" in result
        assert "removed" in result
        memory_manager.vector_store.consolidate.assert_called_once()

    @pytest.mark.asyncio
    async def test_export_memories(self, memory_manager):
        """Test exporting memories."""
        mock_memories = [
            {
                "id": "mem1",
                "content": "Test memory 1",
                "type": "long_term",
                "importance": 0.8,
                "created_at": "2024-01-01T00:00:00Z",
            }
        ]

        memory_manager.vector_store.export_all = AsyncMock(return_value=mock_memories)

        exported = await memory_manager.export_memories()

        assert len(exported) == 1
        assert exported[0]["content"] == "Test memory 1"

    @pytest.mark.asyncio
    async def test_import_memories(self, memory_manager):
        """Test importing memories."""
        memories_data = [
            {
                "content": "Imported memory",
                "type": "semantic",
                "importance": 0.7,
                "metadata": {"source": "import"},
            }
        ]

        memory_manager.vector_store.import_memories = AsyncMock(return_value=True)

        result = await memory_manager.import_memories(memories_data)

        assert result is True
        memory_manager.vector_store.import_memories.assert_called_once()

    def test_memory_type_validation(self, memory_manager):
        """Test memory type validation."""
        # Valid memory types
        valid_types = [
            MemoryType.SHORT_TERM,
            MemoryType.LONG_TERM,
            MemoryType.SEMANTIC,
            MemoryType.EPISODIC,
        ]

        for memory_type in valid_types:
            assert isinstance(memory_type, MemoryType)

    @pytest.mark.asyncio
    async def test_error_handling(self, memory_manager):
        """Test error handling in memory operations."""
        # Test remember with vector store error
        memory_manager.vector_store.add_memory = AsyncMock(
            side_effect=Exception("Vector store error")
        )

        result = await memory_manager.remember("test content")
        assert result is False

        # Test recall with vector store error
        memory_manager.vector_store.search_similar = AsyncMock(
            side_effect=Exception("Search error")
        )

        memories = await memory_manager.recall("test query")
        assert memories == []
