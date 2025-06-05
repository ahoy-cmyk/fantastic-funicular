"""Safe memory operations with user-friendly error handling."""

from collections.abc import Callable
from typing import Any

from src.memory import Memory, MemoryType
from src.memory.manager import MemoryManager
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class SafeMemoryManager:
    """Wrapper around MemoryManager with safe error handling."""

    def __init__(self, memory_manager: MemoryManager, error_callback: Callable | None = None):
        """Initialize safe memory manager.

        Args:
            memory_manager: The underlying memory manager
            error_callback: Function to call on errors (for showing popups)
        """
        self.memory_manager = memory_manager
        self.error_callback = error_callback or self._default_error_handler

    def _default_error_handler(self, operation: str, error: str):
        """Default error handler that just logs."""
        logger.error(f"Memory operation '{operation}' failed: {error}")

    async def safe_remember(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.SHORT_TERM,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
        auto_classify: bool = False,
    ) -> str | None:
        """Safely store a memory with error handling."""
        try:
            if not content or not content.strip():
                self.error_callback("Store Memory", "Cannot store empty content")
                return None

            memory_id = await self.memory_manager.remember(
                content=content,
                memory_type=memory_type,
                importance=importance,
                metadata=metadata,
                auto_classify=auto_classify,
            )

            if memory_id:
                logger.info(f"Successfully stored memory: {memory_id[:8]}...")
                return memory_id
            else:
                self.error_callback("Store Memory", "Failed to store memory - unknown error")
                return None

        except Exception as e:
            error_msg = f"Memory storage failed: {str(e)}"
            self.error_callback("Store Memory", error_msg)
            return None

    async def safe_intelligent_remember(
        self,
        content: str,
        conversation_context: list[str] = None,
    ) -> list[str]:
        """Safely use intelligent memory analysis with error handling."""
        try:
            if not content or not content.strip():
                logger.debug("Empty content provided for intelligent memory analysis")
                return []

            memory_ids = await self.memory_manager.intelligent_remember(
                content=content, conversation_context=conversation_context
            )

            if memory_ids:
                logger.info(f"Intelligent analysis created {len(memory_ids)} memories")

            return memory_ids

        except Exception as e:
            error_msg = f"Intelligent memory analysis failed: {str(e)}"
            self.error_callback("Intelligent Memory Analysis", error_msg)
            return []

    async def safe_recall(
        self,
        query: str,
        memory_types: list[MemoryType] | None = None,
        limit: int = 10,
        threshold: float = 0.7,
    ) -> list[Memory]:
        """Safely recall memories with error handling."""
        try:
            if not query or not query.strip():
                logger.info("Empty query provided for memory recall")
                return []

            memories = await self.memory_manager.recall(
                query=query, memory_types=memory_types, limit=limit, threshold=threshold
            )

            logger.info(f"Successfully recalled {len(memories)} memories")
            return memories

        except Exception as e:
            error_msg = f"Memory recall failed: {str(e)}"
            self.error_callback("Recall Memory", error_msg)
            return []

    async def safe_get_all_memories(
        self,
        memory_types: list[MemoryType] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Memory]:
        """Safely get all memories with error handling using efficient method."""
        try:
            if memory_types and len(memory_types) == 1:
                # Single memory type - use efficient direct method
                memories = await self.memory_manager.get_all_memories(
                    memory_type=memory_types[0], limit=limit, offset=offset
                )
            elif not memory_types:
                # All memory types - use efficient direct method
                memories = await self.memory_manager.get_all_memories(
                    memory_type=None, limit=limit, offset=offset
                )
            else:
                # Multiple specific types - need to combine results
                all_memories = []
                for memory_type in memory_types:
                    type_memories = await self.memory_manager.get_all_memories(
                        memory_type=memory_type, limit=limit, offset=0  # Get all for sorting
                    )
                    all_memories.extend(type_memories)

                # Sort and paginate
                all_memories.sort(key=lambda m: m.created_at, reverse=True)
                memories = all_memories[offset : offset + limit]

            logger.info(f"Successfully retrieved {len(memories)} memories efficiently")
            return memories

        except Exception as e:
            error_msg = f"Failed to get all memories: {str(e)}"
            self.error_callback("Get All Memories", error_msg)
            return []

    async def safe_forget(self, memory_id: str) -> bool:
        """Safely delete a memory with error handling."""
        try:
            if not memory_id or not memory_id.strip():
                self.error_callback("Delete Memory", "Invalid memory ID")
                return False

            success = await self.memory_manager.forget(memory_id)

            if success:
                logger.info(f"Successfully deleted memory: {memory_id[:8]}...")
            else:
                self.error_callback("Delete Memory", "Memory not found or deletion failed")

            return success

        except Exception as e:
            error_msg = f"Memory deletion failed: {str(e)}"
            self.error_callback("Delete Memory", error_msg)
            return False

    async def safe_clear_memories(self, memory_type: MemoryType | None = None) -> int:
        """Safely clear memories with error handling."""
        try:
            count = await self.memory_manager.clear_memories(memory_type)

            type_str = memory_type.value if memory_type else "all"
            logger.info(f"Successfully cleared {count} {type_str} memories")
            return count

        except Exception as e:
            error_msg = f"Memory clearing failed: {str(e)}"
            self.error_callback("Clear Memory", error_msg)
            return 0

    async def safe_get_stats(self) -> dict[str, Any]:
        """Safely get memory statistics with error handling."""
        try:
            stats = await self.memory_manager.get_memory_stats()
            logger.info("Successfully retrieved memory statistics")
            return stats

        except Exception as e:
            error_msg = f"Failed to get memory statistics: {str(e)}"
            self.error_callback("Memory Statistics", error_msg)
            return {"total": 0, "by_type": {}, "avg_importance": 0.0, "error": error_msg}

    async def safe_consolidate(self) -> bool:
        """Safely consolidate memories with error handling."""
        try:
            await self.memory_manager.consolidate()
            logger.info("Successfully consolidated memories")
            return True

        except Exception as e:
            error_msg = f"Memory consolidation failed: {str(e)}"
            self.error_callback("Memory Consolidation", error_msg)
            return False

    def is_healthy(self) -> bool:
        """Check if the memory system is healthy."""
        try:
            # Simple health check
            return hasattr(self.memory_manager, "store") and self.memory_manager.store is not None
        except Exception:
            return False


def create_safe_memory_manager(error_callback: Callable | None = None) -> SafeMemoryManager:
    """Create a safe memory manager with error handling.

    Args:
        error_callback: Function to call on errors (e.g., show popup)

    Returns:
        SafeMemoryManager instance
    """
    try:
        memory_manager = MemoryManager()
        return SafeMemoryManager(memory_manager, error_callback)
    except Exception as e:
        logger.error(f"Failed to create memory manager: {e}")
        if error_callback:
            error_callback("Memory System", f"Failed to initialize memory system: {str(e)}")
        raise
