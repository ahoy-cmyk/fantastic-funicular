"""Memory system for Neuromancer."""

from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol


class MemoryType(Enum):
    """Types of memory storage."""

    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


@dataclass
class Memory:
    """Represents a memory unit."""

    id: str
    content: str
    embedding: list[float] | None = None
    metadata: dict[str, Any] = None
    memory_type: MemoryType = MemoryType.SHORT_TERM
    created_at: datetime = None
    accessed_at: datetime = None
    importance: float = 0.5

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.accessed_at is None:
            self.accessed_at = self.created_at
        if self.metadata is None:
            self.metadata = {}


class MemoryStore(Protocol):
    """Protocol for memory storage backends."""

    @abstractmethod
    async def store(self, memory: Memory) -> str:
        """Store a memory and return its ID."""
        ...

    @abstractmethod
    async def retrieve(self, memory_id: str) -> Memory | None:
        """Retrieve a memory by ID."""
        ...

    @abstractmethod
    async def search(
        self,
        query: str,
        memory_type: MemoryType | None = None,
        limit: int = 10,
        threshold: float = 0.7,
    ) -> list[Memory]:
        """Search memories by semantic similarity."""
        ...

    @abstractmethod
    async def update(self, memory: Memory) -> bool:
        """Update an existing memory."""
        ...

    @abstractmethod
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        ...

    @abstractmethod
    async def clear(self, memory_type: MemoryType | None = None) -> int:
        """Clear memories, optionally filtered by type."""
        ...
