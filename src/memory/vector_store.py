"""Vector-based memory storage using ChromaDB."""

import json
import uuid
from datetime import datetime
from typing import Any

import chromadb
from chromadb.config import Settings

from src.memory import Memory, MemoryStore, MemoryType
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class VectorMemoryStore(MemoryStore):
    """Vector database implementation of memory storage using ChromaDB."""

    def __init__(self, persist_directory: str = "./data/chromadb"):
        """Initialize vector memory store.

        Args:
            persist_directory: Directory to persist ChromaDB data
        """
        self.persist_directory = persist_directory

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory, settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )

        # Create collections for different memory types
        self.collections = {}
        for memory_type in MemoryType:
            collection_name = f"neuromancer_{memory_type.value}"
            self.collections[memory_type] = self.client.get_or_create_collection(
                name=collection_name, metadata={"hnsw:space": "cosine"}
            )

        logger.info(f"Initialized vector memory store at {persist_directory}")

    async def store(self, memory: Memory) -> str:
        """Store a memory in the vector database."""
        try:
            # Generate ID if not provided
            if not memory.id:
                memory.id = str(uuid.uuid4())

            # Get appropriate collection
            collection = self.collections[memory.memory_type]

            # Prepare metadata (convert lists to strings for ChromaDB)
            metadata = {
                "created_at": memory.created_at.isoformat(),
                "accessed_at": memory.accessed_at.isoformat(),
                "importance": memory.importance,
                "memory_type": memory.memory_type.value,
            }

            # Add custom metadata, converting lists to JSON strings
            for key, value in (memory.metadata or {}).items():
                if isinstance(value, list):
                    metadata[key] = json.dumps(value)
                else:
                    metadata[key] = value

            # Store in ChromaDB
            collection.add(
                ids=[memory.id],
                documents=[memory.content],
                embeddings=[memory.embedding] if memory.embedding else None,
                metadatas=[metadata],
            )

            logger.debug(f"Stored memory {memory.id} of type {memory.memory_type.value}")
            return memory.id

        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            raise

    async def retrieve(self, memory_id: str) -> Memory | None:
        """Retrieve a memory by ID."""
        try:
            # Search across all collections
            for memory_type, collection in self.collections.items():
                result = collection.get(ids=[memory_id])

                if result["ids"]:
                    return self._build_memory(
                        id=result["ids"][0],
                        document=result["documents"][0],
                        embedding=result["embeddings"][0] if result["embeddings"] else None,
                        metadata=result["metadatas"][0],
                        memory_type=memory_type,
                    )

            return None

        except Exception as e:
            logger.error(f"Failed to retrieve memory {memory_id}: {e}")
            return None

    async def search(
        self,
        query: str,
        memory_type: MemoryType | None = None,
        limit: int = 10,
        threshold: float = 0.7,
    ) -> list[Memory]:
        """Search memories by semantic similarity."""
        try:
            memories = []

            # Determine which collections to search
            collections_to_search = (
                [(memory_type, self.collections[memory_type])]
                if memory_type
                else self.collections.items()
            )

            # Search each collection
            for mem_type, collection in collections_to_search:
                results = collection.query(query_texts=[query], n_results=limit)

                # Filter by threshold and build Memory objects
                for i, distance in enumerate(results["distances"][0]):
                    # ChromaDB uses cosine distance, convert to similarity
                    similarity = 1 - distance

                    if similarity >= threshold:
                        memory = self._build_memory(
                            id=results["ids"][0][i],
                            document=results["documents"][0][i],
                            embedding=(
                                results["embeddings"][0][i] if results["embeddings"] else None
                            ),
                            metadata=results["metadatas"][0][i],
                            memory_type=mem_type,
                        )
                        memories.append(memory)

            # Sort by importance and recency
            memories.sort(key=lambda m: (m.importance, m.accessed_at.timestamp()), reverse=True)

            return memories[:limit]

        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            return []

    async def update(self, memory: Memory) -> bool:
        """Update an existing memory."""
        try:
            # Update accessed time
            memory.accessed_at = datetime.now()

            # Get collection
            collection = self.collections[memory.memory_type]

            # Update in ChromaDB
            metadata = {
                "created_at": memory.created_at.isoformat(),
                "accessed_at": memory.accessed_at.isoformat(),
                "importance": memory.importance,
                "memory_type": memory.memory_type.value,
            }

            # Add custom metadata, converting lists to JSON strings
            for key, value in (memory.metadata or {}).items():
                if isinstance(value, list):
                    metadata[key] = json.dumps(value)
                else:
                    metadata[key] = value

            collection.update(
                ids=[memory.id],
                documents=[memory.content],
                embeddings=[memory.embedding] if memory.embedding else None,
                metadatas=[metadata],
            )

            logger.debug(f"Updated memory {memory.id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update memory {memory.id}: {e}")
            return False

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        try:
            # Try to delete from all collections
            deleted = False
            for collection in self.collections.values():
                try:
                    collection.delete(ids=[memory_id])
                    deleted = True
                except Exception:
                    pass

            return deleted

        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False

    async def clear(self, memory_type: MemoryType | None = None) -> int:
        """Clear memories, optionally filtered by type."""
        try:
            count = 0

            if memory_type:
                # Clear specific collection
                collection = self.collections[memory_type]
                count = collection.count()
                self.client.delete_collection(f"neuromancer_{memory_type.value}")
                # Recreate empty collection
                self.collections[memory_type] = self.client.create_collection(
                    name=f"neuromancer_{memory_type.value}", metadata={"hnsw:space": "cosine"}
                )
            else:
                # Clear all collections
                for mem_type in MemoryType:
                    collection = self.collections[mem_type]
                    count += collection.count()
                    self.client.delete_collection(f"neuromancer_{mem_type.value}")
                    # Recreate empty collection
                    self.collections[mem_type] = self.client.create_collection(
                        name=f"neuromancer_{mem_type.value}", metadata={"hnsw:space": "cosine"}
                    )

            logger.info(f"Cleared {count} memories")
            return count

        except Exception as e:
            logger.error(f"Failed to clear memories: {e}")
            return 0

    def _build_memory(
        self,
        id: str,
        document: str,
        embedding: list[float] | None,
        metadata: dict[str, Any],
        memory_type: MemoryType,
    ) -> Memory:
        """Build a Memory object from ChromaDB results."""
        # Extract and parse metadata
        parsed_metadata = {}
        for k, v in metadata.items():
            if k not in ["created_at", "accessed_at", "importance", "memory_type"]:
                # Try to parse JSON strings back to lists
                if isinstance(v, str) and v.startswith("["):
                    try:
                        parsed_metadata[k] = json.loads(v)
                    except json.JSONDecodeError:
                        parsed_metadata[k] = v
                else:
                    parsed_metadata[k] = v

        return Memory(
            id=id,
            content=document,
            embedding=embedding,
            memory_type=memory_type,
            created_at=datetime.fromisoformat(
                metadata.get("created_at", datetime.now().isoformat())
            ),
            accessed_at=datetime.fromisoformat(
                metadata.get("accessed_at", datetime.now().isoformat())
            ),
            importance=metadata.get("importance", 0.5),
            metadata=parsed_metadata,
        )
