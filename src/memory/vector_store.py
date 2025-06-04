"""Vector-based memory storage using ChromaDB."""

import json
import time
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
            try:
                collection = self.client.get_or_create_collection(
                    name=collection_name, metadata={"hnsw:space": "cosine"}
                )
                self.collections[memory_type] = collection
                logger.debug(f"Created/found collection for {memory_type.value}: {collection.name}")
            except Exception as e:
                logger.error(f"Failed to create collection for {memory_type.value}: {e}")
                raise

        # Search result cache with TTL
        self._search_cache = {}
        self._cache_ttl = 60  # 60 seconds TTL for search results

        logger.info(f"Initialized vector memory store at {persist_directory}")
        logger.debug(f"Available collections: {list(self.collections.keys())}")

    async def store(self, memory: Memory) -> str:
        """Store a memory in the vector database."""
        try:
            # Generate ID if not provided
            if not memory.id:
                memory.id = str(uuid.uuid4())

            # Get appropriate collection by finding matching memory type
            collection = None
            for mem_type, coll in self.collections.items():
                if mem_type.value == memory.memory_type.value:
                    collection = coll
                    break

            if collection is None:
                raise ValueError(f"No collection found for memory type: {memory.memory_type}")

            logger.debug(f"Using collection for {memory.memory_type.value}: {collection.name}")

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

            # Invalidate search cache when new memory is added
            self._search_cache.clear()

            return memory.id

        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            import traceback

            logger.debug(f"Store memory error traceback: {traceback.format_exc()}")
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

    def _get_cache_key(
        self, query: str, memory_type: MemoryType | None, limit: int, threshold: float
    ) -> str:
        """Generate cache key for search parameters."""
        type_str = memory_type.value if memory_type else "all"
        return f"{query}:{type_str}:{limit}:{threshold}"

    def _is_cache_valid(self, cache_entry: dict) -> bool:
        """Check if cache entry is still valid."""
        return time.time() - cache_entry["timestamp"] < self._cache_ttl

    def _clear_expired_cache(self):
        """Clear expired cache entries."""
        current_time = time.time()
        expired_keys = [
            key
            for key, entry in self._search_cache.items()
            if current_time - entry["timestamp"] >= self._cache_ttl
        ]
        for key in expired_keys:
            del self._search_cache[key]

    async def search(
        self,
        query: str,
        memory_type: MemoryType | None = None,
        limit: int = 10,
        threshold: float = 0.7,
    ) -> list[Memory]:
        """Search memories by semantic similarity with caching."""
        try:
            logger.debug(
                f"Starting search for query: '{query}', type: {memory_type}, limit: {limit}, threshold: {threshold}"
            )

            # Generate cache key
            cache_key = self._get_cache_key(query, memory_type, limit, threshold)
            logger.debug(f"Cache key: {cache_key}")

            # Check cache first
            if cache_key in self._search_cache:
                cache_entry = self._search_cache[cache_key]
                if self._is_cache_valid(cache_entry):
                    logger.debug(f"Cache hit for search query: {query[:50]}...")
                    return cache_entry["results"]
                else:
                    logger.debug("Cache entry expired")
            else:
                logger.debug("No cache entry found")

            # Clear expired cache entries periodically
            self._clear_expired_cache()

            start_time = time.time()
            memories = []

            # Determine which collections to search
            if memory_type:
                # Find collection by value comparison
                collections_to_search = []
                for mem_type, collection in self.collections.items():
                    if mem_type.value == memory_type.value:
                        collections_to_search = [(mem_type, collection)]
                        break
                if not collections_to_search:
                    logger.warning(f"No collection found for memory type: {memory_type}")
                    return []
            else:
                collections_to_search = list(self.collections.items())

            # Use a larger initial search to get better results, then filter
            search_limit = min(limit * 3, 50)  # Search more, then filter for quality

            # Search each collection in parallel-like manner
            all_results = []
            for mem_type, collection in collections_to_search:
                try:
                    # Check if collection has any documents
                    collection_count = collection.count()
                    if collection_count == 0:
                        logger.debug(f"Collection {mem_type} is empty, skipping search")
                        continue

                    results = collection.query(
                        query_texts=[query],
                        n_results=min(
                            search_limit, collection_count
                        ),  # Don't query more than exist
                        include=["documents", "metadatas", "distances", "embeddings"],
                    )

                    # Process results for this collection
                    if (
                        results
                        and results.get("distances")
                        and len(results["distances"]) > 0
                        and results["distances"][0]
                        and len(results["distances"][0]) > 0
                    ):

                        logger.debug(
                            f"Processing {len(results['distances'][0])} results for {mem_type.value}"
                        )

                        for i, distance in enumerate(results["distances"][0]):
                            try:
                                # ChromaDB uses cosine distance, convert to similarity
                                # Handle potential numpy array or scalar
                                if hasattr(distance, "item"):  # numpy scalar
                                    distance_val = distance.item()
                                elif (
                                    hasattr(distance, "__len__") and len(distance) == 1
                                ):  # single-element array
                                    distance_val = distance[0]
                                else:
                                    distance_val = float(distance)

                                similarity = 1 - distance_val
                                logger.debug(
                                    f"  Result {i}: similarity={similarity:.3f}, threshold={threshold}"
                                )

                                if similarity >= threshold:
                                    logger.debug(
                                        f"  Including result {i} (similarity {similarity:.3f} >= {threshold})"
                                    )
                                    try:
                                        # Handle embeddings properly
                                        embedding = None
                                        if (
                                            results.get("embeddings")
                                            and len(results["embeddings"]) > 0
                                            and len(results["embeddings"][0]) > i
                                        ):
                                            embedding = results["embeddings"][0][i]

                                        memory = self._build_memory(
                                            id=results["ids"][0][i],
                                            document=results["documents"][0][i],
                                            embedding=embedding,
                                            metadata=results["metadatas"][0][i],
                                            memory_type=mem_type,
                                        )
                                        # Add similarity score for better sorting
                                        memory._search_similarity = similarity
                                        all_results.append(memory)
                                    except Exception as mem_error:
                                        logger.error(f"Error building memory: {mem_error}")
                                        continue
                                else:
                                    logger.debug(
                                        f"  Skipping result {i} (similarity {similarity:.3f} < {threshold})"
                                    )
                            except (ValueError, TypeError, IndexError) as dist_error:
                                logger.debug(
                                    f"Error processing distance value {distance}: {dist_error}"
                                )
                                continue
                    else:
                        logger.debug(f"No valid results from {mem_type.value} collection")

                except Exception as collection_error:
                    logger.warning(f"Error searching collection {mem_type}: {collection_error}")
                    continue

            # Advanced sorting: combine importance, recency, and similarity
            all_results.sort(
                key=lambda m: (
                    m._search_similarity * 0.4  # Semantic similarity weight
                    + m.importance * 0.4  # Importance weight
                    + (m.accessed_at.timestamp() / 1e9) * 0.2  # Recency weight (normalized)
                ),
                reverse=True,
            )

            # Remove temporary similarity attribute and limit results
            memories = all_results[:limit]
            for memory in memories:
                if hasattr(memory, "_search_similarity"):
                    delattr(memory, "_search_similarity")

            # Cache the results
            self._search_cache[cache_key] = {"results": memories, "timestamp": time.time()}

            search_time = time.time() - start_time
            logger.debug(f"Search completed in {search_time:.3f}s, found {len(memories)} memories")

            return memories

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

            # Handle embeddings properly to avoid array evaluation errors
            embeddings_param = None
            if memory.embedding is not None:
                # Check if embedding has values to avoid boolean evaluation of arrays
                if hasattr(memory.embedding, "__len__") and len(memory.embedding) > 0:
                    embeddings_param = [memory.embedding]
                elif memory.embedding:  # For non-array embeddings
                    embeddings_param = [memory.embedding]

            collection.update(
                ids=[memory.id],
                documents=[memory.content],
                embeddings=embeddings_param,
                metadatas=[metadata],
            )

            logger.debug(f"Updated memory {memory.id}")

            # Invalidate search cache when memory is updated
            self._search_cache.clear()

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

            # Invalidate search cache when memory is deleted
            if deleted:
                self._search_cache.clear()

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
