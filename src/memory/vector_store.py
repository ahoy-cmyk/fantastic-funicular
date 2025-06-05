"""Vector-based memory storage using ChromaDB.

This module implements the storage backend for the Neuromancer memory system
using ChromaDB for vector similarity search. It provides persistent storage
with semantic search capabilities across different memory types.

Key Features:
    - Persistent vector storage with ChromaDB
    - Separate collections per memory type for organization
    - Semantic similarity search with configurable thresholds
    - Result caching for improved performance
    - Metadata preservation and serialization
    - Efficient bulk operations and pagination

Architecture:
    The VectorMemoryStore implements the MemoryStore protocol and uses
    ChromaDB as the backend. Each memory type gets its own collection
    for better organization and search performance.

Performance Optimizations:
    - Search result caching with 60-second TTL
    - Cosine similarity distance for semantic search
    - Batch operations where possible
    - Memory type filtering for targeted searches
    - Periodic cache cleanup

Storage Structure:
    Collections: neuromancer_{memory_type}
    - short_term: Temporary memories
    - long_term: Important persistent memories
    - episodic: Event-based memories
    - semantic: Factual knowledge

Technical Details:
    - Embedding dimension: Determined by EmbeddingGenerator
    - Distance metric: Cosine similarity
    - Persistence: Local filesystem (configurable)
    - Indexing: HNSW for fast approximate search

Example Usage:
    ```python
    store = VectorMemoryStore("./data/memories")

    # Store a memory
    memory = Memory(content="Hello world", embedding=[0.1, 0.2, ...])
    memory_id = await store.store(memory)

    # Search for similar memories
    results = await store.search("greeting", limit=5, threshold=0.7)
    ```
"""

import json
import logging
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
    """Vector database implementation of memory storage using ChromaDB.

    Provides persistent vector storage with semantic search capabilities.
    Implements the MemoryStore protocol for use by MemoryManager.

    Storage Organization:
        - One ChromaDB collection per memory type
        - Each collection has cosine similarity indexing
        - Persistent storage on local filesystem

    Search Strategy:
        - Converts text queries to embeddings
        - Searches relevant collections in parallel
        - Ranks by similarity, importance, and recency
        - Caches results for repeated queries

    Thread Safety:
        ChromaDB operations are thread-safe, but this class
        is designed for single-user access patterns.
    """

    def __init__(self, persist_directory: str = "./data/chromadb"):
        """Initialize vector memory store.

        Args:
            persist_directory: Directory to persist ChromaDB data.
                Will be created if it doesn't exist.

        Initialization:
            1. Create ChromaDB persistent client
            2. Create/connect to collections for each memory type
            3. Configure cosine similarity indexing
            4. Set up result caching

        Collection Naming:
            Collections are named "neuromancer_{memory_type}" to avoid
            conflicts with other applications using the same directory.

        Error Handling:
            Initialization failures raise exceptions as they indicate
            fundamental storage issues that must be resolved.
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
        """Store a memory in the vector database.

        Stores a memory with its embedding and metadata in the appropriate
        collection based on memory type.

        Args:
            memory: Memory object to store. Must have content and embedding.

        Returns:
            Memory ID (UUID) assigned to the stored memory

        Process:
            1. Generate ID if not provided
            2. Select appropriate collection by memory type
            3. Serialize metadata (convert lists to JSON)
            4. Store document, embedding, and metadata
            5. Clear search cache to ensure fresh results

        Metadata Handling:
            List values are serialized to JSON strings for ChromaDB compatibility.
            This is reversed during retrieval to restore original types.

        Error Handling:
            Storage failures are logged with full traceback and re-raised
            as they indicate serious storage issues.

        Example:
            ```python
            memory = Memory(
                content="User prefers dark mode",
                embedding=[0.1, 0.2, 0.3, ...],
                memory_type=MemoryType.LONG_TERM,
                importance=0.8
            )
            memory_id = await store.store(memory)
            ```
        """
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
        """Retrieve a memory by ID.

        Direct retrieval of a specific memory by its unique identifier.

        Args:
            memory_id: UUID of the memory to retrieve

        Returns:
            Memory object if found, None otherwise

        Process:
            Searches across all collections since memory type is unknown.
            Returns the first match found.

        Performance:
            Direct ID lookup is very fast (~1-5ms) as it doesn't require
            embedding comparison.
        """
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
        """Search memories by semantic similarity with caching.

        Core search method that finds memories similar to the query text.
        Uses ChromaDB's vector similarity search with result caching.

        Args:
            query: Natural language query to search for
            memory_type: Filter to specific memory type, None for all types
            limit: Maximum memories to return
            threshold: Minimum similarity score (0.0-1.0)

        Returns:
            List of Memory objects sorted by relevance

        Search Process:
            1. Check cache for identical query
            2. Determine collections to search
            3. Perform vector similarity search
            4. Filter by threshold and process results
            5. Sort by composite score
            6. Cache results

        Similarity Scoring:
            ChromaDB returns cosine distances which are converted to
            similarities: similarity = 1 - distance

        Ranking Algorithm:
            Composite score = similarity(40%) + importance(40%) + recency(20%)
            This balances relevance with importance and freshness.

        Performance:
            - Cached queries: ~1-5ms
            - Fresh searches: ~50-200ms depending on collection size
            - Parallel collection search for multi-type queries

        Error Handling:
            Collection errors are logged but don't stop the search.
            Returns empty list if all collections fail.

        Example:
            ```python
            # Search specific type
            memories = await store.search(
                "user preferences",
                memory_type=MemoryType.LONG_TERM,
                limit=5,
                threshold=0.6
            )

            # Search all types
            memories = await store.search("Python programming")
            ```
        """
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

                        if logger.isEnabledFor(logging.DEBUG):
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

                                if similarity >= threshold:
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
                                # Skip results below threshold
                            except (ValueError, TypeError, IndexError) as dist_error:
                                if logger.isEnabledFor(logging.DEBUG):
                                    logger.debug(
                                        f"Error processing distance value {distance}: {dist_error}"
                                    )
                                continue
                    elif logger.isEnabledFor(logging.DEBUG):
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

            # Cache the results (only if we have meaningful results to avoid cache pollution)
            if memories:
                self._search_cache[cache_key] = {"results": memories, "timestamp": time.time()}

            if logger.isEnabledFor(logging.DEBUG):
                search_time = time.time() - start_time
                logger.debug(
                    f"Search completed in {search_time:.3f}s, found {len(memories)} memories"
                )

            return memories

        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            return []

    async def update(self, memory: Memory) -> bool:
        """Update an existing memory.

        Modifies an existing memory in place, updating content, metadata,
        and access timestamp.

        Args:
            memory: Memory object with updated information

        Returns:
            True if update successful, False otherwise

        Process:
            1. Update access timestamp
            2. Serialize metadata
            3. Handle embeddings carefully to avoid array errors
            4. Update in ChromaDB
            5. Clear search cache

        Embedding Handling:
            Special care is taken with embeddings to avoid numpy array
            boolean evaluation errors that can occur with empty arrays.

        Cache Invalidation:
            Clears the entire search cache as updated memories may
            affect many different search results.
        """
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
        """Delete a memory by ID.

        Permanently removes a memory from storage.

        Args:
            memory_id: UUID of memory to delete

        Returns:
            True if any collection reported successful deletion

        Process:
            Attempts deletion from all collections since we don't
            know which one contains the memory.

        Note:
            ChromaDB delete operations are idempotent - deleting
            a non-existent ID doesn't raise an error.
        """
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
        """Clear memories, optionally filtered by type.

        Bulk deletion operation for memory management.

        Args:
            memory_type: Type to clear, or None to clear all memories

        Returns:
            Number of memories that were cleared

        Process:
            1. Count existing memories
            2. Delete entire collections
            3. Recreate empty collections with same settings

        Warning:
            This is a destructive operation that cannot be undone.
            Use with caution, especially when memory_type is None.

        Collection Recreation:
            Collections are recreated with the same metadata (cosine similarity)
            to maintain consistent behavior after clearing.
        """
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

    async def get_all_memories(
        self, memory_type: MemoryType | None = None, limit: int = 1000, offset: int = 0
    ) -> list[Memory]:
        """Get all memories efficiently without vector search.

        Direct access method that bypasses similarity search for listing operations.
        Much faster than search when you need all memories of a type.

        Args:
            memory_type: Specific memory type or None for all types
            limit: Maximum number of memories to return (default: 1000)
            offset: Number of memories to skip for pagination (default: 0)

        Returns:
            List of Memory objects sorted by creation time (newest first)

        Performance:
            ~10-50ms vs ~50-200ms for semantic search.
            Time scales linearly with collection size.

        Pagination:
            Supports offset-based pagination for large datasets:
            - offset=0, limit=100: first 100 memories
            - offset=100, limit=100: next 100 memories

        Error Handling:
            Collection errors are logged but don't stop retrieval.
            Continues with other collections if some fail.

        Memory Handling:
            Special handling for embeddings to avoid numpy array
            evaluation errors during processing.

        Example:
            ```python
            # Get all long-term memories
            memories = await store.get_all_memories(
                memory_type=MemoryType.LONG_TERM
            )

            # Paginated retrieval
            page1 = await store.get_all_memories(limit=50, offset=0)
            page2 = await store.get_all_memories(limit=50, offset=50)
            ```
        """
        try:
            all_memories = []

            # Determine which collections to query
            if memory_type:
                if memory_type not in self.collections:
                    logger.error(f"Memory type {memory_type} not found in collections")
                    return []
                collections_to_query = [(memory_type, self.collections[memory_type])]
            else:
                collections_to_query = list(self.collections.items())

            for mem_type, collection in collections_to_query:
                # Get all IDs from the collection (much faster than vector search)
                try:
                    # Get collection size first
                    collection_count = collection.count()
                    if collection_count == 0:
                        continue

                    # Get all documents without doing vector search
                    results = collection.get(
                        include=["documents", "metadatas", "embeddings"],
                        limit=min(collection_count, limit * 2),  # Get more than needed for sorting
                    )

                    if results and results.get("ids"):
                        for i, doc_id in enumerate(results["ids"]):
                            # Don't break early - we need to collect from all collections first
                            # The final sorting and pagination will handle the limit

                            try:
                                # Handle embeddings properly to avoid array evaluation errors
                                embedding = None
                                embeddings_data = results.get("embeddings")
                                if embeddings_data is not None:
                                    try:
                                        # Check if we have embeddings and the index is valid
                                        if (
                                            hasattr(embeddings_data, "__len__")
                                            and len(embeddings_data) > i
                                        ):
                                            embedding = embeddings_data[i]
                                        elif (
                                            isinstance(embeddings_data, list)
                                            and len(embeddings_data) > i
                                        ):
                                            embedding = embeddings_data[i]
                                    except (IndexError, TypeError):
                                        # Skip invalid embeddings
                                        embedding = None

                                memory = self._build_memory(
                                    id=doc_id,
                                    document=results["documents"][i],
                                    embedding=embedding,
                                    metadata=results["metadatas"][i],
                                    memory_type=mem_type,
                                )
                                all_memories.append(memory)

                            except Exception as mem_error:
                                logger.warning(f"Error building memory {doc_id}: {mem_error}")
                                continue

                except Exception as collection_error:
                    logger.warning(f"Error querying collection {mem_type}: {collection_error}")
                    continue

            # Sort by creation time (newest first) and apply pagination
            all_memories.sort(key=lambda m: m.created_at, reverse=True)

            # Apply offset and limit
            start_idx = offset
            end_idx = offset + limit
            paginated_memories = all_memories[start_idx:end_idx]

            logger.info(
                f"Retrieved {len(paginated_memories)} memories efficiently (total available: {len(all_memories)})"
            )
            return paginated_memories

        except Exception as e:
            logger.error(f"Failed to get all memories: {e}")
            return []

    def _build_memory(
        self,
        id: str,
        document: str,
        embedding: list[float] | None,
        metadata: dict[str, Any],
        memory_type: MemoryType,
    ) -> Memory:
        """Build a Memory object from ChromaDB results.

        Reconstructs a Memory object from ChromaDB storage format,
        handling metadata deserialization and type conversion.

        Args:
            id: Memory UUID
            document: Memory content text
            embedding: Vector embedding (may be None)
            metadata: Raw metadata from ChromaDB
            memory_type: Memory type for this memory

        Returns:
            Reconstructed Memory object

        Metadata Processing:
            - Extracts standard fields (created_at, importance, etc.)
            - Deserializes JSON strings back to lists
            - Preserves custom metadata fields

        Type Safety:
            Handles missing metadata gracefully with sensible defaults.
            Uses current timestamp if dates are missing.
        """
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
