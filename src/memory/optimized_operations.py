"""Optimized memory operations with caching and performance enhancements."""

import asyncio
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from src.memory import Memory, MemoryStore, MemoryType
from src.memory.manager import MemoryManager
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class CacheEntry:
    """Cache entry for memory operations."""

    data: Any
    timestamp: datetime
    ttl_seconds: int = 300  # 5 minutes default
    access_count: int = 0

    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return datetime.now() - self.timestamp > timedelta(seconds=self.ttl_seconds)

    def access(self):
        """Mark entry as accessed."""
        self.access_count += 1


class MemoryCache:
    """Intelligent caching system for memory operations."""

    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: dict[str, CacheEntry] = {}
        self.access_times: dict[str, datetime] = {}

    def _generate_key(self, operation: str, **kwargs) -> str:
        """Generate cache key from operation and parameters."""
        key_data = {"op": operation, **kwargs}
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]

    def get(self, operation: str, **kwargs) -> Any | None:
        """Get cached result for operation."""
        key = self._generate_key(operation, **kwargs)

        if key in self.cache:
            entry = self.cache[key]
            if not entry.is_expired():
                entry.access()
                self.access_times[key] = datetime.now()
                return entry.data
            else:
                # Remove expired entry
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]

        return None

    def set(self, operation: str, data: Any, ttl: int | None = None, **kwargs):
        """Cache result for operation."""
        key = self._generate_key(operation, **kwargs)
        ttl = ttl or self.default_ttl

        # Evict if cache is full
        if len(self.cache) >= self.max_size:
            self._evict_lru()

        self.cache[key] = CacheEntry(data=data, timestamp=datetime.now(), ttl_seconds=ttl)
        self.access_times[key] = datetime.now()

    def _evict_lru(self):
        """Evict least recently used entries."""
        if not self.access_times:
            return

        # Remove 20% of cache (LRU)
        sorted_keys = sorted(self.access_times.items(), key=lambda x: x[1])
        evict_count = max(1, len(sorted_keys) // 5)

        for key, _ in sorted_keys[:evict_count]:
            if key in self.cache:
                del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]

    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
        self.access_times.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_entries = len(self.cache)
        expired_entries = sum(1 for entry in self.cache.values() if entry.is_expired())

        return {
            "total_entries": total_entries,
            "expired_entries": expired_entries,
            "cache_size_mb": self._estimate_size_mb(),
            "hit_rate": self._calculate_hit_rate(),
            "max_size": self.max_size,
        }

    def _estimate_size_mb(self) -> float:
        """Estimate cache size in MB."""
        # Rough estimation
        return len(self.cache) * 0.001  # ~1KB per entry estimate

    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if not self.cache:
            return 0.0

        total_accesses = sum(entry.access_count for entry in self.cache.values())
        if total_accesses == 0:
            return 0.0

        # This is a simplified calculation
        return min(1.0, total_accesses / (len(self.cache) * 2))


class OptimizedMemoryManager(MemoryManager):
    """Enhanced memory manager with performance optimizations."""

    def __init__(self, store: MemoryStore | None = None, **kwargs):
        super().__init__(store, **kwargs)

        # Performance enhancements
        self.cache = MemoryCache(max_size=2000, default_ttl=600)  # 10 minutes
        self.batch_operations = []
        self.batch_size = 50
        self.background_task_running = False

        # Search optimizations
        self.search_index: dict[str, set[str]] = defaultdict(set)  # word -> memory_ids
        self.importance_index: dict[float, list[str]] = defaultdict(
            list
        )  # importance -> memory_ids
        self.type_index: dict[MemoryType, list[str]] = defaultdict(list)  # type -> memory_ids

        # Performance tracking
        self.operation_stats = {
            "searches": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_operations": 0,
            "avg_response_time": 0.0,
        }

        # Start background optimization tasks
        asyncio.create_task(self._start_background_optimization())

    async def _start_background_optimization(self):
        """Start background optimization tasks."""
        if self.background_task_running:
            return

        self.background_task_running = True

        try:
            while self.background_task_running:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self._background_optimize()
        except Exception as e:
            logger.error(f"Background optimization error: {e}")
        finally:
            self.background_task_running = False

    async def _background_optimize(self):
        """Perform background optimization tasks."""
        try:
            # Clean expired cache entries
            self.cache.clear()

            # Rebuild search indexes periodically
            await self._rebuild_search_indexes()

            # Consolidate memories
            await self.consolidate()

            logger.info("Background optimization completed")

        except Exception as e:
            logger.error(f"Background optimization failed: {e}")

    async def _rebuild_search_indexes(self):
        """Rebuild search indexes for faster queries."""
        try:
            # Clear existing indexes
            self.search_index.clear()
            self.importance_index.clear()
            self.type_index.clear()

            # Get all memories
            all_memories = []
            for memory_type in MemoryType:
                memories = await self.store.search(
                    query="", memory_type=memory_type, limit=10000, threshold=0.0
                )
                all_memories.extend(memories)

            # Build indexes
            for memory in all_memories:
                # Text search index
                words = memory.content.lower().split()
                for word in words:
                    if len(word) > 2:  # Skip short words
                        self.search_index[word].add(memory.id)

                # Importance index
                importance_key = round(memory.importance, 1)
                self.importance_index[importance_key].append(memory.id)

                # Type index
                self.type_index[memory.memory_type].append(memory.id)

            logger.info(f"Rebuilt search indexes for {len(all_memories)} memories")

        except Exception as e:
            logger.error(f"Failed to rebuild search indexes: {e}")

    async def optimized_search(
        self,
        query: str,
        memory_types: list[MemoryType] | None = None,
        limit: int = 10,
        threshold: float = 0.7,
        use_cache: bool = True,
    ) -> list[Memory]:
        """Optimized search with caching and indexing."""
        start_time = time.time()

        try:
            self.operation_stats["searches"] += 1
            self.operation_stats["total_operations"] += 1

            # Check cache first
            if use_cache:
                cached_result = self.cache.get(
                    "search",
                    query=query,
                    memory_types=[t.value for t in (memory_types or [])],
                    limit=limit,
                    threshold=threshold,
                )

                if cached_result is not None:
                    self.operation_stats["cache_hits"] += 1
                    return cached_result
                else:
                    self.operation_stats["cache_misses"] += 1

            # Use index-based search for better performance
            if query and self.search_index:
                result = await self._indexed_search(query, memory_types, limit, threshold)
            else:
                # Fallback to regular search
                result = await self.recall(query, memory_types, limit, threshold)

            # Cache the result
            if use_cache:
                cache_ttl = 300 if len(result) < 100 else 600  # Longer cache for larger results
                self.cache.set(
                    "search",
                    result,
                    ttl=cache_ttl,
                    query=query,
                    memory_types=[t.value for t in (memory_types or [])],
                    limit=limit,
                    threshold=threshold,
                )

            return result

        except Exception as e:
            logger.error(f"Optimized search failed: {e}")
            return []
        finally:
            # Update performance stats
            response_time = time.time() - start_time
            self._update_response_time_stats(response_time)

    async def _indexed_search(
        self, query: str, memory_types: list[MemoryType] | None, limit: int, threshold: float
    ) -> list[Memory]:
        """Perform indexed search for better performance."""
        try:
            # Find candidate memory IDs using word index
            query_words = query.lower().split()
            candidate_ids = set()

            for word in query_words:
                if word in self.search_index:
                    if not candidate_ids:
                        candidate_ids = self.search_index[word].copy()
                    else:
                        candidate_ids &= self.search_index[word]  # Intersection

            # If no indexed matches, fallback to regular search
            if not candidate_ids:
                return await self.recall(query, memory_types, limit, threshold)

            # Retrieve candidate memories
            candidate_memories = []
            for memory_id in candidate_ids:
                memory = await self.store.retrieve(memory_id)
                if memory:
                    candidate_memories.append(memory)

            # Filter by memory type if specified
            if memory_types:
                candidate_memories = [
                    m for m in candidate_memories if m.memory_type in memory_types
                ]

            # Sort by importance and recency
            candidate_memories.sort(
                key=lambda m: (m.importance, m.accessed_at.timestamp()), reverse=True
            )

            return candidate_memories[:limit]

        except Exception as e:
            logger.error(f"Indexed search failed: {e}")
            return await self.recall(query, memory_types, limit, threshold)

    async def batch_remember(self, memories_data: list[dict[str, Any]]) -> list[str]:
        """Batch memory storage for better performance."""
        try:
            memory_ids = []

            # Process in batches
            for i in range(0, len(memories_data), self.batch_size):
                batch = memories_data[i : i + self.batch_size]
                batch_ids = []

                for memory_data in batch:
                    memory_id = await self.remember(
                        content=memory_data.get("content", ""),
                        memory_type=memory_data.get("memory_type", MemoryType.SHORT_TERM),
                        importance=memory_data.get("importance", 0.5),
                        metadata=memory_data.get("metadata", {}),
                    )
                    if memory_id:
                        batch_ids.append(memory_id)

                memory_ids.extend(batch_ids)

                # Small delay between batches to prevent overwhelming the system
                if i + self.batch_size < len(memories_data):
                    await asyncio.sleep(0.01)

            logger.info(f"Batch stored {len(memory_ids)} memories")
            return memory_ids

        except Exception as e:
            logger.error(f"Batch remember failed: {e}")
            return []

    async def get_memories_by_importance(
        self,
        min_importance: float,
        max_importance: float = 1.0,
        memory_types: list[MemoryType] | None = None,
        limit: int = 100,
    ) -> list[Memory]:
        """Get memories filtered by importance range."""
        try:
            # Use importance index for faster retrieval
            candidate_ids = []

            for importance_key in self.importance_index:
                if min_importance <= importance_key <= max_importance:
                    candidate_ids.extend(self.importance_index[importance_key])

            # Retrieve memories
            memories = []
            for memory_id in candidate_ids[: limit * 2]:  # Get more to account for filtering
                memory = await self.store.retrieve(memory_id)
                if memory:
                    # Double-check importance (in case index is slightly stale)
                    if min_importance <= memory.importance <= max_importance:
                        if not memory_types or memory.memory_type in memory_types:
                            memories.append(memory)

            # Sort by importance
            memories.sort(key=lambda m: m.importance, reverse=True)
            return memories[:limit]

        except Exception as e:
            logger.error(f"Get memories by importance failed: {e}")
            return []

    async def get_recent_memories(
        self, days: int = 7, memory_types: list[MemoryType] | None = None, limit: int = 100
    ) -> list[Memory]:
        """Get recent memories efficiently."""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)

            # Use type index for faster retrieval
            candidate_memories = []

            types_to_search = memory_types or list(MemoryType)
            for memory_type in types_to_search:
                if memory_type in self.type_index:
                    for memory_id in self.type_index[memory_type]:
                        memory = await self.store.retrieve(memory_id)
                        if memory and memory.created_at >= cutoff_date:
                            candidate_memories.append(memory)

            # Sort by creation time
            candidate_memories.sort(key=lambda m: m.created_at, reverse=True)
            return candidate_memories[:limit]

        except Exception as e:
            logger.error(f"Get recent memories failed: {e}")
            return []

    def _update_response_time_stats(self, response_time: float):
        """Update response time statistics."""
        try:
            current_avg = self.operation_stats["avg_response_time"]
            total_ops = self.operation_stats["total_operations"]

            # Calculate new average
            new_avg = ((current_avg * (total_ops - 1)) + response_time) / total_ops
            self.operation_stats["avg_response_time"] = new_avg

        except Exception as e:
            logger.error(f"Failed to update response time stats: {e}")

    async def get_performance_stats(self) -> dict[str, Any]:
        """Get comprehensive performance statistics."""
        try:
            base_stats = await self.get_memory_stats()
            cache_stats = self.cache.get_stats()

            performance_stats = {
                **base_stats,
                "cache": cache_stats,
                "operations": self.operation_stats.copy(),
                "indexes": {
                    "search_terms": len(self.search_index),
                    "importance_buckets": len(self.importance_index),
                    "type_buckets": len(self.type_index),
                },
            }

            return performance_stats

        except Exception as e:
            logger.error(f"Failed to get performance stats: {e}")
            return {"error": str(e)}

    async def optimize_storage(self) -> dict[str, Any]:
        """Optimize memory storage and indexes."""
        try:
            optimization_results = {
                "cache_cleared": False,
                "indexes_rebuilt": False,
                "memories_consolidated": 0,
                "storage_optimized": False,
            }

            # Clear cache
            self.cache.clear()
            optimization_results["cache_cleared"] = True

            # Rebuild indexes
            await self._rebuild_search_indexes()
            optimization_results["indexes_rebuilt"] = True

            # Consolidate memories
            await self.consolidate()
            optimization_results["memories_consolidated"] = len(self.batch_operations)

            # Reset performance stats
            self.operation_stats = {
                "searches": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "total_operations": 0,
                "avg_response_time": 0.0,
            }

            optimization_results["storage_optimized"] = True

            logger.info("Memory storage optimization completed")
            return optimization_results

        except Exception as e:
            logger.error(f"Storage optimization failed: {e}")
            return {"error": str(e)}

    def shutdown(self):
        """Clean shutdown of optimized memory manager."""
        try:
            self.background_task_running = False
            self.cache.clear()
            logger.info("Optimized memory manager shutdown completed")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Factory function for creating optimized memory manager
def create_optimized_memory_manager() -> OptimizedMemoryManager:
    """Create an optimized memory manager instance."""
    try:
        return OptimizedMemoryManager()
    except Exception as e:
        logger.error(f"Failed to create optimized memory manager: {e}")
        # Fallback to regular memory manager
        from src.memory.manager import MemoryManager

        return MemoryManager()
