"""Memory management system with intelligent retrieval and storage.

This module provides the core memory management functionality for Neuromancer,
implementing a sophisticated system for storing, retrieving, and managing
memories with semantic search capabilities.

Key Features:
    - Multiple memory types (SHORT_TERM, LONG_TERM, EPISODIC, SEMANTIC)
    - Intelligent memory classification and importance scoring
    - Semantic similarity search using embeddings
    - Memory consolidation from short-term to long-term
    - AI-powered memory analysis and formation
    - Automatic memory lifecycle management

Architecture:
    The MemoryManager acts as a facade over the storage backend (VectorMemoryStore)
    and coordinates with the embedding generator and intelligent analyzer.
    It implements memory formation, retrieval, consolidation, and lifecycle
    management strategies.

Memory Lifecycle:
    1. Formation: Content analyzed and stored with metadata
    2. Access: Retrieved based on semantic similarity
    3. Consolidation: Important short-term memories become long-term
    4. Forgetting: Unimportant memories are cleaned up

Performance Considerations:
    - Embedding generation: ~50-100ms per memory
    - Retrieval: ~50-200ms depending on search scope
    - Background consolidation runs periodically
    - Importance scoring prevents memory bloat

Example Usage:
    ```python
    memory_manager = MemoryManager()

    # Store a memory
    memory_id = await memory_manager.remember(
        "User's name is John Smith",
        memory_type=MemoryType.LONG_TERM,
        importance=0.9
    )

    # Intelligent memory formation
    memory_ids = await memory_manager.intelligent_remember(
        "I prefer Python over Java for most projects"
    )

    # Recall memories
    memories = await memory_manager.recall(
        "What is the user's name?",
        limit=5
    )
    ```
"""

from datetime import datetime, timedelta
from typing import Any

from src.memory import Memory, MemoryStore, MemoryType
from src.memory.intelligent_analyzer import IntelligentMemoryAnalyzer
from src.memory.vector_store import VectorMemoryStore
from src.utils.embeddings import EmbeddingGenerator
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class MemoryManager:
    """Manages memory storage, retrieval, and optimization.

    Central manager for all memory operations, providing high-level
    interfaces for memory formation, retrieval, and lifecycle management.

    Attributes:
        store: Backend storage implementation (VectorMemoryStore)
        embedding_generator: Generates embeddings for semantic search
        intelligent_analyzer: AI-powered memory analysis
        short_term_duration: How long memories stay short-term (24h)
        importance_threshold: Minimum importance for consolidation (0.3)
        consolidation_interval: Time between consolidation runs (6h)
        _conversation_context: Recent conversation for context
        _max_context_length: Maximum context messages to keep (10)

    Design Philosophy:
        The manager implements a human-like memory system with different
        types serving different purposes. Short-term memories capture
        recent interactions, while important information is consolidated
        to long-term storage.
    """

    def __init__(
        self,
        store: MemoryStore | None = None,
        embedding_generator: EmbeddingGenerator | None = None,
    ):
        """Initialize memory manager.

        Args:
            store: Memory storage backend. If not provided, creates a
                VectorMemoryStore with default settings.
            embedding_generator: Generator for text embeddings. If not
                provided, creates default EmbeddingGenerator.

        Initialization:
            Sets up storage, embedding generation, and intelligent analysis
            components. Configures memory lifecycle parameters.

        Note:
            Background tasks are started but not yet implemented.
            Future versions will include automatic consolidation.
        """
        self.store = store or VectorMemoryStore()
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.intelligent_analyzer = IntelligentMemoryAnalyzer()

        # Memory configuration
        self.short_term_duration = timedelta(hours=24)
        self.importance_threshold = 0.3
        self.consolidation_interval = timedelta(hours=6)

        # Conversation context for intelligent analysis
        self._conversation_context = []
        self._max_context_length = 10

        # Start background tasks
        self._start_background_tasks()

    def _start_background_tasks(self):
        """Start background memory management tasks.

        Placeholder for future background task implementation.
        Will include:
            - Periodic memory consolidation
            - Cleanup of old unimportant memories
            - Memory importance recalculation
            - Storage optimization
        """
        # TODO: Implement memory consolidation and cleanup
        pass

    async def remember(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.SHORT_TERM,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
        auto_classify: bool = False,
    ) -> str:
        """Store a new memory with enhanced formation logic.

        Primary method for storing memories with optional intelligent classification.

        Args:
            content: The content to remember. Can be any text that should
                be stored for future retrieval.
            memory_type: Type of memory classification. Defaults to SHORT_TERM
                but can be overridden or auto-classified.
            importance: Importance score between 0.0 and 1.0. Higher scores
                mean the memory is more likely to be retained long-term.
            metadata: Additional metadata to store with the memory. Common keys:
                - source: Where the memory came from
                - context: Additional context
                - entities: Extracted entities
            auto_classify: If True, uses AI to classify memory type and
                calculate importance based on content analysis.

        Returns:
            Memory ID (UUID string) or empty string on failure

        Process:
            1. Generate embedding for semantic search
            2. Optionally classify type and importance
            3. Add formation metadata
            4. Store in backend

        Error Handling:
            Failures are logged but don't raise exceptions.
            Returns empty string on failure to maintain flow.

        Example:
            ```python
            # Manual classification
            memory_id = await memory_manager.remember(
                "Meeting scheduled for 3pm tomorrow",
                memory_type=MemoryType.EPISODIC,
                importance=0.7
            )

            # Auto classification
            memory_id = await memory_manager.remember(
                "Python is a high-level programming language",
                auto_classify=True
            )
            ```
        """
        try:
            # Generate embedding
            embedding = await self.embedding_generator.generate(content)

            # Enhanced metadata with formation context (minimal for speed)
            enhanced_metadata = metadata or {}
            if auto_classify:
                enhanced_metadata.update(
                    {
                        "formation_timestamp": datetime.now().isoformat(),
                        "content_length": len(content),
                        "formation_method": "enhanced_auto",
                    }
                )

                # Auto-classify memory type if requested
                memory_type = self._classify_memory_type(content, memory_type)
                enhanced_metadata["auto_classified_type"] = memory_type.value

                # Auto-calculate importance if requested
                importance = self._calculate_importance(content, importance)
                enhanced_metadata["auto_calculated_importance"] = importance

                # Add minimal content analysis metadata
                enhanced_metadata.update(self._analyze_content_minimal(content))
            else:
                # Minimal metadata for speed
                enhanced_metadata.update(
                    {
                        "formation_timestamp": datetime.now().isoformat(),
                        "formation_method": "manual",
                    }
                )

            # Create memory object
            memory = Memory(
                id="",  # Will be generated by store
                content=content,
                embedding=embedding,
                memory_type=memory_type,
                importance=importance,
                metadata=enhanced_metadata,
            )

            # Store memory
            memory_id = await self.store.store(memory)

            logger.info(
                f"Stored {memory_type.value} memory: {memory_id[:8]}... (importance: {importance:.2f})"
            )
            return memory_id

        except Exception as e:
            logger.error(f"Failed to remember content: {e}")
            # Don't raise - return empty string to indicate failure
            return ""

    async def intelligent_remember(
        self, content: str, conversation_context: list[str] = None
    ) -> list[str]:
        """
        Intelligently analyze content and automatically store significant memories.

        Uses AI-powered analysis to identify important information worth remembering.
        This method can extract multiple memories from a single content piece.

        Args:
            content: The content to analyze for memory-worthy information.
                Can be a single message or longer text.
            conversation_context: Recent conversation messages for context.
                Helps the analyzer understand the conversation flow.

        Returns:
            List of memory IDs that were created (may be empty)

        Analysis Process:
            1. Update conversation context
            2. Analyze content for memory signals
            3. Extract significant information
            4. Create memories with appropriate types
            5. Store with rich metadata

        Memory Signals:
            The analyzer looks for:
            - Personal information (names, preferences)
            - Important facts or data
            - Action items or commitments
            - Learning moments
            - Emotional significance

        Significance Threshold:
            Only memories with significance >= 0.4 are stored
            to prevent memory pollution.

        Example:
            ```python
            # Analyze a complex message
            memory_ids = await memory_manager.intelligent_remember(
                "My name is Sarah and I work at TechCorp. "
                "I'm interested in machine learning and Python."
            )
            # Might create 2-3 memories for different facts
            ```
        """
        try:
            # Update conversation context
            if conversation_context:
                self._conversation_context = conversation_context[-self._max_context_length :]
            else:
                # Add current content to context
                self._conversation_context.append(content)
                if len(self._conversation_context) > self._max_context_length:
                    self._conversation_context = self._conversation_context[
                        -self._max_context_length :
                    ]

            # Analyze content for memory signals
            memory_signals = self.intelligent_analyzer.analyze_memory_significance(
                content, self._conversation_context
            )

            if not memory_signals:
                logger.debug("No significant memory signals detected")
                return []

            created_memory_ids = []

            # Process each memory signal
            for signal in memory_signals:
                if signal.significance >= 0.4:  # Only store significant memories
                    try:
                        # Generate embedding for the memory content
                        embedding = await self.embedding_generator.generate(signal.content)

                        # Create enhanced metadata
                        metadata = {
                            "ai_analyzed": True,
                            "significance": signal.significance,
                            "context": signal.context,
                            "entities": signal.entities,
                            "reasoning": signal.reasoning,
                            "original_content": content,
                            "formation_timestamp": datetime.now().isoformat(),
                            "conversation_turn": len(self._conversation_context),
                        }

                        # Create memory object
                        memory = Memory(
                            id="",  # Will be generated by store
                            content=signal.content,
                            embedding=embedding,
                            memory_type=signal.memory_type,
                            importance=signal.significance,
                            metadata=metadata,
                        )

                        # Store memory
                        memory_id = await self.store.store(memory)

                        if memory_id:
                            created_memory_ids.append(memory_id)
                            logger.info(
                                f"Intelligently stored {signal.memory_type.value} memory: {memory_id[:8]}... "
                                f"(significance: {signal.significance:.2f}, context: {signal.context})"
                            )

                    except Exception as e:
                        logger.error(f"Failed to store intelligent memory signal: {e}")
                        continue

            if created_memory_ids:
                logger.info(
                    f"Intelligent analysis created {len(created_memory_ids)} memories from content analysis"
                )

            return created_memory_ids

        except Exception as e:
            logger.error(f"Intelligent memory analysis failed: {e}")
            return []

    async def recall(
        self,
        query: str,
        memory_types: list[MemoryType] | None = None,
        limit: int = 10,
        threshold: float = 0.3,
    ) -> list[Memory]:
        """Recall relevant memories based on query.

        Semantic search across memories to find relevant information.

        Args:
            query: Search query in natural language. The query is embedded
                and compared against stored memories.
            memory_types: Types of memory to search. If None, searches all
                types. Can filter to specific types for targeted search.
            limit: Maximum memories to return. More memories provide more
                context but increase processing time.
            threshold: Similarity threshold (0.0-1.0). Lower values return
                more results but may be less relevant.

        Returns:
            List of Memory objects sorted by relevance and importance

        Search Strategy:
            1. Convert query to embedding
            2. Search specified memory types
            3. Filter by similarity threshold
            4. Sort by relevance and importance
            5. Update access timestamps

        Personal Information Priority:
            Memories tagged as personal_info get +0.5 importance boost
            to ensure they surface in relevant queries.

        Performance:
            - Single type search: ~50-100ms
            - All types search: ~100-200ms
            - Scales with memory count

        Example:
            ```python
            # Search all memories
            memories = await memory_manager.recall(
                "What is the user's name?"
            )

            # Search specific types
            facts = await memory_manager.recall(
                "Python features",
                memory_types=[MemoryType.SEMANTIC],
                limit=5
            )
            ```
        """
        try:
            memories = []

            if memory_types:
                # Search specific memory types
                for memory_type in memory_types:
                    results = await self.store.search(
                        query=query, memory_type=memory_type, limit=limit, threshold=threshold
                    )
                    memories.extend(results)
            else:
                # Search all memory types
                memories = await self.store.search(query=query, limit=limit, threshold=threshold)

            # Update access times
            for memory in memories:
                memory.accessed_at = datetime.now()
                await self.store.update(memory)

            # Sort by relevance and importance, with priority for personal info
            def memory_priority(memory):
                base_score = (memory.importance, memory.accessed_at.timestamp())
                # Boost personal info memories
                if memory.metadata and memory.metadata.get("type") == "personal_info":
                    return (base_score[0] + 0.5, base_score[1])  # Add 0.5 to importance
                return base_score

            memories.sort(key=memory_priority, reverse=True)

            logger.info(f"Recalled {len(memories)} memories for query: {query[:50]}...")
            return memories[:limit]

        except Exception as e:
            logger.error(f"Failed to recall memories: {e}")
            return []

    async def consolidate(self):
        """Consolidate short-term memories into long-term storage.

        Implements memory consolidation similar to human memory, where
        important short-term memories are converted to long-term storage
        while unimportant ones are forgotten.

        Consolidation Rules:
            1. Short-term memories older than 24 hours are evaluated
            2. Importance >= 0.3: Converted to LONG_TERM
            3. Importance < 0.3: Deleted (forgotten)

        Process:
            1. Query all SHORT_TERM memories
            2. Check age and importance
            3. Convert or delete based on rules
            4. Update metadata with consolidation timestamp

        Design Rationale:
            This mimics human memory consolidation during sleep,
            where the brain decides what to keep long-term.

        Note:
            Currently must be called manually. Future versions will
            run this automatically in the background.
        """
        try:
            # Get old short-term memories
            cutoff_time = datetime.now() - self.short_term_duration

            # Search for memories to consolidate
            all_short_term = await self.store.search(
                query="", memory_type=MemoryType.SHORT_TERM, limit=1000, threshold=0.0  # Get all
            )

            consolidated_count = 0

            for memory in all_short_term:
                # Check if memory should be consolidated
                if (
                    memory.created_at < cutoff_time
                    and memory.importance >= self.importance_threshold
                ):

                    # Convert to long-term memory
                    memory.memory_type = MemoryType.LONG_TERM
                    memory.metadata["consolidated_at"] = datetime.now().isoformat()

                    # Update in store
                    await self.store.update(memory)
                    consolidated_count += 1

                elif memory.created_at < cutoff_time:
                    # Remove unimportant old memories
                    await self.store.delete(memory.id)

            logger.info(f"Consolidated {consolidated_count} memories to long-term storage")

        except Exception as e:
            logger.error(f"Failed to consolidate memories: {e}")

    async def forget(self, memory_id: str) -> bool:
        """Forget a specific memory.

        Permanently removes a memory from storage.

        Args:
            memory_id: UUID of the memory to forget

        Returns:
            True if successfully deleted, False otherwise

        Use Cases:
            - User requests to forget information
            - Removing incorrect memories
            - Privacy compliance

        Note:
            This is permanent deletion. The memory cannot be recovered.
        """
        try:
            return await self.store.delete(memory_id)
        except Exception as e:
            logger.error(f"Failed to forget memory {memory_id}: {e}")
            return False

    async def clear_memories(self, memory_type: MemoryType | None = None) -> int:
        """Clear memories of a specific type or all memories.

        Bulk deletion operation for memory management.

        Args:
            memory_type: Type to clear. If None, clears ALL memories.
                Use with caution as this is irreversible.

        Returns:
            Number of memories cleared

        Warning:
            This permanently deletes memories. Consider archiving
            important data before clearing.

        Example:
            ```python
            # Clear only short-term memories
            count = await memory_manager.clear_memories(MemoryType.SHORT_TERM)

            # Clear everything (dangerous!)
            count = await memory_manager.clear_memories()
            ```
        """
        try:
            return await self.store.clear(memory_type)
        except Exception as e:
            logger.error(f"Failed to clear memories: {e}")
            return 0

    async def get_memory_stats(self) -> dict[str, Any]:
        """Get statistics about stored memories.

        Provides insights into memory usage and distribution.

        Returns:
            Dictionary with statistics:
            - total: Total memory count
            - by_type: Count per memory type
            - avg_importance: Average importance score
            - storage_size: Approximate storage size

        Usage:
            Useful for monitoring memory system health and usage patterns.
            Can help identify if consolidation or cleanup is needed.

        Example:
            ```python
            stats = await memory_manager.get_memory_stats()
            print(f"Total memories: {stats['total']}")
            print(f"Long-term: {stats['by_type']['long_term']}")
            ```
        """
        try:
            stats = {"total": 0, "by_type": {}, "avg_importance": 0.0, "storage_size": 0}

            total_importance = 0.0

            for memory_type in MemoryType:
                memories = await self.store.search(
                    query="", memory_type=memory_type, limit=10000, threshold=0.0
                )

                count = len(memories)
                stats["by_type"][memory_type.value] = count
                stats["total"] += count

                if memories:
                    total_importance += sum(m.importance for m in memories)

            if stats["total"] > 0:
                stats["avg_importance"] = total_importance / stats["total"]

            return stats

        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {"error": str(e)}

    async def get_all_memories(
        self, memory_type: MemoryType | None = None, limit: int = 100, offset: int = 0
    ) -> list[Memory]:
        """Get all memories efficiently without vector search.

        Direct retrieval method for listing memories without semantic search.
        Useful for UI display, exports, and management operations.

        Args:
            memory_type: Specific memory type to filter by, or None for all types
            limit: Maximum number of memories to return (default: 100)
            offset: Number of memories to skip for pagination (default: 0)

        Returns:
            List of Memory objects sorted by creation time (newest first)

        Performance:
            Much faster than semantic search as it bypasses embedding
            comparison. Use this for listing operations.

        Pagination:
            Use limit and offset for paginated retrieval:
            - Page 1: limit=10, offset=0
            - Page 2: limit=10, offset=10
            - Page 3: limit=10, offset=20

        Example:
            ```python
            # Get first 20 long-term memories
            memories = await memory_manager.get_all_memories(
                memory_type=MemoryType.LONG_TERM,
                limit=20
            )

            # Get page 2 of all memories (items 51-100)
            page2 = await memory_manager.get_all_memories(
                limit=50,
                offset=50
            )
            ```
        """
        try:
            if hasattr(self.store, "get_all_memories"):
                # Use the efficient direct method if available
                return await self.store.get_all_memories(memory_type, limit, offset)
            else:
                # Fallback to search method (less efficient)
                logger.warning("Using fallback search method for getting all memories")
                return await self.recall("a", [memory_type] if memory_type else None, limit, 0.0)

        except Exception as e:
            logger.error(f"Failed to get all memories: {e}")
            return []

    def _classify_memory_type(self, content: str, default_type: MemoryType) -> MemoryType:
        """Intelligently classify memory type based on content.

        Analyzes content to determine the most appropriate memory type.
        Uses keyword matching and linguistic patterns.

        Args:
            content: Text content to analyze
            default_type: Fallback type if classification is uncertain

        Returns:
            Classified MemoryType

        Classification Rules:
            - EPISODIC: Time-based events, experiences, conversations
            - SEMANTIC: Facts, definitions, concepts, procedures
            - LONG_TERM: Personal info, preferences, important data
            - SHORT_TERM: Everything else (default)

        Algorithm:
            1. Convert to lowercase for matching
            2. Count keyword matches for each type
            3. Check for personal pronouns and time references
            4. Select type with highest score
        """
        content_lower = content.lower()

        # Keywords for different memory types
        episodic_keywords = [
            "yesterday",
            "today",
            "tomorrow",
            "last week",
            "next week",
            "when",
            "happened",
            "event",
            "meeting",
            "conversation",
            "said",
            "told",
            "experience",
            "remember when",
            "that time",
            "during",
            "while",
        ]

        semantic_keywords = [
            "fact",
            "definition",
            "concept",
            "theory",
            "principle",
            "rule",
            "always",
            "never",
            "generally",
            "usually",
            "knowledge",
            "learn",
            "understand",
            "means",
            "refers to",
            "defined as",
        ]

        long_term_keywords = [
            "important",
            "remember",
            "preference",
            "setting",
            "configuration",
            "always do",
            "never do",
            "my",
            "personal",
            "profile",
            "about me",
            "name is",
            "call me",
            "i am",
            "phone number",
            "email",
            "address",
        ]

        # Count keyword matches
        episodic_score = sum(1 for keyword in episodic_keywords if keyword in content_lower)
        semantic_score = sum(1 for keyword in semantic_keywords if keyword in content_lower)
        long_term_score = sum(1 for keyword in long_term_keywords if keyword in content_lower)

        # Check for personal pronouns and time references
        personal_pronouns = ["i", "my", "me", "myself", "we", "our", "us"]
        time_references = ["am", "pm", "hour", "minute", "day", "week", "month", "year"]

        has_personal = any(pronoun in content_lower.split() for pronoun in personal_pronouns)
        has_time = any(time_ref in content_lower for time_ref in time_references)

        # Classification logic
        if episodic_score >= 2 or (has_personal and has_time):
            return MemoryType.EPISODIC
        elif semantic_score >= 2:
            return MemoryType.SEMANTIC
        elif long_term_score >= 1 or has_personal:
            return MemoryType.LONG_TERM
        else:
            return default_type  # Keep original classification

    def _calculate_importance(self, content: str, default_importance: float) -> float:
        """Calculate importance score based on content analysis.

        Sophisticated importance scoring based on multiple content signals.

        Args:
            content: Text content to analyze
            default_importance: Base importance score to adjust

        Returns:
            Importance score between 0.0 and 1.0

        Scoring Components:
            - Importance keywords: +0.1 per match
            - Personal information: +0.15 per match
            - Content length: +0.2 max for detailed content
            - Structure (lists, steps): +0.1

        Important Patterns:
            - Explicit importance markers ("important", "remember")
            - Personal data (names, contacts, preferences)
            - Structured information
            - Emotional significance

        Design:
            Biased toward personal information and explicit requests
            to ensure critical data is retained.
        """
        base_importance = default_importance

        # Importance boosters
        importance_keywords = [
            "important",
            "critical",
            "urgent",
            "remember",
            "never forget",
            "always",
            "must",
            "required",
            "essential",
            "key",
            "vital",
        ]

        personal_keywords = [
            "password",
            "phone",
            "address",
            "email",
            "name",
            "birthday",
            "preference",
            "setting",
            "like",
            "dislike",
            "hate",
            "love",
            "name is",
            "call me",
            "i am",
            "my name",
            "about me",
        ]

        content_lower = content.lower()

        # Boost for importance keywords
        importance_boost = sum(0.1 for keyword in importance_keywords if keyword in content_lower)

        # Boost for personal information
        personal_boost = sum(0.15 for keyword in personal_keywords if keyword in content_lower)

        # Boost for longer, more detailed content
        length_boost = min(0.2, len(content) / 1000)  # Max 0.2 boost for very long content

        # Boost for structured information (lists, steps, etc.)
        structure_boost = (
            0.1 if any(marker in content for marker in ["\n-", "\n1.", "\n2.", ":\n"]) else 0
        )

        # Calculate final importance
        final_importance = min(
            1.0,
            base_importance + importance_boost + personal_boost + length_boost + structure_boost,
        )

        return round(final_importance, 2)

    def _analyze_content(self, content: str) -> dict[str, Any]:
        """Analyze content and extract useful metadata.

        Comprehensive content analysis for rich metadata extraction.
        Used when auto_classify is enabled.

        Args:
            content: Text to analyze

        Returns:
            Metadata dictionary containing:
            - sentence_count: Number of sentences
            - avg_word_length: Average word length
            - has_questions: Contains questions
            - has_urls: Contains URLs
            - has_code: Contains code snippets
            - language_detected: Programming language or 'natural'
            - entities: Extracted entities (emails, phones, URLs)

        Entity Extraction:
            - Email addresses
            - Phone numbers
            - URLs
            - Limited to 10 entities to prevent bloat
        """
        analysis = {}

        # Basic text analysis
        words = content.split()
        sentences = content.split(".") if "." in content else [content]

        analysis.update(
            {
                "sentence_count": len([s for s in sentences if s.strip()]),
                "avg_word_length": (
                    round(sum(len(word) for word in words) / len(words), 1) if words else 0
                ),
                "has_questions": "?" in content,
                "has_urls": "http" in content.lower() or "www." in content.lower(),
                "has_code": any(
                    marker in content for marker in ["def ", "class ", "import ", "function", "=>"]
                ),
                "language_detected": self._detect_language_hints(content),
            }
        )

        # Extract entities (simple approach)
        entities = self._extract_simple_entities(content)
        if entities:
            analysis["entities"] = entities

        return analysis

    def _detect_language_hints(self, content: str) -> str:
        """Detect programming language hints in content.

        Identifies programming languages mentioned or demonstrated.

        Args:
            content: Text to analyze

        Returns:
            Detected language name or 'natural' for non-code

        Supported Languages:
            - Python: 'def', 'import', '.py'
            - JavaScript: 'function', 'const', '.js'
            - Java: 'public class', '.java'
            - C++: '#include', '.cpp'
            - SQL: 'SELECT', 'FROM'
            - Bash: '#!/bin/bash', '.sh'
        """
        content_lower = content.lower()

        language_patterns = {
            "python": ["def ", "import ", "python", ".py", "pip"],
            "javascript": ["function", "const ", "let ", "var ", ".js", "npm"],
            "java": ["public class", "import java", ".java", "maven"],
            "cpp": ["#include", "int main", ".cpp", ".hpp"],
            "sql": ["select ", "from ", "where ", "insert ", "update "],
            "bash": ["#!/bin/bash", "chmod", "sudo", ".sh"],
        }

        for language, patterns in language_patterns.items():
            if any(pattern in content_lower for pattern in patterns):
                return language

        return "natural"

    def _extract_simple_entities(self, content: str) -> list[str]:
        """Extract simple entities like names, places, etc.

        Basic entity extraction using regex patterns.

        Args:
            content: Text to extract entities from

        Returns:
            List of entity strings with type prefixes

        Extracted Entities:
            - Emails: 'email:address@domain.com'
            - URLs: 'url:https://example.com'
            - Phone numbers: 'phone:123-456-7890'

        Limitations:
            - Simple pattern matching only
            - No NER or advanced extraction
            - Limited to 10 entities
        """
        entities = []

        # Simple patterns for common entities
        import re

        # Email addresses
        emails = re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", content)
        entities.extend([f"email:{email}" for email in emails])

        # URLs
        urls = re.findall(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            content,
        )
        entities.extend([f"url:{url}" for url in urls])

        # Phone numbers (simple pattern)
        phones = re.findall(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", content)
        entities.extend([f"phone:{phone}" for phone in phones])

        return entities[:10]  # Limit to prevent metadata bloat

    def _analyze_content_minimal(self, content: str) -> dict[str, Any]:
        """Minimal content analysis for speed.

        Fast, lightweight analysis when performance is critical.
        Skips expensive operations like entity extraction.

        Args:
            content: Text to analyze

        Returns:
            Basic metadata with:
            - word_count: Number of words
            - has_questions: Contains '?'
            - has_code: Basic code detection

        Performance:
            ~1-2ms vs ~10-20ms for full analysis
        """
        analysis = {}

        # Basic analysis only
        words = content.split()
        analysis.update(
            {
                "word_count": len(words),
                "has_questions": "?" in content,
                "has_code": any(
                    marker in content for marker in ["def ", "class ", "import ", "function"]
                ),
            }
        )

        return analysis
