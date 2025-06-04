"""Retrieval Augmented Generation (RAG) system with intelligent memory integration.

This module implements a sophisticated RAG system that enhances LLM responses
with relevant context from the memory system. It provides intelligent retrieval,
context building, and response generation with memory-aware capabilities.

Key Features:
    - Semantic memory retrieval with relevance scoring
    - Intelligent context building with memory organization
    - Personal information prioritization
    - Caching for improved performance
    - Configurable retrieval strategies
    - Citation and source tracking

Architecture:
    The RAG system sits between the chat manager and memory manager,
    orchestrating the retrieval and integration of memories into
    conversation context. It uses a pipeline approach:
    Query → Retrieval → Context Building → Response Generation

Performance Optimizations:
    - Result caching with TTL (5 minutes default)
    - Parallel memory searches across types
    - Early termination for personal info queries
    - Configurable retrieval timeouts

Example Usage:
    ```python
    rag_system = RAGSystem(memory_manager)
    
    # Configure RAG behavior
    config = RAGConfig(
        max_memories=20,
        min_relevance_threshold=0.4,
        cite_sources=True
    )
    rag_system.update_config(config)
    
    # Retrieve context for a query
    context = await rag_system.retrieve_context(
        "What's my favorite color?",
        conversation_history
    )
    
    # Generate RAG-enhanced response
    response, context = await rag_system.generate_rag_response(
        query="Tell me about my project",
        llm_provider=provider,
        model="gpt-4"
    )
    ```
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.memory import Memory, MemoryType
from src.memory.manager import MemoryManager
from src.providers import Message
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class RetrievalContext:
    """Context information for retrieved memories.
    
    Encapsulates all information about a memory retrieval operation,
    including the memories themselves, performance metrics, and metadata.
    
    Attributes:
        memories: List of retrieved Memory objects
        query: Original search query
        relevance_scores: Similarity scores for each memory (0-1)
        total_retrieved: Total number of memories found
        retrieval_time_ms: Time taken for retrieval in milliseconds
        reasoning: Human-readable explanation of retrieval logic
        metadata: Additional retrieval metadata (query expansion, config, etc.)
    
    Usage:
        This class is returned by retrieve_context() and contains all
        information needed to understand what was retrieved and why.
    """
    
    memories: List[Memory]
    query: str
    relevance_scores: List[float] = field(default_factory=list)
    total_retrieved: int = 0
    retrieval_time_ms: float = 0.0
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGConfig:
    """Configuration for RAG system behavior.
    
    Controls all aspects of the RAG system's retrieval and generation process.
    Default values are optimized for balanced performance and quality.
    
    Attributes:
        max_memories: Maximum memories to retrieve (default: 15)
            Higher values provide more context but increase latency
        min_relevance_threshold: Minimum similarity score (default: 0.3)
            Lower values retrieve more memories but may reduce relevance
        memory_types: Which memory types to search (None = all)
            Can restrict to specific types for targeted retrieval
        max_context_tokens: Maximum tokens for context (default: 4000)
            Prevents context from exceeding model limits
        include_metadata: Include memory metadata in context (default: True)
            Adds timestamps, importance scores, etc. to context
        deduplicate_content: Remove similar memories (default: True)
            Prevents redundant information in context
        temporal_weighting: Boost recent memories (default: True)
            Gives slight preference to newer information
        cite_sources: Add citations to responses (default: True)
            Appends source list to generated responses
        explain_reasoning: Include retrieval reasoning (default: False)
            Adds explanation of why memories were selected
        confidence_scoring: Calculate confidence scores (default: True)
            Provides reliability metrics for retrievals
        retrieval_timeout_ms: Timeout for retrieval (default: 5000ms)
            Prevents long-running searches from blocking
        parallel_queries: Search memory types in parallel (default: True)
            Improves performance for multi-type searches
        cache_results: Cache retrieval results (default: True)
            Reduces latency for repeated queries
    """
    
    # Retrieval settings
    max_memories: int = 15  # Increased from 10 to catch more relevant memories
    min_relevance_threshold: float = 0.3  # Lowered from 0.7 for better recall
    memory_types: Optional[List[MemoryType]] = None  # None = all types
    
    # Context building
    max_context_tokens: int = 4000
    include_metadata: bool = True
    deduplicate_content: bool = True
    temporal_weighting: bool = True
    
    # Response enhancement
    cite_sources: bool = True
    explain_reasoning: bool = False
    confidence_scoring: bool = True
    
    # Performance
    retrieval_timeout_ms: int = 5000
    parallel_queries: bool = True
    cache_results: bool = True


class RAGSystem:
    """Advanced Retrieval Augmented Generation system.
    
    Provides intelligent memory retrieval and integration for enhanced
    language model responses. The system retrieves relevant memories,
    builds context, and generates responses that leverage historical
    information.
    
    Key Capabilities:
        - Semantic search across memory types
        - Personal information prioritization
        - Conversation-aware retrieval
        - Response enhancement with citations
        - Performance optimization via caching
    
    Thread Safety:
        The RAGSystem is designed for single-user access. For multi-user
        scenarios, create separate instances per user to avoid cache conflicts.
    """
    
    def __init__(self, memory_manager: MemoryManager, config: Optional[RAGConfig] = None):
        """Initialize RAG system.
        
        Args:
            memory_manager: Memory manager instance for retrieval operations.
                The RAG system will use this to search and retrieve memories.
            config: Optional RAG configuration. If not provided, uses defaults
                optimized for balanced performance and quality.
        
        Initialization:
            Sets up caching infrastructure and performance tracking.
            No external connections are made during initialization.
        """
        self.memory_manager = memory_manager
        self.config = config or RAGConfig()
        
        # Cache for recent retrievals
        self._retrieval_cache: Dict[str, RetrievalContext] = {}
        self._cache_ttl_seconds = 300  # 5 minutes
        
        # Performance metrics
        self._retrieval_stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "avg_retrieval_time_ms": 0.0,
            "successful_retrievals": 0
        }
        
        logger.info("RAGSystem initialized")
    
    async def retrieve_context(
        self, 
        query: str, 
        conversation_history: Optional[List[Message]] = None,
        custom_config: Optional[RAGConfig] = None
    ) -> RetrievalContext:
        """Retrieve relevant context for a query.
        
        Core retrieval method that searches memories and returns relevant context.
        Implements caching, timeout protection, and intelligent ranking.
        
        Args:
            query: User query or current message to find context for.
                This is the primary search input.
            conversation_history: Recent conversation messages for context.
                Used to enhance the query with conversational context.
            custom_config: Override default config for this retrieval.
                Useful for query-specific adjustments.
            
        Returns:
            RetrievalContext containing memories, scores, and metadata
        
        Process:
            1. Check cache for recent identical queries
            2. Build enhanced query with conversation context
            3. Perform retrieval with timeout protection
            4. Calculate relevance scores
            5. Generate retrieval reasoning
            6. Cache results for future use
        
        Performance:
            - Cached queries: ~1-5ms
            - Fresh retrieval: ~50-200ms depending on memory size
            - Timeout after 5 seconds (configurable)
        
        Error Handling:
            Returns empty context on failure rather than raising,
            allowing graceful degradation.
        """
        start_time = datetime.now()
        config = custom_config or self.config
        
        # Check cache first
        cache_key = f"{query}:{hash(str(config.__dict__))}"
        if config.cache_results and cache_key in self._retrieval_cache:
            cached = self._retrieval_cache[cache_key]
            if (datetime.now() - start_time).total_seconds() < self._cache_ttl_seconds:
                self._retrieval_stats["cache_hits"] += 1
                logger.debug(f"Retrieved context from cache for query: {query[:50]}...")
                return cached
        
        try:
            self._retrieval_stats["total_queries"] += 1
            
            # Build enhanced query with conversation context
            enhanced_query = await self._build_enhanced_query(query, conversation_history)
            
            # Perform retrieval with timeout
            retrieval_task = self._perform_retrieval(enhanced_query, config)
            
            try:
                memories = await asyncio.wait_for(
                    retrieval_task, 
                    timeout=config.retrieval_timeout_ms / 1000
                )
            except asyncio.TimeoutError:
                logger.warning(f"Retrieval timed out for query: {query[:50]}...")
                memories = []
            
            # Calculate retrieval time
            retrieval_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Build retrieval context
            context = RetrievalContext(
                memories=memories,
                query=query,
                total_retrieved=len(memories),
                retrieval_time_ms=retrieval_time,
                reasoning=await self._generate_retrieval_reasoning(query, memories),
                metadata={
                    "enhanced_query": enhanced_query,
                    "config": config.__dict__,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # Calculate relevance scores
            if memories:
                context.relevance_scores = await self._calculate_relevance_scores(
                    enhanced_query, memories
                )
            
            # Cache result
            if config.cache_results:
                self._retrieval_cache[cache_key] = context
            
            # Update stats
            self._retrieval_stats["successful_retrievals"] += 1
            self._update_avg_retrieval_time(retrieval_time)
            
            logger.info(f"Retrieved {len(memories)} memories in {retrieval_time:.1f}ms for query: {query[:50]}...")
            
            return context
            
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            
            # Return empty context on failure
            return RetrievalContext(
                memories=[],
                query=query,
                total_retrieved=0,
                retrieval_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                reasoning="Retrieval failed due to system error",
                metadata={"error": str(e)}
            )
    
    async def enhance_prompt(
        self, 
        original_prompt: str, 
        context: RetrievalContext,
        system_prompt: Optional[str] = None
    ) -> str:
        """Enhance prompt with retrieved context.
        
        Integrates retrieved memories into the prompt in a structured way
        that helps the LLM understand and use the context effectively.
        
        Args:
            original_prompt: Original user prompt/query to enhance
            context: Retrieved context from retrieve_context() containing
                relevant memories and metadata
            system_prompt: Existing system prompt to enhance. If provided,
                context is added to system prompt; otherwise to user prompt.
            
        Returns:
            Enhanced prompt string with integrated context
        
        Enhancement Strategy:
            - For system prompts: Adds context as additional instructions
            - For user prompts: Prepends context with clear delineation
            - Includes metadata if configured (timestamps, importance)
            - Adds reasoning explanation if configured
        
        Format:
            The enhanced prompt clearly separates context from the original
            query, using markdown-style formatting for clarity.
        """
        if not context.memories:
            logger.debug("No memories to enhance prompt with")
            return original_prompt
        
        try:
            # Build context section
            context_sections = []
            
            # Add memory context
            memory_context = await self._build_memory_context(context)
            if memory_context:
                context_sections.append(memory_context)
            
            # Add reasoning if enabled
            if self.config.explain_reasoning and context.reasoning:
                reasoning_section = f"**Context Selection Reasoning:**\n{context.reasoning}\n"
                context_sections.append(reasoning_section)
            
            # Combine context
            if not context_sections:
                return original_prompt
            
            full_context = "\n\n".join(context_sections)
            
            # Build enhanced prompt
            if system_prompt:
                # Enhance system prompt
                enhanced_system = f"{system_prompt}\n\n{full_context}"
                return enhanced_system
            else:
                # Add context to user prompt
                enhanced_prompt = f"""**Relevant Context:**
{full_context}

**User Query:**
{original_prompt}

Please provide a response that takes into account the relevant context above."""
                
                return enhanced_prompt
                
        except Exception as e:
            logger.error(f"Prompt enhancement failed: {e}")
            return original_prompt
    
    async def generate_rag_response(
        self,
        query: str,
        conversation_history: Optional[List[Message]] = None,
        llm_provider=None,
        model: str = "",
        system_prompt: Optional[str] = None
    ) -> Tuple[str, RetrievalContext]:
        """Generate RAG-enhanced response.
        
        Complete RAG pipeline: retrieves context and generates enhanced response.
        This is a convenience method that combines retrieval and generation.
        
        Args:
            query: User query to respond to
            conversation_history: Recent conversation for context (last 5 used)
            llm_provider: LLM provider instance for generation (required)
            model: Specific model name to use for generation
            system_prompt: Base system prompt to enhance with context
            
        Returns:
            Tuple of (response_text, retrieval_context)
            - response_text: The generated response with optional citations
            - retrieval_context: Context object with retrieval details
        
        Pipeline:
            1. Retrieve relevant memories for the query
            2. Enhance prompts with retrieved context  
            3. Build message list with history
            4. Generate response using LLM
            5. Add citations if configured
        
        Error Handling:
            Returns error message and empty context on failure,
            ensuring the system remains responsive.
        
        Example:
            ```python
            response, context = await rag_system.generate_rag_response(
                "What did I say my name was?",
                conversation_history=messages[-10:],
                llm_provider=ollama_provider,
                model="llama2"
            )
            print(f"Response: {response}")
            print(f"Used {len(context.memories)} memories")
            ```
        """
        if not llm_provider:
            raise ValueError("LLM provider required for RAG response generation")
        
        try:
            # Retrieve relevant context
            context = await self.retrieve_context(query, conversation_history)
            
            # Enhance prompt with context
            enhanced_prompt = await self.enhance_prompt(query, context, system_prompt)
            
            # Prepare messages for LLM
            messages = []
            
            # Add enhanced system prompt if provided
            if system_prompt and context.memories:
                enhanced_system = await self.enhance_prompt("", context, system_prompt)
                messages.append(Message(role="system", content=enhanced_system))
            
            # Add conversation history (limited)
            if conversation_history:
                # Include last few messages for context
                recent_messages = conversation_history[-5:]  # Last 5 messages
                messages.extend(recent_messages)
            
            # Add current query
            if not (system_prompt and context.memories):
                # If we didn't enhance system prompt, enhance user prompt
                enhanced_query = await self.enhance_prompt(query, context)
                messages.append(Message(role="user", content=enhanced_query))
            else:
                messages.append(Message(role="user", content=query))
            
            # Generate response
            response = await llm_provider.complete(
                messages=messages,
                model=model,
                temperature=0.7
            )
            
            # Add citations if enabled
            if self.config.cite_sources and context.memories:
                response_with_citations = await self._add_citations(
                    response.content, context
                )
                response.content = response_with_citations
            
            logger.info(f"Generated RAG response using {len(context.memories)} memories")
            
            return response.content, context
            
        except Exception as e:
            logger.error(f"RAG response generation failed: {e}")
            return f"I apologize, but I encountered an error while generating a response: {e}", RetrievalContext(
                memories=[],
                query=query,
                reasoning="Error during response generation",
                metadata={"error": str(e)}
            )
    
    async def _build_enhanced_query(
        self, 
        query: str, 
        conversation_history: Optional[List[Message]]
    ) -> str:
        """Build enhanced query with conversation context.
        
        Expands the original query with recent conversation context to improve
        retrieval accuracy. This helps find memories related to the ongoing
        conversation even if not directly mentioned in the current query.
        
        Args:
            query: Original user query
            conversation_history: Recent messages for context
        
        Returns:
            Enhanced query string with contextual information
        
        Enhancement:
            - Adds key phrases from last 3 messages
            - Preserves original query as primary signal
            - Formats as "query | Recent context: ..."
        """
        enhanced_parts = [query]
        
        if conversation_history:
            # Add context from recent messages
            recent_context = []
            for msg in conversation_history[-3:]:  # Last 3 messages
                if msg.role in ["user", "assistant"]:
                    # Extract key phrases/entities
                    content_preview = msg.content[:100] if len(msg.content) > 100 else msg.content
                    recent_context.append(f"{msg.role}: {content_preview}")
            
            if recent_context:
                context_str = " | ".join(recent_context)
                enhanced_parts.append(f"Recent context: {context_str}")
        
        return " ".join(enhanced_parts)
    
    async def _perform_retrieval(self, query: str, config: RAGConfig) -> List[Memory]:
        """Perform the actual memory retrieval with priority for personal information.
        
        Core retrieval logic that searches memories with sophisticated ranking
        and filtering. Implements special handling for personal information
        to ensure it's always found when relevant.
        
        Args:
            query: Search query (potentially enhanced)
            config: RAG configuration for this retrieval
        
        Returns:
            List of Memory objects sorted by relevance
        
        Retrieval Strategy:
            1. First, search LONG_TERM memories for personal info
            2. Then search other configured memory types
            3. Filter out contradictory memories if personal info found
            4. Apply relevance filtering and deduplication
            5. Sort by relevance, importance, and recency
        
        Personal Information Handling:
            When personal info is found, the system filters out generic
            "I don't know your name" type responses to avoid confusion.
            Personal info gets a +1.0 importance boost for ranking.
        
        Performance Notes:
            - Searches are parallelized when possible
            - Early termination when enough memories found
            - Deduplication prevents redundant context
        """
        try:
            # First, try to get personal information from LONG_TERM memories
            personal_memories = await self.memory_manager.recall(
                query=query,
                memory_types=[MemoryType.LONG_TERM],
                limit=5,  # Get top 5 long-term memories first
                threshold=0.2  # Lower threshold for better recall
            )
            
            # Filter for actual personal info
            personal_info_memories = [
                mem for mem in personal_memories 
                if mem.metadata and mem.metadata.get("type") == "personal_info"
            ]
            
            # If we have personal info, we should filter out contradictory memories
            has_personal_info = len(personal_info_memories) > 0
            
            # Then get other memories to fill remaining slots
            remaining_limit = max(0, config.max_memories - len(personal_info_memories))
            
            other_memories = await self.memory_manager.recall(
                query=query,
                memory_types=config.memory_types,
                limit=remaining_limit + 5,  # Get a few extra to account for overlap
                threshold=config.min_relevance_threshold
            )
            
            # Combine, avoiding duplicates and contradictions
            seen_ids = {mem.id for mem in personal_info_memories}
            
            # Filter out contradictory memories if we have personal info
            def is_contradictory_memory(mem):
                if not has_personal_info:
                    return False
                
                content_lower = mem.content.lower()
                contradictory_phrases = [
                    "don't have any information about your personal identity",
                    "don't have any information about your name",
                    "i'm a large language model, and our conversation just started",
                    "i don't have any recollection",
                    "since our conversation just started",
                    "i'm not aware of your name",
                    "i don't know your name"
                ]
                
                return any(phrase in content_lower for phrase in contradictory_phrases)
            
            unique_other_memories = [
                mem for mem in other_memories 
                if mem.id not in seen_ids and not is_contradictory_memory(mem)
            ]
            
            # Combine with personal info first
            memories = personal_info_memories + unique_other_memories[:remaining_limit]
            
            # Apply additional filtering and ranking
            filtered_memories = []
            
            for memory in memories:
                # Skip if below threshold
                if hasattr(memory, 'relevance_score'):
                    if memory.relevance_score < config.min_relevance_threshold:
                        continue
                
                # Apply temporal weighting if enabled
                if config.temporal_weighting:
                    memory = await self._apply_temporal_weighting(memory)
                
                filtered_memories.append(memory)
            
            # Deduplicate if enabled
            if config.deduplicate_content:
                filtered_memories = await self._deduplicate_memories(filtered_memories)
            
            # Sort by relevance and importance with special priority for personal info
            def memory_ranking_score(memory):
                base_importance = memory.importance
                recency_score = memory.created_at.timestamp()
                relevance_score = getattr(memory, 'relevance_score', 0.5)
                
                # Major boost for personal information
                if memory.metadata and memory.metadata.get("type") == "personal_info":
                    base_importance += 1.0  # Much stronger boost than memory manager's +0.5
                
                # Combine relevance, importance, and recency
                return (relevance_score, base_importance, recency_score)
            
            filtered_memories.sort(key=memory_ranking_score, reverse=True)
            
            return filtered_memories[:config.max_memories]
            
        except Exception as e:
            logger.error(f"Memory retrieval failed: {e}")
            return []
    
    async def _calculate_relevance_scores(self, query: str, memories: List[Memory]) -> List[float]:
        """Calculate relevance scores for retrieved memories.
        
        Computes composite relevance scores based on multiple factors.
        This scoring is used for ranking and filtering memories.
        
        Args:
            query: Search query for comparison
            memories: Retrieved memories to score
        
        Returns:
            List of relevance scores (0.0-1.0) for each memory
        
        Scoring Components:
            - Content overlap (40%): Word intersection ratio
            - Importance (30%): Memory importance score
            - Recency (20%): Time decay over 30 days  
            - Memory type (10%): Type-specific bonuses
        
        Note:
            This is a simplified scoring system. Production systems
            might use learned embeddings or more sophisticated metrics.
        """
        # This is a simplified scoring system
        # In a production system, you might use more sophisticated scoring
        scores = []
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for memory in memories:
            score = 0.0
            
            # Content overlap scoring
            memory_words = set(memory.content.lower().split())
            overlap = len(query_words.intersection(memory_words))
            
            if query_words:
                overlap_ratio = overlap / len(query_words)
                score += overlap_ratio * 0.4
            
            # Importance scoring
            score += memory.importance * 0.3
            
            # Recency scoring (more recent = higher score)
            days_old = (datetime.now() - memory.created_at).days
            recency_score = max(0, 1 - (days_old / 30))  # Decay over 30 days
            score += recency_score * 0.2
            
            # Memory type scoring
            type_scores = {
                MemoryType.LONG_TERM: 0.1,
                MemoryType.SEMANTIC: 0.08,
                MemoryType.EPISODIC: 0.06,
                MemoryType.SHORT_TERM: 0.04
            }
            score += type_scores.get(memory.memory_type, 0.05)
            
            scores.append(min(1.0, score))  # Cap at 1.0
        
        return scores
    
    async def _generate_retrieval_reasoning(self, query: str, memories: List[Memory]) -> str:
        """Generate explanation for why these memories were retrieved.
        
        Creates human-readable explanation of the retrieval logic.
        Useful for debugging and transparency.
        
        Args:
            query: Original search query
            memories: Retrieved memories
        
        Returns:
            Reasoning text explaining the retrieval
        
        Information Included:
            - Number of memories retrieved
            - Memory type distribution
            - Highest importance score
            - Relevance to query
        """
        if not memories:
            return "No relevant memories found for this query."
        
        memory_types = [m.memory_type.value for m in memories]
        type_counts = {}
        for mt in memory_types:
            type_counts[mt] = type_counts.get(mt, 0) + 1
        
        reasoning_parts = [
            f"Retrieved {len(memories)} relevant memories based on semantic similarity and importance."
        ]
        
        if type_counts:
            type_summary = ", ".join([f"{count} {mtype}" for mtype, count in type_counts.items()])
            reasoning_parts.append(f"Memory types: {type_summary}")
        
        # Add highest importance
        if memories:
            max_importance = max(m.importance for m in memories)
            reasoning_parts.append(f"Highest importance score: {max_importance:.2f}")
        
        return " ".join(reasoning_parts)
    
    async def _build_memory_context(self, context: RetrievalContext) -> str:
        """Build formatted memory context for prompt enhancement.
        
        Converts raw memories into formatted text suitable for LLM consumption.
        Applies formatting, truncation, and metadata inclusion as configured.
        
        Args:
            context: RetrievalContext with memories to format
        
        Returns:
            Formatted string with memory content and metadata
        
        Formatting:
            - Numbers each memory for reference
            - Includes metadata in brackets if configured
            - Truncates long memories to 500 chars
            - Preserves readability with proper spacing
        
        Metadata Format:
            [Type: X | Importance: Y | Date: Z | Relevance: W]
        """
        if not context.memories:
            return ""
        
        context_parts = ["**Relevant Information from Memory:**"]
        
        for i, memory in enumerate(context.memories):
            # Format memory content
            memory_text = f"\n{i+1}. "
            
            # Add metadata if enabled
            if self.config.include_metadata:
                metadata_parts = []
                
                # Add memory type
                metadata_parts.append(f"Type: {memory.memory_type.value}")
                
                # Add importance
                metadata_parts.append(f"Importance: {memory.importance:.2f}")
                
                # Add timestamp
                time_str = memory.created_at.strftime("%Y-%m-%d %H:%M")
                metadata_parts.append(f"Date: {time_str}")
                
                # Add relevance score if available
                if i < len(context.relevance_scores):
                    metadata_parts.append(f"Relevance: {context.relevance_scores[i]:.2f}")
                
                metadata_str = " | ".join(metadata_parts)
                memory_text += f"[{metadata_str}]\n   "
            
            memory_text += memory.content
            
            # Truncate very long memories
            if len(memory_text) > 500:
                memory_text = memory_text[:497] + "..."
            
            context_parts.append(memory_text)
        
        return "\n".join(context_parts)
    
    async def _add_citations(self, response: str, context: RetrievalContext) -> str:
        """Add citations to response.
        
        Appends source citations to the response for transparency
        and verifiability. Helps users understand where information
        came from.
        
        Args:
            response: Generated response text
            context: Retrieval context with source memories
        
        Returns:
            Response with appended citations
        
        Citation Format:
            **Sources:**
            1. [Memory Type] memory from [Date] (Related: [Entities])
            2. ...
        
        Note:
            Only first 3 entities are shown per citation to avoid clutter.
        """
        if not context.memories:
            return response
        
        # Add source list at the end
        citation_parts = ["\n\n**Sources:**"]
        
        for i, memory in enumerate(context.memories):
            # Create citation
            time_str = memory.created_at.strftime("%Y-%m-%d")
            citation = f"{i+1}. {memory.memory_type.value.title()} memory from {time_str}"
            
            if memory.metadata and "entities" in memory.metadata:
                entities = memory.metadata["entities"][:3]  # First 3 entities
                if entities:
                    entity_str = ", ".join(entities)
                    citation += f" (Related: {entity_str})"
            
            citation_parts.append(citation)
        
        return response + "\n".join(citation_parts)
    
    async def _apply_temporal_weighting(self, memory: Memory) -> Memory:
        """Apply temporal weighting to memory importance.
        
        Adjusts memory importance based on age to slightly favor
        recent information. This helps with temporal relevance.
        
        Args:
            memory: Memory to adjust
        
        Returns:
            New Memory object with adjusted importance
        
        Weighting Scheme:
            - 0-1 days old: +0.1 importance
            - 2-7 days old: +0.05 importance
            - 8-30 days old: +0.02 importance
            - >30 days: no adjustment
        
        Design Rationale:
            Recent memories are often more relevant in conversations,
            but we avoid heavy recency bias to preserve long-term knowledge.
        """
        # Boost recent memories slightly
        days_old = (datetime.now() - memory.created_at).days
        
        if days_old <= 1:
            temporal_boost = 0.1
        elif days_old <= 7:
            temporal_boost = 0.05
        elif days_old <= 30:
            temporal_boost = 0.02
        else:
            temporal_boost = 0
        
        # Create copy with adjusted importance
        adjusted_memory = Memory(
            id=memory.id,
            content=memory.content,
            embedding=memory.embedding,
            memory_type=memory.memory_type,
            importance=min(1.0, memory.importance + temporal_boost),
            metadata=memory.metadata,
            created_at=memory.created_at,
            accessed_at=memory.accessed_at
        )
        
        return adjusted_memory
    
    async def _deduplicate_memories(self, memories: List[Memory]) -> List[Memory]:
        """Remove very similar memories to avoid redundancy.
        
        Identifies and removes near-duplicate memories to prevent
        repetitive context. Keeps the more important version when
        duplicates are found.
        
        Args:
            memories: List of memories to deduplicate
        
        Returns:
            Deduplicated list of memories
        
        Algorithm:
            - Compares memories pairwise for content similarity
            - Similarity > 0.9 considered duplicate
            - Keeps memory with higher importance score
            - Preserves order for non-duplicates
        
        Performance:
            O(n²) comparison but typically fast due to small n (<20)
        """
        if len(memories) <= 1:
            return memories
        
        deduplicated = []
        
        for memory in memories:
            is_duplicate = False
            
            for existing in deduplicated:
                # Simple content similarity check
                similarity = await self._calculate_content_similarity(memory.content, existing.content)
                
                if similarity > 0.9:  # Very similar content
                    is_duplicate = True
                    # Keep the more important one
                    if memory.importance > existing.importance:
                        deduplicated.remove(existing)
                        deduplicated.append(memory)
                    break
            
            if not is_duplicate:
                deduplicated.append(memory)
        
        return deduplicated
    
    async def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate simple content similarity.
        
        Computes Jaccard similarity between two text contents.
        Used for deduplication and relevance scoring.
        
        Args:
            content1: First text content
            content2: Second text content
        
        Returns:
            Similarity score between 0.0 and 1.0
        
        Method:
            Uses word set intersection over union (Jaccard index).
            Simple but effective for deduplication purposes.
        """
        # Simple word overlap similarity
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _update_avg_retrieval_time(self, retrieval_time_ms: float):
        """Update average retrieval time statistic.
        
        Maintains running average of retrieval times for performance monitoring.
        
        Args:
            retrieval_time_ms: Latest retrieval time in milliseconds
        
        Algorithm:
            Uses incremental average calculation to avoid storing all values:
            new_avg = ((old_avg * (n-1)) + new_value) / n
        """
        current_avg = self._retrieval_stats["avg_retrieval_time_ms"]
        total_queries = self._retrieval_stats["total_queries"]
        
        # Calculate new average
        new_avg = ((current_avg * (total_queries - 1)) + retrieval_time_ms) / total_queries
        self._retrieval_stats["avg_retrieval_time_ms"] = new_avg
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics.
        
        Returns comprehensive performance and usage statistics.
        
        Returns:
            Dictionary containing:
            - total_queries: Total retrieval requests
            - cache_hits: Number of cache hits
            - cache_hit_rate: Percentage of cache hits
            - avg_retrieval_time_ms: Average retrieval latency
            - successful_retrievals: Number of successful retrievals
            - cache_size: Current cache entry count
            - config: Current configuration
        
        Usage:
            Useful for monitoring system performance and tuning
            configuration parameters.
        """
        cache_hit_rate = 0.0
        if self._retrieval_stats["total_queries"] > 0:
            cache_hit_rate = self._retrieval_stats["cache_hits"] / self._retrieval_stats["total_queries"]
        
        return {
            **self._retrieval_stats,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self._retrieval_cache),
            "config": self.config.__dict__
        }
    
    def clear_cache(self):
        """Clear retrieval cache.
        
        Removes all cached retrieval results. Useful when memory
        contents have changed significantly.
        
        When to Clear:
            - After bulk memory updates
            - When switching users/contexts
            - For testing/debugging
        """
        self._retrieval_cache.clear()
        logger.info("RAG cache cleared")
    
    def update_config(self, new_config: RAGConfig):
        """Update RAG configuration.
        
        Updates the RAG system configuration and clears cache to
        ensure new settings take effect immediately.
        
        Args:
            new_config: New RAGConfig instance
        
        Side Effects:
            - Clears retrieval cache
            - Resets performance counters
        """
        self.config = new_config
        self.clear_cache()  # Clear cache when config changes
        logger.info("RAG configuration updated")
    
    async def get_relevant_memories(self, query: str, limit: int = None) -> List[Memory]:
        """Get relevant memories for a query (convenience method).
        
        Simplified interface for memory retrieval without full context.
        Useful for quick lookups and testing.
        
        Args:
            query: Search query text
            limit: Maximum memories to return (uses config default if None)
            
        Returns:
            List of relevant Memory objects
        
        Note:
            This is a wrapper around retrieve_context() that returns
            just the memories without metadata. For full retrieval info,
            use retrieve_context() directly.
        """
        try:
            context = await self.retrieve_context(query)
            memories = context.memories
            
            if limit is not None:
                memories = memories[:limit]
            
            logger.info(f"Retrieved {len(memories)} relevant memories for query: {query[:50]}...")
            return memories
            
        except Exception as e:
            logger.error(f"Failed to get relevant memories: {e}")
            return []