"""Retrieval Augmented Generation (RAG) system with intelligent memory integration."""

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
    """Context information for retrieved memories."""
    
    memories: List[Memory]
    query: str
    relevance_scores: List[float] = field(default_factory=list)
    total_retrieved: int = 0
    retrieval_time_ms: float = 0.0
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGConfig:
    """Configuration for RAG system behavior."""
    
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
    """Advanced Retrieval Augmented Generation system."""
    
    def __init__(self, memory_manager: MemoryManager, config: Optional[RAGConfig] = None):
        """Initialize RAG system.
        
        Args:
            memory_manager: Memory manager for retrieval
            config: RAG configuration
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
        
        Args:
            query: User query or current message
            conversation_history: Recent conversation for context
            custom_config: Override default config for this retrieval
            
        Returns:
            Retrieved context with relevant memories
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
        
        Args:
            original_prompt: Original user prompt/query
            context: Retrieved context from retrieve_context()
            system_prompt: Existing system prompt to enhance
            
        Returns:
            Enhanced prompt with context integration
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
        
        Args:
            query: User query
            conversation_history: Recent conversation
            llm_provider: LLM provider for generation
            model: Model name to use
            system_prompt: System prompt
            
        Returns:
            Tuple of (response, retrieval_context)
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
        """Build enhanced query with conversation context."""
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
        """Perform the actual memory retrieval with priority for personal information."""
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
        """Calculate relevance scores for retrieved memories."""
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
        """Generate explanation for why these memories were retrieved."""
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
        """Build formatted memory context for prompt enhancement."""
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
        """Add citations to response."""
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
        """Apply temporal weighting to memory importance."""
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
        """Remove very similar memories to avoid redundancy."""
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
        """Calculate simple content similarity."""
        # Simple word overlap similarity
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _update_avg_retrieval_time(self, retrieval_time_ms: float):
        """Update average retrieval time statistic."""
        current_avg = self._retrieval_stats["avg_retrieval_time_ms"]
        total_queries = self._retrieval_stats["total_queries"]
        
        # Calculate new average
        new_avg = ((current_avg * (total_queries - 1)) + retrieval_time_ms) / total_queries
        self._retrieval_stats["avg_retrieval_time_ms"] = new_avg
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics."""
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
        """Clear retrieval cache."""
        self._retrieval_cache.clear()
        logger.info("RAG cache cleared")
    
    def update_config(self, new_config: RAGConfig):
        """Update RAG configuration."""
        self.config = new_config
        self.clear_cache()  # Clear cache when config changes
        logger.info("RAG configuration updated")
    
    async def get_relevant_memories(self, query: str, limit: int = None) -> List[Memory]:
        """Get relevant memories for a query (convenience method).
        
        Args:
            query: Search query
            limit: Maximum memories to return (uses config default if None)
            
        Returns:
            List of relevant memories
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