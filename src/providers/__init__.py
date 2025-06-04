"""
LLM Provider System - Enterprise-Grade Abstraction Layer

This module defines the core abstractions for the Neuromancer LLM provider system,
implementing a Protocol-based architecture that enables seamless integration with
multiple Large Language Model providers while maintaining type safety and consistency.

Architecture Philosophy:
======================

The provider system uses Python's Protocol typing to define behavioral contracts
rather than inheritance hierarchies. This design choice offers several advantages:

1. **Duck Typing with Type Safety**: Protocols enable structural subtyping,
   allowing any class that implements the required methods to be considered
   a valid provider without explicit inheritance.

2. **Flexibility**: New providers can be added without modifying existing code,
   and providers can implement additional methods beyond the protocol requirements.

3. **Testability**: Protocols make it easy to create mock providers for testing
   without complex inheritance hierarchies.

4. **Performance**: No runtime overhead from abstract base classes or multiple
   inheritance complications.

5. **Vendor Independence**: The abstraction layer isolates the application from
   vendor-specific APIs, making it easy to switch providers or support new ones.

Design Patterns:
===============

- **Strategy Pattern**: Each provider implements the same interface but with
  different underlying implementations (OpenAI API, Ollama local, etc.)
- **Adapter Pattern**: Providers adapt vendor-specific APIs to our common interface
- **Factory Pattern**: Provider selection and instantiation is handled by factory
  methods based on configuration

Thread Safety & Concurrency:
============================

All provider implementations must be async-first and thread-safe. The protocol
enforces async methods to ensure:
- Non-blocking I/O operations
- Proper resource cleanup
- Graceful error handling
- Responsive UI during long-running operations

Error Handling Strategy:
=======================

Providers should implement comprehensive error handling:
- Network timeouts and connection failures
- API rate limiting and quota exceeded
- Authentication and authorization errors
- Model availability and capability limitations
- Graceful degradation when services are unavailable
"""

from abc import abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol


@dataclass
class Message:
    """
    Represents a single message in a conversation.
    
    This standardized message format ensures consistency across all providers
    while supporting extensibility through metadata.
    
    Attributes:
        role: The message role following OpenAI convention ('user', 'assistant', 'system')
              - 'system': System prompts and instructions
              - 'user': Human input and queries  
              - 'assistant': AI model responses
        content: The actual text content of the message
        metadata: Optional provider-specific or application-specific metadata
                 Examples: timestamps, model parameters, tool calls, attachments
    
    Design Notes:
        - Immutable dataclass for thread safety
        - Role validation should be handled by providers based on their capabilities
        - Content is always string to maintain simplicity, binary data goes in metadata
        - Metadata dict allows for future extensibility without breaking changes
    """

    role: str  # 'user', 'assistant', 'system'
    content: str
    metadata: dict[str, Any] | None = None


@dataclass
class CompletionResponse:
    """
    Standardized response from LLM completion operations.
    
    This unified response format abstracts away provider-specific response
    structures while preserving important metadata for monitoring and analytics.
    
    Attributes:
        content: The generated text response from the model
        model: Identifier of the specific model used (important for billing/analytics)
        usage: Token consumption statistics for cost tracking and optimization
              Standard keys: 'prompt_tokens', 'completion_tokens', 'total_tokens'
        metadata: Provider-specific response metadata
                 Examples: finish_reason, response_id, latency, quality_scores
    
    Usage Patterns:
        - Always check content is not None before using
        - Use usage data for cost monitoring and rate limiting
        - Log metadata for debugging and performance analysis
        - Response objects should be considered immutable after creation
    
    Performance Considerations:
        - Large responses should stream content rather than buffer entirely
        - Usage statistics enable token-based rate limiting
        - Metadata can include timing information for latency monitoring
    """

    content: str
    model: str
    usage: dict[str, int] | None = None
    metadata: dict[str, Any] | None = None


class LLMProvider(Protocol):
    """
    Protocol defining the contract for Large Language Model providers.
    
    This protocol establishes the behavioral interface that all LLM providers
    must implement to integrate with the Neuromancer system. The protocol-based
    approach provides strong typing guarantees while maintaining flexibility.
    
    Implementation Requirements:
    ===========================
    
    All providers MUST implement these methods with the exact signatures.
    Providers MAY implement additional methods for extended functionality.
    
    Async Requirements:
        - All methods must be async to prevent blocking the UI thread
        - Providers should handle their own connection pooling and lifecycle
        - Proper cleanup of resources in finally blocks or context managers
    
    Error Handling:
        - Raise specific exceptions for different error conditions
        - Network errors, authentication failures, rate limits, etc.
        - Include original provider errors in exception chains
        - Log errors with appropriate severity levels
    
    Configuration:
        - Accept configuration through constructor parameters
        - Support environment variable fallbacks for sensitive data
        - Validate configuration during initialization when possible
    
    Testing:
        - Implement health_check for monitoring and diagnostics
        - Support mock/test modes for development and CI/CD
        - Provide deterministic behavior when needed for testing
    
    Performance:
        - Implement connection reuse and pooling where appropriate
        - Support request timeouts and cancellation
        - Handle rate limiting gracefully with exponential backoff
        - Consider caching for model lists and static data
    
    Provider-Specific Considerations:
    ================================
    
    Local Providers (Ollama, LM Studio):
        - Handle service availability checks
        - Manage local model downloads and updates
        - Deal with varying response times based on hardware
        - Support offline operation when possible
    
    Cloud Providers (OpenAI, Anthropic):
        - Implement robust retry logic for network issues
        - Handle API versioning and deprecation gracefully
        - Manage authentication token refresh
        - Respect rate limits and quotas
        - Support multiple API endpoints for load balancing
    """

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> CompletionResponse:
        """
        Generate a complete response from a list of messages.
        
        This method performs a full completion request and returns the entire
        response at once. Use this for shorter interactions where streaming
        is not required.
        
        Args:
            messages: Conversation history as a list of Message objects
                     Order matters - messages should be chronologically ordered
                     System messages typically come first
            model: Model identifier (provider-specific format)
                  Examples: "gpt-4", "llama2:7b", "claude-3-sonnet"
            temperature: Randomness control (0.0-2.0, typically 0.0-1.0)
                        0.0 = deterministic, 1.0 = very creative
            max_tokens: Maximum tokens to generate (None = provider default)
                       Consider context window limits and cost implications
            **kwargs: Provider-specific parameters
                     Examples: top_p, frequency_penalty, presence_penalty
        
        Returns:
            CompletionResponse with the generated content and metadata
        
        Raises:
            ConnectionError: Network or service unavailable
            AuthenticationError: Invalid credentials or expired tokens
            RateLimitError: API quota exceeded or rate limited
            ValidationError: Invalid parameters or model not found
            ModelError: Model-specific errors (context too long, etc.)
        
        Implementation Notes:
            - Validate all parameters before making API calls
            - Convert internal Message format to provider's expected format
            - Handle provider-specific parameter mapping
            - Implement exponential backoff for retryable errors
            - Log request/response for debugging (excluding sensitive data)
            - Clean up resources even if exceptions occur
        
        Example Usage:
            ```python
            messages = [
                Message(role="system", content="You are a helpful assistant"),
                Message(role="user", content="What is 2+2?")
            ]
            response = await provider.complete(messages, "gpt-4", temperature=0.1)
            print(response.content)  # "2+2 equals 4."
            ```
        """
        ...

    @abstractmethod
    async def stream_complete(
        self,
        messages: list[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Generate a streaming response from a list of messages.
        
        This method returns an async iterator that yields content chunks as they
        become available. Use this for longer responses to provide real-time
        feedback to users and reduce perceived latency.
        
        Args:
            messages: Conversation history (same as complete())
            model: Model identifier (same as complete())
            temperature: Randomness control (same as complete())
            max_tokens: Maximum tokens to generate (same as complete())
            **kwargs: Provider-specific parameters (same as complete())
        
        Yields:
            str: Individual content chunks as they're generated
                 Chunks may be words, phrases, or even single characters
                 Empty chunks should be filtered out by the provider
                 Final chunk may be empty to signal completion
        
        Raises:
            Same exceptions as complete(), plus:
            StreamingError: Stream-specific errors (connection dropped, etc.)
        
        Implementation Notes:
            - Yield chunks as soon as they're available
            - Handle connection drops gracefully
            - Support stream cancellation via asyncio cancellation
            - Buffer chunks if necessary but prioritize low latency
            - Clean up streaming connections in finally blocks
            - Consider implementing stream heartbeats for long delays
        
        Stream Management:
            - Streams should be cancellable via asyncio task cancellation
            - Always clean up network connections when stream ends
            - Handle partial responses gracefully
            - Log stream statistics (chunks, duration) for monitoring
        
        Example Usage:
            ```python
            async for chunk in provider.stream_complete(messages, "gpt-4"):
                print(chunk, end="", flush=True)  # Real-time output
            print()  # New line when complete
            ```
        
        Error Recovery:
            - Implement retry logic for transient network errors
            - Provide fallback to complete() if streaming fails
            - Cache partial responses for replay if connection drops
        """
        ...

    @abstractmethod
    async def list_models(self) -> list[str]:
        """
        Retrieve a list of available models from the provider.
        
        This method should return all models that can be used with the
        complete() and stream_complete() methods. The list should be
        filtered to only include compatible models.
        
        Returns:
            list[str]: Model identifiers in provider-specific format
                      Should be sorted by relevance or alphabetically
                      Empty list if no models available or service down
        
        Raises:
            ConnectionError: Cannot reach the provider service
            AuthenticationError: Invalid credentials (for cloud providers)
        
        Implementation Notes:
            - Cache results for a reasonable time (5-15 minutes)
            - Filter out incompatible or deprecated models
            - Handle service unavailability gracefully
            - Sort models by capability or popularity when possible
            - Include only models that support chat completion
        
        Provider-Specific Behavior:
            - OpenAI: Filter to chat models (gpt-*, not davinci-*)
            - Ollama: Return locally downloaded models
            - LM Studio: Return currently loaded models
            - Anthropic: Return available Claude variants
        
        Example Usage:
            ```python
            models = await provider.list_models()
            if "gpt-4" in models:
                # Use GPT-4 for complex tasks
                pass
            ```
        
        Caching Strategy:
            - Implement time-based cache expiration
            - Cache should be instance-specific, not global
            - Invalidate cache on authentication changes
            - Consider cache warming during provider initialization
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Perform a health check to verify provider availability and functionality.
        
        This method should quickly verify that the provider is operational
        and can handle requests. It's used by the system for monitoring,
        provider selection, and error recovery.
        
        Returns:
            bool: True if provider is healthy and operational
                 False if provider is unavailable or malfunctioning
        
        Implementation Strategy:
            - Implement as a lightweight operation (< 5 seconds timeout)
            - Test core functionality without expensive operations
            - Don't make actual completion requests unless necessary
            - Cache results briefly to avoid excessive health checks
        
        Health Check Criteria:
            - Service is reachable and responding
            - Authentication is valid (for cloud providers)
            - At least one model is available
            - Core API endpoints are functional
        
        Example Implementations:
            - OpenAI: Call models.list() endpoint
            - Ollama: Check /api/tags endpoint
            - LM Studio: Verify server is running and has loaded models
        
        Error Handling:
            - Never raise exceptions from health checks
            - Log health check failures for debugging
            - Return False for any error condition
            - Implement request timeout to prevent hanging
        
        Usage in System:
            ```python
            if await provider.health_check():
                response = await provider.complete(messages, model)
            else:
                # Fall back to alternative provider
                pass
            ```
        
        Monitoring Integration:
            - Health check results can be exposed to monitoring systems
            - Track success rates and response times
            - Alert on sustained health check failures
            - Use for automatic provider failover
        """
        ...
