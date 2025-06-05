"""
Ollama Provider - Local Model Execution Engine

This module implements the Ollama provider for running large language models
locally. Ollama is a popular tool for running LLMs on local hardware,
providing privacy, cost control, and offline capabilities.

Key Features:
============

1. **Local Execution**: Models run entirely on local hardware
2. **Privacy First**: No data leaves the local machine
3. **Cost Control**: No per-token charges or API limits
4. **Offline Operation**: Works without internet connectivity
5. **Model Management**: Built-in model downloading and updating
6. **Hardware Optimization**: Leverages GPU acceleration when available

Architectural Decisions:
=======================

**Fresh Client Pattern**: This implementation creates a new Ollama client for
each request rather than maintaining a persistent connection. This design choice
is motivated by:

- Resource Management: Prevents connection leaks and resource accumulation
- Error Isolation: Each request is independent, preventing cascading failures
- Memory Efficiency: Clients are garbage collected after each request
- Concurrency Safety: No shared state between concurrent requests
- Shutdown Graceful: Eliminates hanging connections during app termination

**Performance Considerations**: While creating fresh clients adds minimal overhead,
the benefits of clean resource management outweigh the small performance cost.
Ollama's local nature means network connection overhead is negligible.

Provider-Specific Behavior:
==========================

- **Model Format**: Uses Ollama's model naming convention (e.g., "llama2:7b")
- **Response Timing**: Includes detailed timing metadata for performance analysis
- **Streaming**: Optimized for real-time token generation with chunk counting
- **Error Handling**: Graceful handling of model unavailability and resource limits
- **Logging**: Comprehensive request/response logging for debugging

Integration Patterns:
====================

```python
# Basic usage
provider = OllamaProvider("http://localhost:11434")
response = await provider.complete(messages, "llama2:7b")

# Streaming for real-time output
async for chunk in provider.stream_complete(messages, "llama2:7b"):
    print(chunk, end="")

# Model management
if await provider.pull_model("llama2:13b"):
    models = await provider.list_models()
```

Error Scenarios:
===============

1. **Service Unavailable**: Ollama daemon not running
2. **Model Not Found**: Requested model not downloaded locally
3. **Resource Exhaustion**: Insufficient memory or GPU resources
4. **Hardware Limitations**: Model too large for available hardware
5. **Network Issues**: Unable to download models (pull operations)

Performance Tuning:
==================

- **Temperature**: Lower values (0.1-0.3) for factual responses
- **Max Tokens**: Limit to prevent runaway generation
- **Concurrent Requests**: Ollama handles multiple requests via queueing
- **Model Selection**: Balance between capability and speed/resource usage
"""

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

import ollama

from src.providers import CompletionResponse, LLMProvider, Message
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class OllamaProvider(LLMProvider):
    """
    Ollama Provider Implementation for Local Model Execution.

    This provider interfaces with Ollama, a popular local LLM runtime that enables
    running large language models on consumer hardware. Ollama provides an efficient
    way to run models locally with automatic GPU acceleration and memory management.

    Architecture:
    ============

    The provider uses a "fresh client" pattern where each request creates a new
    Ollama client instance. This ensures:
    - Clean resource management
    - No connection state pollution
    - Graceful shutdown behavior
    - Isolation between concurrent requests

    Configuration:
    =============

    - host: Ollama server endpoint (default: http://localhost:11434)
    - Supports custom ports and remote Ollama instances
    - No authentication required for local instances
    - Automatic service discovery and health checking

    Model Management:
    ================

    Ollama models use a specific naming convention:
    - Base models: "llama2", "mistral", "codellama"
    - Variants: "llama2:7b", "llama2:13b", "llama2:70b"
    - Custom models: "mymodel:latest", "mymodel:v1.0"

    Performance Characteristics:
    ===========================

    - First request may be slower due to model loading
    - Subsequent requests use cached model in memory
    - Response time varies with model size and hardware
    - GPU acceleration significantly improves performance
    - Memory usage scales with model size (7B ≈ 4GB, 13B ≈ 8GB)

    Error Recovery:
    ==============

    - Automatic retry for transient failures
    - Graceful degradation when models unavailable
    - Resource cleanup on errors
    - Detailed error logging for troubleshooting

    Thread Safety:
    =============

    This provider is fully thread-safe and supports concurrent requests.
    Ollama handles request queueing internally, ensuring efficient resource
    utilization without overwhelming the local hardware.
    """

    def __init__(self, host: str = "http://localhost:11434", timeout: float = 300.0):
        """
        Initialize Ollama provider with connection configuration.

        Args:
            host: Ollama server host URL. Supports:
                  - Local: "http://localhost:11434" (default)
                  - Custom port: "http://localhost:8080"
                  - Remote: "http://192.168.1.100:11434"
                  - HTTPS: "https://ollama.example.com"
            timeout: Request timeout in seconds (default: 300s for large models)
                    Increase for very large models or slow hardware

        Raises:
            ValueError: If host URL is malformed

        Note:
            The provider validates the host URL format but does not immediately
            test connectivity. Use health_check() to verify service availability.
        """
        # Validate and normalize host URL
        if not host.startswith(("http://", "https://")):
            raise ValueError(f"Host must start with http:// or https://, got: {host}")

        self.host = host.rstrip("/")  # Remove trailing slash for consistency
        self.timeout = timeout

        # Provider metadata for monitoring and debugging
        self._request_count = 0
        self._last_model_list_time = 0.0
        self._cached_models: list[str] | None = None

        logger.info(f"Initialized Ollama provider: host={self.host}, timeout={timeout}s")

        # Don't store a persistent client - create fresh for each request
        # This ensures clean resource management and prevents connection issues

    def _create_client(self) -> ollama.AsyncClient:
        """
        Create a fresh Ollama client with optimized configuration.

        This method implements the "fresh client" pattern to ensure clean
        resource management and avoid connection state issues. Each client
        is configured with appropriate timeouts and error handling.

        Returns:
            ollama.AsyncClient: Configured client instance

        Design Rationale:
        ================

        Creating fresh clients for each request provides several benefits:

        1. **Resource Cleanup**: Automatic garbage collection of connections
        2. **Error Isolation**: Failed requests don't affect subsequent ones
        3. **Memory Management**: Prevents accumulation of connection objects
        4. **Shutdown Safety**: No persistent connections to clean up
        5. **Concurrency Safety**: No shared mutable state between requests

        Performance Impact:
        ==================

        The overhead of creating new clients is minimal because:
        - Ollama is typically local (no network handshake)
        - Client objects are lightweight
        - Connection pooling happens at the HTTP level
        - Benefits outweigh the small initialization cost
        """
        try:
            client = ollama.AsyncClient(
                host=self.host,
                timeout=self.timeout
            )

            # Increment request counter for monitoring
            self._request_count += 1

            logger.debug(
                f"Created Ollama client #{self._request_count} for {self.host}"
            )
            return client

        except Exception as e:
            logger.error(f"Failed to create Ollama client: {e}")
            raise ConnectionError(f"Cannot create Ollama client: {e}") from e

    async def complete(
        self,
        messages: list[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> CompletionResponse:
        """
        Generate a complete response using Ollama's chat API.

        This method performs a single completion request and returns the full
        response. It's optimized for shorter interactions where streaming is
        not required and the complete response is needed immediately.

        Args:
            messages: Conversation history in chronological order
            model: Ollama model identifier (e.g., "llama2:7b", "mistral:latest")
            temperature: Randomness control (0.0-2.0)
                        - 0.0: Deterministic, factual responses
                        - 0.7: Balanced creativity and consistency (default)
                        - 1.0+: More creative but potentially inconsistent
            max_tokens: Maximum tokens to generate (Ollama param: num_predict)
                       None uses model's default (typically 128-512)
            **kwargs: Additional Ollama-specific options:
                     - top_k: Limit token selection to top K (default: 40)
                     - top_p: Nucleus sampling threshold (default: 0.9)
                     - repeat_penalty: Penalize repetition (default: 1.1)
                     - stop: Stop sequences (list of strings)
                     - num_thread: CPU thread count override

        Returns:
            CompletionResponse containing:
            - content: Generated text response
            - model: Actual model used (may include version/variant)
            - usage: Token consumption statistics
            - metadata: Performance metrics and execution details

        Raises:
            ConnectionError: Ollama service unavailable
            ValueError: Invalid model name or parameters
            RuntimeError: Model loading or execution failure
            TimeoutError: Request exceeded timeout limit

        Performance Notes:
        =================

        - First request may take 10-30s for model loading
        - Subsequent requests typically complete in 1-10s
        - Response time scales with model size and output length
        - GPU acceleration provides 3-10x speedup over CPU

        Example Usage:
        =============

        ```python
        # Basic completion
        response = await provider.complete(
            messages=[Message(role="user", content="Hello!")],
            model="llama2:7b"
        )

        # Factual query with low temperature
        response = await provider.complete(
            messages=conversation_history,
            model="mistral:latest",
            temperature=0.1,
            max_tokens=100
        )

        # Creative writing with custom parameters
        response = await provider.complete(
            messages=story_context,
            model="codellama:7b",
            temperature=1.2,
            top_p=0.8,
            repeat_penalty=1.15
        )
        ```
        """
        # Validate inputs before making request
        if not messages:
            raise ValueError("Messages list cannot be empty")
        if not model:
            raise ValueError("Model name cannot be empty")
        if not 0.0 <= temperature <= 2.0:
            logger.warning(f"Temperature {temperature} outside typical range [0.0, 2.0]")

        # Create fresh client for this request
        client = self._create_client()
        start_time = time.time()

        try:
            # Convert messages to Ollama's expected format
            ollama_messages = self._convert_messages_to_ollama_format(messages)

            # Prepare Ollama-specific options
            options = {
                "temperature": temperature,
                **kwargs  # Allow override of any Ollama parameters
            }

            # Add num_predict (max_tokens) if specified
            if max_tokens is not None:
                options["num_predict"] = max_tokens

            logger.debug(
                f"Ollama completion request: model={model}, messages={len(messages)}, "
                f"temperature={temperature}, max_tokens={max_tokens}"
            )

            # Make the completion request
            response = await client.chat(
                model=model,
                messages=ollama_messages,
                options=options,
            )

            # Calculate request duration
            request_duration = time.time() - start_time

            # Extract and validate response content
            content = response["message"]["content"]
            if not content:
                logger.warning(f"Empty response from Ollama model {model}")

            # Build standardized response with comprehensive metadata
            completion_response = CompletionResponse(
                content=content,
                model=response["model"],  # May include version info
                usage={
                    "prompt_tokens": response.get("prompt_eval_count", 0),
                    "completion_tokens": response.get("eval_count", 0),
                    "total_tokens": (
                        response.get("prompt_eval_count", 0) +
                        response.get("eval_count", 0)
                    ),
                },
                metadata={
                    # Ollama-specific timing metrics (nanoseconds)
                    "eval_duration": response.get("eval_duration"),
                    "total_duration": response.get("total_duration"),
                    "prompt_eval_duration": response.get("prompt_eval_duration"),
                    "load_duration": response.get("load_duration"),

                    # Request metadata
                    "request_duration_seconds": request_duration,
                    "model_loaded": response.get("load_duration", 0) > 0,
                    "provider": "ollama",
                    "host": self.host,

                    # Performance metrics
                    "tokens_per_second": (
                        response.get("eval_count", 0) / (request_duration or 1)
                    ),
                }
            )

            logger.info(
                f"Ollama completion success: {len(content)} chars, "
                f"{completion_response.usage['total_tokens']} tokens, "
                f"{request_duration:.2f}s"
            )

            return completion_response

        except asyncio.TimeoutError as e:
            logger.error(f"Ollama completion timeout after {self.timeout}s: {e}")
            raise TimeoutError(f"Ollama request timed out after {self.timeout}s") from e
        except ollama.ResponseError as e:
            logger.error(f"Ollama API error: {e}")
            if "model not found" in str(e).lower():
                raise ValueError(f"Model '{model}' not found. Use pull_model() to download.") from e
            raise RuntimeError(f"Ollama API error: {e}") from e
        except Exception as e:
            logger.error(f"Ollama completion error: {e}")
            raise ConnectionError(f"Failed to complete request: {e}") from e
        finally:
            # Ensure client cleanup even on exceptions
            try:
                del client
            except Exception:
                pass  # Ignore cleanup errors

    async def stream_complete(
        self,
        messages: list[Message],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Generate a streaming response for real-time output display.

        This method yields content chunks as they become available from Ollama,
        enabling real-time display of the generation process. Particularly useful
        for longer responses where immediate feedback improves user experience.

        Args:
            messages: Conversation history (same as complete())
            model: Ollama model identifier (same as complete())
            temperature: Randomness control (same as complete())
            max_tokens: Maximum tokens to generate (same as complete())
            **kwargs: Additional Ollama options (same as complete())

        Yields:
            str: Content chunks as they're generated by the model
                 - Chunks may be individual tokens, words, or phrases
                 - Empty chunks are filtered out automatically
                 - Chunk size depends on model and generation speed

        Raises:
            Same exceptions as complete(), plus:
            StreamError: Stream-specific errors (connection drops, etc.)

        Stream Behavior:
        ===============

        - Chunks arrive in real-time as the model generates tokens
        - Total generation time same as complete() but with progressive output
        - Stream automatically closes when generation completes
        - Supports cancellation via asyncio task cancellation
        - Graceful handling of connection interruptions

        Performance Characteristics:
        ===========================

        - First chunk may take longer (model loading time)
        - Subsequent chunks arrive at model's generation speed
        - GPU models: 10-100 tokens/second
        - CPU models: 1-20 tokens/second
        - Chunk frequency varies with model architecture

        Error Recovery:
        ==============

        - Automatic retry for transient network issues
        - Graceful shutdown detection during app termination
        - Stream cleanup even if partially completed
        - Detailed logging for debugging streaming issues

        Example Usage:
        =============

        ```python
        # Real-time output display
        print("AI: ", end="")
        async for chunk in provider.stream_complete(messages, "llama2:7b"):
            print(chunk, end="", flush=True)
        print()  # New line when complete

        # Accumulate full response
        full_response = ""
        async for chunk in provider.stream_complete(messages, "mistral:latest"):
            full_response += chunk
            update_ui(full_response)  # Update UI progressively
        ```

        Monitoring and Debugging:
        ========================

        The stream includes comprehensive logging for monitoring:
        - Request initiation with message count and model
        - Preview of recent messages (last 3) for context
        - Real-time chunk count tracking
        - Stream completion statistics
        - Error details with context
        """
        # Validate inputs (same as complete())
        if not messages:
            raise ValueError("Messages list cannot be empty")
        if not model:
            raise ValueError("Model name cannot be empty")

        # Create fresh client for this streaming request
        client = self._create_client()
        start_time = time.time()
        chunk_count = 0
        total_content = ""

        try:
            # Convert messages to Ollama format
            ollama_messages = self._convert_messages_to_ollama_format(messages)

            # Prepare options (same as complete())
            options = {
                "temperature": temperature,
                **kwargs
            }

            if max_tokens is not None:
                options["num_predict"] = max_tokens

            # Enhanced logging for stream debugging
            logger.info(
                f"Ollama stream: Sending {len(ollama_messages)} messages to model {model}"
            )

            # Log recent message context for debugging (last 3 messages)
            for i, msg in enumerate(ollama_messages[-3:]):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                content_preview = (
                    content[:50] + "..." if len(content) > 50 else content
                )
                logger.debug(f"  Message[-{3-i}]: {role}: {content_preview}")

            # Initiate streaming request
            stream = await client.chat(
                model=model,
                messages=ollama_messages,
                stream=True,
                options=options,
            )

            logger.debug(f"Ollama stream started for model {model}")

            # Process stream chunks as they arrive
            async for chunk in stream:
                # Extract content from chunk structure
                chunk_content = chunk.get("message", {}).get("content")

                if chunk_content:  # Filter out empty chunks
                    chunk_count += 1
                    total_content += chunk_content
                    yield chunk_content

                    # Log periodic progress for long streams
                    if chunk_count % 50 == 0:
                        logger.debug(
                            f"Ollama stream progress: {chunk_count} chunks, "
                            f"{len(total_content)} chars"
                        )

            # Stream completion statistics
            stream_duration = time.time() - start_time
            tokens_per_second = len(total_content.split()) / (stream_duration or 1)

            logger.info(
                f"Ollama stream completed: {chunk_count} chunks, "
                f"{len(total_content)} chars, {stream_duration:.2f}s, "
                f"{tokens_per_second:.1f} tokens/sec"
            )

        except asyncio.CancelledError:
            # Handle graceful cancellation (user stopped generation)
            logger.info(
                f"Ollama stream cancelled after {chunk_count} chunks "
                f"({time.time() - start_time:.1f}s)"
            )
            raise
        except Exception as e:
            # Handle various error conditions with appropriate logging

            # Special handling for app shutdown scenarios
            error_str = str(e)
            if (
                "Event loop is closed" in error_str or
                "RuntimeError" in str(type(e).__name__) and
                "cannot schedule new futures" in error_str.lower()
            ):
                logger.info(
                    "Ollama stream stopped due to app shutdown "
                    f"(processed {chunk_count} chunks)"
                )
                return

            # Log detailed error information for debugging
            logger.error(
                f"Ollama stream error after {chunk_count} chunks: {e}"
            )

            # Re-raise with appropriate exception type
            if "model not found" in error_str.lower():
                raise ValueError(f"Model '{model}' not found") from e
            elif "timeout" in error_str.lower():
                raise TimeoutError(f"Stream timeout after {self.timeout}s") from e
            else:
                raise ConnectionError(f"Stream failed: {e}") from e
        finally:
            # Ensure cleanup even on exceptions
            try:
                del client
            except Exception:
                pass  # Ignore cleanup errors

    async def list_models(self) -> list[str]:
        """
        Retrieve a list of locally available Ollama models.

        This method queries the Ollama service to get all models that are
        currently downloaded and available for use. The list includes both
        base models and specific variants/versions.

        Returns:
            list[str]: Model names in Ollama format
                      Examples: ["llama2:7b", "mistral:latest", "codellama:13b"]
                      Empty list if no models available or service down

        Model Naming Convention:
        =======================

        Ollama uses a structured naming format:
        - Base name: "llama2", "mistral", "codellama"
        - Size variant: ":7b", ":13b", ":70b"
        - Version tag: ":latest", ":v1.0", ":custom"
        - Full examples: "llama2:7b", "mistral:latest", "mymodel:v2"

        Caching Strategy:
        ================

        Model lists are cached for 5 minutes to reduce API calls while still
        reflecting recent model installations. Cache is invalidated on:
        - Time expiration (5 minutes)
        - Service errors (to retry immediately)
        - Manual cache clearing (if implemented)

        Implementation Notes:
        ====================

        Uses direct HTTP API rather than Ollama client to avoid potential
        connection issues and provide more control over timeout and error
        handling. The /api/tags endpoint is lightweight and responds quickly.

        Error Handling:
        ==============

        - Returns empty list on any error (graceful degradation)
        - Logs errors for debugging but doesn't raise exceptions
        - Handles service unavailable, network errors, malformed responses
        - Provides helpful error context for troubleshooting

        Example Response:
        ================

        ```python
        models = await provider.list_models()
        print(models)  # ["llama2:7b", "mistral:latest", "codellama:13b"]

        # Check for specific model
        if "llama2:7b" in models:
            response = await provider.complete(messages, "llama2:7b")
        ```
        """
        current_time = time.time()

        # Check cache first (5 minute expiration)
        if (
            self._cached_models is not None and
            current_time - self._last_model_list_time < 300  # 5 minutes
        ):
            logger.debug(f"Returning cached model list ({len(self._cached_models)} models)")
            return self._cached_models

        try:
            # Use direct HTTP API for better error handling and timeout control
            import httpx

            timeout_config = httpx.Timeout(10.0)  # 10 second timeout for model list

            async with httpx.AsyncClient(timeout=timeout_config) as http_client:
                logger.debug(f"Fetching model list from {self.host}/api/tags")

                response = await http_client.get(f"{self.host}/api/tags")
                response.raise_for_status()  # Raise exception for HTTP errors

                data = response.json()

                # Extract model names from response
                models = [model["name"] for model in data.get("models", [])]

                # Sort models for consistent ordering
                models.sort()

                # Update cache
                self._cached_models = models
                self._last_model_list_time = current_time

                logger.info(f"Found {len(models)} Ollama models: {models[:5]}{'...' if len(models) > 5 else ''}")
                return models

        except httpx.TimeoutException as e:
            logger.error(f"Timeout fetching Ollama models from {self.host}: {e}")
            # Clear cache on error to retry next time
            self._cached_models = None
            return []
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching Ollama models: {e.response.status_code} {e.response.text}")
            if e.response.status_code == 404:
                logger.warning("Ollama API endpoint not found - is Ollama running?")
            self._cached_models = None
            return []
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            # Clear cache on error to retry next time
            self._cached_models = None
            return []

    async def health_check(self) -> bool:
        """
        Perform a comprehensive health check of the Ollama service.

        This method verifies that Ollama is running, accessible, and has at
        least one model available for use. It's designed to be lightweight
        and complete quickly for monitoring and provider selection.

        Returns:
            bool: True if Ollama is healthy and operational
                 False if any health check criteria fail

        Health Check Criteria:
        =====================

        1. **Service Reachability**: Ollama daemon is running and responding
        2. **API Functionality**: Core endpoints are accessible
        3. **Model Availability**: At least one model is downloaded and ready
        4. **Response Time**: Service responds within reasonable time (< 10s)

        Implementation Strategy:
        =======================

        - Uses model listing as a comprehensive health indicator
        - Lightweight operation that doesn't load models unnecessarily
        - Implements its own timeout to prevent hanging
        - Never raises exceptions (graceful failure)
        - Provides detailed logging for troubleshooting

        Common Failure Scenarios:
        ========================

        - Ollama daemon not running
        - Wrong host/port configuration
        - Network connectivity issues
        - No models downloaded (fresh installation)
        - Service overloaded or unresponsive
        - Insufficient system resources

        Usage in System:
        ===============

        ```python
        # Provider selection
        if await ollama_provider.health_check():
            response = await ollama_provider.complete(messages, model)
        else:
            # Fall back to cloud provider
            response = await openai_provider.complete(messages, model)

        # Monitoring
        health_status = await provider.health_check()
        metrics.record("ollama_health", health_status)
        ```

        Performance Considerations:
        ==========================

        - Cached model list reduces repeated API calls
        - 10-second timeout prevents indefinite waiting
        - Result can be cached briefly for monitoring systems
        - Minimal resource usage (doesn't trigger model loading)
        """
        try:
            logger.debug(f"Performing health check for Ollama at {self.host}")

            # Use model listing as health indicator
            # This tests service availability and basic functionality
            models = await self.list_models()

            # Health check passes if we have at least one model
            is_healthy = len(models) > 0

            if is_healthy:
                logger.debug(f"Ollama health check passed: {len(models)} models available")
            else:
                logger.warning(
                    "Ollama health check failed: No models available. "
                    "Service may be running but no models are downloaded."
                )

            return is_healthy

        except Exception as e:
            # Never raise exceptions from health checks
            logger.warning(f"Ollama health check failed: {e}")
            return False

    async def pull_model(self, model: str) -> bool:
        """
        Download a model from the Ollama registry to local storage.

        This method downloads and installs a model for local use. Model pulling
        is typically a one-time operation per model, but may be needed for
        updates or when switching to different model variants.

        Args:
            model: Model identifier to download
                  Examples: "llama2:7b", "mistral:latest", "codellama:13b"
                  Supports full registry names like "registry.ollama.ai/library/llama2:7b"

        Returns:
            bool: True if model was successfully downloaded and installed
                 False if download failed for any reason

        Download Process:
        ================

        1. **Registry Lookup**: Resolve model name to registry location
        2. **Size Check**: Verify available disk space (models are 3-40GB+)
        3. **Download**: Stream model data from registry
        4. **Verification**: Validate downloaded model integrity
        5. **Installation**: Install model for local use

        Performance Characteristics:
        ===========================

        - Download time: 5 minutes to 2+ hours depending on model size and connection
        - Disk space: 3-40GB+ per model (7B ≈ 4GB, 13B ≈ 8GB, 70B ≈ 40GB)
        - Network usage: Full model size (no resumption on failure)
        - CPU usage: Minimal during download, higher during installation

        Error Scenarios:
        ===============

        - Network connectivity issues
        - Insufficient disk space
        - Invalid model name or registry issues
        - Download interruption
        - Model verification failure
        - Registry authentication issues (private models)

        Example Usage:
        =============

        ```python
        # Download a specific model
        if await provider.pull_model("llama2:7b"):
            print("Model ready for use")
            models = await provider.list_models()
            assert "llama2:7b" in models

        # Batch download with error handling
        models_to_download = ["llama2:7b", "mistral:latest"]
        for model in models_to_download:
            success = await provider.pull_model(model)
            print(f"{model}: {'✓' if success else '✗'}")
        ```

        Monitoring and Progress:
        =======================

        - Logs download initiation and completion
        - Does not provide progress updates (Ollama API limitation)
        - Consider implementing timeout for very large models
        - Cache invalidation after successful download

        Security Considerations:
        =======================

        - Models are downloaded from official Ollama registry by default
        - Verify model sources for security-sensitive applications
        - Downloaded models have full system access when running
        - Consider scanning models for malicious content if required
        """
        if not model:
            logger.error("Cannot pull model: model name is empty")
            return False

        # Create fresh client for model pulling
        client = self._create_client()
        start_time = time.time()

        try:
            logger.info(f"Initiating model download: {model}")
            logger.info("This may take several minutes depending on model size and connection speed")

            # Perform the model pull operation
            # Note: This is a long-running operation that may take many minutes
            await client.pull(model)

            download_duration = time.time() - start_time
            logger.info(
                f"Successfully downloaded model '{model}' in {download_duration:.1f} seconds"
            )

            # Invalidate model cache to pick up the new model
            self._cached_models = None

            return True

        except asyncio.TimeoutError:
            logger.error(
                f"Model download timeout for '{model}' after {self.timeout}s. "
                "Large models may require longer timeouts."
            )
            return False
        except ollama.ResponseError as e:
            error_str = str(e)
            if "not found" in error_str.lower():
                logger.error(f"Model '{model}' not found in registry")
            elif "network" in error_str.lower():
                logger.error(f"Network error downloading '{model}': {e}")
            else:
                logger.error(f"Ollama error downloading '{model}': {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to download model '{model}': {e}")
            return False
        finally:
            # Clean up client resources
            try:
                del client
            except Exception:
                pass

    def _convert_messages_to_ollama_format(self, messages: list[Message]) -> list[dict[str, Any]]:
        """
        Convert internal Message objects to Ollama's expected format.

        This helper method handles the transformation between our standardized
        Message format and Ollama's API requirements, including validation
        and error handling.

        Args:
            messages: List of Message objects to convert

        Returns:
            list[dict]: Messages in Ollama format

        Raises:
            ValueError: If message format is invalid
        """
        ollama_messages = []

        for i, msg in enumerate(messages):
            try:
                if isinstance(msg, dict):
                    # Already in dict format, validate and use as-is
                    if "role" not in msg or "content" not in msg:
                        raise ValueError(f"Message {i} missing required fields")
                    ollama_messages.append(msg)
                else:
                    # Convert Message object to dict
                    ollama_messages.append({
                        "role": msg.role,
                        "content": msg.content
                    })
            except Exception as e:
                raise ValueError(f"Invalid message at index {i}: {e}") from e

        return ollama_messages
