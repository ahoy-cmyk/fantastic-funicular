"""Chat management with provider integration and memory.

This module provides the core orchestration layer for the Neuromancer application,
coordinating between LLM providers, memory systems, MCP tools, and session management.

Key Components:
    - Provider management: Dynamically initializes and manages multiple LLM providers
    - Session handling: Maintains conversation state and history
    - Memory integration: Intelligent memory storage and recall with RAG enhancement
    - Model management: Dynamic model discovery and intelligent selection
    - System prompts: Configurable prompts with memory integration

Architectural Design:
    The ChatManager follows a facade pattern, providing a unified interface to complex
    subsystems. It uses dependency injection for providers and managers, enabling
    flexible configuration and testing.

Performance Considerations:
    - Async/await throughout for non-blocking operations
    - Memory caching in RAG system reduces retrieval latency
    - Streaming responses minimize time-to-first-token
    - Background tasks handle memory consolidation

Example Usage:
    ```python
    chat_manager = ChatManager()
    await chat_manager.create_session("My Chat")

    # RAG-enhanced message
    response = await chat_manager.send_message_with_rag(
        "What's my name?",
        stream=True
    )

    async for chunk in response:
        print(chunk, end="")
    ```
"""

from datetime import datetime
from typing import Any

from src.core.config import settings
from src.core.model_manager import ModelManager
from src.core.models import MessageRole
from src.core.rag_system import RAGConfig, RAGSystem
from src.core.session_manager import ConversationSession, SessionManager
from src.mcp import MCPResponse, MCPTool
from src.mcp.manager import MCPManager
from src.memory import MemoryType
from src.memory.manager import MemoryManager
from src.providers import LLMProvider, Message
from src.providers.lmstudio_provider import LMStudioProvider
from src.providers.ollama_provider import OllamaProvider
from src.providers.openai_provider import OpenAIProvider
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ChatManager:
    """Enterprise chat manager with sessions, providers, memory, and MCP tools.

    This class serves as the central orchestrator for all chat-related operations,
    managing the complex interactions between various subsystems.

    Attributes:
        providers (dict[str, LLMProvider]): Registered LLM provider instances
        current_provider (str): Active provider name for chat operations
        current_model (str): Active model name within the provider
        memory_manager (MemoryManager): Handles memory storage and retrieval
        mcp_manager (MCPManager): Manages MCP tool connections
        session_manager (SessionManager): Manages conversation sessions
        model_manager (ModelManager): Handles model discovery and selection
        rag_system (RAGSystem): Provides retrieval-augmented generation
        rag_enabled (bool): Whether RAG enhancement is active
        current_session (ConversationSession | None): Active conversation session
        system_prompt (str): Base system prompt for conversations
        system_prompt_memory_integration (bool): Include memories in system prompt

    Thread Safety:
        While the class uses async methods, it's not designed for concurrent
        access to the same instance. Each user session should have its own
        ChatManager instance in a multi-user scenario.
    """

    def __init__(self):
        """Initialize chat manager.

        Sets up all subsystems and establishes provider connections.
        The initialization order is important:
        1. Core attributes and configuration
        2. Manager instances (memory, MCP, session)
        3. Model management and RAG (depend on memory manager)
        4. Provider initialization (may depend on config)
        5. Cross-references between managers

        Note:
            Provider initialization failures are logged but don't prevent
            the chat manager from starting. This allows graceful degradation
            when some providers are unavailable.
        """
        self.providers: dict[str, LLMProvider] = {}
        self.current_provider = settings.DEFAULT_PROVIDER
        self.current_model = settings.DEFAULT_MODEL

        # Initialize managers
        self.memory_manager = MemoryManager()
        self.mcp_manager = MCPManager()
        self.session_manager = SessionManager()

        # Initialize model management and RAG
        self.model_manager = ModelManager(self)
        self.rag_system = RAGSystem(self.memory_manager)
        self.rag_enabled = True

        # Current session
        self.current_session: ConversationSession | None = None
        self._session_context = None

        # MCP servers to connect when event loop is available
        self._pending_mcp_servers = None

        # System prompt configuration
        self.system_prompt = ""
        self.system_prompt_memory_integration = True
        self._load_system_prompt_config()

        # Initialize providers
        self._initialize_providers()

        # Set provider reference for model manager
        self.model_manager.set_chat_manager(self)

        # Initialize MCP servers if configured
        self._initialize_mcp_servers()

        # Session manager will be started on first use

    def _initialize_providers(self):
        """Initialize LLM providers based on configuration.

        Reads provider configuration and attempts to initialize each enabled provider.
        Failures are logged but don't prevent other providers from initializing.

        Provider Priority:
            1. Ollama - Default local provider, usually most reliable
            2. OpenAI - Requires API key, highest quality
            3. LM Studio - Alternative local provider

        Configuration Keys:
            - providers.<provider>_enabled: Whether to initialize the provider
            - providers.<provider>_host: API endpoint (for local providers)
            - providers.<provider>_api_key: Authentication (for cloud providers)
            - providers.<provider>_base_url: Custom endpoints (OpenAI-compatible)
            - providers.<provider>_organization: Org ID (OpenAI-specific)

        Error Handling:
            Each provider initialization is wrapped in try-except to ensure
            one provider's failure doesn't affect others.
        """
        from src.core.config import _config_manager

        # Ollama
        try:
            if _config_manager.get("providers.ollama_enabled", True):
                ollama_host = _config_manager.get("providers.ollama_host", "http://localhost:11434")
                self.providers["ollama"] = OllamaProvider(host=ollama_host)
                logger.info(f"Initialized Ollama provider at {ollama_host}")
        except Exception as e:
            logger.warning(f"Failed to initialize Ollama provider: {e}")

        # OpenAI
        try:
            if _config_manager.get("providers.openai_enabled", False):
                api_key = _config_manager.get("providers.openai_api_key")
                if api_key:
                    openai_params = {"api_key": api_key}

                    base_url = _config_manager.get("providers.openai_base_url")
                    if base_url:
                        openai_params["base_url"] = base_url

                    organization = _config_manager.get("providers.openai_organization")
                    if organization:
                        openai_params["organization"] = organization

                    self.providers["openai"] = OpenAIProvider(**openai_params)
                    logger.info("Initialized OpenAI provider")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI provider: {e}")

        # LM Studio
        try:
            if _config_manager.get("providers.lmstudio_enabled", False):
                lmstudio_host = _config_manager.get(
                    "providers.lmstudio_host", "http://localhost:1234"
                )
                self.providers["lmstudio"] = LMStudioProvider(host=lmstudio_host)
                logger.info(f"Initialized LM Studio provider at {lmstudio_host}")
        except Exception as e:
            logger.warning(f"Failed to initialize LM Studio provider: {e}")

        logger.info(f"Initialized {len(self.providers)} LLM providers")

    def _initialize_mcp_servers(self):
        """Initialize MCP servers from configuration.

        Reads MCP server configurations and attempts to connect to each one
        if auto_connect is enabled. Failures are logged but don't prevent
        the chat manager from starting.

        Configuration Structure:
            mcp.servers: {
                "server_name": {
                    "url": "ws://localhost:8080",
                    "enabled": true,
                    "description": "My MCP Server"
                }
            }
        """
        try:
            from src.core.config import _config_manager

            # Check if MCP is enabled
            if not _config_manager.get("mcp.enabled", True):
                logger.info("MCP integration is disabled")
                return

            # Get server configurations
            servers = _config_manager.get("mcp.servers", {})
            auto_connect = _config_manager.get("mcp.auto_connect", True)

            if not servers:
                logger.info("No MCP servers configured")
                return

            logger.info(f"Found {len(servers)} MCP server configurations")

            if auto_connect:
                # Schedule connection for when event loop is available
                self._pending_mcp_servers = servers
                logger.info("MCP auto-connect scheduled for event loop startup")
            else:
                logger.info("MCP auto-connect is disabled")

        except Exception as e:
            logger.error(f"Failed to initialize MCP servers: {e}")

    async def connect_pending_mcp_servers(self):
        """Connect any pending MCP servers when event loop is available."""
        if self._pending_mcp_servers:
            logger.info("Connecting pending MCP servers...")
            await self._connect_mcp_servers(self._pending_mcp_servers)
            self._pending_mcp_servers = None

    async def _connect_mcp_servers(self, servers: dict[str, dict[str, Any]]):
        """Connect to MCP servers asynchronously.

        Args:
            servers: Dictionary of server configurations
        """
        from src.core.config import _config_manager

        # Get global SSL configuration
        global_ssl_config = {
            "verify": _config_manager.get("mcp.ssl_verify", True),
            "ca_bundle": _config_manager.get("mcp.ssl_ca_bundle"),
            "allow_self_signed": _config_manager.get("mcp.allow_self_signed", False),
        }

        for name, config in servers.items():
            try:
                if not config.get("enabled", True):
                    logger.info(f"MCP server '{name}' is disabled")
                    continue

                # Handle WebSocket servers
                url = config.get("url")
                command = config.get("command")
                args = config.get("args", [])

                if url:
                    # WebSocket MCP server
                    # Merge server-specific SSL config with global config
                    ssl_config = global_ssl_config.copy()
                    if "ssl" in config:
                        ssl_config.update(config["ssl"])

                    logger.info(f"Connecting to MCP WebSocket server '{name}' at {url}")
                    success = await self.mcp_manager.add_server(
                        name, server_url=url, ssl_config=ssl_config
                    )

                elif command:
                    # Subprocess MCP server
                    logger.info(
                        f"Connecting to MCP subprocess server '{name}': {command} {' '.join(args)}"
                    )
                    success = await self.mcp_manager.add_server(name, command=command, args=args)

                else:
                    logger.warning(f"MCP server '{name}' missing both URL and command")
                    continue

                if success:
                    logger.info(f"Successfully connected to MCP server '{name}'")
                else:
                    logger.warning(f"Failed to connect to MCP server '{name}'")

            except Exception as e:
                logger.error(f"Error connecting to MCP server '{name}': {e}")

    def _load_system_prompt_config(self):
        """Load system prompt configuration.

        Loads the system prompt and memory integration settings from configuration.
        The system prompt defines the AI's personality and capabilities.

        Configuration Keys:
            - system_prompt: Base prompt text
            - system_prompt_memory_integration: Whether to include memories

        Note:
            If loading fails, the system continues with default values rather
            than crashing, ensuring robustness.
        """
        try:
            from src.core.config import _config_manager

            self.system_prompt = _config_manager.get("system_prompt", "")
            self.system_prompt_memory_integration = _config_manager.get(
                "system_prompt_memory_integration", True
            )
            logger.info(f"Loaded system prompt: {len(self.system_prompt)} chars")
        except Exception as e:
            logger.error(f"Failed to load system prompt config: {e}")

    def update_system_prompt(self, prompt: str, memory_integration: bool = True):
        """Update the system prompt configuration.

        Args:
            prompt: The new system prompt text. Can include personality,
                capabilities, and behavioral instructions.
            memory_integration: Whether to include relevant memories in the
                system prompt. When True, memories are dynamically added
                based on the conversation context.

        Effects:
            - Immediately affects all new conversations
            - Existing conversations will use the new prompt on next message
            - Does not persist across application restarts (use config for that)

        Example:
            ```python
            chat_manager.update_system_prompt(
                "You are a helpful coding assistant with expertise in Python.",
                memory_integration=True
            )
            ```
        """
        self.system_prompt = prompt
        self.system_prompt_memory_integration = memory_integration
        logger.info(
            f"Updated system prompt: {len(prompt)} chars, memory integration: {memory_integration}"
        )

    def refresh_providers(self):
        """Refresh provider configurations.

        Clears all existing providers and reinitializes from configuration.
        Useful when provider settings have changed at runtime.

        Warning:
            This will interrupt any ongoing operations with the providers.
            Ensure no active streaming or requests before calling.
        """
        self.providers.clear()
        self._initialize_providers()

    def get_available_providers(self) -> list[str]:
        """Get list of available provider names.

        Returns:
            List of provider names that were successfully initialized.
            Names correspond to configuration keys (e.g., 'ollama', 'openai').

        Note:
            A provider being in this list doesn't guarantee it's currently
            healthy, just that it was initialized. Use get_provider_health()
            for current status.
        """
        return list(self.providers.keys())

    def is_provider_available(self, provider_name: str) -> bool:
        """Check if a provider is available.

        Args:
            provider_name: Name of the provider to check

        Returns:
            True if the provider was successfully initialized

        Note:
            This checks initialization status, not current health.
            A provider may be available but temporarily unreachable.
        """
        return provider_name in self.providers

    # Model Management Methods
    # These methods delegate to the ModelManager for model discovery,
    # selection, and configuration. They provide a convenient interface
    # at the ChatManager level.

    async def discover_models(self, force_refresh: bool = False) -> bool:
        """Discover available models from all providers.

        Args:
            force_refresh: Force rediscovery even if recently performed.
                By default, discovery is cached for 1 minute.

        Returns:
            True if at least one provider returned models

        Performance:
            Discovery queries all providers in parallel. Results are cached
            to avoid repeated API calls. First discovery may take 1-3 seconds.
        """
        return await self.model_manager.discover_models(force_refresh)

    def get_available_models(self, provider: str = None):
        """Get available models, optionally filtered by provider.

        Args:
            provider: Filter models by provider name. None returns all models.

        Returns:
            List of ModelInfo objects with detailed model information

        Note:
            Models are sorted by provider, then by name for consistent display.
        """
        return self.model_manager.get_available_models(provider)

    def get_model_info(self, model_name: str, provider: str = None):
        """Get detailed information about a model.

        Args:
            model_name: Name of the model (e.g., 'llama2', 'gpt-4')
            provider: Provider name if known, helps disambiguate models
                with the same name across providers

        Returns:
            ModelInfo object with capabilities, parameters, etc., or None

        Search Order:
            1. Exact match with provider:model_name
            2. Full name search if model_name contains ':'
            3. Search across all providers for model_name
        """
        return self.model_manager.get_model_info(model_name, provider)

    async def select_best_model(self, strategy=None, requirements=None):
        """Automatically select the best available model.

        Args:
            strategy: ModelSelectionStrategy enum value or None for default
            requirements: Dict of requirements like {'capabilities': ['code'],
                'min_context_length': 8192, 'max_cost_per_token': 0.001}

        Returns:
            ModelInfo of the selected model or None if no match

        Strategies:
            - MANUAL: Use preferred_models list
            - FASTEST: Select from provider with best response time
            - CHEAPEST: Prefer free models, then lowest cost
            - BEST_QUALITY: Prefer larger parameter counts
            - BALANCED: Balance quality, speed, and cost
            - MOST_CAPABLE: Most capabilities/features
        """
        return await self.model_manager.select_best_model(strategy, requirements)

    def set_model(self, model_name: str, provider: str = None) -> bool:
        """Set the current model and provider.

        Args:
            model_name: Name of the model to use. Can be a full name
                like 'ollama:llama2' or just 'llama2'.
            provider: Explicit provider name. If None, attempts to
                determine from model_name or searches all providers.

        Returns:
            True if model was successfully set

        Validation:
            1. Model must exist in the model registry
            2. Model status must be AVAILABLE
            3. Provider must be initialized

        Side Effects:
            Updates current_model and current_provider attributes

        Example:
            ```python
            # Explicit provider
            chat_manager.set_model('gpt-4', 'openai')

            # Auto-detect provider
            chat_manager.set_model('llama2:latest')
            ```
        """
        from src.core.model_manager import ModelStatus

        logger.info(f"Attempting to switch to model: {model_name} (provider: {provider})")

        model_info = self.model_manager.get_model_info(model_name, provider)
        if not model_info:
            logger.error(f"Model not found: {model_name} (provider: {provider})")
            # Debug: show what models are actually available
            self.model_manager.debug_model_registry()
            return False

        if model_info.status != ModelStatus.AVAILABLE:
            logger.error(
                f"Model not available: {model_info.full_name} (status: {model_info.status})"
            )
            return False

        # Check if the provider instance exists
        if model_info.provider not in self.providers:
            logger.error(f"Provider not initialized: {model_info.provider}")
            return False

        self.current_model = model_info.name
        self.current_provider = model_info.provider
        logger.info(f"Successfully switched to model: {model_info.full_name}")
        return True

    def get_provider_health(self):
        """Get health status of all providers.

        Returns:
            Dict mapping provider names to ProviderInfo objects with
            health status, response times, and available models.

        Note:
            This uses cached discovery data. For real-time health checks,
            call discover_models(force_refresh=True) first.
        """
        return {name: self.model_manager.get_provider_info(name) for name in self.providers.keys()}

    # RAG System Methods
    # These methods manage the Retrieval-Augmented Generation system,
    # which enhances responses with relevant memory context.

    def enable_rag(self, enabled: bool = True):
        """Enable or disable RAG system.

        Args:
            enabled: Whether to use RAG enhancement for messages

        Effects:
            When enabled, messages will:
            - Search memory for relevant context
            - Include retrieved memories in prompts
            - Track retrieval metrics
            - Store conversations for future retrieval

        Performance:
            RAG adds 50-200ms latency for memory retrieval but
            significantly improves response relevance.
        """
        self.rag_enabled = enabled
        logger.info(f"RAG system {'enabled' if enabled else 'disabled'}")

    def configure_rag(self, config: RAGConfig):
        """Configure RAG system behavior.

        Args:
            config: RAGConfig object with retrieval parameters

        Key Configuration Options:
            - max_memories: Number of memories to retrieve (default: 15)
            - min_relevance_threshold: Similarity threshold (default: 0.3)
            - cite_sources: Add memory citations to responses
            - explain_reasoning: Include retrieval reasoning
            - cache_results: Cache retrieval results for performance
        """
        self.rag_system.update_config(config)
        logger.info("RAG system configuration updated")

    def get_rag_stats(self):
        """Get RAG system performance statistics.

        Returns:
            Dict with metrics including:
            - total_queries: Number of RAG retrievals
            - cache_hits: Number of cache hits
            - avg_retrieval_time_ms: Average retrieval latency
            - successful_retrievals: Number of successful retrievals
            - cache_hit_rate: Percentage of cache hits
        """
        return self.rag_system.get_stats()

    def send_message_with_rag(
        self, content: str, provider: str = None, model: str = None, stream: bool = False
    ):
        """Send message with RAG enhancement.

        Args:
            content: User message content
            provider: Override current provider
            model: Override current model
            stream: Whether to stream the response

        Returns:
            For stream=False: Response string
            For stream=True: Async generator yielding response chunks

        RAG Process:
            1. Retrieve relevant memories based on query
            2. Build enhanced prompt with memory context
            3. Generate response using LLM
            4. Store conversation for future retrieval

        Fallback:
            If RAG fails, automatically falls back to standard messaging
        """
        if not self.rag_enabled:
            if stream:
                return self._send_message_stream_generator(content, provider, model)
            else:
                return self._send_message_non_stream(content, provider, model)

        if stream:
            # Return the async generator directly
            return self._send_rag_message_stream_generator(content, provider, model)
        else:
            return self._send_rag_message_non_stream(content, provider, model)

    async def _send_rag_message_stream_generator(
        self, content: str, provider: str = None, model: str = None
    ):
        """Async generator for RAG-enhanced streaming.

        This wrapper ensures proper async generator protocol for streaming responses.
        It delegates to the actual implementation while maintaining the generator interface.
        """
        async for chunk in self._send_rag_message_stream(content, provider, model):
            yield chunk

    async def _send_rag_message_stream(self, content: str, provider: str = None, model: str = None):
        """Send RAG-enhanced message with streaming response.

        Implements the full RAG pipeline with streaming support:
        1. Memory retrieval with conversation context
        2. Context enhancement with retrieved memories
        3. Streaming generation with periodic UI updates
        4. Memory storage for future retrieval

        Error Handling:
            - Timeout protection for retrieval (5s default)
            - Fallback to standard streaming on RAG failure
            - Graceful handling of stream interruptions

        Performance Notes:
            - Retrieval happens before streaming starts (adds initial latency)
            - UI updates occur with each chunk for responsiveness
            - Memory storage happens asynchronously after completion
        """
        # Ensure we have a session
        if not self.current_session:
            await self.create_session()

        try:
            provider = provider or self.current_provider
            model = model or self.current_model

            # Get provider instance
            if provider not in self.providers:
                raise ValueError(f"Provider '{provider}' not available")

            llm_provider = self.providers[provider]

            # Get conversation history for RAG context
            conversation_history = []
            if self.current_session:
                recent_messages = await self.current_session.get_messages(limit=10)
                conversation_history = [
                    Message(
                        role=msg.role.value if hasattr(msg.role, "value") else str(msg.role),
                        content=msg.content,
                    )
                    for msg in recent_messages
                ]

            # Get RAG context first
            logger.info(f"Retrieving RAG context for query: {content[:50]}...")
            context = await self.rag_system.retrieve_context(
                query=content, conversation_history=conversation_history
            )

            logger.info(f"Retrieved {len(context.memories)} memories for RAG context")

            # Build enhanced messages with RAG context
            enhanced_messages = await self._build_rag_enhanced_messages(content, context)

            # Add user message to session with RAG metadata
            await self.current_session.add_message(
                role=MessageRole.USER,
                content=content,
                metadata={
                    "rag_enabled": True,
                    "memories_used": len(context.memories),
                    "retrieval_time_ms": context.retrieval_time_ms,
                },
            )

            # Create placeholder assistant message
            assistant_msg = await self.current_session.add_message(
                role=MessageRole.ASSISTANT,
                content="",
                model=model,
                metadata={"status": "pending", "rag_enabled": True},
            )

            response_content = ""

            # Stream the response with RAG-enhanced context and tool calling
            logger.info(
                f"Starting RAG-enhanced stream with {len(enhanced_messages)} context messages"
            )
            chunk_count = 0
            try:
                # Create the original streaming generator
                original_stream = llm_provider.stream_complete(
                    messages=enhanced_messages, model=model, temperature=0.7, max_tokens=1000
                )

                # Use the tool-calling aware streaming handler
                async for chunk in self._handle_streaming_with_tool_calls(
                    original_stream, assistant_msg.id
                ):
                    chunk_count += 1
                    response_content += chunk
                    # Update message periodically (less frequently to avoid conflicts)
                    if chunk_count % 5 == 0:  # Update every 5 chunks instead of every chunk
                        await self.current_session.update_message(
                            assistant_msg.id, content=response_content
                        )
                    yield chunk  # Yield for UI streaming

                logger.info(
                    f"RAG stream completed with {chunk_count} chunks, "
                    f"total length: {len(response_content)}"
                )
            except Exception as stream_error:
                logger.error(
                    f"Error during RAG streaming (after {chunk_count} chunks): {stream_error}"
                )
                raise

            # Final update with RAG metadata
            await self.current_session.update_message(
                assistant_msg.id,
                content=response_content,
                metadata={
                    "status": "completed",
                    "rag_enabled": True,
                    "memories_used": len(context.memories),
                    "retrieval_time_ms": context.retrieval_time_ms,
                },
            )

            # Store conversation in memory for future context
            await self._store_conversation_memory(content, response_content)

        except Exception as e:
            logger.error(f"Error sending RAG-enhanced streaming message: {e}")
            # Fallback to regular streaming
            async for chunk in self._send_message_stream(content, provider, model):
                yield chunk

    async def _send_rag_message_non_stream(
        self, content: str, provider: str = None, model: str = None
    ):
        """Send RAG-enhanced message with non-streaming response.

        Similar to streaming version but returns complete response at once.
        Better for programmatic use where streaming isn't needed.

        Returns:
            Complete response string with RAG enhancement

        Metadata Tracking:
            Messages include metadata about:
            - Number of memories used
            - Retrieval time
            - RAG reasoning (if enabled)
        """
        # Get conversation history for context
        conversation_history = []
        if self.current_session:
            recent_messages = await self.current_session.get_messages(limit=10)
            conversation_history = [
                Message(
                    role=msg.role.value if hasattr(msg.role, "value") else str(msg.role),
                    content=msg.content,
                )
                for msg in recent_messages
            ]

        # Use RAG to generate enhanced response
        try:
            provider = provider or self.current_provider
            model = model or self.current_model
            llm_provider = self.providers.get(provider)

            if not llm_provider:
                raise ValueError(f"Provider '{provider}' not available")

            response, context = await self.rag_system.generate_rag_response(
                query=content,
                conversation_history=conversation_history,
                llm_provider=llm_provider,
                model=model,
                system_prompt=self.system_prompt if self.system_prompt_memory_integration else None,
            )

            # Check for tool calls in response and execute them
            final_response = await self._parse_and_execute_tool_calls(response)

            # Store response in session if we have one
            if self.current_session:
                await self.current_session.add_message(
                    role=MessageRole.USER,
                    content=content,
                    metadata={"rag_enabled": True, "memories_used": len(context.memories)},
                )

                await self.current_session.add_message(
                    role=MessageRole.ASSISTANT,
                    content=final_response,
                    model=model,
                    metadata={
                        "rag_enabled": True,
                        "memories_used": len(context.memories),
                        "retrieval_time_ms": context.retrieval_time_ms,
                        "rag_reasoning": context.reasoning,
                    },
                )

            # Store in memory for future retrieval
            await self._store_conversation_memory(content, final_response)

            return final_response

        except Exception as e:
            logger.error(f"RAG-enhanced message failed: {e}")
            # Fallback to regular message sending
            return await self._send_message_non_stream(content, provider, model)

    async def _build_rag_enhanced_messages(self, query: str, rag_context) -> list[Message]:
        """Build messages enhanced with RAG context.

        Args:
            query: Current user query
            rag_context: RetrievalContext with memories and metadata

        Returns:
            List of Message objects with context-enhanced system prompt

        Context Organization:
            Memories are organized by type for clarity:
            1. Personal Information (highest priority)
            2. Uploaded File Content
            3. Other Relevant Information

        Design Decision:
            We enhance the system prompt rather than the user message to:
            - Maintain clear conversation flow
            - Avoid confusing the model with injected context
            - Allow the model to naturally reference the context
        """
        messages = []

        # Start with system prompt
        system_content = (
            self.system_prompt
            if self.system_prompt
            else (
                "You are Neuromancer, an advanced AI assistant with exceptional memory "
                "and tool-use capabilities. You have access to long-term memory and can "
                "execute various tools through MCP (Model Context Protocol) servers."
            )
        )

        # Add RAG context to system prompt
        if rag_context.memories:
            system_content += (
                "\n\n**IMPORTANT: You have access to the following relevant information "
                "from your memory. Use this information to answer the user's question "
                "accurately:**\n"
            )

            # Organize memories by type for better presentation
            personal_memories = []
            file_memories = []
            other_memories = []

            for memory in rag_context.memories:
                if memory.metadata and memory.metadata.get("type") == "personal_info":
                    personal_memories.append(memory)
                elif memory.metadata and memory.metadata.get("source") == "file_upload":
                    file_memories.append(memory)
                else:
                    other_memories.append(memory)

            # Add personal info first (most important)
            if personal_memories:
                system_content += "\n**Personal Information:**\n"
                for memory in personal_memories:
                    system_content += f"- {memory.content}\n"

            # Add file information
            if file_memories:
                system_content += "\n**Uploaded File Content:**\n"
                for memory in file_memories:
                    file_name = memory.metadata.get("file_name", "uploaded file")
                    content_preview = memory.content[:200]
                    suffix = "..." if len(memory.content) > 200 else ""
                    system_content += f"- From {file_name}: {content_preview}{suffix}\n"

            # Add other relevant memories
            if other_memories:
                system_content += "\n**Other Relevant Information:**\n"
                for memory in other_memories[:3]:  # Limit to avoid clutter
                    system_content += (
                        f"- {memory.content[:150]}{'...' if len(memory.content) > 150 else ''}\n"
                    )

            system_content += (
                "\n**CRITICAL: Always use the above information to provide accurate answers. "
                "If the user asks about their name or personal details, refer to the "
                "Personal Information section. If they ask about uploaded files, refer to the "
                "Uploaded File Content section.**"
            )

        messages.append(Message(role="system", content=system_content))

        # Add conversation history
        if self.current_session:
            session_messages = await self.current_session.get_messages(limit=20)
            for msg in session_messages:
                # Handle role conversion safely
                if hasattr(msg.role, "value"):
                    role_str = msg.role.value
                else:
                    role_str = str(msg.role)
                messages.append(Message(role=role_str, content=msg.content))

        # Add current user query
        messages.append(Message(role="user", content=query))

        return messages

    async def create_session(self, title: str | None = None, template_id: int | None = None):
        """Create a new chat session.

        Args:
            title: Optional session title. Auto-generated if not provided.
            template_id: Optional template ID for pre-configured sessions.

        Effects:
            - Closes any existing session
            - Creates new session in database
            - Sets current_session attribute

        Note:
            Sessions are automatically saved and can be resumed later
            using load_session().
        """
        if self.current_session:
            await self.close_session()

        self._session_context = self.session_manager.create_session(title, template_id)
        self.current_session = await self._session_context.__aenter__()
        logger.info(f"Created new session: {self.current_session.id}")

        # Connect any pending MCP servers now that we have an event loop
        await self.connect_pending_mcp_servers()

    async def load_session(self, conversation_id: str):
        """Load an existing chat session.

        Args:
            conversation_id: UUID of the conversation to load

        Effects:
            - Closes any existing session
            - Loads session from database
            - Restores conversation history
            - Sets current_session attribute

        Raises:
            ValueError: If conversation_id doesn't exist
        """
        if self.current_session:
            await self.close_session()

        self._session_context = self.session_manager.load_session(conversation_id)
        self.current_session = await self._session_context.__aenter__()
        logger.info(f"Loaded session: {self.current_session.id}")

        # Connect any pending MCP servers now that we have an event loop
        await self.connect_pending_mcp_servers()

    async def close_session(self):
        """Close the current session.

        Properly closes the session context manager and cleans up resources.
        Safe to call even if no session is active.
        """
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
            self.current_session = None
            self._session_context = None

    def send_message(
        self,
        content: str,
        provider: str | None = None,
        model: str | None = None,
        stream: bool = False,
        parent_message_id: str | None = None,
    ):
        """Send a message and get a response.

        This is the primary interface for sending messages without RAG.
        For RAG-enhanced messages, use send_message_with_rag().

        Args:
            content: Message content from the user
            provider: Override the current provider (e.g., 'openai')
            model: Override the current model (e.g., 'gpt-4')
            stream: If True, returns async generator for streaming
            parent_message_id: For conversation threading (future feature)

        Returns:
            For stream=False: Response object with content
            For stream=True: Async generator yielding text chunks

        Session Management:
            Automatically creates a session if none exists.
            Messages are persisted to the session for history.

        Example:
            ```python
            # Non-streaming
            response = await chat_manager.send_message("Hello!")
            print(response.content)

            # Streaming
            async for chunk in chat_manager.send_message("Hello!", stream=True):
                print(chunk, end="")
            ```
        """
        if stream:
            return self._send_message_stream_generator(content, provider, model, parent_message_id)
        else:
            return self._send_message_non_stream(content, provider, model, parent_message_id)

    async def _send_message_stream_generator(
        self,
        content: str,
        provider: str | None = None,
        model: str | None = None,
        parent_message_id: str | None = None,
    ):
        """Async generator wrapper for streaming message sending.

        This thin wrapper ensures proper async generator protocol compliance.
        Python requires async generators to be defined with async def and yield.
        """
        async for chunk in self._send_message_stream(content, provider, model, parent_message_id):
            yield chunk

    async def _send_message_stream(
        self,
        content: str,
        provider: str | None = None,
        model: str | None = None,
        parent_message_id: str | None = None,
    ):
        """Send a message with streaming response.

        Core implementation of streaming message sending.

        Process Flow:
            1. Ensure session exists (auto-create if needed)
            2. Build context messages with history and memory
            3. Add user message to session
            4. Stream response from LLM
            5. Update assistant message progressively
            6. Store conversation in memory

        Context Building:
            - Includes system prompt (with optional memory)
            - Adds conversation history (last 20 messages)
            - Appends current user message

        Error Recovery:
            Special handling for "Event loop is closed" during shutdown
            to prevent error spam when the app is closing.

        Performance:
            - Yields chunks immediately for low latency
            - Updates stored message periodically
            - Memory storage is async and non-blocking
        """
        # Ensure we have a session
        if not self.current_session:
            await self.create_session()

        try:
            # Use specified or default provider/model
            provider = provider or self.current_provider
            model = model or self.current_model

            # Get provider instance
            if provider not in self.providers:
                raise ValueError(f"Provider '{provider}' not available")

            llm_provider = self.providers[provider]

            # Build context messages with memory and history BEFORE adding current message
            context_messages = await self._build_context_messages(content, exclude_current=False)

            # Add the current user message as the final message
            context_messages.append(Message(role="user", content=content))

            # Add user message to session
            await self.current_session.add_message(
                role=MessageRole.USER,
                content=content,
                metadata={
                    "parent_message_id": parent_message_id,
                    "provider": provider,
                    "model": model,
                },
            )

            # Create placeholder assistant message
            assistant_msg = await self.current_session.add_message(
                role=MessageRole.ASSISTANT, content="", model=model, metadata={"status": "pending"}
            )

            response_content = ""

            # Stream the response with full context and tool calling
            logger.info(f"Starting stream with {len(context_messages)} context messages")
            chunk_count = 0
            try:
                # Create the original streaming generator
                original_stream = llm_provider.stream_complete(
                    messages=context_messages, model=model, temperature=0.7, max_tokens=1000
                )

                # Use the tool-calling aware streaming handler
                async for chunk in self._handle_streaming_with_tool_calls(
                    original_stream, assistant_msg.id
                ):
                    chunk_count += 1
                    response_content += chunk
                    # Update message periodically (less frequently to avoid conflicts)
                    if chunk_count % 5 == 0:  # Update every 5 chunks instead of every chunk
                        await self.current_session.update_message(
                            assistant_msg.id, content=response_content
                        )
                    yield chunk  # Yield for UI streaming

                logger.info(
                    f"Stream completed with {chunk_count} chunks, "
                    f"total length: {len(response_content)}"
                )
            except Exception as stream_error:
                logger.error(f"Error during streaming (after {chunk_count} chunks): {stream_error}")
                raise

            # Final update
            await self.current_session.update_message(
                assistant_msg.id, content=response_content, metadata={"status": "completed"}
            )

            # Store conversation in memory for future context
            await self._store_conversation_memory(content, response_content)

        except Exception as e:
            # Handle event loop closure gracefully during app shutdown
            if "Event loop is closed" in str(e):
                logger.info("Chat manager stream stopped due to app shutdown")
                return
            logger.error(f"Error sending streaming message: {e}")
            raise

    async def _send_message_non_stream(
        self,
        content: str,
        provider: str | None = None,
        model: str | None = None,
        parent_message_id: str | None = None,
    ):
        """Send a message with non-streaming response.

        Simpler version that waits for complete response before returning.

        Use Cases:
            - Programmatic interactions
            - Testing and debugging
            - When you need the full response for processing

        Note:
            This method doesn't include memory context building like the
            streaming version. It's a simpler direct message-response flow.
            For memory-aware responses, use streaming or RAG methods.
        """
        # Ensure we have a session
        if not self.current_session:
            await self.create_session()

        try:
            # Use specified or default provider/model
            provider = provider or self.current_provider
            model = model or self.current_model

            # Get provider instance
            if provider not in self.providers:
                raise ValueError(f"Provider '{provider}' not available")

            llm_provider = self.providers[provider]

            # Add user message to session
            user_msg = await self.current_session.add_message(
                role=MessageRole.USER,
                content=content,
                metadata={
                    "parent_message_id": parent_message_id,
                    "provider": provider,
                    "model": model,
                },
            )

            # Get completion
            response = await llm_provider.complete(
                messages=[Message(role="user", content=content)],
                model=model,
                temperature=0.7,
                max_tokens=1000,
            )

            # Check for tool calls in response and execute them
            final_content = await self._parse_and_execute_tool_calls(response.content)

            # Add assistant message with potentially modified content
            await self.current_session.add_message(
                role=MessageRole.ASSISTANT,
                content=final_content,
                model=model,
                metadata={"parent_message_id": user_msg.id},
            )

            # Return response with updated content
            response.content = final_content
            return response

        except Exception as e:
            logger.error(f"Error sending message: {e}")
            raise

    async def _build_context_messages(
        self, query: str, exclude_current: bool = False
    ) -> list[Message]:
        """Build messages with memory context.

        Constructs the full message context for the LLM, including system prompt,
        relevant memories, conversation history, and available tools.

        Args:
            query: Current user query for memory retrieval
            exclude_current: If True, exclude the most recent message from history.
                Used to avoid duplicating the current message.

        Returns:
            List of Message objects forming the complete context

        Context Structure:
            1. System prompt (base or user-configured)
            2. Conversation metadata (title, session info)
            3. Relevant memories (if memory integration enabled)
            4. Available MCP tools summary
            5. Conversation history (up to 20 messages)

        Memory Integration:
            When enabled, uses enhanced memory recall to find relevant memories
            including special handling for personal information queries.
        """
        messages = []

        # Start with user-configured system prompt or default
        if self.system_prompt:
            system_content = self.system_prompt
        else:
            system_content = (
                "You are Neuromancer, an advanced AI assistant with exceptional memory "
                "and tool-use capabilities. You have access to long-term memory and can "
                "execute various tools through MCP (Model Context Protocol) servers."
            )

        # Add conversation context
        if self.current_session:
            system_content += f"\n\nConversation: {self.current_session.title}"

        # Add relevant memories if integration is enabled
        if self.system_prompt_memory_integration:
            # Enhanced memory recall for better retrieval
            memories = await self._enhanced_memory_recall(query)

            if memories:
                system_content += "\n\nRelevant memories:\n"
                for memory in memories:
                    system_content += f"- {memory.content}\n"

        # Add available MCP tools with detailed descriptions
        tools = await self.mcp_manager.list_all_tools()
        if tools:
            system_content += f"\n\nAvailable tools ({len(tools)}):"
            system_content += (
                "\nWhen users ask for current information, recent data, or web searches, "
                "you MUST use tools instead of your training data."
            )
            system_content += "\n\nExecute tools by responding with tool calls in this EXACT format (include 'tool_name:'):"
            system_content += "\n```tool_call"
            system_content += "\ntool_name: exa:web_search_exa"  # Use server:tool format
            system_content += "\nparameters:"
            system_content += "\n  query: your search query"
            system_content += "\n  num_results: 5"
            system_content += "\n```"
            system_content += "\n\nCRITICAL: Always include 'tool_name:' prefix and use server:tool format exactly as shown!"
            system_content += "\n\nAvailable tools:"
            for cached_name, tool in self.mcp_manager.tools_cache.items():
                # cached_name is already in server:tool format like "exa:web_search_exa"
                system_content += f"\n- {cached_name}: {tool.description}"
                if tool.parameters:
                    params = ", ".join(tool.parameters.keys())
                    system_content += f" (params: {params})"

        messages.append(Message(role="system", content=system_content))

        # Add conversation history from session
        if self.current_session:
            session_messages = await self.current_session.get_messages(limit=20)

            # Exclude the most recent message if requested (to avoid duplication)
            if exclude_current and session_messages:
                session_messages = session_messages[:-1]

            for msg in session_messages:
                # Ensure role is converted to string if it's an enum
                role_str = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
                messages.append(Message(role=role_str, content=msg.content))

        return messages

    async def _enhanced_memory_recall(self, query: str) -> list:
        """Enhanced memory recall with better personal information retrieval.

        Implements an intelligent memory retrieval strategy that prioritizes
        personal information and expands searches for better recall.

        Args:
            query: The user's query to search for

        Returns:
            List of up to 8 most relevant Memory objects

        Search Strategy:
            1. Initial semantic search with standard parameters
            2. Personal information detection and expanded search
            3. Keyword-based fallback for personal queries
            4. Importance and recency-based ranking

        Personal Query Detection:
            Identifies queries about personal information like names,
            contact details, preferences, etc., and performs additional
            targeted searches to ensure this critical information is found.

        Performance:
            - Primary search: ~50-100ms
            - Expanded search (if needed): +50-100ms
            - Returns max 8 memories to balance context and relevance
        """
        try:
            # Start with direct semantic search
            memories = await self.memory_manager.recall(query=query, limit=8, threshold=0.4)

            # For personal information queries, expand search
            personal_queries = [
                "name",
                "my name",
                "what is my name",
                "who am i",
                "about me",
                "phone",
                "number",
                "email",
                "address",
                "birthday",
                "preference",
            ]

            query_lower = query.lower()
            is_personal_query = any(term in query_lower for term in personal_queries)

            if is_personal_query and len(memories) < 3:
                # Search for memories containing personal keywords
                personal_keywords = [
                    "name",
                    "phone",
                    "email",
                    "address",
                    "birthday",
                    "preference",
                    "like",
                    "dislike",
                    "i am",
                    "my",
                ]

                for keyword in personal_keywords:
                    additional_memories = await self.memory_manager.recall(
                        query=keyword, limit=5, threshold=0.3
                    )

                    # Add unique memories
                    existing_ids = {m.id for m in memories}
                    for mem in additional_memories:
                        if mem.id not in existing_ids:
                            memories.append(mem)
                            existing_ids.add(mem.id)

                    if len(memories) >= 8:  # Limit total memories
                        break

            # Sort by importance and recency
            memories.sort(key=lambda m: (m.importance, m.accessed_at.timestamp()), reverse=True)

            return memories[:8]  # Limit to top 8

        except Exception as e:
            logger.error(f"Enhanced memory recall failed: {e}")
            # Fallback to basic recall
            return await self.memory_manager.recall(query=query, limit=5, threshold=0.4)

    async def get_conversations(self, **kwargs) -> list[dict[str, Any]]:
        """Get list of conversations.

        Args:
            **kwargs: Filtering options passed to SessionManager
                - archived: Include archived conversations
                - limit: Maximum number to return
                - offset: Pagination offset

        Returns:
            List of conversation dictionaries with metadata
        """
        return await self.session_manager.list_conversations(**kwargs)

    async def get_conversation_stats(self, conversation_id: str) -> dict[str, Any]:
        """Get conversation statistics.

        Args:
            conversation_id: UUID of the conversation

        Returns:
            Dict with statistics like message count, duration,
            tokens used, most active times, etc.
        """
        return await self.session_manager.get_conversation_stats(conversation_id)

    async def export_conversation(
        self, conversation_id: str, format: str = "json"
    ) -> dict[str, Any]:
        """Export a conversation.

        Args:
            conversation_id: UUID of the conversation
            format: Export format ('json', 'markdown', 'txt')

        Returns:
            Exported data in requested format

        Formats:
            - json: Complete data with metadata
            - markdown: Human-readable with formatting
            - txt: Plain text transcript
        """
        return await self.session_manager.export_conversation(conversation_id, format)

    # MCP Tool Methods
    # These methods provide access to Model Context Protocol tools,
    # enabling the AI to execute external functions and integrations.

    async def list_mcp_servers(self) -> list[dict[str, Any]]:
        """List all connected MCP servers.

        Returns:
            List of server information including name, URL, connection status,
            and number of available tools.
        """
        return await self.mcp_manager.list_servers()

    async def list_mcp_tools(self) -> list[MCPTool]:
        """List all available MCP tools from all servers.

        Returns:
            List of MCPTool objects with full tool information including
            names, descriptions, parameters, and server prefixes.
        """
        return await self.mcp_manager.list_all_tools()

    async def execute_mcp_tool(self, tool_name: str, parameters: dict[str, Any]) -> MCPResponse:
        """Execute an MCP tool.

        Args:
            tool_name: Name of the tool in format 'server:tool'
            parameters: Tool parameters as a dictionary

        Returns:
            MCPResponse with execution result or error information

        Example:
            ```python
            response = await chat_manager.execute_mcp_tool(
                "filesystem:read_file",
                {"path": "/tmp/data.txt"}
            )
            if response.success:
                print(response.result)
            ```
        """
        return await self.mcp_manager.execute_tool(tool_name, parameters)

    async def add_mcp_server(
        self,
        name: str,
        server_url: str = None,
        ssl_config: dict[str, Any] = None,
        command: str = None,
        args: list[str] = None,
    ) -> bool:
        """Add and connect to a new MCP server.

        Args:
            name: Unique name for the server
            server_url: WebSocket URL of the MCP server (for WebSocket servers)
            ssl_config: Optional SSL configuration for secure connections
            command: Command to execute (for subprocess servers)
            args: Arguments for the command (for subprocess servers)

        Returns:
            True if successfully connected
        """
        # If no SSL config provided, use defaults from configuration
        if ssl_config is None and server_url:
            from src.core.config import _config_manager

            ssl_config = {
                "verify": _config_manager.get("mcp.ssl_verify", True),
                "ca_bundle": _config_manager.get("mcp.ssl_ca_bundle"),
                "allow_self_signed": _config_manager.get("mcp.allow_self_signed", False),
            }

        success = await self.mcp_manager.add_server(
            name, server_url=server_url, ssl_config=ssl_config, command=command, args=args
        )

        # Save to configuration if successful
        if success:
            await self._save_mcp_server_to_config(name, server_url, ssl_config, command, args)

        return success

    async def _save_mcp_server_to_config(
        self,
        name: str,
        server_url: str = None,
        ssl_config: dict = None,
        command: str = None,
        args: list[str] = None,
    ):
        """Save MCP server configuration to persistent config."""
        try:
            from src.core.config import _config_manager

            # Get current MCP servers configuration
            servers = _config_manager.get("mcp.servers", {})

            # Build server config
            server_config = {"enabled": True, "description": f"MCP server: {name}"}

            if server_url:
                # WebSocket server
                server_config["url"] = server_url
                if ssl_config:
                    server_config["ssl"] = ssl_config
            elif command and args:
                # Subprocess server
                server_config["command"] = command
                server_config["args"] = args

            # Add to servers
            servers[name] = server_config

            # Save back to config using async method
            success = await _config_manager.set("mcp.servers", servers)
            if success:
                # The config manager will auto-save if enabled, but let's ensure it's saved
                _config_manager.save()
                logger.info(f"Saved MCP server '{name}' to configuration")
            else:
                logger.error(f"Failed to set MCP server '{name}' in configuration")

        except Exception as e:
            logger.error(f"Failed to save MCP server '{name}' to config: {e}")
            import traceback

            logger.debug(f"Full error traceback: {traceback.format_exc()}")

    async def remove_mcp_server(self, name: str) -> bool:
        """Disconnect and remove an MCP server.

        Args:
            name: Name of the server to remove

        Returns:
            True if successfully removed
        """
        success = await self.mcp_manager.remove_server(name)

        # Remove from configuration if successful
        if success:
            await self._remove_mcp_server_from_config(name)

        return success

    async def _remove_mcp_server_from_config(self, name: str):
        """Remove MCP server from persistent config."""
        try:
            from src.core.config import _config_manager

            # Get current MCP servers configuration
            servers = _config_manager.get("mcp.servers", {})

            # Remove server if it exists
            if name in servers:
                del servers[name]

                # Save back to config using async method
                success = await _config_manager.set("mcp.servers", servers)
                if success:
                    _config_manager.save()
                    logger.info(f"Removed MCP server '{name}' from configuration")
                else:
                    logger.error(f"Failed to update configuration after removing server '{name}'")

        except Exception as e:
            logger.error(f"Failed to remove MCP server '{name}' from config: {e}")
            import traceback

            logger.debug(f"Full error traceback: {traceback.format_exc()}")

    async def check_mcp_health(self) -> dict[str, bool]:
        """Check health status of all MCP servers.

        Returns:
            Dictionary mapping server names to health status
        """
        return await self.mcp_manager.health_check_all()

    async def archive_conversation(self, conversation_id: str) -> bool:
        """Archive a conversation.

        Args:
            conversation_id: UUID of the conversation

        Returns:
            True if successfully archived

        Effects:
            Archived conversations are hidden from normal lists
            but remain searchable and can be unarchived.
        """
        return await self.session_manager.archive_conversation(conversation_id)

    async def delete_conversation(self, conversation_id: str, hard_delete: bool = False) -> bool:
        """Delete a conversation.

        Args:
            conversation_id: UUID of the conversation
            hard_delete: If True, permanently delete. If False, soft delete.

        Returns:
            True if successfully deleted

        Soft vs Hard Delete:
            - Soft: Marks as deleted but keeps data (recoverable)
            - Hard: Permanently removes from database (unrecoverable)
        """
        return await self.session_manager.delete_conversation(conversation_id, hard_delete)

    async def set_provider(self, provider: str) -> bool:
        """Set the current LLM provider.

        Args:
            provider: Provider name ('ollama', 'openai', 'lmstudio')

        Returns:
            True if provider exists and was set

        Note:
            This only changes the provider, not the model. You may need
            to also call set_model() if the current model isn't available
            in the new provider.
        """
        if provider in self.providers:
            self.current_provider = provider
            logger.info(f"Set provider to: {provider}")
            return True
        return False

    async def list_providers(self) -> list[str]:
        """List available providers.

        Returns:
            List of initialized provider names

        Note:
            This returns the same as get_available_providers() but
            maintains the async interface for consistency.
        """
        return list(self.providers.keys())

    async def list_models(self, provider: str | None = None) -> list[str]:
        """List available models for a provider.

        Args:
            provider: Provider name (uses current if not specified)

        Returns:
            List of model names available from the provider

        Note:
            This queries the provider directly. For cached model info
            with metadata, use get_available_models() instead.
        """
        provider = provider or self.current_provider

        if provider in self.providers:
            return await self.providers[provider].list_models()
        return []

    async def _store_conversation_memory(self, user_content: str, assistant_content: str):
        """Store conversation exchange in memory for future context with intelligent filtering.

        Analyzes conversation content and selectively stores important information
        in the memory system for future retrieval.

        Args:
            user_content: The user's message
            assistant_content: The assistant's response

        Storage Logic:
            1. Calculate importance scores for both messages
            2. Classify memory types based on content analysis
            3. Store messages that meet importance thresholds
            4. Create conversation pair memories for valuable exchanges

        Importance Thresholds:
            - User messages: 0.4 (more inclusive)
            - Assistant responses: 0.5 (more selective)
            - Conversation pairs: 0.6 (only meaningful exchanges)

        Memory Types:
            - EPISODIC: Time-based events and experiences
            - SEMANTIC: Facts, procedures, general knowledge
            - LONG_TERM: Important persistent information
            - SHORT_TERM: Temporary information (default)

        Performance:
            Designed to be fast and non-blocking. Memory storage happens
            asynchronously and failures don't affect the conversation flow.
        """
        try:
            # Calculate importance for both messages
            user_importance = self._calculate_importance(user_content)
            assistant_importance = self._calculate_importance(assistant_content)

            # Enhanced memory type determination based on content analysis
            def determine_memory_type(content: str, importance: float) -> MemoryType:
                content_lower = content.lower()

                # Episodic memory for specific events, conversations, and experiences
                if any(
                    term in content_lower
                    for term in [
                        "today",
                        "yesterday",
                        "last week",
                        "remember when",
                        "that time",
                        "conversation",
                        "meeting",
                        "discussion",
                        "session",
                    ]
                ):
                    return MemoryType.EPISODIC

                # Semantic memory for facts, procedures, and general knowledge
                if any(
                    term in content_lower
                    for term in [
                        "how to",
                        "what is",
                        "definition",
                        "explain",
                        "concept",
                        "algorithm",
                        "method",
                        "procedure",
                        "process",
                    ]
                ):
                    return MemoryType.SEMANTIC

                # Long-term memory for important persistent information
                if importance >= 0.7 or any(
                    term in content_lower
                    for term in [
                        "my name",
                        "i am",
                        "my company",
                        "my project",
                        "preference",
                        "always",
                        "never",
                        "important",
                        "remember",
                    ]
                ):
                    return MemoryType.LONG_TERM

                # Default to short-term for temporary information
                return MemoryType.SHORT_TERM

            # Only store user message if it meets minimum importance threshold
            if user_importance >= 0.4:  # Raised threshold to be more selective
                memory_type = determine_memory_type(user_content, user_importance)

                # Create more descriptive content for user messages
                content_summary = self._create_memory_summary(user_content, "user")

                await self.memory_manager.remember(
                    content=content_summary,
                    memory_type=memory_type,
                    importance=user_importance,
                    metadata={
                        "type": "user_message",
                        "session_id": self.current_session.id if self.current_session else None,
                        "session_title": (
                            self.current_session.title if self.current_session else None
                        ),
                        "timestamp": datetime.now().isoformat(),
                        "word_count": len(user_content.split()),
                        "original_length": len(user_content),
                    },
                )

            # Only store assistant response if it's particularly informative or important
            if assistant_importance >= 0.5:  # Higher threshold for assistant responses
                memory_type = determine_memory_type(assistant_content, assistant_importance)

                # Create more descriptive content for assistant messages
                content_summary = self._create_memory_summary(assistant_content, "assistant")

                await self.memory_manager.remember(
                    content=content_summary,
                    memory_type=memory_type,
                    importance=assistant_importance,
                    metadata={
                        "type": "assistant_response",
                        "session_id": self.current_session.id if self.current_session else None,
                        "session_title": (
                            self.current_session.title if self.current_session else None
                        ),
                        "timestamp": datetime.now().isoformat(),
                        "word_count": len(assistant_content.split()),
                        "original_length": len(assistant_content),
                    },
                )

            # Store conversation pairs only for highly valuable exchanges
            pair_importance = max(user_importance, assistant_importance)

            if pair_importance >= 0.6:  # Only store meaningful exchanges
                # Create a more structured conversation summary
                conversation_summary = self._create_conversation_summary(
                    user_content, assistant_content
                )

                await self.memory_manager.remember(
                    content=conversation_summary,
                    memory_type=MemoryType.SEMANTIC,  # Conversation pairs are semantic knowledge
                    importance=pair_importance,
                    metadata={
                        "type": "conversation_pair",
                        "session_id": self.current_session.id if self.current_session else None,
                        "session_title": (
                            self.current_session.title if self.current_session else None
                        ),
                        "timestamp": datetime.now().isoformat(),
                        "user_importance": user_importance,
                        "assistant_importance": assistant_importance,
                        "combined_length": len(user_content) + len(assistant_content),
                    },
                )

            # Log memory storage decisions
            logger.debug(
                f"Memory storage - User: {user_importance:.2f} ({'stored' if user_importance >= 0.4 else 'skipped'}), "
                f"Assistant: {assistant_importance:.2f} ({'stored' if assistant_importance >= 0.5 else 'skipped'}), "
                f"Pair: {pair_importance:.2f} ({'stored' if pair_importance >= 0.6 else 'skipped'})"
            )

        except Exception as e:
            logger.error(f"Failed to store conversation memory: {e}")

    def _create_memory_summary(self, content: str, speaker: str) -> str:
        """Create a more descriptive summary for memory storage.

        Args:
            content: Original message content
            speaker: 'user' or 'assistant'

        Returns:
            Formatted content with speaker context

        Truncation:
            Long content is truncated to ~400 chars, preserving
            both beginning and end for better context.
        """
        # Truncate very long content but preserve key information
        if len(content) > 500:
            # Try to preserve the beginning and end, which often contain key info
            content = content[:250] + " ... " + content[-150:]

        # Add speaker context
        if speaker == "user":
            return f"User said: {content}"
        else:
            return f"Assistant explained: {content}"

    def _create_conversation_summary(self, user_content: str, assistant_content: str) -> str:
        """Create a structured summary of a conversation exchange.

        Args:
            user_content: The user's message
            assistant_content: The assistant's response

        Returns:
            Formatted conversation summary for memory storage

        Format:
            Creates a structured Q&A format that's easy to parse
            and understand when retrieved from memory.
        """
        # Truncate long messages for summary
        user_summary = user_content[:200] + "..." if len(user_content) > 200 else user_content
        assistant_summary = (
            assistant_content[:200] + "..." if len(assistant_content) > 200 else assistant_content
        )

        return f"Conversation exchange:\nUser asked: {user_summary}\nAssistant replied: {assistant_summary}"

    def _calculate_importance(self, content: str) -> float:
        """Calculate importance score for content based on various factors.

        Implements a sophisticated scoring algorithm that considers multiple
        signals to determine if content should be stored in memory.

        Args:
            content: Text content to analyze

        Returns:
            Importance score between 0.0 and 1.0

        Scoring Factors:
            1. Base score: 0.3 (conservative default)
            2. Length/complexity: Up to +0.25 for substantial content
            3. Questions: +0.1-0.2 (indicates learning opportunity)
            4. Memory commands: +0.3 (explicit storage request)
            5. Personal identity: +0.25 (critical information)
            6. Preferences: +0.2 (personalization data)
            7. Project/goals: +0.15 (work context)
            8. Learning requests: +0.1 (knowledge building)
            9. Technical content: +0.03-0.15 (domain expertise)
            10. Priority indicators: +0.1 (urgency/importance)
            11. Emotional content: +0.05 (significant interactions)
            12. Structured data: +0.05-0.1 (URLs, emails, etc.)

        Design Philosophy:
            The algorithm is intentionally conservative (base 0.3) to avoid
            memory pollution. Content must have clear value signals to be
            stored. Personal information and explicit requests get highest priority.
        """
        importance = 0.3  # Lower base importance to be more selective

        # Length and complexity factors
        word_count = len(content.split())
        if word_count > 20:
            importance += 0.1
        if word_count > 50:
            importance += 0.1
        if word_count > 100:
            importance += 0.05

        # Question factor (questions often indicate learning needs)
        question_count = content.count("?")
        if question_count > 0:
            importance += min(0.2, question_count * 0.1)

        # High-priority memory indicators
        memory_commands = [
            "remember",
            "don't forget",
            "keep in mind",
            "note that",
            "important",
            "save this",
            "store this",
            "make a note",
            "remind me",
        ]

        # Personal identity and preferences (very important)
        identity_terms = [
            "my name is",
            "i am",
            "i'm called",
            "call me",
            "i work at",
            "my company",
            "my role",
            "my job",
            "my position",
            "my title",
        ]

        # User preferences and settings (important for personalization)
        preference_terms = [
            "i prefer",
            "i like",
            "i don't like",
            "i hate",
            "my favorite",
            "i always",
            "i never",
            "i usually",
            "i typically",
            "i tend to",
        ]

        # Project and goal information (important for context)
        project_terms = [
            "working on",
            "my project",
            "my goal",
            "objective",
            "deadline",
            "requirement",
            "specification",
            "feature",
            "bug",
            "issue",
        ]

        # Learning and knowledge (important for assistance)
        learning_terms = [
            "how to",
            "help me",
            "explain",
            "what is",
            "why does",
            "show me",
            "teach me",
            "guide me",
            "walk me through",
        ]

        # Technical context (moderately important)
        technical_terms = [
            "code",
            "function",
            "class",
            "method",
            "algorithm",
            "database",
            "api",
            "config",
            "configuration",
            "setup",
            "install",
            "debug",
            "error",
            "framework",
            "library",
            "package",
            "module",
            "script",
        ]

        content_lower = content.lower()

        # Check for explicit memory commands (highest priority)
        for term in memory_commands:
            if term in content_lower:
                importance += 0.3
                break

        # Check for identity information (very high priority)
        for term in identity_terms:
            if term in content_lower:
                importance += 0.25
                break

        # Check for preferences (high priority)
        for term in preference_terms:
            if term in content_lower:
                importance += 0.2
                break

        # Check for project/goal information (high priority)
        for term in project_terms:
            if term in content_lower:
                importance += 0.15
                break

        # Check for learning requests (moderate priority)
        for term in learning_terms:
            if term in content_lower:
                importance += 0.1
                break

        # Technical content (moderate priority, but cap accumulation)
        tech_matches = sum(1 for term in technical_terms if term in content_lower)
        if tech_matches > 0:
            importance += min(0.15, tech_matches * 0.03)

        # Context indicators that suggest importance
        context_indicators = [
            "critical",
            "urgent",
            "priority",
            "key",
            "essential",
            "must",
            "required",
            "necessary",
            "vital",
            "crucial",
        ]

        for indicator in context_indicators:
            if indicator in content_lower:
                importance += 0.1
                break

        # Emotional content often indicates importance
        emotional_terms = [
            "frustrated",
            "excited",
            "worried",
            "concerned",
            "happy",
            "sad",
            "angry",
            "disappointed",
            "pleased",
            "surprised",
        ]

        for term in emotional_terms:
            if term in content_lower:
                importance += 0.05
                break

        # Numbers and specific data (might be important)
        import re

        if re.search(r"\b\d+\b", content):  # Contains numbers
            importance += 0.05

        # URLs, emails, or specific identifiers
        if re.search(r"https?://|@|\.com|\.org|\.net", content):
            importance += 0.1

        # Cap at 1.0 and ensure minimum threshold for storage
        final_importance = min(1.0, importance)

        # Log detailed importance calculation for debugging
        if final_importance > 0.6:
            logger.debug(f"High importance content ({final_importance:.2f}): {content[:100]}...")

        return final_importance

    async def _parse_and_execute_tool_calls(self, response_content: str) -> str:
        """Parse tool calls from AI response and execute them.

        Args:
            response_content: The AI's response content that may contain tool calls

        Returns:
            Modified response content with tool results integrated
        """
        import re

        import yaml

        # Pattern to match tool call blocks (primary format)
        tool_call_pattern = r"```tool_call\n(.*?)\n```"
        matches = re.findall(tool_call_pattern, response_content, re.DOTALL)

        # Also try alternative format: **Tool call: tool_name** with parameters
        if not matches:
            alt_pattern = r"\*\*Tool call: ([^*]+)\*\*.*?\*\*Parameters:\*\*\s*```([^`]+)```"
            alt_matches = re.findall(alt_pattern, response_content, re.DOTALL | re.IGNORECASE)

            # Convert alternative format to standard format
            for tool_name, params in alt_matches:
                tool_spec = f"tool_name: {tool_name.strip()}\nparameters:\n{params.strip()}"
                matches.append(tool_spec)

        if not matches:
            return response_content

        modified_content = response_content

        for match in matches:
            try:
                logger.debug(f"Raw tool call match: {repr(match)}")

                # Handle case where tool name appears without "tool_name:" prefix
                if match.strip() and not match.startswith("tool_name:"):
                    lines = match.strip().split("\n")
                    if lines and ":" not in lines[0]:
                        # First line is likely the tool name without prefix
                        tool_name_line = f"tool_name: {lines[0]}"
                        rest_lines = lines[1:]
                        match = tool_name_line + "\n" + "\n".join(rest_lines)
                        logger.debug(f"Modified tool call: {repr(match)}")

                # Parse the YAML-like tool call specification
                tool_spec = yaml.safe_load(match)

                if not isinstance(tool_spec, dict):
                    continue

                tool_name = tool_spec.get("tool_name")
                parameters = tool_spec.get("parameters", {})

                if not tool_name:
                    continue

                logger.info(f"Executing MCP tool: {tool_name} with parameters: {parameters}")

                # Execute the tool
                result = await self.execute_mcp_tool(tool_name, parameters)

                # Format the result
                if result.success:
                    result_text = f"\n**Tool Result ({tool_name}):**\n{result.result}\n"
                else:
                    result_text = f"\n**Tool Error ({tool_name}):** {result.error}\n"

                # Replace the tool call block with the result
                tool_block = f"```tool_call\n{match}\n```"
                modified_content = modified_content.replace(tool_block, result_text)

            except Exception as e:
                logger.error(f"Error executing tool call: {e}")
                # Replace with error message
                tool_block = f"```tool_call\n{match}\n```"
                error_text = f"\n**Tool Execution Error:** {str(e)}\n"
                modified_content = modified_content.replace(tool_block, error_text)

        return modified_content

    async def _handle_streaming_with_tool_calls(self, original_generator, session_message_id: str):
        """Handle streaming response with potential tool calls.

        Args:
            original_generator: The original streaming generator
            session_message_id: ID of the message to update

        Yields:
            Response chunks with tool results integrated
        """
        accumulated_content = ""

        # Collect all chunks first
        async for chunk in original_generator:
            accumulated_content += chunk
            yield chunk

        # Check for tool calls after streaming is complete
        if "```tool_call" in accumulated_content:
            logger.info("Detected tool calls in response, executing...")

            # Execute tool calls and get modified content
            modified_content = await self._parse_and_execute_tool_calls(accumulated_content)

            # If content was modified (tools were executed), update the message
            if modified_content != accumulated_content:
                if self.current_session and session_message_id:
                    await self.current_session.update_message(
                        session_message_id, content=modified_content
                    )

                # Yield the additional content (tool results)
                additional_content = modified_content[len(accumulated_content) :]
                if additional_content:
                    # Stream the tool results
                    for char in additional_content:
                        yield char
                        # Small delay to make tool results visible
                        import asyncio

                        await asyncio.sleep(0.01)
