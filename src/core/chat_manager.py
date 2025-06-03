"""Chat management with provider integration and memory."""

from datetime import datetime
from typing import Any

from src.core.config import settings
from src.core.models import MessageRole
from src.core.session_manager import ConversationSession, SessionManager
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
    """Enterprise chat manager with sessions, providers, memory, and MCP tools."""

    def __init__(self):
        """Initialize chat manager."""
        self.providers: dict[str, LLMProvider] = {}
        self.current_provider = settings.DEFAULT_PROVIDER
        self.current_model = settings.DEFAULT_MODEL

        # Initialize managers
        self.memory_manager = MemoryManager()
        self.mcp_manager = MCPManager()
        self.session_manager = SessionManager()

        # Current session
        self.current_session: ConversationSession | None = None
        self._session_context = None

        # Initialize providers
        self._initialize_providers()

        # Session manager will be started on first use

    def _initialize_providers(self):
        """Initialize LLM providers based on configuration."""
        from src.core.config import config_manager

        # Ollama
        try:
            if config_manager.get("providers.ollama_enabled", True):
                ollama_host = config_manager.get("providers.ollama_host", "http://localhost:11434")
                self.providers["ollama"] = OllamaProvider(host=ollama_host)
                logger.info(f"Initialized Ollama provider at {ollama_host}")
        except Exception as e:
            logger.warning(f"Failed to initialize Ollama provider: {e}")

        # OpenAI
        try:
            if config_manager.get("providers.openai_enabled", False):
                api_key = config_manager.get("providers.openai_api_key")
                if api_key:
                    openai_params = {"api_key": api_key}

                    base_url = config_manager.get("providers.openai_base_url")
                    if base_url:
                        openai_params["base_url"] = base_url

                    organization = config_manager.get("providers.openai_organization")
                    if organization:
                        openai_params["organization"] = organization

                    self.providers["openai"] = OpenAIProvider(**openai_params)
                    logger.info("Initialized OpenAI provider")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI provider: {e}")

        # LM Studio
        try:
            if config_manager.get("providers.lmstudio_enabled", False):
                lmstudio_host = config_manager.get(
                    "providers.lmstudio_host", "http://localhost:1234"
                )
                self.providers["lmstudio"] = LMStudioProvider(host=lmstudio_host)
                logger.info(f"Initialized LM Studio provider at {lmstudio_host}")
        except Exception as e:
            logger.warning(f"Failed to initialize LM Studio provider: {e}")

        logger.info(f"Initialized {len(self.providers)} LLM providers")

    def refresh_providers(self):
        """Refresh provider configurations."""
        self.providers.clear()
        self._initialize_providers()

    def get_available_providers(self) -> list[str]:
        """Get list of available provider names."""
        return list(self.providers.keys())

    def is_provider_available(self, provider_name: str) -> bool:
        """Check if a provider is available."""
        return provider_name in self.providers

    async def create_session(self, title: str | None = None, template_id: int | None = None):
        """Create a new chat session."""
        if self.current_session:
            await self.close_session()

        self._session_context = self.session_manager.create_session(title, template_id)
        self.current_session = await self._session_context.__aenter__()
        logger.info(f"Created new session: {self.current_session.id}")

    async def load_session(self, conversation_id: str):
        """Load an existing chat session."""
        if self.current_session:
            await self.close_session()

        self._session_context = self.session_manager.load_session(conversation_id)
        self.current_session = await self._session_context.__aenter__()
        logger.info(f"Loaded session: {self.current_session.id}")

    async def close_session(self):
        """Close the current session."""
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

        Args:
            content: Message content
            provider: Override provider
            model: Override model
            stream: Stream the response
            parent_message_id: ID of parent message for threading

        Returns:
            Completion response (non-streaming) or async generator (streaming)
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
        """Async generator wrapper for streaming message sending."""
        async for chunk in self._send_message_stream(content, provider, model, parent_message_id):
            yield chunk

    async def _send_message_stream(
        self,
        content: str,
        provider: str | None = None,
        model: str | None = None,
        parent_message_id: str | None = None,
    ):
        """Send a message with streaming response."""
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
            user_msg = await self.current_session.add_message(
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

            # Stream the response with full context
            logger.info(f"Starting stream with {len(context_messages)} context messages")
            chunk_count = 0
            try:
                async for chunk in llm_provider.stream_complete(
                    messages=context_messages, model=model, temperature=0.7, max_tokens=1000
                ):
                    chunk_count += 1
                    response_content += chunk
                    # Update message periodically
                    await self.current_session.update_message(
                        assistant_msg.id, content=response_content
                    )
                    yield chunk  # Yield for UI streaming

                logger.info(
                    f"Stream completed with {chunk_count} chunks, total length: {len(response_content)}"
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
        """Send a message with non-streaming response."""
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

            # Add assistant message
            assistant_msg = await self.current_session.add_message(
                role=MessageRole.ASSISTANT,
                content=response.content,
                model=model,
                metadata={"parent_message_id": user_msg.id},
            )

            return response

        except Exception as e:
            logger.error(f"Error sending message: {e}")
            raise

    async def _build_context_messages(
        self, query: str, exclude_current: bool = False
    ) -> list[Message]:
        """Build messages with memory context.

        Args:
            query: Current user query
            exclude_current: If True, exclude the most recent message from history

        Returns:
            Messages with context
        """
        messages = []

        # System message
        system_content = (
            "You are Neuromancer, an advanced AI assistant with exceptional memory "
            "and tool-use capabilities. You have access to long-term memory and can "
            "execute various tools through MCP (Model Context Protocol) servers."
        )

        # Add conversation context
        if self.current_session:
            system_content += f"\n\nConversation: {self.current_session.title}"

        # Add relevant memories
        memories = await self.memory_manager.recall(query=query, limit=5, threshold=0.6)

        if memories:
            system_content += "\n\nRelevant memories:\n"
            for memory in memories:
                system_content += f"- {memory.content}\n"

        # Add available MCP tools
        tools = await self.mcp_manager.list_all_tools()
        if tools:
            system_content += f"\n\nAvailable tools: {len(tools)}"

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

    async def get_conversations(self, **kwargs) -> list[dict[str, Any]]:
        """Get list of conversations."""
        return await self.session_manager.list_conversations(**kwargs)

    async def get_conversation_stats(self, conversation_id: str) -> dict[str, Any]:
        """Get conversation statistics."""
        return await self.session_manager.get_conversation_stats(conversation_id)

    async def export_conversation(
        self, conversation_id: str, format: str = "json"
    ) -> dict[str, Any]:
        """Export a conversation."""
        return await self.session_manager.export_conversation(conversation_id, format)

    async def archive_conversation(self, conversation_id: str) -> bool:
        """Archive a conversation."""
        return await self.session_manager.archive_conversation(conversation_id)

    async def delete_conversation(self, conversation_id: str, hard_delete: bool = False) -> bool:
        """Delete a conversation."""
        return await self.session_manager.delete_conversation(conversation_id, hard_delete)

    async def set_provider(self, provider: str) -> bool:
        """Set the current LLM provider.

        Args:
            provider: Provider name

        Returns:
            True if successful
        """
        if provider in self.providers:
            self.current_provider = provider
            logger.info(f"Set provider to: {provider}")
            return True
        return False

    async def set_model(self, model: str):
        """Set the current model.

        Args:
            model: Model name
        """
        self.current_model = model
        logger.info(f"Set model to: {model}")

    async def list_providers(self) -> list[str]:
        """List available providers.

        Returns:
            List of provider names
        """
        return list(self.providers.keys())

    async def list_models(self, provider: str | None = None) -> list[str]:
        """List available models for a provider.

        Args:
            provider: Provider name (uses current if not specified)

        Returns:
            List of model names
        """
        provider = provider or self.current_provider

        if provider in self.providers:
            return await self.providers[provider].list_models()
        return []

    async def _store_conversation_memory(self, user_content: str, assistant_content: str):
        """Store conversation exchange in memory for future context."""
        try:
            # Store user message with high importance if it contains key information
            user_importance = self._calculate_importance(user_content)
            await self.memory_manager.remember(
                content=f"User: {user_content}",
                memory_type=(
                    MemoryType.SHORT_TERM if user_importance < 0.7 else MemoryType.LONG_TERM
                ),
                importance=user_importance,
                metadata={
                    "type": "user_message",
                    "session_id": self.current_session.id if self.current_session else None,
                    "timestamp": datetime.now().isoformat(),
                },
            )

            # Store assistant response with contextual importance
            assistant_importance = self._calculate_importance(assistant_content)
            await self.memory_manager.remember(
                content=f"Assistant: {assistant_content}",
                memory_type=(
                    MemoryType.SHORT_TERM if assistant_importance < 0.8 else MemoryType.LONG_TERM
                ),
                importance=assistant_importance,
                metadata={
                    "type": "assistant_response",
                    "session_id": self.current_session.id if self.current_session else None,
                    "timestamp": datetime.now().isoformat(),
                },
            )

            # Store the conversation pair as semantic memory for better context
            conversation_pair = f"Q: {user_content}\nA: {assistant_content}"
            pair_importance = max(user_importance, assistant_importance)

            if pair_importance > 0.6:  # Only store meaningful exchanges
                await self.memory_manager.remember(
                    content=conversation_pair,
                    memory_type=MemoryType.SEMANTIC,
                    importance=pair_importance,
                    metadata={
                        "type": "conversation_pair",
                        "session_id": self.current_session.id if self.current_session else None,
                        "timestamp": datetime.now().isoformat(),
                    },
                )

            logger.debug(
                f"Stored conversation memory - User: {user_importance:.2f}, Assistant: {assistant_importance:.2f}"
            )

        except Exception as e:
            logger.error(f"Failed to store conversation memory: {e}")

    def _calculate_importance(self, content: str) -> float:
        """Calculate importance score for content based on various factors."""
        importance = 0.5  # Base importance

        # Length factor (longer messages might be more important)
        if len(content) > 100:
            importance += 0.1
        if len(content) > 500:
            importance += 0.1

        # Question factor (questions are often important)
        if "?" in content:
            importance += 0.2

        # Key terms that indicate importance
        important_terms = [
            "remember",
            "important",
            "preference",
            "always",
            "never",
            "name",
            "project",
            "goal",
            "objective",
            "requirement",
            "deadline",
            "priority",
            "critical",
            "urgent",
            "key",
        ]

        content_lower = content.lower()
        for term in important_terms:
            if term in content_lower:
                importance += 0.15

        # Personal information indicators
        personal_indicators = [
            "i am",
            "my name",
            "i work",
            "i like",
            "i prefer",
            "i need",
            "i want",
            "my company",
            "my project",
            "my team",
        ]

        for indicator in personal_indicators:
            if indicator in content_lower:
                importance += 0.2
                break  # Only add once

        # Technical/specific content
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
        ]

        tech_count = sum(1 for term in technical_terms if term in content_lower)
        if tech_count > 0:
            importance += min(0.2, tech_count * 0.05)

        # Cap at 1.0
        return min(1.0, importance)
