"""Comprehensive configuration schema with validation."""

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator


class LogLevel(str, Enum):
    """Logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class MemoryStrategy(str, Enum):
    """Memory management strategies."""

    AGGRESSIVE = "aggressive"  # Keep everything, high memory usage
    BALANCED = "balanced"  # Smart consolidation
    CONSERVATIVE = "conservative"  # Minimal memory usage
    CUSTOM = "custom"  # User-defined rules


class ThemeMode(str, Enum):
    """UI theme modes."""

    LIGHT = "Light"
    DARK = "Dark"
    AUTO = "Auto"  # Follow system theme
    CUSTOM = "Custom"


class ModelSelectionStrategy(str, Enum):
    """How to select models."""

    MANUAL = "manual"
    CHEAPEST = "cheapest"
    FASTEST = "fastest"
    BEST_QUALITY = "best_quality"
    BALANCED = "balanced"


class GeneralConfig(BaseModel):
    """General application settings."""

    app_name: str = Field(default="Neuromancer", description="Application name")
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging verbosity")
    auto_save: bool = Field(default=True, description="Auto-save conversations")
    auto_save_interval: int = Field(default=300, description="Auto-save interval in seconds", ge=30)
    check_updates: bool = Field(default=True, description="Check for updates on startup")
    telemetry: bool = Field(default=False, description="Send anonymous usage statistics")
    language: str = Field(default="en", description="UI language code")
    timezone: str = Field(default="auto", description="Timezone (auto detects from system)")

    # Advanced
    worker_threads: int = Field(default=4, description="Number of worker threads", ge=1, le=32)
    async_timeout: int = Field(default=30, description="Async operation timeout in seconds", ge=5)


class ProviderConfig(BaseModel):
    """LLM provider configuration."""

    # Provider selection
    default_provider: str = Field(default="ollama", description="Default LLM provider")
    fallback_providers: list[str] = Field(default=[], description="Fallback providers in order")
    model_selection: ModelSelectionStrategy = Field(
        default=ModelSelectionStrategy.MANUAL, description="How to select models"
    )

    # Ollama
    ollama_enabled: bool = Field(default=True, description="Enable Ollama provider")
    ollama_host: str = Field(default="http://localhost:11434", description="Ollama server URL")
    ollama_timeout: int = Field(default=120, description="Ollama request timeout", ge=10)
    ollama_models: list[str] = Field(
        default=["llama3.2", "mistral"], description="Preferred Ollama models"
    )
    ollama_auto_pull: bool = Field(default=True, description="Auto-pull missing models")

    # OpenAI
    openai_enabled: bool = Field(default=False, description="Enable OpenAI provider")
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")
    openai_base_url: str | None = Field(default=None, description="Custom OpenAI base URL")
    openai_organization: str | None = Field(default=None, description="OpenAI organization ID")
    openai_models: list[str] = Field(
        default=["gpt-4", "gpt-3.5-turbo"], description="Preferred OpenAI models"
    )
    openai_max_retries: int = Field(default=3, description="Max retry attempts", ge=0, le=10)

    # LM Studio
    lmstudio_enabled: bool = Field(default=False, description="Enable LM Studio provider")
    lmstudio_host: str = Field(default="http://localhost:1234", description="LM Studio server URL")
    lmstudio_models: list[str] = Field(default=[], description="Available LM Studio models")

    # Model parameters
    temperature: float = Field(default=0.7, description="Default temperature", ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, description="Max tokens per response")
    top_p: float = Field(default=1.0, description="Top-p sampling", ge=0.0, le=1.0)
    frequency_penalty: float = Field(default=0.0, description="Frequency penalty", ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, description="Presence penalty", ge=-2.0, le=2.0)

    # Cost management
    daily_cost_limit: float | None = Field(default=None, description="Daily spending limit in USD")
    warn_cost_threshold: float = Field(
        default=1.0, description="Warn when cost exceeds this per day"
    )


class MemoryConfig(BaseModel):
    """Memory system configuration."""

    # General memory settings
    enabled: bool = Field(default=True, description="Enable memory system")
    strategy: MemoryStrategy = Field(default=MemoryStrategy.BALANCED, description="Memory strategy")
    persist_dir: Path = Field(
        default=Path.home() / ".neuromancer" / "memory", description="Memory storage directory"
    )

    # Memory types
    short_term_enabled: bool = Field(default=True, description="Enable short-term memory")
    long_term_enabled: bool = Field(default=True, description="Enable long-term memory")
    episodic_enabled: bool = Field(default=True, description="Enable episodic memory")
    semantic_enabled: bool = Field(default=True, description="Enable semantic memory")

    # Memory limits
    max_memories: int = Field(default=100000, description="Maximum total memories", ge=100)
    max_memory_size_mb: int = Field(default=1024, description="Max memory storage in MB", ge=10)

    # Short-term memory
    short_term_duration_hours: int = Field(
        default=24, description="Short-term memory duration", ge=1
    )
    short_term_capacity: int = Field(default=1000, description="Short-term memory capacity", ge=10)

    # Long-term memory
    importance_threshold: float = Field(
        default=0.3, description="Minimum importance for long-term storage", ge=0.0, le=1.0
    )
    consolidation_interval_hours: int = Field(
        default=6, description="Memory consolidation interval", ge=1
    )

    # Vector search
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2", description="Sentence transformer model"
    )
    similarity_threshold: float = Field(
        default=0.7, description="Similarity threshold for retrieval", ge=0.0, le=1.0
    )
    search_limit: int = Field(default=20, description="Max search results", ge=1, le=100)

    # Advanced
    use_gpu: bool = Field(default=False, description="Use GPU for embeddings if available")
    batch_size: int = Field(default=32, description="Embedding batch size", ge=1, le=256)
    cache_embeddings: bool = Field(default=True, description="Cache computed embeddings")
    compression: bool = Field(default=False, description="Compress stored memories")


class MCPConfig(BaseModel):
    """MCP (Model Context Protocol) configuration."""

    enabled: bool = Field(default=True, description="Enable MCP integration")
    auto_connect: bool = Field(default=True, description="Auto-connect to configured servers")

    # Server configurations
    servers: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="MCP server configurations"
    )

    # Connection settings
    connection_timeout: int = Field(default=10, description="Connection timeout in seconds", ge=1)
    reconnect_attempts: int = Field(default=3, description="Reconnection attempts", ge=0)
    reconnect_delay: int = Field(default=5, description="Delay between reconnects", ge=1)

    # Tool execution
    tool_timeout: int = Field(default=30, description="Tool execution timeout", ge=5)
    parallel_tools: int = Field(default=5, description="Max parallel tool executions", ge=1, le=20)

    # Security
    allowed_tools: list[str] = Field(default=["*"], description="Allowed tool patterns")
    blocked_tools: list[str] = Field(default=[], description="Blocked tool patterns")
    require_confirmation: bool = Field(default=False, description="Require confirmation for tools")


class UIConfig(BaseModel):
    """User interface configuration."""

    # Theme
    theme_mode: ThemeMode = Field(default=ThemeMode.DARK, description="UI theme mode")
    primary_color: str = Field(default="DeepPurple", description="Primary color")
    accent_color: str = Field(default="Cyan", description="Accent color")
    custom_theme: dict[str, Any] = Field(default_factory=dict, description="Custom theme values")

    # Window
    window_width: int = Field(default=1200, description="Default window width", ge=800)
    window_height: int = Field(default=800, description="Default window height", ge=600)
    start_maximized: bool = Field(default=False, description="Start with maximized window")
    always_on_top: bool = Field(default=False, description="Keep window on top")

    # Chat interface
    font_size: int = Field(default=14, description="Base font size", ge=8, le=32)
    font_family: str = Field(default="Roboto", description="Font family")
    message_bubble_style: str = Field(default="modern", description="Message bubble style")
    show_timestamps: bool = Field(default=True, description="Show message timestamps")
    show_token_count: bool = Field(default=False, description="Show token usage")

    # Animations
    enable_animations: bool = Field(default=True, description="Enable UI animations")
    animation_speed: float = Field(
        default=1.0, description="Animation speed multiplier", ge=0.1, le=5.0
    )

    # Accessibility
    high_contrast: bool = Field(default=False, description="High contrast mode")
    screen_reader_support: bool = Field(default=True, description="Screen reader support")
    keyboard_navigation: bool = Field(default=True, description="Full keyboard navigation")

    # Advanced
    render_backend: str = Field(default="auto", description="Kivy render backend")
    fps_limit: int = Field(default=60, description="FPS limit", ge=30, le=144)


class PerformanceConfig(BaseModel):
    """Performance tuning configuration."""

    # Caching
    enable_cache: bool = Field(default=True, description="Enable response caching")
    cache_size_mb: int = Field(default=100, description="Cache size in MB", ge=10)
    cache_ttl_hours: int = Field(default=24, description="Cache TTL in hours", ge=1)

    # Resource limits
    max_concurrent_requests: int = Field(default=3, description="Max concurrent LLM requests", ge=1)
    request_queue_size: int = Field(default=10, description="Request queue size", ge=1)

    # Optimization
    lazy_load_models: bool = Field(default=True, description="Lazy load embedding models")
    preload_providers: bool = Field(default=False, description="Preload all providers on startup")

    # Monitoring
    enable_profiling: bool = Field(default=False, description="Enable performance profiling")
    collect_metrics: bool = Field(default=True, description="Collect performance metrics")


class PrivacyConfig(BaseModel):
    """Privacy and security configuration."""

    # Data retention
    store_conversations: bool = Field(default=True, description="Store conversation history")
    conversation_retention_days: int = Field(
        default=90, description="Days to retain conversations", ge=1
    )

    # Encryption
    encrypt_storage: bool = Field(default=True, description="Encrypt local storage")
    encryption_key: str | None = Field(default=None, description="Custom encryption key")

    # Privacy
    redact_api_keys: bool = Field(default=True, description="Redact API keys in logs")
    disable_telemetry: bool = Field(default=True, description="Disable all telemetry")
    local_only_mode: bool = Field(default=False, description="Disable all network features")

    # Security
    verify_ssl: bool = Field(default=True, description="Verify SSL certificates")
    allowed_hosts: list[str] = Field(default=["*"], description="Allowed hosts for connections")


class ExperimentalConfig(BaseModel):
    """Experimental features configuration."""

    # Features
    enable_experimental: bool = Field(default=False, description="Enable experimental features")

    # Advanced memory
    semantic_compression: bool = Field(default=False, description="Enable semantic compression")
    memory_clustering: bool = Field(default=False, description="Enable memory clustering")
    auto_forget: bool = Field(default=False, description="Auto-forget irrelevant memories")

    # Advanced MCP
    mcp_chaining: bool = Field(default=False, description="Enable MCP tool chaining")
    mcp_caching: bool = Field(default=False, description="Cache MCP tool results")

    # UI experiments
    voice_input: bool = Field(default=False, description="Enable voice input")
    voice_output: bool = Field(default=False, description="Enable voice output")
    ar_mode: bool = Field(default=False, description="Enable AR mode (requires camera)")

    # AI experiments
    multi_agent: bool = Field(default=False, description="Enable multi-agent conversations")
    self_reflection: bool = Field(default=False, description="Enable self-reflection loops")
    continuous_learning: bool = Field(default=False, description="Enable continuous learning")


class NeuromancerConfig(BaseModel):
    """Complete Neuromancer configuration."""

    # Allow extra fields for custom settings like system_prompt
    model_config = {"extra": "allow"}

    general: GeneralConfig = Field(default_factory=GeneralConfig)
    providers: ProviderConfig = Field(default_factory=ProviderConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    experimental: ExperimentalConfig = Field(default_factory=ExperimentalConfig)

    # Metadata
    version: int = Field(default=1, description="Configuration version")
    profile: str = Field(default="default", description="Configuration profile name")

    # Custom system prompt fields
    system_prompt: str = Field(default="", description="Custom system prompt")
    system_prompt_memory_integration: bool = Field(
        default=True, description="Include memory in system prompt"
    )

    @model_validator(mode="before")
    @classmethod
    def validate_config(cls, values):
        """Validate configuration consistency."""
        # Ensure memory persist dir exists
        memory = values.get("memory")
        if memory and hasattr(memory, "enabled") and memory.enabled:
            if hasattr(memory, "persist_dir"):
                memory.persist_dir.mkdir(parents=True, exist_ok=True)

        return values
