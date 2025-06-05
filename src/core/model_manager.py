"""Advanced model management system with dynamic discovery and intelligent selection.

This module provides sophisticated model management capabilities for the Neuromancer
application, including dynamic model discovery, health monitoring, intelligent
selection, and performance tracking across multiple LLM providers.

Key Features:
    - Dynamic model discovery from all configured providers
    - Provider health monitoring and response time tracking
    - Intelligent model selection based on various strategies
    - Model capability inference and metadata extraction
    - Performance metrics and error tracking
    - Cost optimization for paid APIs

Architecture:
    The ModelManager acts as an abstraction layer over multiple LLM providers,
    providing a unified interface for model operations. It maintains a registry
    of available models with rich metadata and implements various selection
    strategies.

Design Patterns:
    - Registry Pattern: Central model registry with metadata
    - Strategy Pattern: Pluggable model selection strategies
    - Observer Pattern: Performance tracking and metrics
    - Cache Pattern: Discovery results cached to reduce API calls

Performance Considerations:
    - Parallel discovery across providers (~1-3s total)
    - Discovery results cached for 1 minute
    - Lazy initialization of provider connections
    - Minimal overhead for model selection (<1ms)

Example Usage:
    ```python
    model_manager = ModelManager(chat_manager)

    # Discover all available models
    await model_manager.discover_models()

    # Get models for a specific provider
    ollama_models = model_manager.get_available_models('ollama')

    # Select best model automatically
    best_model = await model_manager.select_best_model(
        strategy=ModelSelectionStrategy.BALANCED,
        requirements={'capabilities': ['code'], 'min_context_length': 8192}
    )

    # Get model performance stats
    stats = model_manager.get_model_performance('gpt-4')
    ```
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from src.providers import LLMProvider
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ModelStatus(str, Enum):
    """Model availability status.

    Represents the current state of a model within a provider.

    Values:
        AVAILABLE: Model is ready for use
        UNAVAILABLE: Model exists but cannot be used (e.g., not downloaded)
        DOWNLOADING: Model is being downloaded/prepared
        ERROR: Model encountered an error
        UNKNOWN: Status cannot be determined
    """

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    DOWNLOADING = "downloading"
    ERROR = "error"
    UNKNOWN = "unknown"


class ProviderStatus(str, Enum):
    """Provider health status.

    Represents the operational state of an LLM provider.

    Values:
        HEALTHY: Provider is fully operational
        DEGRADED: Provider is operational but with issues
        UNAVAILABLE: Provider cannot be reached
        ERROR: Provider encountered a critical error
        UNKNOWN: Status has not been determined
    """

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class ModelInfo:
    """Comprehensive model information.

    Rich metadata container for language models, including capabilities,
    performance characteristics, and availability status.

    Attributes:
        name: Model identifier within the provider (e.g., 'llama2')
        provider: Provider name (e.g., 'ollama', 'openai')
        display_name: Human-readable name for UI display
        description: Detailed model description
        status: Current availability status
        size_bytes: Model size in bytes (for local models)
        parameters: Parameter count string (e.g., '7B', '13B')
        capabilities: List of model capabilities (e.g., ['chat', 'code'])
        cost_per_token: Cost per token in USD (for paid APIs)
        context_length: Maximum context window size
        last_updated: Timestamp of last status update
        metadata: Additional provider-specific metadata

    Properties:
        full_name: Provider-qualified name (e.g., 'ollama:llama2')
        is_available: Quick check for availability
        size_human: Human-readable size string

    Usage:
        ModelInfo objects are created during discovery and cached
        in the model registry for quick access.
    """

    name: str
    provider: str
    display_name: str = ""
    description: str = ""
    status: ModelStatus = ModelStatus.UNKNOWN
    size_bytes: int | None = None
    parameters: str | None = None  # e.g., "7B", "13B"
    capabilities: list[str] = field(default_factory=list)  # e.g., ["chat", "code", "reasoning"]
    cost_per_token: float | None = None  # For paid APIs
    context_length: int | None = None
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def full_name(self) -> str:
        """Get full model name with provider prefix."""
        return f"{self.provider}:{self.name}"

    @property
    def is_available(self) -> bool:
        """Check if model is currently available."""
        return self.status == ModelStatus.AVAILABLE

    @property
    def size_human(self) -> str:
        """Human-readable size."""
        if not self.size_bytes:
            return "Unknown"

        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if self.size_bytes < 1024.0:
                return f"{self.size_bytes:.1f} {unit}"
            self.size_bytes /= 1024.0
        return f"{self.size_bytes:.1f} PB"


@dataclass
class ProviderInfo:
    """Provider health and status information.

    Tracks the operational status and performance metrics of an LLM provider.

    Attributes:
        name: Provider identifier (e.g., 'ollama')
        status: Current operational status
        last_health_check: Timestamp of last health check
        response_time_ms: Average API response time in milliseconds
        available_models: List of model names available from provider
        error_message: Error details if status is ERROR
        capabilities: Provider-specific capabilities
        metadata: Additional provider-specific information

    Properties:
        is_healthy: Quick check for provider health

    Usage:
        Updated during discovery and health checks to track
        provider availability and performance.
    """

    name: str
    status: ProviderStatus = ProviderStatus.UNKNOWN
    last_health_check: datetime | None = None
    response_time_ms: float | None = None
    available_models: list[str] = field(default_factory=list)
    error_message: str | None = None
    capabilities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_healthy(self) -> bool:
        """Check if provider is healthy."""
        return self.status == ProviderStatus.HEALTHY


class ModelSelectionStrategy(str, Enum):
    """Model selection strategies.

    Different strategies for automatically selecting the best model
    based on various criteria.

    Values:
        MANUAL: Use user-defined preference list
        FASTEST: Select from provider with best response time
        CHEAPEST: Minimize cost (free models preferred)
        BEST_QUALITY: Maximize model quality/parameters
        BALANCED: Balance quality, speed, and cost
        MOST_CAPABLE: Maximum features/capabilities

    Strategy Details:
        - MANUAL: Checks preferred_models list in order
        - FASTEST: Uses provider response time metrics
        - CHEAPEST: Free > lowest cost per token
        - BEST_QUALITY: Larger parameter counts preferred
        - BALANCED: Weighted scoring across factors
        - MOST_CAPABLE: Most capability tags
    """

    MANUAL = "manual"
    FASTEST = "fastest"
    CHEAPEST = "cheapest"
    BEST_QUALITY = "best_quality"
    BALANCED = "balanced"
    MOST_CAPABLE = "most_capable"


class ModelManager:
    """Advanced model management with dynamic discovery and intelligent selection.

    Central manager for all model-related operations across providers.
    Maintains a unified registry of models and implements intelligent
    selection strategies.

    Attributes:
        chat_manager: Reference to ChatManager for provider access
        _models: Registry of discovered models by full name
        _providers: Registry of provider health information
        _provider_instances: Provider instance references
        selection_strategy: Default selection strategy
        preferred_models: Ordered list of preferred model names
        model_blacklist: Models to exclude from selection
        _response_times: Response time tracking by model
        _error_counts: Error count tracking by model
        health_check_interval: Time between health checks
        last_discovery: Timestamp of last discovery

    Thread Safety:
        Not thread-safe. Use separate instances for concurrent access.

    Lifecycle:
        1. Initialize with ChatManager reference
        2. Call discover_models() to populate registry
        3. Use get/select methods to work with models
        4. Track performance during usage
    """

    def __init__(self, chat_manager=None):
        """Initialize model manager.

        Args:
            chat_manager: Reference to chat manager for provider access.
                Can be set later via set_chat_manager() if not available
                during initialization.

        Initialization:
            Sets up empty registries and default configuration.
            No external calls are made during init.
        """
        self.chat_manager = chat_manager
        self._models: dict[str, ModelInfo] = {}
        self._providers: dict[str, ProviderInfo] = {}
        self._provider_instances: dict[str, LLMProvider] = {}

        # Selection preferences
        self.selection_strategy = ModelSelectionStrategy.MANUAL
        self.preferred_models: list[str] = []
        self.model_blacklist: list[str] = []

        # Performance tracking
        self._response_times: dict[str, list[float]] = {}
        self._error_counts: dict[str, int] = {}

        # Health check settings
        self.health_check_interval = timedelta(minutes=5)
        self.last_discovery = None

        logger.info("ModelManager initialized")

    def set_chat_manager(self, chat_manager):
        """Set chat manager reference after initialization.

        Args:
            chat_manager: ChatManager instance with provider references

        Usage:
            Called by ChatManager after bidirectional references are
            established. Extracts provider instances for model operations.
        """
        self.chat_manager = chat_manager
        if hasattr(chat_manager, 'providers'):
            self._provider_instances = chat_manager.providers

    async def discover_models(self, force_refresh: bool = False) -> bool:
        """Discover all available models across providers.

        Queries all configured providers in parallel to build a comprehensive
        model registry with metadata and capabilities.

        Args:
            force_refresh: Force rediscovery even if recently done.
                By default, skips if discovery was done within 1 minute.

        Returns:
            True if at least one provider returned models successfully

        Process:
            1. Check if discovery needed (cache timeout or forced)
            2. Clear existing registry
            3. Query each provider in parallel
            4. Build ModelInfo for each discovered model
            5. Extract metadata and infer capabilities
            6. Update provider health status

        Performance:
            - First discovery: 1-3 seconds (depends on providers)
            - Cached discovery: <1ms (skipped)
            - Parallel execution minimizes total time

        Error Handling:
            Provider failures are logged but don't stop discovery.
            At least one successful provider is required to return True.
        """
        if (not force_refresh and self.last_discovery and
            datetime.now() - self.last_discovery < timedelta(minutes=1)):
            logger.debug("Skipping model discovery - too recent")
            return True

        logger.info("Starting model discovery across all providers")

        try:
            # Clear existing data
            self._models.clear()
            self._providers.clear()

            # Discover from each provider
            discovery_tasks = []
            for provider_name, provider in self._provider_instances.items():
                task = self._discover_provider_models(provider_name, provider)
                discovery_tasks.append(task)

            # Run discovery in parallel
            results = await asyncio.gather(*discovery_tasks, return_exceptions=True)

            # Process results
            successful_providers = 0
            for i, result in enumerate(results):
                provider_name = list(self._provider_instances.keys())[i]
                if isinstance(result, Exception):
                    logger.error(f"Failed to discover models from {provider_name}: {result}")
                    self._providers[provider_name] = ProviderInfo(
                        name=provider_name,
                        status=ProviderStatus.ERROR,
                        error_message=str(result),
                        last_health_check=datetime.now()
                    )
                else:
                    successful_providers += 1

            self.last_discovery = datetime.now()

            logger.info(f"Model discovery completed. Found {len(self._models)} models "
                       f"from {successful_providers}/{len(self._provider_instances)} providers")

            return successful_providers > 0

        except Exception as e:
            logger.error(f"Model discovery failed: {e}")
            return False

    async def _discover_provider_models(self, provider_name: str, provider: LLMProvider):
        """Discover models from a specific provider.

        Queries a single provider for available models and builds metadata.

        Args:
            provider_name: Name of the provider (e.g., 'ollama')
            provider: LLMProvider instance to query

        Process:
            1. Health check the provider
            2. List available models if healthy
            3. Create ModelInfo for each model
            4. Infer capabilities from model names
            5. Add provider-specific metadata
            6. Store in model registry

        Model Naming:
            Models are stored with full names (provider:model) to handle
            cases where multiple providers offer the same model.

        Error Recovery:
            Provider errors result in ERROR status but don't raise.
            This allows partial discovery when some providers fail.
        """
        start_time = datetime.now()

        try:
            # Check provider health
            is_healthy = await provider.health_check()

            # Get available models
            model_names = []
            if is_healthy:
                model_names = await provider.list_models()

            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds() * 1000

            # Create provider info
            provider_info = ProviderInfo(
                name=provider_name,
                status=ProviderStatus.HEALTHY if is_healthy else ProviderStatus.UNAVAILABLE,
                last_health_check=datetime.now(),
                response_time_ms=response_time,
                available_models=model_names,
                capabilities=self._get_provider_capabilities(provider_name)
            )

            self._providers[provider_name] = provider_info

            # Create model info for each model
            for model_name in model_names:
                model_info = ModelInfo(
                    name=model_name,
                    provider=provider_name,
                    display_name=self._generate_display_name(model_name),
                    description=self._generate_description(provider_name, model_name),
                    status=ModelStatus.AVAILABLE,
                    capabilities=self._infer_model_capabilities(model_name),
                    context_length=self._estimate_context_length(model_name),
                    parameters=self._extract_parameters(model_name),
                    metadata={
                        "discovered_at": datetime.now().isoformat(),
                        "provider_response_time_ms": response_time
                    }
                )

                # Add cost information for paid APIs
                if provider_name == "openai":
                    model_info.cost_per_token = self._get_openai_cost(model_name)

                self._models[model_info.full_name] = model_info
                logger.debug(f"Stored model: {model_info.full_name} -> {model_info.name}")

            logger.info(f"Discovered {len(model_names)} models from {provider_name} "
                       f"(response time: {response_time:.1f}ms)")
            logger.debug(f"Total models in registry: {len(self._models)}")
            logger.debug(f"Model keys: {list(self._models.keys())}")

        except Exception as e:
            logger.error(f"Failed to discover models from {provider_name}: {e}")
            self._providers[provider_name] = ProviderInfo(
                name=provider_name,
                status=ProviderStatus.ERROR,
                error_message=str(e),
                last_health_check=datetime.now()
            )
            raise

    def debug_model_registry(self):
        """Debug helper to show what models are in the registry.

        Logs the complete model registry for troubleshooting.
        Useful when models aren't being found as expected.

        Output Format:
            Total models: X
            provider:model -> model_name (status: STATUS)
        """
        logger.info("=== MODEL REGISTRY DEBUG ===")
        logger.info(f"Total models: {len(self._models)}")
        for key, model in self._models.items():
            logger.info(f"  {key} -> {model.name} (status: {model.status})")
        logger.info("=== END DEBUG ===")

    def get_available_models(self, provider: str | None = None) -> list[ModelInfo]:
        """Get list of available models.

        Args:
            provider: Filter by provider name. If None, returns models
                from all providers.

        Returns:
            List of ModelInfo objects for available models,
            sorted by provider then name.

        Filtering:
            Only returns models with AVAILABLE status.
            Use the model registry directly for all models including
            unavailable ones.
        """
        models = []
        for model_info in self._models.values():
            if provider and model_info.provider != provider:
                continue
            if model_info.is_available:
                models.append(model_info)

        # Sort by provider, then by name
        models.sort(key=lambda m: (m.provider, m.name))
        return models

    def get_model_info(self, model_name: str, provider: str | None = None) -> ModelInfo | None:
        """Get information about a specific model.

        Flexible model lookup supporting various naming formats.

        Args:
            model_name: Name of the model. Can be:
                - Simple name: 'llama2'
                - Full name: 'ollama:llama2'
                - With tag: 'llama2:latest'
            provider: Provider name if not included in model_name.
                Helps disambiguate when multiple providers have same model.

        Returns:
            ModelInfo object or None if not found

        Search Order:
            1. Exact match with provider:model_name (if provider given)
            2. Exact match with model_name (if contains ':')
            3. Search all providers for model_name

        Example:
            ```python
            # All of these could work:
            model_manager.get_model_info('llama2', 'ollama')
            model_manager.get_model_info('ollama:llama2')
            model_manager.get_model_info('llama2')  # finds first match
            ```
        """
        logger.info(f"Searching for model: {model_name} (provider: {provider})")
        logger.info(f"Available models: {list(self._models.keys())}")

        # Handle explicit provider first (this handles model names that might have colons)
        if provider:
            full_name = f"{provider}:{model_name}"
            result = self._models.get(full_name)
            logger.info(f"Provider:model search for '{full_name}': {'Found' if result else 'Not found'}")
            if result:
                return result

        # Handle full model names (provider:model) - only if no provider was specified
        if ":" in model_name and not provider:
            result = self._models.get(model_name)
            logger.info(f"Full name search for '{model_name}': {'Found' if result else 'Not found'}")
            if result:
                return result

        # Search across all providers for the model name
        for model_info in self._models.values():
            if model_info.name == model_name:
                logger.info(f"Found model '{model_name}' as '{model_info.full_name}'")
                return model_info

        logger.info(f"Model '{model_name}' not found in any provider")
        return None

    def get_provider_info(self, provider: str) -> ProviderInfo | None:
        """Get information about a provider.

        Args:
            provider: Provider name (e.g., 'ollama')

        Returns:
            ProviderInfo with health status or None if not found
        """
        return self._providers.get(provider)

    def get_healthy_providers(self) -> list[ProviderInfo]:
        """Get list of healthy providers.

        Returns:
            List of ProviderInfo for providers with HEALTHY status

        Usage:
            Useful for failover scenarios or selecting optimal provider.
        """
        return [p for p in self._providers.values() if p.is_healthy]

    async def select_best_model(
        self,
        strategy: ModelSelectionStrategy | None = None,
        requirements: dict[str, Any] | None = None
    ) -> ModelInfo | None:
        """Select the best model based on strategy and requirements.

        Intelligent model selection using various strategies and filters.

        Args:
            strategy: Selection strategy to use. If None, uses the
                instance's default strategy (MANUAL by default).
            requirements: Model requirements dictionary. Supported keys:
                - capabilities: List of required capabilities
                - min_context_length: Minimum context window size
                - provider: Specific provider requirement
                - max_cost_per_token: Maximum acceptable cost

        Returns:
            Best ModelInfo matching requirements or None if no match

        Selection Process:
            1. Get all available models
            2. Filter by requirements
            3. Apply selection strategy
            4. Return best match

        Strategies Explained:
            - MANUAL: Uses preferred_models list order
            - FASTEST: Minimizes provider response time
            - CHEAPEST: Free models first, then lowest cost
            - BEST_QUALITY: Maximizes parameter count
            - BALANCED: Weighted scoring across factors
            - MOST_CAPABLE: Maximum capability count

        Example:
            ```python
            model = await model_manager.select_best_model(
                strategy=ModelSelectionStrategy.BALANCED,
                requirements={
                    'capabilities': ['code', 'chat'],
                    'min_context_length': 4096,
                    'max_cost_per_token': 0.01
                }
            )
            ```
        """
        strategy = strategy or self.selection_strategy
        available_models = self.get_available_models()

        if not available_models:
            logger.warning("No available models for selection")
            return None

        # Filter by requirements
        if requirements:
            available_models = self._filter_by_requirements(available_models, requirements)

        if not available_models:
            logger.warning("No models match the requirements")
            return None

        # Apply selection strategy
        if strategy == ModelSelectionStrategy.MANUAL:
            # Return first preferred model that's available
            for preferred in self.preferred_models:
                for model in available_models:
                    if model.full_name == preferred or model.name == preferred:
                        return model
            # Fallback to first available
            return available_models[0]

        elif strategy == ModelSelectionStrategy.FASTEST:
            # Select model from provider with best response time
            provider_times = {p.name: p.response_time_ms or float('inf')
                            for p in self._providers.values()}
            available_models.sort(key=lambda m: provider_times.get(m.provider, float('inf')))
            return available_models[0]

        elif strategy == ModelSelectionStrategy.CHEAPEST:
            # Select model with lowest cost
            paid_models = [m for m in available_models if m.cost_per_token is not None]
            free_models = [m for m in available_models if m.cost_per_token is None]

            if free_models:
                return free_models[0]  # Free models are cheapest
            elif paid_models:
                paid_models.sort(key=lambda m: m.cost_per_token)
                return paid_models[0]

        elif strategy == ModelSelectionStrategy.BEST_QUALITY:
            # Prefer models with more parameters or specific high-quality models
            quality_score = {}
            for model in available_models:
                score = 0

                # Parameter-based scoring
                if model.parameters:
                    if "70B" in model.parameters or "65B" in model.parameters:
                        score += 100
                    elif "30B" in model.parameters or "34B" in model.parameters:
                        score += 80
                    elif "13B" in model.parameters or "14B" in model.parameters:
                        score += 60
                    elif "7B" in model.parameters or "8B" in model.parameters:
                        score += 40
                    elif "3B" in model.parameters:
                        score += 20

                # Known high-quality models
                if any(name in model.name.lower() for name in ["gpt-4", "claude", "llama-3"]):
                    score += 50

                quality_score[model.full_name] = score

            available_models.sort(key=lambda m: quality_score.get(m.full_name, 0), reverse=True)
            return available_models[0]

        elif strategy == ModelSelectionStrategy.BALANCED:
            # Balance between quality, speed, and cost
            scored_models = []

            for model in available_models:
                score = 50  # Base score

                # Quality bonus
                if "gpt-4" in model.name.lower():
                    score += 30
                elif "llama" in model.name.lower():
                    score += 20
                elif "mistral" in model.name.lower():
                    score += 15

                # Speed bonus (lower response time is better)
                provider_info = self._providers.get(model.provider)
                if provider_info and provider_info.response_time_ms:
                    if provider_info.response_time_ms < 1000:  # < 1 second
                        score += 20
                    elif provider_info.response_time_ms < 3000:  # < 3 seconds
                        score += 10

                # Cost penalty for paid models
                if model.cost_per_token:
                    score -= 15

                scored_models.append((model, score))

            scored_models.sort(key=lambda x: x[1], reverse=True)
            return scored_models[0][0]

        elif strategy == ModelSelectionStrategy.MOST_CAPABLE:
            # Select model with most capabilities
            available_models.sort(key=lambda m: len(m.capabilities), reverse=True)
            return available_models[0]

        # Fallback
        return available_models[0]

    def _filter_by_requirements(self, models: list[ModelInfo], requirements: dict[str, Any]) -> list[ModelInfo]:
        """Filter models by requirements.

        Applies requirement filters to model list.

        Args:
            models: List of models to filter
            requirements: Requirements dictionary

        Returns:
            Filtered list of models meeting all requirements

        Supported Filters:
            - capabilities: Model must have all listed capabilities
            - min_context_length: Model context >= specified value
            - provider: Model must be from specified provider
            - max_cost_per_token: Model cost <= specified value
        """
        filtered = []

        for model in models:
            # Check required capabilities
            if "capabilities" in requirements:
                required_caps = requirements["capabilities"]
                if not all(cap in model.capabilities for cap in required_caps):
                    continue

            # Check context length
            if "min_context_length" in requirements:
                min_context = requirements["min_context_length"]
                if not model.context_length or model.context_length < min_context:
                    continue

            # Check provider
            if "provider" in requirements:
                if model.provider != requirements["provider"]:
                    continue

            # Check cost
            if "max_cost_per_token" in requirements:
                max_cost = requirements["max_cost_per_token"]
                if model.cost_per_token and model.cost_per_token > max_cost:
                    continue

            filtered.append(model)

        return filtered

    def track_model_performance(self, model_name: str, response_time_ms: float, success: bool):
        """Track model performance metrics.

        Records performance data for models to enable performance-based selection.

        Args:
            model_name: Full model name (provider:model)
            response_time_ms: Response time in milliseconds (if successful)
            success: Whether the request succeeded

        Tracking:
            - Success: Adds to response time history (last 100 kept)
            - Failure: Increments error counter

        Usage:
            Called by ChatManager after each model interaction to build
            performance profile over time.
        """
        if success:
            if model_name not in self._response_times:
                self._response_times[model_name] = []

            self._response_times[model_name].append(response_time_ms)

            # Keep only last 100 measurements
            if len(self._response_times[model_name]) > 100:
                self._response_times[model_name] = self._response_times[model_name][-100:]
        else:
            self._error_counts[model_name] = self._error_counts.get(model_name, 0) + 1

    def get_model_performance(self, model_name: str) -> dict[str, Any]:
        """Get performance statistics for a model.

        Args:
            model_name: Full model name (provider:model)

        Returns:
            Dictionary with performance metrics:
            - error_count: Number of failed requests
            - total_requests: Total requests made
            - success_rate: Percentage of successful requests
            - avg_response_time_ms: Average response time
            - min_response_time_ms: Fastest response
            - max_response_time_ms: Slowest response

        Note:
            Returns zeros for models with no recorded data.
        """
        response_times = self._response_times.get(model_name, [])
        error_count = self._error_counts.get(model_name, 0)

        stats = {
            "error_count": error_count,
            "total_requests": len(response_times) + error_count,
            "success_rate": 0.0,
            "avg_response_time_ms": 0.0,
            "min_response_time_ms": 0.0,
            "max_response_time_ms": 0.0
        }

        total_requests = stats["total_requests"]
        if total_requests > 0:
            stats["success_rate"] = len(response_times) / total_requests

        if response_times:
            stats["avg_response_time_ms"] = sum(response_times) / len(response_times)
            stats["min_response_time_ms"] = min(response_times)
            stats["max_response_time_ms"] = max(response_times)

        return stats

    # Helper methods for model metadata inference
    # These methods analyze model names and provider characteristics
    # to infer capabilities and metadata when not explicitly provided.

    def _get_provider_capabilities(self, provider: str) -> list[str]:
        """Get capabilities for a provider.

        Returns known capabilities for each provider type.

        Args:
            provider: Provider name

        Returns:
            List of capability tags for the provider

        Known Capabilities:
            - chat: Basic conversation support
            - local: Runs on local hardware
            - offline: Works without internet
            - free: No API costs
            - function_calling: OpenAI function calling
            - vision: Image understanding
            - paid: Requires API payment
        """
        capabilities = {
            "ollama": ["chat", "local", "offline", "free"],
            "openai": ["chat", "function_calling", "vision", "paid"],
            "lmstudio": ["chat", "local", "offline", "free"]
        }
        return capabilities.get(provider, ["chat"])

    def _generate_display_name(self, model_name: str) -> str:
        """Generate human-readable display name.

        Cleans up model names for UI display.

        Args:
            model_name: Raw model name from provider

        Returns:
            Cleaned display name

        Transformations:
            - Replace underscores/hyphens with spaces
            - Capitalize known acronyms (GPT, LLM, AI)
            - Title case known model families
        """
        # Clean up common model name patterns
        display = model_name.replace("_", " ").replace("-", " ")

        # Capitalize words
        words = []
        for word in display.split():
            if word.upper() in ["GPT", "LLM", "AI"]:
                words.append(word.upper())
            elif word.lower() in ["llama", "mistral", "codellama"]:
                words.append(word.capitalize())
            else:
                words.append(word)

        return " ".join(words)

    def _generate_description(self, provider: str, model_name: str) -> str:
        """Generate model description.

        Creates descriptive text for models based on known characteristics.

        Args:
            provider: Provider name
            model_name: Model name

        Returns:
            Human-readable description

        Description Sources:
            1. Known model descriptions (hard-coded)
            2. Generic description with provider name
        """
        descriptions = {
            "gpt-4": "Advanced large language model with superior reasoning capabilities",
            "gpt-3.5": "Fast and efficient language model for general tasks",
            "llama": "Open-source large language model",
            "mistral": "High-performance open-source model",
            "codellama": "Specialized model for code generation and analysis"
        }

        model_lower = model_name.lower()
        for key, desc in descriptions.items():
            if key in model_lower:
                return f"{desc} via {provider.capitalize()}"

        return f"Language model provided by {provider.capitalize()}"

    def _infer_model_capabilities(self, model_name: str) -> list[str]:
        """Infer model capabilities from name.

        Analyzes model names to determine likely capabilities.

        Args:
            model_name: Model name to analyze

        Returns:
            List of inferred capability tags

        Inference Rules:
            - 'code' in name → code, programming capabilities
            - 'instruct' in name → instruction_following
            - 'chat' in name → conversational
            - 'vision'/'multimodal' in name → vision
            - Large models (70B+) → reasoning, complex_tasks

        Note:
            This is a heuristic approach. Actual capabilities may differ.
        """
        capabilities = ["chat"]

        model_lower = model_name.lower()

        if "code" in model_lower:
            capabilities.extend(["code", "programming"])

        if "instruct" in model_lower:
            capabilities.append("instruction_following")

        if "chat" in model_lower:
            capabilities.append("conversational")

        if "vision" in model_lower or "multimodal" in model_lower:
            capabilities.append("vision")

        if any(size in model_lower for size in ["70b", "65b"]):
            capabilities.extend(["reasoning", "complex_tasks"])

        return capabilities

    def _estimate_context_length(self, model_name: str) -> int | None:
        """Estimate context length from model name.

        Infers context window size from model naming patterns.

        Args:
            model_name: Model name to analyze

        Returns:
            Estimated context length in tokens or None

        Patterns:
            - Explicit: '32k', '16k', '8k', '4k' in name
            - Known models: GPT-4 (8192), GPT-3.5 (4096), Llama (4096)
            - Default: None (unknown)
        """
        model_lower = model_name.lower()

        # Common context lengths
        if "32k" in model_lower:
            return 32768
        elif "16k" in model_lower:
            return 16384
        elif "8k" in model_lower:
            return 8192
        elif "4k" in model_lower:
            return 4096
        elif "gpt-4" in model_lower:
            return 8192  # Default for GPT-4
        elif "gpt-3.5" in model_lower:
            return 4096  # Default for GPT-3.5
        elif "llama" in model_lower:
            return 4096  # Default for Llama models

        return None

    def _extract_parameters(self, model_name: str) -> str | None:
        """Extract parameter count from model name.

        Uses regex to find parameter specifications in model names.

        Args:
            model_name: Model name to analyze

        Returns:
            Parameter string (e.g., '7B', '13B') or None

        Pattern:
            Matches patterns like: 7B, 13B, 70B, 1.5B, etc.
        """
        import re

        # Look for parameter patterns like 7B, 13B, etc.
        match = re.search(r'(\d+\.?\d*)[Bb]', model_name)
        if match:
            return f"{match.group(1)}B"

        return None

    def _get_openai_cost(self, model_name: str) -> float | None:
        """Get cost per token for OpenAI models.

        Returns estimated cost per token for known OpenAI models.

        Args:
            model_name: OpenAI model name

        Returns:
            Cost per token in USD or None if unknown

        Costs:
            Based on rough estimates for input tokens:
            - GPT-4: $0.03/1K tokens
            - GPT-4-turbo: $0.01/1K tokens
            - GPT-3.5-turbo: $0.001/1K tokens

        Note:
            These are estimates. Check OpenAI pricing for current rates.
        """
        # Rough estimates for input tokens (USD per 1K tokens)
        costs = {
            "gpt-4": 0.03,
            "gpt-4-turbo": 0.01,
            "gpt-3.5-turbo": 0.001,
        }

        model_lower = model_name.lower()
        for key, cost in costs.items():
            if key in model_lower:
                return cost / 1000  # Convert to per-token cost

        return None
