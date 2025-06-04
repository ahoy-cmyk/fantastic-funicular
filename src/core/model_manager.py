"""Advanced model management system with dynamic discovery and intelligent selection."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from src.core.config import config_manager, settings
from src.providers import LLMProvider
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ModelStatus(str, Enum):
    """Model availability status."""
    
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    DOWNLOADING = "downloading"
    ERROR = "error"
    UNKNOWN = "unknown"


class ProviderStatus(str, Enum):
    """Provider health status."""
    
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class ModelInfo:
    """Comprehensive model information."""
    
    name: str
    provider: str
    display_name: str = ""
    description: str = ""
    status: ModelStatus = ModelStatus.UNKNOWN
    size_bytes: Optional[int] = None
    parameters: Optional[str] = None  # e.g., "7B", "13B"
    capabilities: List[str] = field(default_factory=list)  # e.g., ["chat", "code", "reasoning"]
    cost_per_token: Optional[float] = None  # For paid APIs
    context_length: Optional[int] = None
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
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
    """Provider health and status information."""
    
    name: str
    status: ProviderStatus = ProviderStatus.UNKNOWN
    last_health_check: Optional[datetime] = None
    response_time_ms: Optional[float] = None
    available_models: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_healthy(self) -> bool:
        """Check if provider is healthy."""
        return self.status == ProviderStatus.HEALTHY


class ModelSelectionStrategy(str, Enum):
    """Model selection strategies."""
    
    MANUAL = "manual"
    FASTEST = "fastest"
    CHEAPEST = "cheapest"
    BEST_QUALITY = "best_quality"
    BALANCED = "balanced"
    MOST_CAPABLE = "most_capable"


class ModelManager:
    """Advanced model management with dynamic discovery and intelligent selection."""
    
    def __init__(self, chat_manager=None):
        """Initialize model manager.
        
        Args:
            chat_manager: Reference to chat manager for provider access
        """
        self.chat_manager = chat_manager
        self._models: Dict[str, ModelInfo] = {}
        self._providers: Dict[str, ProviderInfo] = {}
        self._provider_instances: Dict[str, LLMProvider] = {}
        
        # Selection preferences
        self.selection_strategy = ModelSelectionStrategy.MANUAL
        self.preferred_models: List[str] = []
        self.model_blacklist: List[str] = []
        
        # Performance tracking
        self._response_times: Dict[str, List[float]] = {}
        self._error_counts: Dict[str, int] = {}
        
        # Health check settings
        self.health_check_interval = timedelta(minutes=5)
        self.last_discovery = None
        
        logger.info("ModelManager initialized")
    
    def set_chat_manager(self, chat_manager):
        """Set chat manager reference after initialization."""
        self.chat_manager = chat_manager
        if hasattr(chat_manager, 'providers'):
            self._provider_instances = chat_manager.providers
    
    async def discover_models(self, force_refresh: bool = False) -> bool:
        """Discover all available models across providers.
        
        Args:
            force_refresh: Force rediscovery even if recently done
            
        Returns:
            True if discovery completed successfully
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
        """Discover models from a specific provider."""
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
        """Debug helper to show what models are in the registry."""
        logger.info(f"=== MODEL REGISTRY DEBUG ===")
        logger.info(f"Total models: {len(self._models)}")
        for key, model in self._models.items():
            logger.info(f"  {key} -> {model.name} (status: {model.status})")
        logger.info(f"=== END DEBUG ===")

    def get_available_models(self, provider: Optional[str] = None) -> List[ModelInfo]:
        """Get list of available models.
        
        Args:
            provider: Filter by provider (None = all providers)
            
        Returns:
            List of available model information
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
    
    def get_model_info(self, model_name: str, provider: Optional[str] = None) -> Optional[ModelInfo]:
        """Get information about a specific model.
        
        Args:
            model_name: Name of the model
            provider: Provider name (if not included in model_name)
            
        Returns:
            Model information or None if not found
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
    
    def get_provider_info(self, provider: str) -> Optional[ProviderInfo]:
        """Get information about a provider."""
        return self._providers.get(provider)
    
    def get_healthy_providers(self) -> List[ProviderInfo]:
        """Get list of healthy providers."""
        return [p for p in self._providers.values() if p.is_healthy]
    
    async def select_best_model(
        self, 
        strategy: Optional[ModelSelectionStrategy] = None,
        requirements: Optional[Dict[str, Any]] = None
    ) -> Optional[ModelInfo]:
        """Select the best model based on strategy and requirements.
        
        Args:
            strategy: Selection strategy to use
            requirements: Model requirements (capabilities, context_length, etc.)
            
        Returns:
            Best model or None if no suitable model found
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
    
    def _filter_by_requirements(self, models: List[ModelInfo], requirements: Dict[str, Any]) -> List[ModelInfo]:
        """Filter models by requirements."""
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
        """Track model performance metrics."""
        if success:
            if model_name not in self._response_times:
                self._response_times[model_name] = []
            
            self._response_times[model_name].append(response_time_ms)
            
            # Keep only last 100 measurements
            if len(self._response_times[model_name]) > 100:
                self._response_times[model_name] = self._response_times[model_name][-100:]
        else:
            self._error_counts[model_name] = self._error_counts.get(model_name, 0) + 1
    
    def get_model_performance(self, model_name: str) -> Dict[str, Any]:
        """Get performance statistics for a model."""
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
    
    def _get_provider_capabilities(self, provider: str) -> List[str]:
        """Get capabilities for a provider."""
        capabilities = {
            "ollama": ["chat", "local", "offline", "free"],
            "openai": ["chat", "function_calling", "vision", "paid"],
            "lmstudio": ["chat", "local", "offline", "free"]
        }
        return capabilities.get(provider, ["chat"])
    
    def _generate_display_name(self, model_name: str) -> str:
        """Generate human-readable display name."""
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
        """Generate model description."""
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
    
    def _infer_model_capabilities(self, model_name: str) -> List[str]:
        """Infer model capabilities from name."""
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
    
    def _estimate_context_length(self, model_name: str) -> Optional[int]:
        """Estimate context length from model name."""
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
    
    def _extract_parameters(self, model_name: str) -> Optional[str]:
        """Extract parameter count from model name."""
        import re
        
        # Look for parameter patterns like 7B, 13B, etc.
        match = re.search(r'(\d+\.?\d*)[Bb]', model_name)
        if match:
            return f"{match.group(1)}B"
        
        return None
    
    def _get_openai_cost(self, model_name: str) -> Optional[float]:
        """Get cost per token for OpenAI models."""
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