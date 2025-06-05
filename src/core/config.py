"""Configuration management with advanced features."""

from pathlib import Path
from typing import Any

from src import __version__
from src.config.manager import ConfigManager

# Initialize the advanced configuration manager
_config_manager = ConfigManager()


# For backward compatibility, expose a settings object
class Settings:
    """Legacy settings interface for backward compatibility."""

    def __init__(self, config_manager: ConfigManager):
        self._cm = config_manager
        self._config = config_manager.config

    # Legacy properties
    @property
    def APP_NAME(self) -> str:
        return self._config.general.app_name

    @property
    def APP_VERSION(self) -> str:
        return __version__

    @property
    def DEBUG(self) -> bool:
        return self._config.general.log_level == "DEBUG"

    @property
    def DATA_DIR(self) -> Path:
        return Path.home() / ".neuromancer"

    @property
    def LOG_DIR(self) -> Path:
        return Path.home() / ".neuromancer" / "logs"

    @property
    def DEFAULT_PROVIDER(self) -> str:
        return self._config.providers.default_provider

    @property
    def DEFAULT_MODEL(self) -> str:
        # Use last used model if available
        if self._config.providers.last_used_model:
            return self._config.providers.last_used_model
            
        # Otherwise get default model based on provider
        provider = self._config.providers.default_provider
        if provider == "ollama" and self._config.providers.ollama_models:
            return self._config.providers.ollama_models[0]
        elif provider == "openai" and self._config.providers.openai_models:
            return self._config.providers.openai_models[0]
        return "llama3.2"

    @property
    def OLLAMA_HOST(self) -> str:
        return self._config.providers.ollama_host

    @property
    def OPENAI_API_KEY(self) -> str | None:
        return self._config.providers.openai_api_key

    @property
    def OPENAI_BASE_URL(self) -> str | None:
        return self._config.providers.openai_base_url

    @property
    def LMSTUDIO_HOST(self) -> str:
        return self._config.providers.lmstudio_host

    @property
    def MEMORY_PERSIST_DIR(self) -> Path:
        return self._config.memory.persist_dir

    @property
    def MEMORY_SHORT_TERM_HOURS(self) -> int:
        return self._config.memory.short_term_duration_hours

    @property
    def MEMORY_IMPORTANCE_THRESHOLD(self) -> float:
        return self._config.memory.importance_threshold

    @property
    def MCP_SERVERS(self) -> dict[str, dict[str, Any]]:
        return self._config.mcp.servers

    @property
    def THEME_STYLE(self) -> str:
        return self._config.ui.theme_mode

    @property
    def PRIMARY_PALETTE(self) -> str:
        return self._config.ui.primary_color

    @property
    def ACCENT_PALETTE(self) -> str:
        return self._config.ui.accent_color

    @property
    def WINDOW_WIDTH(self) -> int:
        return self._config.ui.window_width

    @property
    def WINDOW_HEIGHT(self) -> int:
        return self._config.ui.window_height

    def get_provider_config(self, provider: str) -> dict[str, Any]:
        """Get configuration for a specific provider."""
        if provider == "ollama":
            return {
                "host": self._config.providers.ollama_host,
                "timeout": self._config.providers.ollama_timeout,
            }
        elif provider == "openai":
            return {
                "api_key": self._config.providers.openai_api_key,
                "base_url": self._config.providers.openai_base_url,
                "organization": self._config.providers.openai_organization,
                "max_retries": self._config.providers.openai_max_retries,
            }
        elif provider == "lmstudio":
            return {
                "host": self._config.providers.lmstudio_host,
            }
        return {}


# Global instances
config_manager = _config_manager
config = config_manager.config
settings = Settings(config_manager)

# Configuration will be initialized when needed
# Lazy initialization to avoid circular imports


# Convenience functions
def get_config(path: str, default: Any = None) -> Any:
    """Get configuration value by path."""
    return config_manager.get(path, default)


async def set_config(path: str, value: Any) -> bool:
    """Set configuration value by path."""
    return await config_manager.set(path, value)


def reload_config():
    """Reload configuration from disk."""
    import asyncio

    asyncio.create_task(config_manager.reload())
