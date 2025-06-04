"""Advanced configuration management system."""

import json
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import toml
import yaml


class ConfigFormat(Enum):
    """Supported configuration formats."""

    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    ENV = "env"


class ConfigProfile(Enum):
    """Configuration profiles."""

    DEFAULT = "default"
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    POWER_USER = "power_user"
    MINIMAL = "minimal"
    CUSTOM = "custom"


class ConfigSection(Enum):
    """Configuration sections."""

    GENERAL = "general"
    PROVIDERS = "providers"
    MEMORY = "memory"
    MCP = "mcp"
    UI = "ui"
    PERFORMANCE = "performance"
    PRIVACY = "privacy"
    ADVANCED = "advanced"
    EXPERIMENTAL = "experimental"


class ConfigValidator(ABC):
    """Base class for configuration validators."""

    @abstractmethod
    def validate(self, value: Any) -> tuple[bool, str | None]:
        """Validate a configuration value.

        Returns:
            Tuple of (is_valid, error_message)
        """
        pass


class ConfigObserver(ABC):
    """Observer for configuration changes."""

    @abstractmethod
    async def on_config_changed(self, section: str, key: str, old_value: Any, new_value: Any):
        """Called when configuration changes."""
        pass
