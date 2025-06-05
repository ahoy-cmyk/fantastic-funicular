"""Configuration validators for runtime validation."""

import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from src.config import ConfigValidator


class URLValidator(ConfigValidator):
    """Validate URL format."""

    def __init__(self, schemes: list[str] | None = None):
        self.schemes = schemes or ["http", "https"]

    def validate(self, value: Any) -> tuple[bool, str | None]:
        if not isinstance(value, str):
            return False, "Value must be a string"

        try:
            result = urlparse(value)
            if not result.scheme:
                return False, "URL must have a scheme"
            if self.schemes and result.scheme not in self.schemes:
                return False, f"URL scheme must be one of: {', '.join(self.schemes)}"
            if not result.netloc:
                return False, "URL must have a network location"
            return True, None
        except Exception as e:
            return False, str(e)


class PathValidator(ConfigValidator):
    """Validate file system paths."""

    def __init__(self, must_exist: bool = False, must_be_dir: bool = False):
        self.must_exist = must_exist
        self.must_be_dir = must_be_dir

    def validate(self, value: Any) -> tuple[bool, str | None]:
        if not isinstance(value, str | Path):
            return False, "Value must be a string or Path"

        path = Path(value)

        if self.must_exist and not path.exists():
            return False, f"Path does not exist: {path}"

        if self.must_be_dir and path.exists() and not path.is_dir():
            return False, f"Path is not a directory: {path}"

        return True, None


class RangeValidator(ConfigValidator):
    """Validate numeric ranges."""

    def __init__(self, min_value: float | None = None, max_value: float | None = None):
        self.min_value = min_value
        self.max_value = max_value

    def validate(self, value: Any) -> tuple[bool, str | None]:
        if not isinstance(value, int | float):
            return False, "Value must be numeric"

        if self.min_value is not None and value < self.min_value:
            return False, f"Value must be >= {self.min_value}"

        if self.max_value is not None and value > self.max_value:
            return False, f"Value must be <= {self.max_value}"

        return True, None


class RegexValidator(ConfigValidator):
    """Validate against regex pattern."""

    def __init__(self, pattern: str, message: str | None = None):
        self.pattern = re.compile(pattern)
        self.message = message or f"Value must match pattern: {pattern}"

    def validate(self, value: Any) -> tuple[bool, str | None]:
        if not isinstance(value, str):
            return False, "Value must be a string"

        if not self.pattern.match(value):
            return False, self.message

        return True, None


class APIKeyValidator(ConfigValidator):
    """Validate API key format."""

    def __init__(self, prefix: str | None = None, length: int | None = None):
        self.prefix = prefix
        self.length = length

    def validate(self, value: Any) -> tuple[bool, str | None]:
        if not value:
            return True, None  # Empty is OK (not enabled)

        if not isinstance(value, str):
            return False, "API key must be a string"

        if self.prefix and not value.startswith(self.prefix):
            return False, f"API key must start with: {self.prefix}"

        if self.length and len(value) != self.length:
            return False, f"API key must be {self.length} characters"

        return True, None


class ModelNameValidator(ConfigValidator):
    """Validate model names."""

    def __init__(self, allowed_patterns: list[str] | None = None):
        self.allowed_patterns = [re.compile(p) for p in (allowed_patterns or [])]

    def validate(self, value: Any) -> tuple[bool, str | None]:
        if not isinstance(value, str):
            return False, "Model name must be a string"

        if not value:
            return False, "Model name cannot be empty"

        if self.allowed_patterns:
            if not any(p.match(value) for p in self.allowed_patterns):
                return False, "Model name does not match allowed patterns"

        return True, None


class ListValidator(ConfigValidator):
    """Validate list values."""

    def __init__(
        self,
        min_length: int | None = None,
        max_length: int | None = None,
        item_validator: ConfigValidator | None = None,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.item_validator = item_validator

    def validate(self, value: Any) -> tuple[bool, str | None]:
        if not isinstance(value, list):
            return False, "Value must be a list"

        if self.min_length is not None and len(value) < self.min_length:
            return False, f"List must have at least {self.min_length} items"

        if self.max_length is not None and len(value) > self.max_length:
            return False, f"List must have at most {self.max_length} items"

        if self.item_validator:
            for i, item in enumerate(value):
                is_valid, error = self.item_validator.validate(item)
                if not is_valid:
                    return False, f"Item {i}: {error}"

        return True, None


class DependencyValidator(ConfigValidator):
    """Validate configuration dependencies."""

    def __init__(self, depends_on: str, condition: Any, config_getter):
        self.depends_on = depends_on
        self.condition = condition
        self.config_getter = config_getter

    def validate(self, value: Any) -> tuple[bool, str | None]:
        dependency_value = self.config_getter(self.depends_on)

        if dependency_value == self.condition and not value:
            return False, f"This field is required when {self.depends_on} is {self.condition}"

        return True, None


class ConfigValidatorRegistry:
    """Registry for configuration validators."""

    def __init__(self):
        self.validators: dict[str, ConfigValidator] = {}
        self._register_default_validators()

    def _register_default_validators(self):
        """Register default validators."""
        # URLs
        self.register("providers.ollama_host", URLValidator())
        self.register("providers.openai_base_url", URLValidator())
        self.register("providers.lmstudio_host", URLValidator())

        # API Keys
        self.register("providers.openai_api_key", APIKeyValidator(prefix="sk-", length=51))

        # Paths
        self.register("memory.persist_dir", PathValidator(must_be_dir=True))

        # Ranges
        self.register("providers.temperature", RangeValidator(0.0, 2.0))
        self.register("providers.top_p", RangeValidator(0.0, 1.0))
        self.register("memory.importance_threshold", RangeValidator(0.0, 1.0))
        self.register("memory.similarity_threshold", RangeValidator(0.0, 1.0))

        # Lists
        self.register("providers.ollama_models", ListValidator(item_validator=ModelNameValidator()))
        self.register(
            "providers.openai_models",
            ListValidator(
                item_validator=ModelNameValidator(allowed_patterns=[r"^gpt-.*", r"^text-.*"])
            ),
        )

        # Language codes
        self.register(
            "general.language",
            RegexValidator(
                r"^[a-z]{2}(-[A-Z]{2})?$", "Language must be ISO 639-1 code (e.g., 'en' or 'en-US')"
            ),
        )

    def register(self, path: str, validator: ConfigValidator):
        """Register a validator for a configuration path."""
        self.validators[path] = validator

    def unregister(self, path: str):
        """Unregister a validator."""
        self.validators.pop(path, None)

    def get_validator(self, path: str) -> ConfigValidator | None:
        """Get validator for a path."""
        return self.validators.get(path)

    def validate_all(self, config_getter) -> list[tuple[str, str]]:
        """Validate all registered paths.

        Returns list of (path, error) tuples.
        """
        errors = []

        for path, validator in self.validators.items():
            value = config_getter(path)
            if value is not None:
                is_valid, error = validator.validate(value)
                if not is_valid:
                    errors.append((path, error))

        return errors
