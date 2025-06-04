"""Advanced configuration management with profiles, validation, and persistence."""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import appdirs
import toml
import yaml
from cryptography.fernet import Fernet
from watchdog.events import FileModifiedEvent, FileSystemEventHandler
from watchdog.observers import Observer

from src.config import ConfigFormat, ConfigObserver, ConfigSection
from src.config.schema import NeuromancerConfig
from src.config.validators import ConfigValidatorRegistry
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ConfigFileWatcher(FileSystemEventHandler):
    """Watch configuration files for changes."""

    def __init__(self, config_manager: "ConfigManager"):
        self.config_manager = config_manager

    def on_modified(self, event: FileModifiedEvent):
        """Handle file modification events."""
        if not event.is_directory and event.src_path.endswith((".json", ".yaml", ".toml")):
            logger.info(f"Configuration file modified: {event.src_path}")
            asyncio.create_task(self.config_manager.reload())


class ConfigManager:
    """Advanced configuration manager with all the bells and whistles."""

    def __init__(self, app_name: str = "Neuromancer"):
        """Initialize configuration manager."""
        self.app_name = app_name
        self.config: NeuromancerConfig = NeuromancerConfig()
        self.profiles: dict[str, NeuromancerConfig] = {}
        self.observers: set[ConfigObserver] = set()
        self.validator_registry = ConfigValidatorRegistry()

        # Paths
        self.config_dir = Path(appdirs.user_config_dir(app_name))
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self.config_file = self.config_dir / "config.json"
        self.profiles_dir = self.config_dir / "profiles"
        self.profiles_dir.mkdir(exist_ok=True)

        # Encryption
        self._fernet = None
        self._init_encryption()

        # File watcher
        self.observer = Observer()
        self.observer.schedule(ConfigFileWatcher(self), str(self.config_dir), recursive=True)

        # History
        self.history: list[dict[str, Any]] = []
        self.max_history = 50

        # Load configuration
        self.load()

    def _init_encryption(self):
        """Initialize encryption for sensitive data."""
        key_file = self.config_dir / ".key"

        if key_file.exists():
            key = key_file.read_bytes()
        else:
            key = Fernet.generate_key()
            key_file.write_bytes(key)
            key_file.chmod(0o600)  # Read/write for owner only

        self._fernet = Fernet(key)

    def start_watching(self):
        """Start watching configuration files."""
        self.observer.start()
        logger.info("Started configuration file watcher")

    def stop_watching(self):
        """Stop watching configuration files."""
        self.observer.stop()
        self.observer.join()
        logger.info("Stopped configuration file watcher")

    def add_observer(self, observer: ConfigObserver):
        """Add a configuration change observer."""
        self.observers.add(observer)

    def remove_observer(self, observer: ConfigObserver):
        """Remove a configuration change observer."""
        self.observers.discard(observer)

    async def _notify_observers(self, section: str, key: str, old_value: Any, new_value: Any):
        """Notify all observers of configuration changes."""
        tasks = []
        for observer in self.observers:
            task = asyncio.create_task(
                observer.on_config_changed(section, key, old_value, new_value)
            )
            tasks.append(task)

        await asyncio.gather(*tasks, return_exceptions=True)

    def get(self, path: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated path.

        Example: config.get("providers.ollama.host")
        """
        parts = path.split(".")
        value = self.config.dict()

        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default

        return value

    def set_sync(self, path: str, value: Any, validate: bool = True) -> bool:
        """Set configuration value synchronously."""
        parts = path.split(".")
        if not parts:
            return False

        # Get old value for logging
        old_value = self.get(path)

        # Update configuration
        config_dict = self.config.dict()
        current = config_dict

        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[parts[-1]] = value

        try:
            # Recreate config from updated dict
            self.config = NeuromancerConfig(**config_dict)

            # Auto-save if enabled
            if self.config.general.auto_save:
                self.save()

            logger.debug(f"Updated config {path}: {old_value} -> {value}")
            return True

        except Exception as e:
            logger.error(f"Failed to set {path} = {value}: {e}")
            return False

    async def set(self, path: str, value: Any, validate: bool = True) -> bool:
        """Set configuration value by dot-separated path.

        Returns True if successful.
        """
        parts = path.split(".")
        if not parts:
            return False

        # Validate if requested
        if validate:
            validator = self.validator_registry.get_validator(path)
            if validator:
                is_valid, error = validator.validate(value)
                if not is_valid:
                    logger.error(f"Validation failed for {path}: {error}")
                    return False

        # Get old value for notification
        old_value = self.get(path)

        # Update configuration
        config_dict = self.config.dict()
        current = config_dict

        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[parts[-1]] = value

        # Recreate config from updated dict
        self.config = NeuromancerConfig(**config_dict)

        # Add to history
        self._add_to_history(
            {
                "timestamp": datetime.now().isoformat(),
                "path": path,
                "old_value": old_value,
                "new_value": value,
            }
        )

        # Notify observers
        section = parts[0] if parts else "unknown"
        key = ".".join(parts[1:]) if len(parts) > 1 else parts[0]
        await self._notify_observers(section, key, old_value, value)

        # Auto-save if enabled
        if self.config.general.auto_save:
            self.save()

        return True

    def save(self, path: Path | None = None, format: ConfigFormat = ConfigFormat.JSON):
        """Save configuration to file."""
        path = path or self.config_file

        # Prepare data
        data = self.config.dict()

        # Encrypt sensitive fields
        data = self._encrypt_sensitive(data)

        # Save based on format
        if format == ConfigFormat.JSON:
            # Custom JSON encoder to handle Path objects
            class PathEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, Path):
                        return str(obj)
                    return super().default(obj)

            path.write_text(json.dumps(data, indent=2, cls=PathEncoder))
        elif format == ConfigFormat.YAML:
            path.write_text(yaml.dump(data, default_flow_style=False))
        elif format == ConfigFormat.TOML:
            path.write_text(toml.dumps(data))

        logger.info(f"Saved configuration to {path}")

    def load(self, path: Path | None = None) -> bool:
        """Load configuration from file."""
        path = path or self.config_file

        if not path.exists():
            logger.info("No configuration file found, using defaults")
            return False

        try:
            # Detect format
            suffix = path.suffix.lower()

            if suffix == ".json":
                data = json.loads(path.read_text())
            elif suffix in (".yaml", ".yml"):
                data = yaml.safe_load(path.read_text())
            elif suffix == ".toml":
                data = toml.loads(path.read_text())
            else:
                logger.error(f"Unknown configuration format: {suffix}")
                return False

            # Decrypt sensitive fields
            data = self._decrypt_sensitive(data)

            # Load into configuration
            self.config = NeuromancerConfig(**data)

            logger.info(f"Loaded configuration from {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return False

    async def reload(self):
        """Reload configuration from file."""
        old_config = self.config.dict()

        if self.load():
            # Find and notify changes
            new_config = self.config.dict()
            await self._notify_changes(old_config, new_config)

    def _notify_changes(self, old_dict: dict, new_dict: dict, path: str = ""):
        """Recursively find and notify configuration changes."""
        tasks = []

        for key in set(old_dict.keys()) | set(new_dict.keys()):
            current_path = f"{path}.{key}" if path else key

            old_val = old_dict.get(key)
            new_val = new_dict.get(key)

            if old_val != new_val:
                if isinstance(old_val, dict) and isinstance(new_val, dict):
                    # Recurse into nested dicts
                    tasks.extend(self._notify_changes(old_val, new_val, current_path))
                else:
                    # Value changed
                    parts = current_path.split(".")
                    section = parts[0]
                    key = ".".join(parts[1:])

                    task = asyncio.create_task(
                        self._notify_observers(section, key, old_val, new_val)
                    )
                    tasks.append(task)

        return tasks

    def save_profile(self, name: str, profile: NeuromancerConfig | None = None):
        """Save a configuration profile."""
        profile = profile or self.config
        profile_path = self.profiles_dir / f"{name}.json"

        data = profile.dict()
        data["profile"] = name

        profile_path.write_text(json.dumps(data, indent=2))
        logger.info(f"Saved profile: {name}")

    def load_profile(self, name: str) -> bool:
        """Load a configuration profile."""
        profile_path = self.profiles_dir / f"{name}.json"

        if not profile_path.exists():
            logger.error(f"Profile not found: {name}")
            return False

        try:
            data = json.loads(profile_path.read_text())
            self.config = NeuromancerConfig(**data)
            logger.info(f"Loaded profile: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to load profile {name}: {e}")
            return False

    def list_profiles(self) -> list[str]:
        """List available configuration profiles."""
        profiles = []

        for file in self.profiles_dir.glob("*.json"):
            profiles.append(file.stem)

        return sorted(profiles)

    def export_config(
        self, path: Path, format: ConfigFormat = ConfigFormat.JSON, include_sensitive: bool = False
    ):
        """Export configuration to file."""
        data = self.config.dict()

        if not include_sensitive:
            # Remove sensitive fields
            data = self._remove_sensitive(data)

        # Save based on format
        if format == ConfigFormat.JSON:
            path.write_text(json.dumps(data, indent=2))
        elif format == ConfigFormat.YAML:
            path.write_text(yaml.dump(data, default_flow_style=False))
        elif format == ConfigFormat.TOML:
            path.write_text(toml.dumps(data))

        logger.info(f"Exported configuration to {path}")

    def import_config(self, path: Path, merge: bool = False) -> bool:
        """Import configuration from file."""
        if not path.exists():
            logger.error(f"Import file not found: {path}")
            return False

        try:
            # Load the import data
            if self.load(path):
                if merge:
                    # TODO: Implement intelligent merge
                    pass

                logger.info(f"Imported configuration from {path}")
                return True

        except Exception as e:
            logger.error(f"Failed to import configuration: {e}")

        return False

    def reset_section(self, section: ConfigSection):
        """Reset a configuration section to defaults."""
        section_name = section.value

        if hasattr(self.config, section_name):
            # Create default instance of the section
            section_class = type(getattr(self.config, section_name))
            setattr(self.config, section_name, section_class())

            logger.info(f"Reset configuration section: {section_name}")

            if self.config.general.auto_save:
                self.save()

    def validate(self) -> list[str]:
        """Validate entire configuration.

        Returns list of validation errors.
        """
        errors = []

        # Use Pydantic validation
        try:
            self.config = NeuromancerConfig(**self.config.dict())
        except Exception as e:
            errors.append(str(e))

        # Custom validators
        for path, validator in self.validator_registry.validators.items():
            value = self.get(path)
            if value is not None:
                is_valid, error = validator.validate(value)
                if not is_valid:
                    errors.append(f"{path}: {error}")

        return errors

    def _encrypt_sensitive(self, data: dict) -> dict:
        """Encrypt sensitive fields in configuration."""
        # List of paths to encrypt
        sensitive_paths = [
            "providers.openai_api_key",
            "providers.openai_organization",
            "privacy.encryption_key",
        ]

        result = data.copy()

        for path in sensitive_paths:
            value = self._get_nested(result, path)
            if value and isinstance(value, str):
                encrypted = self._fernet.encrypt(value.encode()).decode()
                self._set_nested(result, path, f"ENC:{encrypted}")

        return result

    def _decrypt_sensitive(self, data: dict) -> dict:
        """Decrypt sensitive fields in configuration."""
        result = data.copy()

        def decrypt_value(value):
            if isinstance(value, str) and value.startswith("ENC:"):
                try:
                    encrypted = value[4:]
                    decrypted = self._fernet.decrypt(encrypted.encode()).decode()
                    return decrypted
                except Exception as e:
                    logger.error(f"Failed to decrypt value: {e}")
                    return None
            return value

        def walk_dict(d):
            for key, value in d.items():
                if isinstance(value, dict):
                    walk_dict(value)
                else:
                    d[key] = decrypt_value(value)

        walk_dict(result)
        return result

    def _remove_sensitive(self, data: dict) -> dict:
        """Remove sensitive fields from configuration."""
        sensitive_paths = [
            "providers.openai_api_key",
            "providers.openai_organization",
            "privacy.encryption_key",
        ]

        result = data.copy()

        for path in sensitive_paths:
            self._set_nested(result, path, None)

        return result

    def _get_nested(self, data: dict, path: str) -> Any:
        """Get nested value from dict by dot path."""
        parts = path.split(".")
        current = data

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None

        return current

    def _set_nested(self, data: dict, path: str, value: Any):
        """Set nested value in dict by dot path."""
        parts = path.split(".")
        current = data

        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        if value is None and parts[-1] in current:
            del current[parts[-1]]
        else:
            current[parts[-1]] = value

    def _add_to_history(self, entry: dict[str, Any]):
        """Add entry to configuration history."""
        self.history.append(entry)

        # Limit history size
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]

    def get_history(self) -> list[dict[str, Any]]:
        """Get configuration change history."""
        return self.history.copy()

    def create_default_profiles(self):
        """Create default configuration profiles."""
        # Development profile
        dev_config = NeuromancerConfig()
        dev_config.general.log_level = "DEBUG"
        dev_config.general.telemetry = False
        dev_config.experimental.enable_experimental = True
        self.save_profile("development", dev_config)

        # Production profile
        prod_config = NeuromancerConfig()
        prod_config.general.log_level = "INFO"
        prod_config.performance.enable_cache = True
        prod_config.performance.preload_providers = True
        prod_config.privacy.encrypt_storage = True
        self.save_profile("production", prod_config)

        # Power user profile
        power_config = NeuromancerConfig()
        power_config.memory.strategy = "aggressive"
        power_config.memory.max_memories = 1000000
        power_config.ui.show_token_count = True
        power_config.experimental.enable_experimental = True
        self.save_profile("power_user", power_config)

        # Minimal profile
        minimal_config = NeuromancerConfig()
        minimal_config.memory.enabled = False
        minimal_config.mcp.enabled = False
        minimal_config.ui.enable_animations = False
        minimal_config.providers.fallback_providers = []
        self.save_profile("minimal", minimal_config)

        logger.info("Created default configuration profiles")
