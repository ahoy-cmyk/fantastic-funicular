"""Logging configuration for Neuromancer."""

import logging
import os
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler


def setup_logger(name: str, level: str = None) -> logging.Logger:
    """Set up a logger with Rich formatting.

    Args:
        name: Logger name (usually __name__)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Set level from parameter, config, environment, or default
    if level is None:
        # Try to get from config first
        try:
            from src.core.config import config_manager
            level = config_manager.get("general.log_level", "INFO")

            # Handle enum values from config
            if hasattr(level, 'value'):
                level = level.value
            elif hasattr(level, 'name'):
                level = level.name

        except (ImportError, Exception):
            # Fallback to environment or default
            debug = os.getenv("DEBUG", "false").lower() == "true"
            level = "DEBUG" if debug else "INFO"

    # Ensure level is a string and convert to logging level
    level = str(level).upper()
    numeric_level = getattr(logging, level, logging.INFO)

    # Always update the logger level (in case config changed)
    logger.setLevel(numeric_level)

    # Update existing handlers if they exist
    if logger.handlers:
        for handler in logger.handlers:
            if isinstance(handler, RichHandler):  # Console handler
                handler.setLevel(numeric_level)
        return logger

    # Console handler with Rich
    show_locals = numeric_level <= logging.DEBUG
    console_handler = RichHandler(
        console=Console(stderr=True), rich_tracebacks=True, tracebacks_show_locals=show_locals
    )
    console_handler.setLevel(numeric_level)

    # File handler
    log_dir = Path.home() / ".neuromancer" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"neuromancer_{datetime.now():%Y%m%d}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)  # Always log everything to file

    # Formatters
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    )

    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def update_log_level(level: str):
    """Update log level for all existing loggers.

    Args:
        level: New log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Update root logger level
    logging.getLogger().setLevel(numeric_level)

    # Update all loggers that start with 'src.' and their handlers
    for logger_name in list(logging.Logger.manager.loggerDict.keys()):
        if logger_name.startswith('src.'):
            logger = logging.getLogger(logger_name)
            logger.setLevel(numeric_level)

            # Update handler levels too
            for handler in logger.handlers:
                if isinstance(handler, RichHandler):  # Console handler
                    handler.setLevel(numeric_level)
                # Keep file handler at DEBUG to capture everything


def refresh_all_loggers():
    """Refresh all loggers to respect current configuration."""
    try:
        current_level = get_current_log_level()  # Use helper that handles enums
        update_log_level(current_level)
    except (ImportError, Exception):
        pass  # Fallback gracefully


def get_current_log_level() -> str:
    """Get the current log level.

    Returns:
        Current log level as string
    """
    try:
        from src.core.config import config_manager
        level = config_manager.get("general.log_level", "INFO")

        # Debug: Print the raw level value and its type
        # Note: Only do this once to avoid spam
        if not hasattr(get_current_log_level, '_debug_logged'):
            print(f"[DEBUG] Raw log level from config: {level} (type: {type(level)})")
            get_current_log_level._debug_logged = True

        # Handle enum values from config
        if hasattr(level, 'value'):
            return level.value
        elif hasattr(level, 'name'):
            return level.name
        else:
            return str(level)

    except (ImportError, Exception) as e:
        print(f"[DEBUG] Exception in get_current_log_level: {e}")
        return "INFO"


def debug_logger_info() -> dict:
    """Get debug information about current loggers.

    Returns:
        Dictionary with logger debug information
    """
    info = {
        "config_level": get_current_log_level(),
        "root_level": logging.getLevelName(logging.getLogger().level),
        "src_loggers": {},
        "total_loggers": len(logging.Logger.manager.loggerDict)
    }

    # Get info about src loggers
    for logger_name in logging.Logger.manager.loggerDict:
        if logger_name.startswith('src.'):
            logger = logging.getLogger(logger_name)
            handler_info = []
            for handler in logger.handlers:
                handler_info.append({
                    "type": type(handler).__name__,
                    "level": logging.getLevelName(handler.level)
                })

            info["src_loggers"][logger_name] = {
                "level": logging.getLevelName(logger.level),
                "handlers": handler_info,
                "effective_level": logging.getLevelName(logger.getEffectiveLevel())
            }

    return info
