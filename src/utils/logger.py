"""Logging configuration for Neuromancer."""

import logging
import os
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler


def setup_logger(name: str, debug: bool = None) -> logging.Logger:
    """Set up a logger with Rich formatting.

    Args:
        name: Logger name (usually __name__)
        debug: Whether to enable debug logging (defaults to env var)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Only configure if not already configured
    if logger.handlers:
        return logger

    # Set level based on debug setting or environment
    if debug is None:
        debug = os.getenv("DEBUG", "false").lower() == "true"
    level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(level)

    # Console handler with Rich
    console_handler = RichHandler(
        console=Console(stderr=True), rich_tracebacks=True, tracebacks_show_locals=debug
    )
    console_handler.setLevel(level)

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
