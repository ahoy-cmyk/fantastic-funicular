#!/usr/bin/env python3
"""Main entry point for Neuromancer application."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import suppression first to catch all warnings
from src.utils import suppress_warnings  # noqa: F401

from src.core.config import settings
from src.gui.app import NeuromancerApp
from src.gui.utils.error_handling import setup_global_error_handling
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    """Main entry point for the application."""
    try:
        # Setup global error handling to prevent crashes
        setup_global_error_handling()

        logger.info(f"Starting Neuromancer v{settings.APP_VERSION}")
        logger.info("Global error handling initialized - crash protection active")

        # Initialize and run the app
        app = NeuromancerApp()
        app.run()

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
