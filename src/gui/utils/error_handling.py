"""Comprehensive error handling utilities for the GUI."""

import functools
import traceback
from collections.abc import Callable
from typing import Any

from kivymd.uix.button import MDRaisedButton
from kivymd.uix.dialog import MDDialog

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ErrorHandler:
    """Centralized error handling for GUI operations."""

    @staticmethod
    def show_error_dialog(title: str, message: str, details: str | None = None):
        """Show an error dialog to the user."""
        try:
            full_message = message
            if details:
                full_message += f"\n\nDetails: {details}"

            dialog = MDDialog(
                title=f"Error: {title}",
                text=full_message,
                buttons=[MDRaisedButton(text="OK", on_release=lambda x: dialog.dismiss())],
            )
            dialog.open()
        except Exception as e:
            # Fallback to notification if dialog fails
            logger.error(f"Failed to show error dialog: {e}")
            try:
                from src.gui.utils.notifications import Notification

                Notification.error(f"{title}: {message}")
            except Exception:
                # Ultimate fallback - just log
                logger.error(f"Error dialog fallback failed: {title} - {message}")

    @staticmethod
    def safe_execute(
        func: Callable,
        error_title: str = "Operation Failed",
        show_dialog: bool = False,
        default_return: Any = None,
    ):
        """Execute a function safely with error handling."""
        try:
            return func()
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Safe execution failed in {error_title}: {error_msg}")
            logger.debug(f"Traceback: {traceback.format_exc()}")

            if show_dialog:
                ErrorHandler.show_error_dialog(error_title, error_msg)
            else:
                try:
                    from src.gui.utils.notifications import Notification

                    Notification.error(f"{error_title}: {error_msg}")
                except Exception:
                    pass  # Silent fallback

            return default_return


def gui_error_handler(
    title: str = "GUI Error", show_dialog: bool = False, default_return: Any = None
):
    """Decorator for GUI methods to handle errors gracefully."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = str(e)
                func_name = func.__name__
                logger.error(f"GUI error in {func_name}: {error_msg}")
                logger.debug(f"Traceback: {traceback.format_exc()}")

                if show_dialog:
                    ErrorHandler.show_error_dialog(f"{title} - {func_name}", error_msg)
                else:
                    try:
                        from src.gui.utils.notifications import Notification

                        Notification.error(f"{title}: {error_msg}")
                    except Exception:
                        pass  # Silent fallback

                return default_return

        return wrapper

    return decorator


def safe_widget_operation(operation: Callable, widget_name: str = "Widget"):
    """Safely perform widget operations with error handling."""
    try:
        return operation()
    except AttributeError as e:
        logger.error(f"{widget_name} operation failed - attribute error: {e}")
        return None
    except Exception as e:
        logger.error(f"{widget_name} operation failed: {e}")
        return None


class CrashPrevention:
    """Utilities to prevent common GUI crashes."""

    @staticmethod
    def safe_widget_access(widget, attribute: str, default_value: Any = None):
        """Safely access widget attributes."""
        try:
            if hasattr(widget, attribute):
                return getattr(widget, attribute)
            else:
                logger.warning(f"Widget missing attribute: {attribute}")
                return default_value
        except Exception as e:
            logger.error(f"Failed to access widget attribute {attribute}: {e}")
            return default_value

    @staticmethod
    def safe_widget_method(widget, method_name: str, *args, **kwargs):
        """Safely call widget methods."""
        try:
            if hasattr(widget, method_name):
                method = getattr(widget, method_name)
                if callable(method):
                    return method(*args, **kwargs)
                else:
                    logger.warning(f"Widget attribute {method_name} is not callable")
                    return None
            else:
                logger.warning(f"Widget missing method: {method_name}")
                return None
        except Exception as e:
            logger.error(f"Failed to call widget method {method_name}: {e}")
            return None

    @staticmethod
    def safe_dialog_creation(dialog_class, *args, **kwargs):
        """Safely create dialog widgets."""
        try:
            return dialog_class(*args, **kwargs)
        except Exception as e:
            logger.error(f"Failed to create dialog: {e}")
            # Return a minimal fallback
            try:
                from kivymd.uix.boxlayout import MDBoxLayout

                return MDBoxLayout()  # Minimal fallback
            except Exception:
                return None

    @staticmethod
    def safe_button_creation(button_class, text: str = "Button", **kwargs):
        """Safely create button widgets."""
        try:
            return button_class(text=text, **kwargs)
        except Exception as e:
            logger.error(f"Failed to create button: {e}")
            # Return a minimal fallback
            try:
                from kivymd.uix.button import MDFlatButton

                return MDFlatButton(text=text)
            except Exception:
                return None


# Global error handling setup
def setup_global_error_handling():
    """Setup global error handling for the application."""
    import sys

    def handle_exception(exc_type, exc_value, exc_traceback):
        """Handle uncaught exceptions."""
        if issubclass(exc_type, KeyboardInterrupt):
            # Allow keyboard interrupts to pass through
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        logger.critical(f"Uncaught exception: {error_msg}")

        # Try to show error dialog
        try:
            ErrorHandler.show_error_dialog(
                "Critical Error",
                f"An unexpected error occurred: {exc_value}",
                "Please check the logs for more details. The application will continue running.",
            )
        except Exception:
            # Ultimate fallback - print to console
            print(f"Critical error (fallback): {exc_value}")

    # Set the exception handler
    sys.excepthook = handle_exception


# Convenience decorators for common GUI operations
def safe_ui_operation(func):
    """Decorator for UI operations that should never crash the app."""
    return gui_error_handler("UI Operation", show_dialog=False)(func)


def safe_dialog_operation(func):
    """Decorator for dialog operations with user-visible error handling."""
    return gui_error_handler("Dialog Operation", show_dialog=True)(func)


def safe_button_callback(func):
    """Decorator for button callbacks that should handle errors gracefully."""
    return gui_error_handler("Button Action", show_dialog=False)(func)
