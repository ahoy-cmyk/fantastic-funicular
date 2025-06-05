"""Notification utilities for the application."""

from kivy.animation import Animation
from kivy.clock import Clock
from kivy.metrics import dp
from kivymd.uix.card import MDCard
from kivymd.uix.label import MDLabel


class Toast:
    """Simple toast notification system."""

    @staticmethod
    def show(text: str, duration: float = 3.0):
        """Show a toast notification."""
        # Ensure this runs on the main thread
        def _show_on_main_thread(dt):
            try:
                # Get the root widget (screen manager)
                from kivymd.app import MDApp

                app = MDApp.get_running_app()

                if not hasattr(app, "screen_manager"):
                    return

                # Create toast container
                toast_container = MDCard(
                    orientation="horizontal",
                    spacing=dp(10),
                    padding=dp(15),
                    size_hint=(None, None),
                    size=(dp(300), dp(50)),
                    elevation=8,
                    md_bg_color=(0.2, 0.2, 0.2, 0.95),
                    pos_hint={"center_x": 0.5, "y": 0.1},
                    opacity=0,
                )

                # Toast text
                toast_label = MDLabel(
                    text=text, theme_text_color="Primary", font_style="Body2", adaptive_width=True
                )

                toast_container.add_widget(toast_label)

                # Add to current screen
                current_screen = app.screen_manager.current_screen
                current_screen.add_widget(toast_container)

                # Animate in
                Animation(opacity=1, duration=0.3).start(toast_container)

                # Schedule removal
                def remove_toast(dt):
                    Animation(opacity=0, duration=0.3).start(toast_container)
                    Clock.schedule_once(lambda dt2: current_screen.remove_widget(toast_container), 0.3)

                Clock.schedule_once(remove_toast, duration)
            except Exception:
                # Fallback to console logging if UI toast fails
                print(f"Toast notification: {text}")

        # Always schedule on main thread
        Clock.schedule_once(_show_on_main_thread, 0)


class Notification:
    """Enhanced notification system with different types."""

    @staticmethod
    def success(text: str):
        """Show success notification."""
        Toast.show(f"[OK] {text}", 2.0)

    @staticmethod
    def error(text: str):
        """Show error notification."""
        Toast.show(f"[ERROR] {text}", 4.0)

    @staticmethod
    def info(text: str):
        """Show info notification."""
        Toast.show(f"[INFO] {text}", 3.0)

    @staticmethod
    def warning(text: str):
        """Show warning notification."""
        Toast.show(f"[WARNING] {text}", 3.5)
