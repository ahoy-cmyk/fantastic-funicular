"""Loading widgets for the application."""

from kivy.animation import Animation
from kivy.metrics import dp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.card import MDCard
from kivymd.uix.label import MDLabel
from kivymd.uix.spinner import MDSpinner

from src.gui.theme import UIConstants


class LoadingWidget(MDBoxLayout):
    """Reusable loading widget with spinner and text."""

    def __init__(self, text="Loading...", **kwargs):
        super().__init__(**kwargs)
        self.orientation = "vertical"
        self.spacing = UIConstants.SPACING_MEDIUM
        self.adaptive_height = True
        self.size_hint_x = None
        self.width = dp(200)
        self.pos_hint = {"center_x": 0.5, "center_y": 0.5}

        # Spinner
        self.spinner = MDSpinner(
            size_hint=(None, None), size=(dp(48), dp(48)), pos_hint={"center_x": 0.5}, active=True
        )

        # Text
        self.label = MDLabel(
            text=text,
            halign="center",
            theme_text_color="Secondary",
            font_style="Body2",
            adaptive_height=True,
        )

        self.add_widget(self.spinner)
        self.add_widget(self.label)

    def set_text(self, text: str):
        """Update loading text."""
        self.label.text = text

    def show(self):
        """Show the loading widget with animation."""
        self.opacity = 0
        Animation(opacity=1, duration=UIConstants.ANIMATION_FAST).start(self)
        self.spinner.active = True

    def hide(self):
        """Hide the loading widget with animation."""
        Animation(opacity=0, duration=UIConstants.ANIMATION_FAST).start(self)
        self.spinner.active = False


class LoadingOverlay(MDCard):
    """Full-screen loading overlay."""

    def __init__(self, text="Loading...", **kwargs):
        super().__init__(**kwargs)
        self.md_bg_color = (0, 0, 0, 0.7)
        self.elevation = 0
        self.size_hint = (1, 1)
        self.opacity = 0

        # Loading widget
        self.loading_widget = LoadingWidget(text=text)
        self.add_widget(self.loading_widget)

    def show(self, text=None):
        """Show the overlay."""
        if text:
            self.loading_widget.set_text(text)
        Animation(opacity=1, duration=UIConstants.ANIMATION_MEDIUM).start(self)
        self.loading_widget.show()

    def hide(self):
        """Hide the overlay."""
        Animation(opacity=0, duration=UIConstants.ANIMATION_MEDIUM).start(self)
        self.loading_widget.hide()


class TypingIndicator(MDBoxLayout):
    """Typing indicator for chat messages."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "horizontal"
        self.spacing = UIConstants.SPACING_SMALL
        self.adaptive_height = True
        self.size_hint_y = None
        self.height = dp(30)

        # Typing text
        self.label = MDLabel(
            text="AI is thinking",
            theme_text_color="Secondary",
            font_style="Caption",
            adaptive_height=True,
        )

        # Small spinner
        self.spinner = MDSpinner(size_hint=(None, None), size=(dp(20), dp(20)), active=True)

        self.add_widget(self.label)
        self.add_widget(self.spinner)

        # Animate dots
        self._animate_dots()

    def _animate_dots(self):
        """Animate typing dots."""
        import threading
        import time

        def animate():
            dots = ["", ".", "..", "..."]
            i = 0
            while hasattr(self, "spinner") and self.spinner.active:
                base_text = "AI is thinking"
                self.label.text = f"{base_text}{dots[i % len(dots)]}"
                i += 1
                time.sleep(0.5)

        thread = threading.Thread(target=animate)
        thread.daemon = True
        thread.start()

    def stop(self):
        """Stop the typing indicator."""
        self.spinner.active = False
