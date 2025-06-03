"""Professional splash screen with loading progress."""

import asyncio
from collections.abc import Callable

from kivy.animation import Animation
from kivy.metrics import dp
from kivy.properties import NumericProperty, StringProperty
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.label import MDLabel
from kivymd.uix.progressbar import MDProgressBar
from kivymd.uix.screen import MDScreen

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class SplashScreen(MDScreen):
    """Professional splash screen with loading progress."""

    progress_text = StringProperty("Initializing...")
    progress_value = NumericProperty(0)

    def __init__(self, on_complete: Callable | None = None, **kwargs):
        super().__init__(**kwargs)
        self.on_complete = on_complete
        self.build_ui()

    def build_ui(self):
        """Build the splash screen UI."""
        # Main container
        container = MDBoxLayout(
            orientation="vertical",
            spacing=dp(30),
            adaptive_height=True,
            pos_hint={"center_x": 0.5, "center_y": 0.5},
            size_hint_y=None,
        )

        # Logo/Title section
        title_container = MDBoxLayout(
            orientation="vertical", spacing=dp(10), adaptive_height=True, size_hint_y=None
        )

        # Main title
        self.title_label = MDLabel(
            text="NEUROMANCER",
            font_style="H3",
            theme_text_color="Primary",
            halign="center",
            size_hint_y=None,
            height=dp(60),
        )

        # Subtitle
        self.subtitle_label = MDLabel(
            text="AI Assistant with Exceptional Memory",
            font_style="H6",
            theme_text_color="Secondary",
            halign="center",
            size_hint_y=None,
            height=dp(30),
        )

        title_container.add_widget(self.title_label)
        title_container.add_widget(self.subtitle_label)

        # Progress section
        progress_container = MDBoxLayout(
            orientation="vertical",
            spacing=dp(15),
            adaptive_height=True,
            size_hint_y=None,
            size_hint_x=0.6,
            pos_hint={"center_x": 0.5},
        )

        # Progress bar
        self.progress_bar = MDProgressBar(
            value=0, size_hint_y=None, height=dp(6), color=(0.2, 0.6, 1.0, 1.0)  # Professional blue
        )

        # Progress text
        self.progress_label = MDLabel(
            text=self.progress_text,
            font_style="Body2",
            theme_text_color="Secondary",
            halign="center",
            size_hint_y=None,
            height=dp(20),
        )

        progress_container.add_widget(self.progress_bar)
        progress_container.add_widget(self.progress_label)

        # Version info
        self.version_label = MDLabel(
            text="v0.1.0",
            font_style="Caption",
            theme_text_color="Hint",
            halign="center",
            size_hint_y=None,
            height=dp(20),
        )

        # Assembly
        container.add_widget(title_container)
        container.add_widget(progress_container)
        container.add_widget(self.version_label)

        self.add_widget(container)

        # Bind properties
        self.bind(progress_text=self._update_progress_text)
        self.bind(progress_value=self._update_progress_value)

    def _update_progress_text(self, instance, value):
        """Update progress text."""
        if hasattr(self, "progress_label"):
            self.progress_label.text = value

    def _update_progress_value(self, instance, value):
        """Update progress value with smooth animation."""
        if hasattr(self, "progress_bar"):
            Animation(value=value, duration=0.3, t="out_cubic").start(self.progress_bar)

    def set_progress(self, value: float, text: str = None):
        """Set progress value and optional text."""
        self.progress_value = min(100, max(0, value))
        if text:
            self.progress_text = text

    async def start_loading_sequence(self):
        """Start the loading sequence with realistic progress."""
        loading_steps = [
            (10, "Loading configuration..."),
            (25, "Initializing vector memory..."),
            (40, "Setting up LLM providers..."),
            (60, "Creating database models..."),
            (75, "Building user interface..."),
            (90, "Finalizing setup..."),
            (100, "Ready!"),
        ]

        for progress, text in loading_steps:
            self.set_progress(progress, text)
            await asyncio.sleep(0.3)  # Realistic loading feel

        # Wait a moment on "Ready!" then complete
        await asyncio.sleep(0.5)

        if self.on_complete:
            self.on_complete()
