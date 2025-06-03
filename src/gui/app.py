"""Main Kivy application for Neuromancer."""

# Import warning suppression first to catch all warnings from startup

import asyncio

from kivy.clock import Clock
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager
from kivymd.app import MDApp

from src.core.config import config
from src.gui.screens.enhanced_chat_screen import EnhancedChatScreen
from src.gui.screens.memory_screen import MemoryScreen
from src.gui.screens.provider_config_screen import ProviderConfigScreen
from src.gui.screens.settings_screen import SettingsScreen
from src.gui.screens.simple_memory_screen import SimpleMemoryScreen
from src.gui.screens.splash_screen import SplashScreen
from src.gui.theme import NeuromancerTheme
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Set default window size from config
Window.size = (config.ui.window_width, config.ui.window_height)


class NeuromancerApp(MDApp):
    """Main application class."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.title = config.general.app_name + " - AI Assistant"
        self.main_screens_loaded = False

        # Apply professional theme
        NeuromancerTheme.apply_theme(self)

        # Apply window settings
        if config.ui.start_maximized:
            Window.maximize()
        if config.ui.always_on_top:
            Window.always_on_top = True

    def build(self):
        """Build the application UI."""
        try:
            # Load KV files
            self.load_kv_files()

            # Create screen manager
            self.screen_manager = ScreenManager()

            # Add splash screen first
            self.splash_screen = SplashScreen(name="splash", on_complete=self.on_splash_complete)
            self.screen_manager.add_widget(self.splash_screen)

            # Set splash as initial screen
            self.screen_manager.current = "splash"

            # Start loading sequence after UI is ready
            Clock.schedule_once(lambda dt: self.start_async_loading(), 0.1)

            return self.screen_manager

        except Exception as e:
            logger.error(f"Failed to build application: {e}", exc_info=True)
            raise

    def start_async_loading(self):
        """Start the async loading process."""
        # Create a simple event loop for the loading sequence
        import threading

        def run_loading():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.splash_screen.start_loading_sequence())
            loop.close()

        thread = threading.Thread(target=run_loading)
        thread.daemon = True
        thread.start()

    def on_splash_complete(self):
        """Called when splash screen loading is complete."""
        Clock.schedule_once(lambda dt: self.load_main_screens(), 0.1)

    def load_main_screens(self):
        """Load the main application screens."""
        if self.main_screens_loaded:
            return

        try:
            # Add main screens (use only enhanced chat screen)
            self.screen_manager.add_widget(EnhancedChatScreen(name="enhanced_chat"))
            self.screen_manager.add_widget(SettingsScreen(name="settings"))
            self.screen_manager.add_widget(MemoryScreen(name="memory"))
            self.screen_manager.add_widget(SimpleMemoryScreen(name="advanced_memory"))
            self.screen_manager.add_widget(ProviderConfigScreen(name="provider_config"))
            # TODO: Fix MDSwitch compatibility issue in AdvancedSettingsScreen
            # self.screen_manager.add_widget(AdvancedSettingsScreen(name='advanced_settings'))

            self.main_screens_loaded = True

            # Transition to enhanced chat screen
            self.screen_manager.current = "enhanced_chat"

            logger.info("Main screens loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load main screens: {e}", exc_info=True)
            # Stay on splash and show error
            self.splash_screen.set_progress(100, f"Error: {str(e)}")

    def load_kv_files(self):
        """Load Kivy design files."""
        from pathlib import Path

        kv_dir = Path(__file__).parent / "kv"
        if kv_dir.exists():
            for kv_file in kv_dir.glob("*.kv"):
                try:
                    Builder.load_file(str(kv_file))
                    logger.debug(f"Loaded KV file: {kv_file.name}")
                except Exception as e:
                    logger.error(f"Failed to load KV file {kv_file}: {e}")

    def on_start(self):
        """Called when the application starts."""
        logger.info("Neuromancer application started")

    def on_stop(self):
        """Called when the application stops."""
        logger.info("Neuromancer application stopped")
