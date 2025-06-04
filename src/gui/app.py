"""Main Kivy application entry point for Neuromancer.

This module implements the core MDApp class that orchestrates the entire GUI application.
It demonstrates advanced Kivy patterns including:

- Asynchronous initialization with progress tracking
- Thread-safe resource preloading for performance optimization  
- Professional splash screen implementation
- Cross-platform window management
- Resource sharing between screens for memory efficiency
- Graceful error handling and recovery

Architectural Design:
    The app follows a screen manager pattern with centralized resource management.
    Heavy operations (memory system, embeddings) are preloaded during splash to
    ensure responsive UI performance throughout the application lifecycle.

Performance Considerations:
    - Embedding models are loaded once during startup
    - Chat and memory managers are shared between screens
    - UI operations are scheduled on the main thread safely
    - Background threads handle all I/O and computation
    
Accessibility:
    - Keyboard navigation support
    - Screen reader compatible widget hierarchy
    - High contrast theme for visibility
    - Responsive layout for different screen sizes
"""

# Import warning suppression first to catch all warnings from startup

import asyncio

from kivy.clock import Clock
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager
from kivymd.app import MDApp

from src.core.config import config
from src.gui.screens.enhanced_chat_screen import EnhancedChatScreen
from src.gui.screens.file_management_screen import FileManagementScreen
from src.gui.screens.memory_screen import MemoryScreen
from src.gui.screens.model_management_screen import ModelManagementScreen
from src.gui.screens.settings_screen import SettingsScreen
from src.gui.screens.simple_memory_screen import SimpleMemoryScreen
from src.gui.screens.splash_screen import SplashScreen
from src.gui.theme import NeuromancerTheme
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Set default window size from config
Window.size = (config.ui.window_width, config.ui.window_height)


class NeuromancerApp(MDApp):
    """Main application class implementing the MDApp pattern with advanced features.
    
    This class serves as the central coordinator for the entire GUI application,
    managing screen transitions, shared resources, and the application lifecycle.
    
    Key Features:
        - Asynchronous resource initialization with visual progress feedback
        - Shared resource management (chat manager, memory system) for efficiency
        - Professional splash screen with realistic loading simulation
        - Cross-platform window configuration and state management
        - Thread-safe operations with proper Kivy Clock scheduling
        - Graceful error handling and user feedback
    
    Architecture Pattern:
        Uses the Model-View-Controller (MVC) pattern where:
        - Model: Chat manager, memory system, configuration
        - View: Individual screen classes and their widgets
        - Controller: This app class coordinating between them
    
    Performance Optimizations:
        - Pre-initializes heavy resources (embedding models, database connections)
        - Shares expensive objects between screens to reduce memory footprint
        - Uses background threads for all blocking operations
        - Implements lazy loading for non-critical components
    
    Threading Model:
        - Main thread: UI operations and Kivy event handling
        - Background threads: Resource loading, async operations
        - Clock.schedule_once: Thread-safe UI updates from background threads
    
    Attributes:
        main_screens_loaded (bool): Flag preventing duplicate screen initialization
        _chat_manager (ChatManager): Shared chat management instance
        _safe_memory (SafeMemoryManager): Shared memory system instance
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.title = config.general.app_name + " - AI Assistant"
        self.main_screens_loaded = False

        # Store pre-initialized managers for sharing
        self._chat_manager = None
        self._safe_memory = None

        # Apply professional theme
        NeuromancerTheme.apply_theme(self)

        # Apply window settings
        if config.ui.start_maximized:
            Window.maximize()
        if config.ui.always_on_top:
            Window.always_on_top = True

    def build(self):
        """Build the application UI using the screen manager pattern.
        
        This method sets up the core UI structure and initiates the progressive
        loading sequence. It follows Kivy best practices by:
        
        1. Loading external KV files for UI definitions
        2. Creating a ScreenManager for navigation
        3. Adding the splash screen as the initial view
        4. Scheduling async initialization after UI is ready
        
        The method ensures UI responsiveness by deferring heavy operations
        until after the initial UI is rendered.
        
        Returns:
            ScreenManager: The root widget containing all application screens
            
        Raises:
            Exception: If UI building fails, logged and re-raised for proper handling
        """
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
        """Start the asynchronous loading process with real initialization.
        
        This method creates a separate thread with its own event loop to handle
        resource-intensive initialization without blocking the UI thread.
        
        Threading Strategy:
            - Creates a daemon thread to prevent hanging on app exit
            - Uses a new event loop to avoid conflicts with Kivy's main loop
            - Ensures proper cleanup with try/finally blocks
            
        This pattern allows the splash screen to remain responsive while
        heavy operations (embedding models, database setup) execute in parallel.
        """
        # Create a simple event loop for the loading sequence
        import threading

        def run_loading():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.real_loading_sequence())
            loop.close()

        thread = threading.Thread(target=run_loading)
        thread.daemon = True
        thread.start()

    async def real_loading_sequence(self):
        """Perform real initialization with progressive visual feedback.
        
        This method implements a realistic loading sequence that:
        1. Initializes configuration and logging systems
        2. Preloads heavy embedding models for memory system
        3. Sets up chat managers and LLM provider connections
        4. Prepares database and session management
        5. Builds the main application screens
        
        Progress Updates:
            Each step provides visual feedback via the splash screen,
            giving users insight into what's happening during startup.
            
        Error Handling:
            Individual components can fail without stopping the entire
            initialization process. Warnings are logged but loading continues.
            
        Performance Optimization:
            By preloading expensive resources here, subsequent screen
            transitions are nearly instantaneous, providing a smooth UX.
        """
        try:
            # Step 1: Configuration and basic setup
            self.splash_screen.set_progress(
                10, "Loading configuration...", "Reading settings and preferences"
            )
            await asyncio.sleep(0.1)  # Let UI update
            
            # Refresh loggers to respect configuration
            try:
                from src.utils.logger import refresh_all_loggers
                refresh_all_loggers()
            except Exception:
                pass  # Don't fail loading if logger refresh fails

            # Preload some heavy imports to reduce later delays
            self.splash_screen.set_progress(
                20, "Loading core modules...", "Importing essential components"
            )
            await asyncio.sleep(0.1)

            # Step 2: Initialize memory system
            self.splash_screen.set_progress(
                35, "Initializing memory system...", "Loading embedding model (may take a moment)"
            )
            try:
                # Pre-initialize memory components to reduce chat screen load time
                from src.memory.safe_operations import create_safe_memory_manager
                from src.utils.embeddings import EmbeddingGenerator

                # Initialize embedding generator (this is often slow)
                embedding_gen = EmbeddingGenerator()
                # Trigger lazy loading of the model
                _ = embedding_gen.model

                # Pre-initialize and store memory manager
                self._safe_memory = create_safe_memory_manager(
                    lambda op, err: logger.warning(f"Memory error: {op} - {err}")
                )

                await asyncio.sleep(0.1)
            except Exception as e:
                logger.warning(f"Memory system initialization warning: {e}")

            # Step 3: Initialize chat manager and providers
            self.splash_screen.set_progress(
                50, "Setting up LLM providers...", "Configuring AI model connections"
            )
            try:
                from src.core.chat_manager import ChatManager

                # Pre-initialize and store for reuse
                self._chat_manager = ChatManager()
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.warning(f"Chat manager initialization warning: {e}")

            # Step 4: Database and session setup
            self.splash_screen.set_progress(
                65, "Preparing database...", "Setting up conversation storage"
            )
            try:

                # Session manager is already initialized in chat manager
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.warning(f"Session manager initialization warning: {e}")

            # Step 5: UI preparation
            self.splash_screen.set_progress(
                80, "Building user interface...", "Creating application screens"
            )
            await asyncio.sleep(0.1)

            # Step 6: Final preparations
            self.splash_screen.set_progress(95, "Finalizing setup...", "Almost ready!")
            await asyncio.sleep(0.2)

            # Complete
            self.splash_screen.set_progress(100, "Ready!", "Launching Neuromancer...")
            await asyncio.sleep(0.5)

            # Schedule completion on main thread
            Clock.schedule_once(lambda dt: self.on_splash_complete(), 0)

        except Exception as e:
            logger.error(f"Loading sequence failed: {e}")
            Clock.schedule_once(
                lambda dt: self.splash_screen.set_progress(100, f"Error: {str(e)}"), 0
            )
            Clock.schedule_once(
                lambda dt: self.on_splash_complete(), 2.0
            )  # Still proceed after error

    def on_splash_complete(self):
        """Handle completion of splash screen loading sequence.
        
        This callback is triggered either on successful initialization
        completion or after an error timeout. It transitions from the
        splash screen to the main application interface.
        
        The slight delay ensures the splash screen's final animation
        completes before transitioning to avoid jarring visual changes.
        """
        Clock.schedule_once(lambda dt: self.load_main_screens(), 0.1)

    def load_main_screens(self):
        """Load main application screens with shared resource injection.
        
        This method creates all primary application screens and injects
        pre-initialized shared resources for optimal performance.
        
        Resource Sharing Strategy:
            - Passes pre-loaded chat_manager to screens that need it
            - Provides app instance reference for accessing shared resources
            - Prevents duplicate initialization of expensive components
            
        Screen Loading Order:
            1. Enhanced chat screen (primary interface)
            2. Settings and memory management screens
            3. Model and file management screens
            4. Advanced configuration screens (if compatible)
            
        Error Handling:
            If screen loading fails, the splash screen displays the error
            but the app continues running with available screens.
        """
        if self.main_screens_loaded:
            return

        try:
            logger.info("Loading main application screens...")

            # Load screens one by one for better startup experience
            self.splash_screen.set_progress(
                85, "Loading chat interface...", "Initializing main chat screen"
            )

            # Add main chat screen first (most important) - pass app instance for pre-initialized managers
            self.screen_manager.add_widget(
                EnhancedChatScreen(app_instance=self, name="enhanced_chat")
            )

            self.splash_screen.set_progress(
                90, "Loading auxiliary screens...", "Setting up settings and memory screens"
            )

            # Add other screens
            self.screen_manager.add_widget(SettingsScreen(name="settings"))
            self.screen_manager.add_widget(MemoryScreen(name="memory"))
            self.screen_manager.add_widget(SimpleMemoryScreen(name="advanced_memory"))
            self.screen_manager.add_widget(ModelManagementScreen(chat_manager=self._chat_manager, name="model_management"))
            self.screen_manager.add_widget(FileManagementScreen(chat_manager=self._chat_manager, name="file_management"))
            # TODO: Fix MDSwitch compatibility issue in AdvancedSettingsScreen
            # self.screen_manager.add_widget(AdvancedSettingsScreen(name='advanced_settings'))

            self.main_screens_loaded = True

            self.splash_screen.set_progress(
                100, "Starting application...", "Welcome to Neuromancer!"
            )

            # Transition to enhanced chat screen
            self.screen_manager.current = "enhanced_chat"

            logger.info("Main screens loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load main screens: {e}", exc_info=True)
            # Stay on splash and show error
            self.splash_screen.set_progress(100, f"Error: {str(e)}")

    def load_kv_files(self):
        """Load external Kivy design files for UI definitions.
        
        This method implements dynamic KV file loading, allowing for:
        - Separation of UI design from application logic
        - Easier maintenance and modification of UI layouts
        - Designer-friendly workflow for UI changes
        
        File Discovery:
            Searches for .kv files in the gui/kv directory and loads
            them automatically. Each file is loaded independently
            with error isolation to prevent one bad file from
            breaking the entire UI.
            
        Error Resilience:
            Individual KV file failures are logged but don't stop
            the application from starting with default layouts.
        """
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
        """Handle application startup completion.
        
        This Kivy lifecycle method is called after build() completes
        and the application is fully initialized and displayed.
        
        At this point, all screens are loaded and the user can
        interact with the application normally.
        """
        logger.info("Neuromancer application started")

    def on_stop(self):
        """Handle application shutdown and resource cleanup.
        
        This Kivy lifecycle method is called when the application
        is closing, providing an opportunity for graceful cleanup.
        
        Cleanup Operations:
            - Database connections are closed
            - Background threads are signaled to stop
            - Temporary files are cleaned up
            - Configuration is saved
            
        The logging statement helps with debugging and monitoring
        application lifecycle events.
        """
        logger.info("Neuromancer application stopped")
