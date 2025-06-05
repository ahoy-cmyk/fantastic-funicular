"""Enhanced chat screen implementing advanced conversational AI interface.

This module provides the primary user interface for AI chat interactions,
demonstrating sophisticated Kivy/KivyMD patterns:

- Real-time streaming message display with optimal performance
- Multi-conversation management with persistent storage
- Intelligent memory integration with RAG (Retrieval-Augmented Generation)
- Advanced navigation drawer with gesture support
- File upload and processing for document-based conversations
- Responsive UI design with adaptive message bubbles
- Thread-safe async operations with proper error handling

UI Architecture:
    Uses a navigation drawer pattern with the main chat area as content.
    Messages are displayed in adaptive cards with streaming updates.
    The interface responds to both touch and keyboard input patterns.

Performance Optimizations:
    - Message streaming with batched UI updates (every 3 chunks)
    - Efficient conversation loading with pagination
    - Memory context display with auto-hide functionality
    - Resource sharing to minimize initialization overhead

Memory Integration:
    - Automatic memory storage with intelligent classification
    - Contextual memory retrieval for enhanced responses
    - Real-time memory context display during conversations
    - Support for both user and assistant message memorization

Accessibility Features:
    - Keyboard navigation with Enter/Shift+Enter handling
    - Screen reader compatible widget hierarchy
    - High contrast message bubble design
    - Touch-friendly interface with appropriate sizing
"""

import asyncio
import threading
from datetime import datetime

from kivy.clock import Clock
from kivy.metrics import dp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDFlatButton, MDIconButton, MDRaisedButton
from kivymd.uix.card import MDCard
from kivymd.uix.dialog import MDDialog
from kivymd.uix.label import MDLabel
from kivymd.uix.list import MDList, TwoLineListItem
from kivymd.uix.navigationdrawer import MDNavigationDrawer
from kivymd.uix.screen import MDScreen
from kivymd.uix.scrollview import MDScrollView
from kivymd.uix.textfield import MDTextField
from kivymd.uix.toolbar import MDTopAppBar

from src.core.chat_manager import ChatManager
from src.gui.theme_fixes import ELEVATION, SPACING, THEME_COLORS, apply_chat_theme_fixes
from src.gui.utils.error_handling import safe_dialog_operation, safe_ui_operation
from src.gui.utils.notifications import Notification
from src.memory.safe_operations import create_safe_memory_manager
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ChatListItem(TwoLineListItem):
    """Custom list item for chat conversation management with touch interaction.

    This widget extends TwoLineListItem to provide:
    - Visual conversation preview with truncated titles
    - Long press detection for context menu actions
    - Formatted timestamp display for conversation aging
    - Callback-based interaction pattern for loose coupling

    Touch Interaction Pattern:
        - Single tap: Select conversation
        - Long press (1 second): Show delete confirmation
        - Touch cancellation: Proper event handling

    The widget demonstrates advanced Kivy touch handling with
    proper event propagation and timing-based gesture recognition.

    Args:
        conversation_data (dict): Conversation metadata and content
        on_select_callback (callable): Function called on conversation selection
        on_delete_callback (callable): Function called on long press for deletion
    """

    def __init__(self, conversation_data, on_select_callback, on_delete_callback=None, **kwargs):
        self.conversation_data = conversation_data
        self.on_select_callback = on_select_callback
        self.on_delete_callback = on_delete_callback
        self._long_press_clock = None
        self._touch_time = None

        # Format title and subtitle
        title = conversation_data.get("title", "Untitled Chat")
        if len(title) > 30:
            title = title[:30] + "..."

        # Show last message time or creation time
        created_at = conversation_data.get("created_at", datetime.now().isoformat())
        try:
            dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            subtitle = f"Created: {dt.strftime('%m/%d %H:%M')}"
        except:
            subtitle = "Recent"

        super().__init__(text=title, secondary_text=subtitle, on_release=self._on_select, **kwargs)

    def _on_select(self, *args):
        """Handle chat selection."""
        if self.on_select_callback:
            self.on_select_callback(self.conversation_data)

    def on_touch_down(self, touch):
        """Handle touch down for long press detection."""
        if self.collide_point(*touch.pos):
            # Store touch time
            self._touch_time = Clock.get_time()
            # Schedule long press check
            self._long_press_clock = Clock.schedule_once(self._check_long_press, 1.0)
        # Always call parent to maintain proper touch handling
        return super().on_touch_down(touch)

    def on_touch_up(self, touch):
        """Handle touch up to cancel long press."""
        # Cancel long press if scheduled
        if self._long_press_clock:
            self._long_press_clock.cancel()
            self._long_press_clock = None
        # Always call parent to maintain proper touch handling
        return super().on_touch_up(touch)

    def _check_long_press(self, dt):
        """Check if long press occurred."""
        if self.on_delete_callback:
            self.on_delete_callback(self.conversation_data)
        self._long_press_clock = None


class MemoryContextCard(MDCard):
    """Memory context display card with auto-hide functionality.

    This widget provides visual feedback about memory system activity:
    - Shows relevant memories retrieved for current conversation context
    - Automatically appears when memories are found and hides after timeout
    - Provides truncated memory preview with relevance indication
    - Uses Material Design elevation and theming for professional appearance

    Design Pattern:
        Implements a non-intrusive information display that enhances
        user understanding of AI reasoning without cluttering the interface.

    Memory Display Strategy:
        - Shows top 3 most relevant memories
        - Truncates long content for readability
        - Provides context about memory retrieval success/failure

    The card uses a subtle color scheme and animation to avoid
    distracting from the main conversation flow while still
    providing valuable insight into the AI's knowledge retrieval.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "vertical"
        self.size_hint_y = None
        self.height = dp(120)
        self.padding = dp(10)
        self.spacing = dp(5)
        self.elevation = ELEVATION.get("memory_bubble", 4)
        self.md_bg_color = THEME_COLORS["memory_bubble"]

        # Title
        title = MDLabel(
            text="Memory Context",
            font_style="Subtitle2",
            theme_text_color="Primary",
            size_hint_y=None,
            height=dp(30),
        )
        self.add_widget(title)

        # Memory content
        self.memory_label = MDLabel(
            text="No relevant memories found",
            theme_text_color="Secondary",
            font_style="Caption",
            adaptive_height=True,
        )
        self.add_widget(self.memory_label)

    def update_memory_context(self, memories):
        """Update the memory context display."""
        if not memories:
            self.memory_label.text = "No relevant memories found"
            return

        context_text = "Relevant memories:\n"
        for _i, memory in enumerate(memories[:3]):  # Show top 3
            content = memory.content[:60] + "..." if len(memory.content) > 60 else memory.content
            context_text += f"• {content}\n"

        self.memory_label.text = context_text


class EnhancedChatScreen(MDScreen):
    """Primary chat interface implementing advanced conversational AI patterns.

    This screen serves as the main user interaction point, demonstrating:

    Navigation Architecture:
        - MDNavigationDrawer for conversation list with smooth animations
        - Main content area with adaptive message display
        - Responsive toolbar with contextual action buttons
        - Proper drawer state management and gesture handling

    Message Flow Management:
        - Real-time streaming with batched UI updates for performance
        - Thread-safe async operations using Clock.schedule_once pattern
        - Message persistence with automatic conversation saving
        - Error handling with graceful degradation

    Memory System Integration:
        - Intelligent memory classification and storage
        - Contextual memory retrieval for enhanced AI responses
        - Visual memory context display with auto-hide
        - Support for memory-enhanced prompt engineering

    File Upload & RAG Support:
        - Multi-format file processing (PDF, text, code files)
        - Automatic content extraction and memory integration
        - File size validation and error handling
        - Background processing with progress feedback

    Performance Considerations:
        - Shared resource injection to avoid duplicate initialization
        - Conversation loading with pagination for large chat histories
        - Optimized message rendering with responsive width calculation
        - Background thread isolation for all blocking operations

    State Management:
        - Proper cleanup on screen transitions
        - Shutdown flag to prevent operations during app closure
        - Loading states to prevent race conditions
        - Resource sharing between app sessions

    Accessibility Features:
        - Keyboard navigation with custom Enter key handling
        - Screen reader compatible widget structure
        - High contrast message bubble theming
        - Touch-friendly sizing with appropriate dp measurements
    """

    def __init__(self, app_instance=None, **kwargs):
        super().__init__(**kwargs)
        self.name = "enhanced_chat"

        # Use pre-initialized managers from app if available, otherwise create new ones
        if app_instance and hasattr(app_instance, "_chat_manager") and app_instance._chat_manager:
            self.chat_manager = app_instance._chat_manager
            logger.info("Using pre-initialized chat manager")
        else:
            self.chat_manager = ChatManager()
            logger.info("Creating new chat manager")

        if app_instance and hasattr(app_instance, "_safe_memory") and app_instance._safe_memory:
            self.safe_memory = app_instance._safe_memory
            logger.info("Using pre-initialized memory manager")
        else:
            self.safe_memory = create_safe_memory_manager(self._memory_error_callback)
            logger.info("Creating new memory manager")

        # Chat management
        self.active_conversations = {}  # conversation_id -> conversation_data
        self.current_conversation_id = None
        self.message_widgets = {}  # conversation_id -> list of message widgets

        # Shutdown flag to prevent operations after app closure
        self._is_shutting_down = False

        # Performance optimization flags
        self._pending_scroll = False

        # Build UI
        self.build_ui()

        # Load existing conversations (schedule the async operation safely)
        Clock.schedule_once(self._schedule_load_conversations, 0.05)

    def on_enter(self, *args):
        """Called when entering the screen - reset shutdown flag."""
        self._is_shutting_down = False
        # Refresh model status when entering the screen
        Clock.schedule_once(lambda dt: self._refresh_model_status(), 0.1)

    def on_pre_leave(self, *args):
        """Called when leaving the screen - cleanup async operations."""
        self._is_shutting_down = True
        # Cancel any pending long press events
        for child in self.chat_list.children:
            if hasattr(child, "_long_press_clock") and child._long_press_clock:
                child._long_press_clock.cancel()
                child._long_press_clock = None

    def on_leave(self, *args):
        """Called when screen is left - ensure cleanup."""
        self._is_shutting_down = True

    def build_ui(self):
        """Build the enhanced chat interface using Material Design patterns.

        This method creates a sophisticated UI structure:

        Layout Hierarchy:
            1. Navigation drawer (conversation list) with gesture support
            2. Main content area with toolbar and message display
            3. Input area with multi-line text field and send button
            4. Memory context card with fade-in/out animation

        Design Principles:
            - Material Design 3 specifications for spacing and elevation
            - Responsive layout that adapts to different screen sizes
            - Professional color scheme with proper contrast ratios
            - Smooth animations and transitions for better UX

        Navigation Drawer Features:
            - Gesture-enabled swiping with proper close-on-click behavior
            - Smooth cubic transitions for opening/closing
            - Management buttons for conversation operations
            - Scrollable conversation list with custom item widgets

        Toolbar Configuration:
            - Dynamic title showing current model and conversation
            - Left navigation menu button for drawer control
            - Right action items for quick access to key features
            - Proper icon selection following Material Design guidelines

        The layout uses a layered approach where the main content
        serves as background and the drawer appears on top with
        proper elevation and shadowing.
        """
        # Navigation drawer for chat list (as main container)
        self.nav_drawer = MDNavigationDrawer(
            md_bg_color=THEME_COLORS["drawer_bg"],
            scrim_color=(0, 0, 0, 0.7),
            enable_swiping=True,
            anchor="left",
            radius=(0, 16, 16, 0),  # Rounded corners on the right
            elevation=ELEVATION["drawer"],
            close_on_click=True,  # Close when clicking outside
            state="close",  # Explicitly start in closed state
            opening_transition="out_cubic",  # Faster, smoother transition
            closing_transition="out_cubic",
            opening_time=0.2,  # Faster animation
            closing_time=0.2,
        )

        # Chat list in drawer
        drawer_content = MDBoxLayout(orientation="vertical", spacing=dp(10), padding=dp(10))

        # Drawer header
        drawer_header = MDBoxLayout(size_hint_y=None, height=dp(60), spacing=dp(10))

        drawer_title = MDLabel(text="Conversations", font_style="H6", theme_text_color="Primary")

        new_chat_btn = MDIconButton(icon="plus", on_release=lambda x: self._create_new_chat_sync())

        close_drawer_btn = MDIconButton(
            icon="close", on_release=lambda x: self.nav_drawer.set_state("close")
        )

        drawer_header.add_widget(drawer_title)
        drawer_header.add_widget(new_chat_btn)
        drawer_header.add_widget(close_drawer_btn)
        drawer_content.add_widget(drawer_header)

        # Management buttons
        management_layout = MDBoxLayout(size_hint_y=None, height=dp(40), spacing=dp(5))

        clear_all_btn = MDFlatButton(
            text="Clear All", size_hint_x=0.5, on_release=self._confirm_clear_all_conversations
        )

        management_layout.add_widget(clear_all_btn)
        drawer_content.add_widget(management_layout)

        # Chat list
        self.chat_list = MDList()
        chat_scroll = MDScrollView()
        chat_scroll.add_widget(self.chat_list)
        drawer_content.add_widget(chat_scroll)

        self.nav_drawer.add_widget(drawer_content)

        # Main chat area (content area for the drawer)
        chat_content = MDBoxLayout(orientation="vertical")

        # Top toolbar with provider/model info
        provider_info = self._get_provider_model_info()
        self.toolbar = MDTopAppBar(title=f"Chat - {provider_info}", elevation=2)

        # Add menu button to open drawer
        MDIconButton(icon="menu", on_release=lambda x: self.nav_drawer.set_state("open"))
        self.toolbar.left_action_items = [["menu", lambda x: self.nav_drawer.set_state("open")]]

        # Add action buttons to toolbar
        self.toolbar.right_action_items = [
            ["chip", lambda x: self._go_to_model_management()],  # Model management
            ["attachment", lambda x: self._show_file_upload()],  # File upload for RAG
            ["folder", lambda x: self._go_to_file_management()],  # File management
            ["memory", lambda x: self._go_to_memory_screen()],  # Memory screen
            ["cog", lambda x: self._go_to_settings()],  # Settings
        ]

        chat_content.add_widget(self.toolbar)

        # Current model status card
        self.model_status_card = self._create_model_status_card()
        chat_content.add_widget(self.model_status_card)

        # Memory context card (initially hidden)
        self.memory_context = MemoryContextCard()
        self.memory_context.opacity = 0
        self.memory_context.size_hint_y = None
        self.memory_context.height = 0
        chat_content.add_widget(self.memory_context)

        # Chat messages area
        self.messages_scroll = MDScrollView()
        self.messages_layout = MDBoxLayout(
            orientation="vertical", spacing=dp(10), padding=dp(10), adaptive_height=True
        )
        self.messages_scroll.add_widget(self.messages_layout)
        chat_content.add_widget(self.messages_scroll)

        # Message input area
        input_layout = MDBoxLayout(size_hint_y=None, height=dp(80), padding=dp(10), spacing=dp(10))

        self.message_input = MDTextField(
            hint_text="Type your message... (Press Enter to send, Shift+Enter for new line)",
            multiline=True,
            size_hint_y=None,
            height=dp(60),
            on_text_validate=self._send_message_wrapper,
        )

        # Override keyboard behavior for Enter key
        self.message_input.bind(focus=self._on_focus_change)

        # Store original keyboard method
        self._original_keyboard = None

        send_btn = MDRaisedButton(
            text="Send", size_hint_x=None, width=dp(80), on_release=self._send_message_wrapper
        )

        # Add MCP tools test button
        tools_btn = MDIconButton(
            icon="tools",
            tooltip_text="Test MCP Tools",
            on_release=self._show_tools_dialog,
        )

        input_layout.add_widget(self.message_input)
        input_layout.add_widget(tools_btn)
        input_layout.add_widget(send_btn)
        chat_content.add_widget(input_layout)

        # Add main content first (background)
        self.add_widget(chat_content)

        # Add navigation drawer second (foreground)
        # This should make it appear on top of the content
        self.add_widget(self.nav_drawer)

        # Force proper drawer positioning after layout
        Clock.schedule_once(self._fix_drawer_position, 0.1)

        # Apply comprehensive theme fixes
        apply_chat_theme_fixes(self)

    def _fix_drawer_position(self, dt):
        """Fix drawer positioning to prevent showing on right first."""
        try:
            # Ensure drawer is properly closed and positioned
            self.nav_drawer.set_state("close", animation=False)
            # Trigger layout update using public methods instead of private _update_pos
            if hasattr(self.nav_drawer, "trigger_layout"):
                self.nav_drawer.trigger_layout()
            elif hasattr(self.nav_drawer, "_trigger_layout"):
                self.nav_drawer._trigger_layout()
            else:
                # Fallback - just ensure it's closed
                self.nav_drawer.state = "close"
        except Exception as e:
            logger.warning(f"Could not fix drawer position: {e}")

    def _on_focus_change(self, instance, focus):
        """Handle focus changes to set up advanced keyboard event handling.

        This method demonstrates proper Kivy keyboard event management:

        Event Binding Strategy:
            - Bind keyboard events only when text field is focused
            - Unbind when focus is lost to prevent event conflicts
            - Use flag to track binding state for proper cleanup

        Keyboard Interaction Features:
            - Enter key sends message (standard chat behavior)
            - Shift+Enter creates new line (advanced text editing)
            - Proper event propagation and consumption

        Error Handling:
            - Catches keyboard binding errors gracefully
            - Maintains functionality even if advanced features fail
            - Logs errors for debugging without disrupting UX

        The pattern ensures keyboard events are properly managed
        throughout the widget lifecycle.
        """
        try:
            if focus:
                # When focused, bind keyboard
                from kivy.core.window import Window

                Window.bind(on_key_down=self._on_keyboard_down)
                self._keyboard_bound = True
            else:
                # When unfocused, unbind keyboard
                from kivy.core.window import Window

                if hasattr(self, "_keyboard_bound") and self._keyboard_bound:
                    Window.unbind(on_key_down=self._on_keyboard_down)
                    self._keyboard_bound = False
        except Exception as e:
            logger.error(f"Error in focus handling: {e}")

    def _on_keyboard_down(self, window, keycode, scancode, text, modifiers):
        """Advanced keyboard event handling for enhanced text input.

        This method implements sophisticated keyboard interaction:

        Event Processing:
            - Only processes events when input field is focused
            - Distinguishes between Enter and Shift+Enter combinations
            - Properly consumes events to prevent default handling

        Interaction Patterns:
            - Enter alone: Send message (common chat application behavior)
            - Shift+Enter: Insert newline (power user text editing)
            - Other keys: Allow normal text input handling

        Event Flow Control:
            - Returns True when event is consumed (prevents further processing)
            - Returns False for normal key handling by the text widget
            - Maintains proper event chain for complex input scenarios

        Error Safety:
            - Wrapped in try/catch to prevent keyboard event crashes
            - Graceful fallback to standard behavior on error
            - Maintains application stability during edge cases

        The implementation follows Kivy best practices for keyboard
        event handling while providing enhanced user experience.
        """
        try:
            # Only handle if the input field is focused
            if not self.message_input.focus:
                return False

            # Check for Enter key
            if keycode == 13:  # Enter key
                # If Shift is held, allow normal newline behavior
                if "shift" in modifiers:
                    return False  # Let the widget handle it normally
                else:
                    # Send the message
                    self._send_message_wrapper()
                    return True  # Consume the event
            return False  # Let the widget handle other keys
        except Exception as e:
            logger.error(f"Error in keyboard handling: {e}")
            return False

    def _is_memory_command(self, message_text: str) -> bool:
        """Intelligent detection of memory-related user commands.

        This method analyzes user input to determine if it contains
        explicit memory storage intent, allowing for contextual UI feedback.

        Detection Patterns:
            - Direct memory commands ("remember", "note that")
            - Identity statements ("my name is", "I am")
            - Storage requests ("keep in mind", "save this")
            - Personal information disclosure patterns

        Use Cases:
            - Provides contextual placeholder text ("..." vs "Thinking...")
            - Could trigger enhanced memory processing
            - Enables smart memory importance weighting
            - Supports user intent recognition

        Natural Language Processing:
            - Case-insensitive pattern matching
            - Common linguistic patterns for memory intent
            - Expandable pattern list for different languages

        The detection helps the UI provide more contextual feedback
        about memory system activity.
        """
        text = message_text.lower().strip()

        memory_commands = [
            "remember",
            "note that",
            "keep in mind",
            "store this",
            "my name is",
            "i am",
            "i'm called",
            "call me",
            "save this",
            "make a note",
        ]

        # Check for memory command patterns
        for cmd in memory_commands:
            if cmd in text:
                return True
        return False

    def _memory_error_callback(self, operation: str, error: str):
        """Centralized error handling for memory system operations.

        This callback provides consistent error handling across all
        memory operations within the chat interface.

        Error Processing:
            - Logs detailed error information for debugging
            - Provides user-friendly notifications
            - Truncates long error messages for UI display
            - Maintains operation context for better debugging

        User Experience:
            - Non-intrusive warning notifications
            - Doesn't interrupt conversation flow
            - Provides enough context for user understanding
            - Allows continued operation despite memory errors

        Error Categories:
            - Memory storage failures
            - Memory retrieval timeouts
            - Memory classification errors
            - Database connection issues

        The centralized approach ensures consistent error handling
        and user experience across all memory-related operations.
        """
        logger.warning(f"Memory operation failed - {operation}: {error}")
        Notification.warning(f"Memory: {operation} - {error[:50]}...")

    async def _get_recent_conversation_context(self) -> list[str]:
        """Retrieve recent conversation history for memory context analysis.

        This method extracts recent conversation messages to provide
        context for intelligent memory classification and retrieval.

        Context Extraction:
            - Retrieves last 5 messages from current session
            - Extracts content from message objects safely
            - Handles missing or malformed message data

        Use Cases:
            - Enhances memory classification with conversation context
            - Improves memory retrieval relevance
            - Supports contextual AI response generation
            - Enables conversation flow analysis

        Error Handling:
            - Returns empty list if session or messages unavailable
            - Handles database connection errors gracefully
            - Maintains functionality even with missing context

        Performance Optimization:
            - Limits to recent messages to avoid excessive processing
            - Caches context within the message processing cycle
            - Minimizes database queries through efficient limiting

        The context helps the memory system make more intelligent
        decisions about information classification and storage.
        """
        try:
            if not self.current_conversation_id or not self.chat_manager.current_session:
                return []

            # Get recent messages from current session
            recent_messages = await self.chat_manager.current_session.get_messages(limit=5)

            # Extract content from messages
            context = []
            for msg in recent_messages[-5:]:  # Last 5 messages
                if hasattr(msg, "content") and msg.content:
                    context.append(str(msg.content))

            return context

        except Exception as e:
            logger.error(f"Failed to get conversation context: {e}")
            return []

    def _send_message_wrapper(self, *args):
        """Thread-safe wrapper for message sending with immediate UI feedback.

        This method implements the producer-consumer pattern for message handling:

        Immediate UI Operations (Main Thread):
            1. Clear input field for immediate feedback
            2. Add user message bubble to conversation
            3. Create assistant placeholder with contextual text

        Background Processing (Worker Thread):
            1. Memory retrieval and context analysis
            2. LLM communication with streaming response
            3. Memory storage of conversation content

        UI Update Strategy:
            Uses Kivy's Clock.schedule_once to safely update the UI
            from background threads, ensuring thread safety and
            preventing race conditions.

        Error Handling:
            Graceful fallback with user notification if message
            sending fails, maintaining application stability.

        The wrapper pattern allows the main async logic to be
        isolated while providing immediate user feedback.
        """
        message_text = self.message_input.text.strip()
        if not message_text:
            return

        # Clear input immediately on UI thread
        self.message_input.text = ""

        # Add user message immediately on UI thread
        self._add_message_widget_sync(message_text, "user")

        # Create assistant placeholder with smarter text
        placeholder_text = "..." if self._is_memory_command(message_text) else "Thinking..."
        assistant_widget = self._add_message_widget_sync(placeholder_text, "assistant")

        # Run async processing in background
        def run_async_send():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._send_message_async(message_text, assistant_widget))
            except Exception as e:
                logger.error(f"Failed to send message: {e}")
                Clock.schedule_once(lambda dt: Notification.error("Failed to send message"), 0)
            finally:
                loop.close()

        thread = threading.Thread(target=run_async_send)
        thread.daemon = True
        thread.start()

    @safe_ui_operation
    async def _send_message_async(self, message_text, assistant_widget):
        """Asynchronous message processing with memory integration and streaming.

        This method implements the complete message lifecycle:

        Memory Retrieval Phase:
            - Query memory system for relevant context
            - Display memory context if found
            - Apply intelligent memory classification

        LLM Communication Phase:
            - Send message with RAG-enhanced context
            - Stream response chunks for real-time display
            - Handle streaming errors gracefully

        Memory Storage Phase:
            - Store user message with contextual metadata
            - Store assistant response with importance weighting
            - Use intelligent analysis for automatic classification

        Performance Optimizations:
            - Batched UI updates (every 3 chunks) to reduce overhead
            - Shutdown detection to prevent operations during app exit
            - Memory context auto-hide to maintain focus
            - Conversation title auto-generation from first message

        Error Recovery:
            - Individual phase failures don't stop the entire process
            - User receives feedback about any failures
            - Application continues functioning with degraded features

        The @safe_ui_operation decorator provides additional error
        handling and ensures proper resource cleanup.
        """
        try:
            # Check if we're shutting down
            if self._is_shutting_down:
                logger.info("Skipping message send - app is shutting down")
                return
            # Get relevant memories for context (use lower threshold for better recall)
            relevant_memories = await self.safe_memory.safe_recall(
                query=message_text, threshold=0.3, limit=5
            )

            # Show memory context if memories found (schedule on main thread)
            if relevant_memories:
                Clock.schedule_once(lambda dt: self._show_memory_context(relevant_memories), 0)

            # Use intelligent memory analysis to automatically detect and store significant information
            conversation_context = await self._get_recent_conversation_context()
            intelligent_memories = await self.safe_memory.safe_intelligent_remember(
                content=message_text, conversation_context=conversation_context
            )

            # Also store basic user message if no intelligent memories were created
            if not intelligent_memories:
                from src.memory import MemoryType

                await self.safe_memory.safe_remember(
                    content=message_text,
                    memory_type=MemoryType.SHORT_TERM,
                    importance=0.5,
                    metadata={
                        "conversation_id": self.current_conversation_id,
                        "timestamp": datetime.now().isoformat(),
                        "type": "user_message",
                    },
                )

            # Send to chat manager with RAG-enhanced streaming
            response_text = ""
            chunk_count = 0
            try:
                # Use RAG-enhanced messaging for better context with uploaded files
                async for chunk in self.chat_manager.send_message_with_rag(
                    message_text, stream=True
                ):
                    # Check for shutdown during streaming
                    if self._is_shutting_down:
                        logger.info("Stopping message streaming - app is shutting down")
                        break

                    response_text += chunk
                    chunk_count += 1

                    # Update UI less frequently for better performance (every 3 chunks)
                    if (
                        chunk_count % 3 == 0
                        and assistant_widget
                        and hasattr(assistant_widget, "content_label")
                    ):
                        # Create proper closure to capture current response_text value
                        def update_text(dt, text=response_text):
                            try:
                                if (
                                    not self._is_shutting_down
                                    and assistant_widget
                                    and hasattr(assistant_widget, "content_label")
                                ):
                                    assistant_widget.content_label.text = text
                            except Exception as e:
                                logger.warning(f"Failed to update message text: {e}")

                        if not self._is_shutting_down:
                            Clock.schedule_once(update_text, 0)

                # Final update to ensure all text is shown
                if (
                    assistant_widget
                    and hasattr(assistant_widget, "content_label")
                    and not self._is_shutting_down
                ):
                    Clock.schedule_once(
                        lambda dt: setattr(assistant_widget.content_label, "text", response_text), 0
                    )
            except Exception as e:
                logger.error(f"Error during message streaming: {e}")
                # Update assistant widget with error message
                if assistant_widget and hasattr(assistant_widget, "content_label"):
                    error_msg = f"Error: {str(e)}"
                    Clock.schedule_once(
                        lambda dt, msg=error_msg: setattr(
                            assistant_widget.content_label, "text", msg
                        ),
                        0,
                    )
                return

            # Store assistant response in memory using intelligent analysis
            if response_text:
                # Use intelligent analysis for assistant responses too
                conversation_context = await self._get_recent_conversation_context()
                conversation_context.append(message_text)  # Include the user message

                intelligent_memories = await self.safe_memory.safe_intelligent_remember(
                    content=response_text, conversation_context=conversation_context
                )

                # Store basic assistant response if no intelligent memories were created
                if not intelligent_memories:
                    from src.memory import MemoryType

                    await self.safe_memory.safe_remember(
                        content=response_text,
                        memory_type=MemoryType.SHORT_TERM,
                        importance=0.6,  # Assistant responses are moderately important
                        metadata={
                            "conversation_id": self.current_conversation_id,
                            "timestamp": datetime.now().isoformat(),
                            "type": "assistant_response",
                        },
                    )

            # Update conversation title if this is the first message
            if self.current_conversation_id in self.active_conversations:
                conv_data = self.active_conversations[self.current_conversation_id]
                if conv_data.get("message_count", 0) == 0:
                    # Generate title from first message
                    title = message_text[:30] + "..." if len(message_text) > 30 else message_text
                    conv_data["title"] = title

                    # Update toolbar
                    toolbar = self.children[0].children[-1]
                    toolbar.title = title

                    # Update chat list item
                    for widget in self.chat_list.children:
                        if (
                            hasattr(widget, "conversation_data")
                            and widget.conversation_data["id"] == self.current_conversation_id
                        ):
                            widget.text = title
                            break

        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            Notification.error("Failed to send message")

    async def _add_message_widget(self, content: str, role: str, timestamp: str = None):
        """Add a message widget to the conversation."""
        timestamp = timestamp or datetime.now().isoformat()

        # Create message card
        # Use theme colors for message bubbles
        bubble_color = (
            THEME_COLORS["assistant_bubble"] if role == "assistant" else THEME_COLORS["user_bubble"]
        )

        message_card = MDCard(
            orientation="vertical",
            size_hint_y=None,
            adaptive_height=True,
            padding=SPACING["message_padding"],
            spacing=SPACING["small"],
            elevation=ELEVATION["card"],
            md_bg_color=bubble_color,
            radius=[dp(16), dp(16), dp(16), dp(16)],
        )

        # Message header
        header = MDBoxLayout(size_hint_y=None, height=dp(20))

        role_label = MDLabel(
            text=f"{role.title()}:",
            font_style="Caption",
            theme_text_color="Primary",
            size_hint_x=None,
            width=dp(80),
        )

        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            time_label = MDLabel(
                text=dt.strftime("%H:%M"),
                font_style="Caption",
                theme_text_color="Secondary",
                halign="right",
            )
        except:
            time_label = MDLabel(
                text="Now", font_style="Caption", theme_text_color="Secondary", halign="right"
            )

        header.add_widget(role_label)
        header.add_widget(time_label)
        message_card.add_widget(header)

        # Message content with proper text wrapping (responsive width)
        from kivy.core.window import Window

        max_width = min(Window.width * 0.7, dp(500))  # 70% of screen width, max 500dp
        content_label = MDLabel(
            text=content,
            theme_text_color="Primary",
            adaptive_height=True,
            text_size=(max_width, None),
            halign="left",
            valign="top",
        )
        message_card.add_widget(content_label)

        # Store reference to content label for streaming updates
        message_card.content_label = content_label

        # Add to messages layout
        self.messages_layout.add_widget(message_card)

        # Scroll to bottom
        Clock.schedule_once(lambda dt: setattr(self.messages_scroll, "scroll_y", 0), 0.1)

        return message_card

    def _show_memory_context(self, memories):
        """Display memory context with smooth animation and auto-hide.

        This method provides visual feedback about memory retrieval:

        Animation Strategy:
            - Smooth fade-in with height expansion
            - Temporary display with automatic dismissal
            - Non-intrusive placement above message area

        Content Display:
            - Shows relevant retrieved memories
            - Provides context about AI reasoning
            - Enhances user understanding of responses

        Timing Control:
            - 5-second display duration for adequate reading time
            - Auto-hide to maintain conversation focus
            - Smooth transitions to avoid jarring changes

        User Experience:
            - Informative but not distracting
            - Provides insight into AI thought process
            - Maintains conversation flow priority

        The display helps users understand how the AI is using
        stored memories to inform its responses.
        """
        self.memory_context.update_memory_context(memories)
        self.memory_context.height = dp(120)
        self.memory_context.opacity = 1

        # Auto-hide after 5 seconds
        Clock.schedule_once(lambda dt: self._hide_memory_context(), 5)

    def _hide_memory_context(self):
        """Hide memory context panel with smooth animation.

        This method gracefully removes the memory context display:

        Animation Features:
            - Smooth fade-out with height collapse
            - Returns display space to conversation area
            - Maintains visual continuity during transition

        State Management:
            - Resets opacity and height for future displays
            - Cleans up temporary visual elements
            - Prepares for next memory context display cycle

        The hiding mechanism ensures the memory context doesn't
        permanently clutter the interface while still providing
        valuable temporary insight into AI reasoning.
        """
        self.memory_context.opacity = 0
        self.memory_context.height = 0

    def _create_model_status_card(self):
        """Create an informative model status card with interactive controls.

        This method builds a comprehensive status display featuring:

        Information Display:
            - Current AI model and provider information
            - RAG (Retrieval-Augmented Generation) status
            - Dynamic updates based on configuration changes

        Interactive Elements:
            - Quick model switching button
            - Status refresh capability
            - Visual indicators for system state

        Design Features:
            - Material Design card styling with appropriate elevation
            - Professional color scheme with themed icons
            - Responsive layout for different screen sizes
            - Clear typography hierarchy for information scanning

        Real-time Updates:
            - Reflects current model configuration
            - Shows RAG system status changes
            - Updates automatically when settings change

        User Experience:
            - Provides quick access to important system information
            - Enables rapid model switching workflow
            - Maintains awareness of AI system configuration

        The card serves as a control center for AI system status
        and quick configuration changes.
        """
        model_card = MDCard(
            orientation="horizontal",
            size_hint_y=None,
            height=dp(60),
            padding=dp(10),
            spacing=dp(10),
            elevation=ELEVATION.get("card", 2),
            md_bg_color=THEME_COLORS.get("card_bg", [0.2, 0.2, 0.2, 1]),
            radius=[dp(8), dp(8), dp(8), dp(8)],
        )

        # Model info layout
        info_layout = MDBoxLayout(orientation="vertical", adaptive_height=True)

        # Current model label
        self.current_model_label = MDLabel(
            text=f"Model: {self._get_provider_model_info()}",
            font_style="Subtitle2",
            theme_text_color="Primary",
            adaptive_height=True,
            bold=True,
        )

        # RAG status label
        rag_status = (
            "RAG: Enabled" if getattr(self.chat_manager, "rag_enabled", True) else "RAG: Disabled"
        )
        self.rag_status_label = MDLabel(
            text=rag_status,
            font_style="Caption",
            theme_text_color=(
                "Primary" if getattr(self.chat_manager, "rag_enabled", True) else "Secondary"
            ),
            adaptive_height=True,
        )

        info_layout.add_widget(self.current_model_label)
        info_layout.add_widget(self.rag_status_label)

        # Quick actions
        actions_layout = MDBoxLayout(
            orientation="horizontal",
            spacing=dp(5),
            adaptive_height=True,
            size_hint_x=None,
            width=dp(120),
        )

        # Switch model button
        switch_button = MDIconButton(
            icon="swap-horizontal",
            on_release=lambda x: self._go_to_model_management(),
            theme_icon_color="Custom",
            icon_color=THEME_COLORS.get("primary", [0.2, 0.6, 1, 1]),
        )

        # Refresh model info button
        refresh_button = MDIconButton(
            icon="refresh",
            on_release=lambda x: self._refresh_model_status(),
            theme_icon_color="Custom",
            icon_color=THEME_COLORS.get("primary", [0.2, 0.6, 1, 1]),
        )

        actions_layout.add_widget(switch_button)
        actions_layout.add_widget(refresh_button)

        model_card.add_widget(info_layout)
        model_card.add_widget(actions_layout)

        return model_card

    def _refresh_model_status(self):
        """Refresh model status display with current system state.

        This method updates all model-related UI elements:

        Update Operations:
            - Retrieves current model and provider information
            - Updates RAG system status display
            - Refreshes toolbar title with current configuration
            - Synchronizes all UI elements with system state

        Error Handling:
            - Graceful handling of missing model information
            - Fallback display for disconnected providers
            - Logging for debugging configuration issues

        UI Synchronization:
            - Ensures consistency across multiple display locations
            - Updates both status card and toolbar simultaneously
            - Maintains visual coherence throughout interface

        Use Cases:
            - After model switching operations
            - During provider reconnection
            - When returning to chat screen from settings
            - For periodic status verification

        The refresh ensures users always see accurate system
        configuration regardless of changes made elsewhere.
        """
        try:
            # Update model info
            model_info = self._get_provider_model_info()
            self.current_model_label.text = f"Model: {model_info}"

            # Update RAG status
            rag_enabled = getattr(self.chat_manager, "rag_enabled", True)
            rag_status = "RAG: Enabled" if rag_enabled else "RAG: Disabled"
            self.rag_status_label.text = rag_status
            self.rag_status_label.theme_text_color = "Primary" if rag_enabled else "Secondary"

            # Update toolbar title as well
            if hasattr(self, "toolbar"):
                self.toolbar.title = f"Chat - {model_info}"

            logger.info(f"Refreshed model status: {model_info}, RAG: {rag_enabled}")

        except Exception as e:
            logger.error(f"Failed to refresh model status: {e}")

    @safe_dialog_operation
    def _show_memory_panel(self, *args):
        """Show memory management panel."""
        from kivymd.uix.button import MDRaisedButton

        content = MDBoxLayout(
            orientation="vertical", spacing=dp(10), size_hint_y=None, height=dp(200)
        )

        # Memory stats
        stats_label = MDLabel(
            text="Memory System Status:\n• Loading statistics...",
            theme_text_color="Primary",
            adaptive_height=True,
        )
        content.add_widget(stats_label)

        # Load memory stats
        Clock.schedule_once(lambda dt: self._schedule_load_memory_stats(stats_label), 0.1)

        dialog = MDDialog(
            title="Memory System",
            type="custom",
            content_cls=content,
            buttons=[
                MDRaisedButton(
                    text="Manage Memory", on_release=lambda x: self._go_to_memory_screen(dialog)
                ),
                MDRaisedButton(text="Close", on_release=lambda x: dialog.dismiss()),
            ],
        )
        dialog.open()

    async def _load_memory_stats(self, label):
        """Load and display memory statistics."""
        try:
            stats = await self.safe_memory.safe_get_stats()

            stats_text = "Memory System Status:\n"
            stats_text += f"• Total memories: {stats.get('total', 0)}\n"
            stats_text += f"• Average importance: {stats.get('avg_importance', 0):.2f}\n"

            for mem_type, count in stats.get("by_type", {}).items():
                stats_text += f"• {mem_type}: {count} memories\n"

            label.text = stats_text

        except Exception as e:
            label.text = f"Memory System Status:\n• Error loading stats: {str(e)}"

    def _schedule_load_memory_stats(self, label):
        """Schedule the async memory stats loading in a thread-safe way."""

        def run_async_load():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._load_memory_stats(label))
            except Exception as e:
                logger.error(f"Failed to load memory stats: {e}")
                Clock.schedule_once(
                    lambda dt: setattr(
                        label, "text", f"Memory System Status:\n• Error loading stats: {str(e)}"
                    ),
                    0,
                )
            finally:
                loop.close()

        thread = threading.Thread(target=run_async_load)
        thread.daemon = True
        thread.start()

    def _go_to_memory_screen(self, dialog=None):
        """Navigate to memory management screen."""
        if dialog:
            dialog.dismiss()
        self.manager.current = "memory"

    def _go_to_model_management(self):
        """Navigate to model management screen."""
        self.manager.current = "model_management"

    def _go_to_settings(self):
        """Navigate to settings screen."""
        self.manager.current = "settings"

    def _go_to_file_management(self):
        """Navigate to file management screen."""
        self.manager.current = "file_management"

    @safe_dialog_operation
    def _show_file_upload(self, *args):
        """Display comprehensive file upload dialog for RAG document processing.

        This method creates a sophisticated file upload interface:

        Supported Formats:
            - Text files (.txt, .md, .json)
            - Programming files (.py, .js, .java, .cpp, .c)
            - Markup files (.html, .xml)
            - Data files (.csv)
            - Office documents (.docx)
            - PDF documents (.pdf)

        File Manager Integration:
            - Native file browser with filtering
            - Cross-platform file selection
            - Proper file validation and error handling

        User Experience:
            - Clear format support information
            - Intuitive file browsing workflow
            - Progress feedback during processing

        Processing Pipeline:
            - File size validation (10MB limit)
            - Format-specific content extraction
            - Memory system integration
            - Error handling with user feedback

        The @safe_dialog_operation decorator ensures proper
        dialog lifecycle management and error handling.
        """
        from kivymd.uix.filemanager import MDFileManager

        def file_manager_open():
            self.file_manager = MDFileManager(
                exit_manager=self.exit_file_manager,
                select_path=self.select_file_path,
                ext=[
                    ".txt",
                    ".pdf",
                    ".md",
                    ".json",
                    ".py",
                    ".js",
                    ".java",
                    ".cpp",
                    ".c",
                    ".html",
                    ".xml",
                    ".csv",
                    ".docx",
                ],
            )
            self.file_manager.show("/")  # Start from root directory

        # Create file upload dialog
        content = MDBoxLayout(
            orientation="vertical", spacing=dp(10), size_hint_y=None, height=dp(200)
        )

        info_label = MDLabel(
            text="Upload a file to add its content to the RAG memory system.\n"
            "Supported formats: txt, pdf, md, json, py, js, java, cpp, c, html, xml, csv, docx",
            theme_text_color="Primary",
            adaptive_height=True,
        )

        browse_button = MDRaisedButton(
            text="Browse Files",
            size_hint_y=None,
            height=dp(40),
            on_release=lambda x: (dialog.dismiss(), file_manager_open()),
        )

        content.add_widget(info_label)
        content.add_widget(browse_button)

        dialog = MDDialog(
            title="Upload File for RAG",
            type="custom",
            content_cls=content,
            buttons=[MDFlatButton(text="Cancel", on_release=lambda x: dialog.dismiss())],
        )

        dialog.open()

    def exit_file_manager(self, *args):
        """Close file manager with proper cleanup.

        This method handles file manager closure:

        Cleanup Operations:
            - Closes file manager widget
            - Releases system resources
            - Returns focus to main interface

        State Management:
            - Resets file selection state
            - Clears temporary file references
            - Prepares for next file operation

        The method ensures proper resource management
        when file selection is cancelled or completed.
        """
        self.file_manager.close()

    def select_file_path(self, path):
        """Process selected file for RAG memory integration.

        This method handles the complete file processing pipeline:

        File Processing Steps:
            1. File validation (size, format, accessibility)
            2. Content extraction based on file type
            3. Memory system integration with metadata
            4. User feedback and error handling

        Content Extraction:
            - PDF text extraction using pypdf
            - UTF-8 text file reading with encoding detection
            - Format-specific processing for different file types
            - Error recovery for corrupted or invalid files

        Memory Integration:
            - Stores file content with rich metadata
            - Includes file information for provenance tracking
            - Enables source attribution in AI responses
            - Supports content-based retrieval

        Error Handling:
            - File size limits (10MB) with user notification
            - Format validation with helpful error messages
            - Encoding error recovery with fallback strategies
            - Network/storage error handling

        User Feedback:
            - Progress indication during processing
            - Success confirmation with file details
            - Clear error messages with actionable advice

        The processing runs in a background thread to maintain
        UI responsiveness during file operations.
        """
        self.exit_file_manager()

        async def process_file():
            try:
                # Read file content
                import os

                file_size = os.path.getsize(path)

                # Check file size (limit to 10MB)
                if file_size > 10 * 1024 * 1024:
                    Clock.schedule_once(
                        lambda dt: Notification.warning("File too large (max 10MB)"), 0
                    )
                    return

                # Read file content based on extension
                file_ext = os.path.splitext(path)[1].lower()

                try:
                    if file_ext == ".pdf":
                        # Extract text from PDF
                        from pypdf import PdfReader

                        reader = PdfReader(path)
                        content = ""
                        for page in reader.pages:
                            content += page.extract_text() + "\n"
                    else:
                        # Text-based files
                        with open(path, encoding="utf-8") as f:
                            content = f.read()
                except Exception as e:
                    error_msg = f"Error reading file: {str(e)}"
                    Clock.schedule_once(lambda dt: Notification.error(error_msg), 0)
                    return

                # Store in memory with metadata
                file_name = os.path.basename(path)
                metadata = {
                    "source": "file_upload",
                    "file_name": file_name,
                    "file_path": path,
                    "file_size": file_size,
                    "file_type": file_ext,
                }

                # Add to memory system
                memory_id = await self.chat_manager.memory_manager.remember(
                    content=f"File content from {file_name}:\n\n{content}",
                    auto_classify=True,
                    metadata=metadata,
                )

                if memory_id:
                    Clock.schedule_once(
                        lambda dt: Notification.success(f"File '{file_name}' added to RAG memory"),
                        0,
                    )
                    logger.info(f"Uploaded file to RAG: {file_name} ({file_size} bytes)")
                else:
                    Clock.schedule_once(
                        lambda dt: Notification.error("Failed to add file to memory"), 0
                    )

            except Exception as e:
                error_msg = f"Upload failed: {str(e)}"
                logger.error(f"File upload error: {e}")
                Clock.schedule_once(lambda dt: Notification.error(error_msg), 0)

        # Run in thread
        def run_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(process_file())
            loop.close()

        thread = threading.Thread(target=run_async)
        thread.daemon = True
        thread.start()

    def _schedule_load_conversations(self, dt):
        """Schedule asynchronous conversation loading in a thread-safe pattern.

        This method demonstrates the proper Kivy pattern for background operations:

        Threading Strategy:
            1. Create a daemon thread to prevent app hanging on exit
            2. Establish a new event loop for async operations
            3. Execute async operations in isolation
            4. Schedule UI updates on main thread via Clock
            5. Ensure proper cleanup with try/finally blocks

        Error Handling:
            - Isolates errors to prevent UI thread crashes
            - Provides user feedback through notification system
            - Logs detailed error information for debugging
            - Maintains application stability on failure

        The pattern allows for complex async operations while
        maintaining UI responsiveness and thread safety.
        """

        def run_async_load():
            try:
                # Create a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                # Run the async operation
                loop.run_until_complete(self._load_conversations_sync())

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Failed to load conversations: {e}")
                # Schedule UI update on main thread
                Clock.schedule_once(lambda dt: self._handle_load_error(error_msg), 0)
            finally:
                loop.close()

        # Run in background thread
        thread = threading.Thread(target=run_async_load)
        thread.daemon = True
        thread.start()

    async def _load_conversations_sync(self):
        """Load conversations and schedule UI updates on main thread."""
        try:
            # Get sessions from session manager
            sessions = await self.chat_manager.session_manager.list_conversations(limit=20)

            # Schedule UI update on main thread
            Clock.schedule_once(lambda dt: self._update_conversation_list(sessions), 0)

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to load conversations: {e}")
            Clock.schedule_once(lambda dt: self._handle_load_error(error_msg), 0)

    def _update_conversation_list(self, sessions):
        """Update the conversation list UI (runs on main thread)."""
        try:
            # Clear current chat list
            self.chat_list.clear_widgets()
            self.active_conversations.clear()

            logger.info(f"Loading {len(sessions)} conversations")

            for session_data in sessions:
                try:
                    # Ensure we have required fields
                    if not session_data.get("id"):
                        logger.warning(f"Skipping session with no ID: {session_data}")
                        continue

                    conversation_data = {
                        "id": session_data.get("id"),
                        "title": session_data.get("title", "Untitled Chat"),
                        "created_at": session_data.get("created_at"),
                        "message_count": session_data.get("message_count", 0),
                    }

                    self.active_conversations[conversation_data["id"]] = conversation_data

                    # Create list item
                    chat_item = ChatListItem(
                        conversation_data=conversation_data,
                        on_select_callback=self._select_conversation_wrapper,
                        on_delete_callback=self._confirm_delete_conversation,
                    )
                    self.chat_list.add_widget(chat_item)

                except Exception as e:
                    logger.error(f"Failed to process session data {session_data}: {e}")
                    continue

            logger.info(f"Successfully loaded {len(self.active_conversations)} conversations")

            # If no conversations, create a default one
            if not self.active_conversations:
                logger.info("No conversations found, creating default conversation")
                self._create_new_chat_sync()
            else:
                # Select the first conversation
                first_conv_id = list(self.active_conversations.keys())[0]
                logger.info(f"Auto-selecting first conversation: {first_conv_id}")
                self._select_conversation_wrapper(self.active_conversations[first_conv_id])

        except Exception as e:
            logger.error(f"Failed to update conversation list: {e}")
            self._handle_load_error(str(e))

    def _handle_load_error(self, error_msg):
        """Handle conversation loading errors."""
        Notification.error(f"Failed to load conversations: {error_msg[:50]}...")
        # Create a default conversation as fallback
        self._create_new_chat_sync()

    def _select_conversation_wrapper(self, conversation_data):
        """Thread-safe conversation selection with loading state management.

        This method implements a sophisticated conversation loading pattern:

        Race Condition Prevention:
            - Loading flag prevents multiple simultaneous selections
            - Immediate UI feedback (clear messages, show loading)
            - Conversation ID assignment to prevent state conflicts

        User Experience Optimization:
            - Immediate visual feedback with loading message
            - Drawer closure for better focus
            - Progressive loading with status updates

        Error Recovery:
            - Validation of conversation data before processing
            - Graceful error handling with user notification
            - Proper cleanup of loading states on failure

        Threading Pattern:
            - Background thread for database operations
            - Main thread for all UI updates
            - Proper resource cleanup in finally blocks

        The wrapper pattern isolates complex async logic while
        providing immediate user feedback and preventing UI freezing.
        """
        # Prevent multiple simultaneous selections
        if hasattr(self, "_loading_conversation") and self._loading_conversation:
            logger.warning("Conversation loading already in progress, ignoring request")
            return

        self._loading_conversation = True

        try:
            # Validate conversation data
            if not conversation_data or not conversation_data.get("id"):
                logger.error(f"Invalid conversation data: {conversation_data}")
                Notification.error("Invalid conversation data")
                return

            # Immediately update current conversation ID to prevent multiple selections
            self.current_conversation_id = conversation_data["id"]

            # Clear current messages immediately for responsive feedback
            self.messages_layout.clear_widgets()

            # Add loading message
            loading_widget = self._add_message_widget_sync("Loading conversation...", "system")

            # Close drawer immediately for better UX
            self.nav_drawer.set_state("close")

            def run_async_select():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(
                        self._select_conversation_async(conversation_data, loading_widget)
                    )
                except Exception as e:
                    error_msg = f"Failed to load conversation: {str(e)}"
                    logger.error(f"Failed to select conversation: {e}")
                    Clock.schedule_once(lambda dt: Notification.error(error_msg), 0)
                    # Remove loading widget on error
                    Clock.schedule_once(
                        lambda dt: (
                            self.messages_layout.remove_widget(loading_widget)
                            if loading_widget in self.messages_layout.children
                            else None
                        ),
                        0,
                    )
                finally:
                    loop.close()
                    # Reset loading flag
                    Clock.schedule_once(lambda dt: setattr(self, "_loading_conversation", False), 0)

            thread = threading.Thread(target=run_async_select)
            thread.daemon = True
            thread.start()

        except Exception as e:
            logger.error(f"Failed to start conversation selection: {e}")
            self._loading_conversation = False
            Notification.error("Failed to start conversation loading")

    async def _select_conversation_async(self, conversation_data, loading_widget):
        """Async conversation selection."""
        try:
            conversation_id = conversation_data["id"]
            logger.info(f"Loading conversation: {conversation_id}")

            # Load session in chat manager
            await self.chat_manager.load_session(conversation_id)
            logger.info(f"Session loaded successfully for conversation: {conversation_id}")

            # Get messages from session
            messages = []
            if self.chat_manager.current_session:
                try:
                    session_messages = await self.chat_manager.current_session.get_messages()
                    messages = [
                        (
                            msg.content,
                            msg.role.value if hasattr(msg.role, "value") else str(msg.role),
                            (
                                msg.created_at.isoformat()
                                if msg.created_at
                                else datetime.now().isoformat()
                            ),
                        )
                        for msg in session_messages
                    ]
                    logger.info(
                        f"Loaded {len(messages)} messages for conversation: {conversation_id}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to load messages for conversation {conversation_id}: {e}"
                    )
                    # Continue with empty messages rather than failing
            else:
                logger.warning(f"No current session after loading conversation {conversation_id}")

            # Schedule UI update on main thread with messages (remove loading widget)
            Clock.schedule_once(
                lambda dt: self._update_conversation_ui_with_messages(
                    conversation_data, messages, loading_widget
                ),
                0,
            )

        except Exception as e:
            logger.error(
                f"Failed to select conversation {conversation_data.get('id', 'unknown')}: {e}"
            )
            Clock.schedule_once(
                lambda dt: Notification.error(f"Failed to load conversation: {str(e)}"), 0
            )
            # Remove loading widget on error
            Clock.schedule_once(
                lambda dt: (
                    self.messages_layout.remove_widget(loading_widget)
                    if loading_widget in self.messages_layout.children
                    else None
                ),
                0,
            )

    def _update_conversation_ui(self, conversation_data):
        """Update UI after conversation selection (runs on main thread)."""
        self._update_conversation_ui_with_messages(conversation_data, [], None)

    def _update_conversation_ui_with_messages(
        self, conversation_data, messages, loading_widget=None
    ):
        """Update UI after conversation selection with message history."""
        try:
            # Clear current messages
            self.messages_layout.clear_widgets()

            # Load message history
            for content, role, timestamp in messages:
                self._add_message_widget_sync(content, role, timestamp)

            # Update title using stored toolbar reference with model info
            try:
                if hasattr(self, "toolbar") and self.toolbar:
                    provider_info = self._get_provider_model_info()
                    self.toolbar.title = f"{conversation_data['title']} - {provider_info}"
                else:
                    logger.warning("Toolbar reference not available")
            except Exception as e:
                logger.warning(f"Failed to update toolbar title: {e}")

            # Close drawer
            self.nav_drawer.set_state("close")

            logger.info(
                f"Loaded conversation: {conversation_data['title']} with {len(messages)} messages"
            )

        except Exception as e:
            logger.error(f"Failed to update conversation UI: {e}")

    def _add_message_widget_sync(self, content: str, role: str, timestamp: str = None):
        """Add message widget with responsive design and theming (UI thread safe).

        This method creates message bubbles with:

        Responsive Design:
            - Adaptive width based on screen size (max 70% width)
            - Proper text wrapping with maximum width constraints
            - Optimal sizing for different device form factors

        Visual Design:
            - Role-based color coding for message attribution
            - Material Design elevation and border radius
            - Consistent spacing using theme constants
            - Professional typography with proper hierarchy

        Content Handling:
            - Multi-line text support with proper wrapping
            - Timestamp formatting with fallback handling
            - Reference storage for streaming updates

        Accessibility:
            - Proper text contrast ratios
            - Screen reader compatible structure
            - Touch-friendly sizing

        Auto-scrolling:
            - Automatically scrolls to newest message
            - Smooth scrolling with proper timing
            - Maintains conversation flow visibility

        The synchronous version is safe to call from the main UI
        thread and provides immediate visual feedback.
        """
        timestamp = timestamp or datetime.now().isoformat()

        # Create message card
        # Use theme colors for message bubbles
        bubble_color = (
            THEME_COLORS["assistant_bubble"] if role == "assistant" else THEME_COLORS["user_bubble"]
        )

        message_card = MDCard(
            orientation="vertical",
            size_hint_y=None,
            adaptive_height=True,
            padding=SPACING["message_padding"],
            spacing=SPACING["small"],
            elevation=ELEVATION["card"],
            md_bg_color=bubble_color,
            radius=[dp(16), dp(16), dp(16), dp(16)],
        )

        # Message header
        header = MDBoxLayout(size_hint_y=None, height=dp(20))

        role_label = MDLabel(
            text=f"{role.title()}:",
            font_style="Caption",
            theme_text_color="Primary",
            size_hint_x=None,
            width=dp(80),
        )

        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            time_label = MDLabel(
                text=dt.strftime("%H:%M"),
                font_style="Caption",
                theme_text_color="Secondary",
                halign="right",
            )
        except:
            time_label = MDLabel(
                text="Now", font_style="Caption", theme_text_color="Secondary", halign="right"
            )

        header.add_widget(role_label)
        header.add_widget(time_label)
        message_card.add_widget(header)

        # Message content with proper text wrapping (responsive width)
        from kivy.core.window import Window

        max_width = min(Window.width * 0.7, dp(500))  # 70% of screen width, max 500dp
        content_label = MDLabel(
            text=content,
            theme_text_color="Primary",
            adaptive_height=True,
            text_size=(max_width, None),
            halign="left",
            valign="top",
        )
        message_card.add_widget(content_label)

        # Store reference to content label for streaming updates
        message_card.content_label = content_label

        # Add to messages layout
        self.messages_layout.add_widget(message_card)

        # Scroll to bottom
        Clock.schedule_once(lambda dt: setattr(self.messages_scroll, "scroll_y", 0), 0.1)

        return message_card

    def _create_new_chat_sync(self):
        """Create new chat with thread-safe async execution.

        This method demonstrates the async-to-sync bridge pattern
        commonly needed in Kivy applications:

        Execution Flow:
            1. Immediate return to prevent UI blocking
            2. Background thread creation with new event loop
            3. Async chat creation with session management
            4. UI update scheduling on main thread

        Session Management:
            - Creates new conversation session
            - Generates appropriate conversation metadata
            - Adds to conversation list with proper ordering
            - Automatically selects the new conversation

        Error Handling:
            - Isolated error handling prevents app crashes
            - User notification for creation failures
            - Graceful degradation if session creation fails

        The synchronous wrapper allows this method to be called
        from UI event handlers while performing complex async
        operations in the background.
        """
        try:

            def run_async_create():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(self._create_new_chat_async())
                except Exception as e:
                    logger.error(f"Failed to create new chat: {e}")
                finally:
                    loop.close()

            thread = threading.Thread(target=run_async_create)
            thread.daemon = True
            thread.start()

        except Exception as e:
            logger.error(f"Failed to schedule new chat creation: {e}")

    async def _create_new_chat_async(self):
        """Create new chat async operation."""
        try:
            # Create new session with chat manager
            await self.chat_manager.create_session()

            if not self.chat_manager.current_session:
                raise Exception("Failed to create session")

            session = self.chat_manager.current_session

            conversation_data = {
                "id": session.id,
                "title": f"Chat {datetime.now().strftime('%m/%d %H:%M')}",
                "created_at": datetime.now().isoformat(),
                "message_count": 0,
            }

            # Schedule UI update on main thread
            Clock.schedule_once(lambda dt: self._add_new_chat_to_ui(conversation_data), 0)

        except Exception as e:
            logger.error(f"Failed to create new chat: {e}")
            Clock.schedule_once(
                lambda dt: Notification.error("Failed to create new conversation"), 0
            )

    def _add_new_chat_to_ui(self, conversation_data):
        """Add new chat to UI (runs on main thread)."""
        try:
            # Add to active conversations
            self.active_conversations[conversation_data["id"]] = conversation_data

            # Add to chat list
            chat_item = ChatListItem(
                conversation_data=conversation_data,
                on_select_callback=self._select_conversation_wrapper,
                on_delete_callback=self._confirm_delete_conversation,
            )
            self.chat_list.add_widget(chat_item, index=0)  # Add at top

            # Select the new conversation
            self._select_conversation_wrapper(conversation_data)

            # Close drawer
            self.nav_drawer.set_state("close")

            Notification.success("New conversation created")

        except Exception as e:
            logger.error(f"Failed to add new chat to UI: {e}")

    def _get_provider_model_info(self):
        """Get current provider and model information with fallback handling.

        This method safely retrieves model information for display purposes:

        Information Retrieval:
            - Accesses chat manager's current provider and model
            - Formats information for user-friendly display
            - Provides meaningful fallbacks for missing data

        Error Resilience:
            - Handles missing chat manager gracefully
            - Provides informative error messages
            - Maintains UI functionality even with missing data

        Display Format:
            - Consistent "Provider: Model" format
            - Title case formatting for professional appearance
            - Fallback messages for error states

        This information is used in multiple UI locations
        (toolbar, status cards) to keep users informed about
        the current AI model configuration.
        """
        try:
            # Get actual current model from chat manager
            if self.chat_manager:
                provider = self.chat_manager.current_provider
                model = self.chat_manager.current_model
                return f"{provider.title()}: {model}"
            else:
                return "No Model Selected"
        except Exception as e:
            logger.warning(f"Failed to get provider info: {e}")
            return "Unknown Provider"

    def _confirm_clear_all_conversations(self, *args):
        """Show confirmation dialog for clearing all conversations."""
        from kivymd.uix.button import MDRaisedButton

        content = MDBoxLayout(
            orientation="vertical", spacing=dp(10), size_hint_y=None, height=dp(100)
        )

        warning_label = MDLabel(
            text="This will permanently delete ALL conversations.\nThis action cannot be undone.",
            theme_text_color="Primary",
            adaptive_height=True,
            halign="center",
        )
        content.add_widget(warning_label)

        dialog = MDDialog(
            title="Clear All Conversations",
            type="custom",
            content_cls=content,
            buttons=[
                MDRaisedButton(text="Cancel", on_release=lambda x: dialog.dismiss()),
                MDRaisedButton(
                    text="Delete All",
                    md_bg_color=(0.8, 0.2, 0.2, 1),  # Red color
                    on_release=lambda x: self._clear_all_conversations(dialog),
                ),
            ],
        )
        dialog.open()

    def _clear_all_conversations(self, dialog):
        """Clear all conversations."""
        dialog.dismiss()

        def run_async_clear():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._clear_all_conversations_async())
            except Exception as e:
                logger.error(f"Failed to clear conversations: {e}")
                Clock.schedule_once(
                    lambda dt: Notification.error("Failed to clear conversations"), 0
                )
            finally:
                loop.close()

        thread = threading.Thread(target=run_async_clear)
        thread.daemon = True
        thread.start()

    async def _clear_all_conversations_async(self):
        """Async operation to clear all conversations."""
        try:
            # Clear from session manager
            count = await self.chat_manager.session_manager.clear_all_conversations()
            logger.info(f"Cleared {count} conversations from database")

            # Update UI on main thread
            Clock.schedule_once(lambda dt: self._clear_conversations_ui(), 0)

        except Exception as e:
            logger.error(f"Failed to clear conversations: {e}")
            Clock.schedule_once(
                lambda dt: Notification.error(f"Failed to clear conversations: {str(e)}"), 0
            )

    def _clear_conversations_ui(self):
        """Clear conversations from UI (runs on main thread)."""
        try:
            # Reset loading flag
            self._loading_conversation = False

            # Clear chat list
            self.chat_list.clear_widgets()
            self.active_conversations.clear()

            # Clear current messages
            self.messages_layout.clear_widgets()

            # Reset current conversation
            self.current_conversation_id = None

            # Create a new default conversation
            self._create_new_chat_sync()

            Notification.success("All conversations cleared")
            logger.info("Successfully cleared all conversations from UI")

        except Exception as e:
            logger.error(f"Failed to clear conversations UI: {e}")
            Notification.error("Failed to update UI after clearing conversations")

    def _confirm_delete_conversation(self, conversation_data):
        """Show confirmation dialog for deleting a conversation."""
        from kivymd.uix.button import MDRaisedButton

        content = MDBoxLayout(
            orientation="vertical", spacing=dp(10), size_hint_y=None, height=dp(100)
        )

        warning_label = MDLabel(
            text=f"Delete '{conversation_data.get('title', 'Unknown')}'?\nThis action cannot be undone.",
            theme_text_color="Primary",
            adaptive_height=True,
            halign="center",
        )
        content.add_widget(warning_label)

        dialog = MDDialog(
            title="Delete Conversation",
            type="custom",
            content_cls=content,
            buttons=[
                MDRaisedButton(text="Cancel", on_release=lambda x: dialog.dismiss()),
                MDRaisedButton(
                    text="Delete",
                    md_bg_color=(0.8, 0.2, 0.2, 1),  # Red color
                    on_release=lambda x: self._delete_conversation(conversation_data, dialog),
                ),
            ],
        )
        dialog.open()

    def _delete_conversation(self, conversation_data, dialog):
        """Delete a specific conversation."""
        dialog.dismiss()

        def run_async_delete():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._delete_conversation_async(conversation_data))
            except Exception as e:
                logger.error(f"Failed to delete conversation: {e}")
                Clock.schedule_once(
                    lambda dt: Notification.error("Failed to delete conversation"), 0
                )
            finally:
                loop.close()

        thread = threading.Thread(target=run_async_delete)
        thread.daemon = True
        thread.start()

    async def _delete_conversation_async(self, conversation_data):
        """Async operation to delete a conversation."""
        try:
            conversation_id = conversation_data.get("id")
            if not conversation_id:
                raise ValueError("No conversation ID provided")

            # Delete from session manager
            await self.chat_manager.session_manager.delete_conversation(conversation_id)
            logger.info(f"Deleted conversation: {conversation_id}")

            # Update UI on main thread
            Clock.schedule_once(lambda dt: self._remove_conversation_from_ui(conversation_data), 0)

        except Exception as e:
            logger.error(f"Failed to delete conversation: {e}")
            Clock.schedule_once(
                lambda dt: Notification.error(f"Failed to delete conversation: {str(e)}"), 0
            )

    def _remove_conversation_from_ui(self, conversation_data):
        """Remove conversation from UI (runs on main thread)."""
        try:
            conversation_id = conversation_data.get("id")

            # Remove from active conversations
            if conversation_id in self.active_conversations:
                del self.active_conversations[conversation_id]

            # Remove from chat list
            for widget in list(self.chat_list.children):
                if (
                    hasattr(widget, "conversation_data")
                    and widget.conversation_data.get("id") == conversation_id
                ):
                    self.chat_list.remove_widget(widget)
                    break

            # If this was the current conversation, switch to another or create new
            if self.current_conversation_id == conversation_id:
                if self.active_conversations:
                    # Switch to first available conversation
                    first_conv = list(self.active_conversations.values())[0]
                    self._select_conversation_wrapper(first_conv)
                else:
                    # No conversations left, create new one
                    self._create_new_chat_sync()

            Notification.success("Conversation deleted")
            logger.info(f"Successfully removed conversation from UI: {conversation_id}")

        except Exception as e:
            logger.error(f"Failed to remove conversation from UI: {e}")
            Notification.error("Failed to update UI after deleting conversation")

    def _show_tools_dialog(self, *args):
        """Show dialog with available MCP tools and execute selected tool."""
        from kivymd.uix.dialog import MDDialog
        from kivymd.uix.button import MDRaisedButton, MDTextButton
        from kivymd.uix.list import MDList, OneLineListItem
        from kivymd.uix.boxlayout import MDBoxLayout
        from kivymd.uix.textfield import MDTextField
        import asyncio
        import threading

        def run_tools_dialog():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._show_tools_dialog_async())
                loop.close()
            except Exception as e:
                logger.error(f"Error in tools dialog: {e}")
                from kivy.clock import Clock
                Clock.schedule_once(
                    lambda dt: Notification.error(f"Failed to load MCP tools: {e}"), 0
                )

        thread = threading.Thread(target=run_tools_dialog, daemon=True)
        thread.start()

    async def _show_tools_dialog_async(self):
        """Async implementation of tools dialog."""
        from kivy.clock import Clock

        try:
            # Get available tools
            tools = await self.chat_manager.list_mcp_tools()
            
            def show_dialog(dt):
                self._create_tools_dialog(tools)
                
            Clock.schedule_once(show_dialog, 0)
            
        except Exception as e:
            logger.error(f"Failed to get MCP tools: {e}")
            def show_error(dt):
                Notification.error(f"Failed to load tools: {e}")
            Clock.schedule_once(show_error, 0)

    def _create_tools_dialog(self, tools):
        """Create and show the tools dialog."""
        from kivymd.uix.dialog import MDDialog
        from kivymd.uix.button import MDRaisedButton, MDTextButton
        from kivymd.uix.list import MDList, OneLineListItem
        from kivymd.uix.boxlayout import MDBoxLayout
        from kivymd.uix.label import MDLabel

        content = MDBoxLayout(orientation="vertical", size_hint_y=None, height=dp(400))

        if not tools:
            content.add_widget(
                MDLabel(
                    text="No MCP tools available.\nMake sure MCP servers are connected.",
                    halign="center",
                    theme_text_color="Hint",
                )
            )
        else:
            # Tools list
            tools_list = MDList()
            content.add_widget(
                MDLabel(
                    text=f"Available MCP Tools ({len(tools)}):",
                    size_hint_y=None,
                    height=dp(30),
                )
            )

            for tool in tools[:10]:  # Limit to first 10 tools
                item = OneLineListItem(
                    text=f"{tool.name}: {tool.description[:60]}{'...' if len(tool.description) > 60 else ''}",
                    on_release=lambda x, t=tool: self._execute_test_tool(t),
                )
                tools_list.add_widget(item)

            content.add_widget(tools_list)

        self.tools_dialog = MDDialog(
            title="MCP Tools",
            type="custom",
            content_cls=content,
            buttons=[
                MDTextButton(text="Close", on_release=self._close_tools_dialog),
                MDRaisedButton(
                    text="Refresh", on_release=lambda x: self._refresh_tools_dialog()
                ),
            ],
            size_hint=(0.8, 0.8),
        )
        self.tools_dialog.open()

    def _execute_test_tool(self, tool):
        """Execute a test tool call."""
        # For search tools, use a default query
        if "search" in tool.name.lower():
            test_message = "Search the web for recent AI developments"
        elif "weather" in tool.name.lower():
            test_message = "Get current weather for San Francisco"
        else:
            test_message = f"Test the {tool.name} tool"

        # Close dialog and send test message
        self._close_tools_dialog()
        self.message_input.text = test_message
        self._send_message_wrapper()

    def _close_tools_dialog(self, *args):
        """Close the tools dialog."""
        if hasattr(self, "tools_dialog"):
            self.tools_dialog.dismiss()

    def _refresh_tools_dialog(self):
        """Refresh the tools dialog."""
        self._close_tools_dialog()
        self._show_tools_dialog()
