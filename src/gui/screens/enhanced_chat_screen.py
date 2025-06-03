"""Enhanced chat screen with multi-chat support and memory integration."""

import asyncio
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
    """List item for chat conversations."""

    def __init__(self, conversation_data, on_select_callback, on_delete_callback=None, **kwargs):
        self.conversation_data = conversation_data
        self.on_select_callback = on_select_callback
        self.on_delete_callback = on_delete_callback

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

        # Add delete functionality on long press
        self.bind(on_long_press=self._on_long_press)

    def _on_select(self):
        """Handle chat selection."""
        if self.on_select_callback:
            self.on_select_callback(self.conversation_data)

    def _on_long_press(self, *args):
        """Handle long press for delete option."""
        if self.on_delete_callback:
            self.on_delete_callback(self.conversation_data)

    def on_touch_down(self, touch):
        """Handle touch events for long press detection."""
        if self.collide_point(*touch.pos):
            touch.grab(self)
            # Schedule long press check
            Clock.schedule_once(lambda dt: self._check_long_press(touch), 1.0)
            return True
        return super().on_touch_down(touch)

    def on_touch_up(self, touch):
        """Handle touch up to cancel long press."""
        if touch.grab_current is self:
            touch.ungrab(self)
            Clock.unschedule(self._check_long_press)
        return super().on_touch_up(touch)

    def _check_long_press(self, touch):
        """Check if long press occurred."""
        if touch.grab_current is self:
            self._on_long_press()


class MemoryContextCard(MDCard):
    """Card showing relevant memory context for the current conversation."""

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
        for i, memory in enumerate(memories[:3]):  # Show top 3
            content = memory.content[:60] + "..." if len(memory.content) > 60 else memory.content
            context_text += f"• {content}\n"

        self.memory_label.text = context_text


class EnhancedChatScreen(MDScreen):
    """Enhanced chat screen with multi-chat support and memory integration."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "enhanced_chat"

        # Initialize managers
        self.chat_manager = ChatManager()
        self.safe_memory = create_safe_memory_manager(self._memory_error_callback)

        # Chat management
        self.active_conversations = {}  # conversation_id -> conversation_data
        self.current_conversation_id = None
        self.message_widgets = {}  # conversation_id -> list of message widgets

        # Shutdown flag to prevent operations after app closure
        self._is_shutting_down = False

        # Build UI
        self.build_ui()

        # Load existing conversations (schedule the async operation safely)
        Clock.schedule_once(self._schedule_load_conversations, 0.1)

    def on_enter(self, *args):
        """Called when entering the screen - reset shutdown flag."""
        self._is_shutting_down = False

    def on_pre_leave(self, *args):
        """Called when leaving the screen - cleanup async operations."""
        self._is_shutting_down = True

    def on_leave(self, *args):
        """Called when screen is left - ensure cleanup."""
        self._is_shutting_down = True

    def build_ui(self):
        """Build the enhanced chat interface."""
        # Navigation drawer for chat list (as main container)
        self.nav_drawer = MDNavigationDrawer(
            md_bg_color=THEME_COLORS["drawer_bg"],
            scrim_color=(0, 0, 0, 0.7),
            enable_swiping=True,
            anchor="left",
            radius=(0, 16, 16, 0),  # Rounded corners on the right
            elevation=ELEVATION["drawer"],
            close_on_click=True,  # Close when clicking outside
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
        self.toolbar = MDTopAppBar(title=f"Enhanced Chat - {provider_info}", elevation=2)

        # Add menu button to open drawer
        menu_btn = MDIconButton(icon="menu", on_release=lambda x: self.nav_drawer.set_state("open"))
        self.toolbar.left_action_items = [["menu", lambda x: self.nav_drawer.set_state("open")]]

        # Add action buttons to toolbar
        self.toolbar.right_action_items = [
            ["swap-horizontal", lambda x: self._show_provider_switcher()],
            ["memory", lambda x: self._show_memory_panel()],
            ["cog", lambda x: self._go_to_settings()],
        ]

        chat_content.add_widget(self.toolbar)

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

        input_layout.add_widget(self.message_input)
        input_layout.add_widget(send_btn)
        chat_content.add_widget(input_layout)

        # Add main content first (background)
        self.add_widget(chat_content)

        # Add navigation drawer second (foreground)
        # This should make it appear on top of the content
        self.add_widget(self.nav_drawer)

        # Apply comprehensive theme fixes
        apply_chat_theme_fixes(self)

    def _on_focus_change(self, instance, focus):
        """Handle focus changes to set up keyboard handling."""
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
        """Handle keyboard events for message input."""
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
        """Check if this is likely a memory storage command."""
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
        """Handle memory operation errors."""
        logger.warning(f"Memory operation failed - {operation}: {error}")
        Notification.warning(f"Memory: {operation} - {error[:50]}...")

    def _send_message_wrapper(self, *args):
        """Wrapper for send message that handles async safely."""
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
        import threading

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
        """Send a message with memory-enhanced context."""
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

                # Store this user message in memory
                from src.memory import MemoryType

                await self.safe_memory.safe_remember(
                    content=message_text,
                    memory_type=MemoryType.SHORT_TERM,
                    metadata={
                        "conversation_id": self.current_conversation_id,
                        "timestamp": datetime.now().isoformat(),
                        "type": "user_message",
                    },
                )

            # Send to chat manager with streaming
            response_text = ""
            try:
                async for chunk in self.chat_manager.send_message(message_text, stream=True):
                    # Check for shutdown during streaming
                    if self._is_shutting_down:
                        logger.info("Stopping message streaming - app is shutting down")
                        break

                    response_text += chunk
                    if assistant_widget and hasattr(assistant_widget, "content_label"):
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

            # Store assistant response in memory
            if response_text:
                from src.memory import MemoryType

                await self.safe_memory.safe_remember(
                    content=response_text,
                    memory_type=MemoryType.SHORT_TERM,
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
        """Show memory context panel."""
        self.memory_context.update_memory_context(memories)
        self.memory_context.height = dp(120)
        self.memory_context.opacity = 1

        # Auto-hide after 5 seconds
        Clock.schedule_once(lambda dt: self._hide_memory_context(), 5)

    def _hide_memory_context(self):
        """Hide memory context panel."""
        self.memory_context.opacity = 0
        self.memory_context.height = 0

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
        import threading

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

    def _go_to_memory_screen(self, dialog):
        """Navigate to memory management screen."""
        dialog.dismiss()
        self.manager.current = "advanced_memory"

    def _go_to_settings(self):
        """Navigate to settings screen."""
        self.manager.current = "settings"

    def _schedule_load_conversations(self, dt):
        """Schedule the async conversation loading in a thread-safe way."""
        import threading

        def run_async_load():
            try:
                # Create a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                # Run the async operation
                loop.run_until_complete(self._load_conversations_sync())

            except Exception as e:
                logger.error(f"Failed to load conversations: {e}")
                # Schedule UI update on main thread
                Clock.schedule_once(lambda dt: self._handle_load_error(str(e)), 0)
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
            logger.error(f"Failed to load conversations: {e}")
            Clock.schedule_once(lambda dt: self._handle_load_error(str(e)), 0)

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
        """Wrapper for conversation selection that handles async safely."""
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

            import threading

            def run_async_select():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(
                        self._select_conversation_async(conversation_data, loading_widget)
                    )
                except Exception as e:
                    logger.error(f"Failed to select conversation: {e}")
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

            # Update title using stored toolbar reference
            try:
                if hasattr(self, "toolbar") and self.toolbar:
                    self.toolbar.title = conversation_data["title"]
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
        """Add a message widget synchronously (for UI thread)."""
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
        """Create new chat synchronously (fallback)."""
        try:
            import threading

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
        """Get current provider and model information."""
        try:
            from src.core.config import config

            provider = config.providers.default_provider

            # Get model info based on provider
            if provider == "ollama":
                models = config.providers.ollama_models
                model = models[0] if models else "Unknown"
            elif provider == "openai":
                models = config.providers.openai_models
                model = models[0] if models else "Unknown"
            elif provider == "lmstudio":
                model = "LMStudio"
            else:
                model = "Unknown"

            return f"{provider.title()}/{model}"
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

        import threading

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

    def _show_provider_switcher(self, *args):
        """Show provider/model switcher dialog."""
        from kivymd.uix.button import MDFlatButton, MDRaisedButton

        content = MDBoxLayout(
            orientation="vertical", spacing=dp(10), size_hint_y=None, height=dp(300)
        )

        # Current provider info
        current_provider = self._get_provider_model_info()
        current_label = MDLabel(
            text=f"Current: {current_provider}",
            theme_text_color="Primary",
            size_hint_y=None,
            height=dp(30),
        )
        content.add_widget(current_label)

        # Provider options
        self.selected_provider_value = None
        try:
            from src.core.config import config

            providers = []
            if config.providers.ollama_enabled:
                for model in config.providers.ollama_models:
                    providers.append(("ollama", model))
            if config.providers.openai_enabled:
                for model in config.providers.openai_models:
                    providers.append(("openai", model))
            if config.providers.lmstudio_enabled:
                providers.append(("lmstudio", "LMStudio"))

            # Button selection for providers
            for provider, model in providers:
                provider_button = MDFlatButton(
                    text=f"{provider.title()}/{model}",
                    size_hint_y=None,
                    height=dp(40),
                    on_release=lambda x, p=provider, m=model: self._select_provider_option(x, p, m),
                )

                # Highlight current provider
                if provider == config.providers.default_provider and model in getattr(
                    config.providers, f"{provider}_models", []
                ):
                    provider_button.md_bg_color = (0.2, 0.6, 0.8, 0.3)  # Light blue highlight

                content.add_widget(provider_button)

        except Exception as e:
            logger.error(f"Failed to load provider options: {e}")
            error_label = MDLabel(text="Failed to load provider options", theme_text_color="Error")
            content.add_widget(error_label)

        self.provider_dialog = MDDialog(
            title="Switch Provider/Model",
            type="custom",
            content_cls=content,
            buttons=[
                MDRaisedButton(text="Cancel", on_release=lambda x: self.provider_dialog.dismiss()),
                MDRaisedButton(text="Switch", on_release=lambda x: self._switch_provider()),
            ],
        )
        self.provider_dialog.open()

    def _select_provider_option(self, button, provider, model):
        """Select a provider option."""
        self.selected_provider_value = f"{provider}:{model}"

        # Update button colors to show selection
        parent = button.parent
        if parent:
            for child in parent.children:
                if hasattr(child, "md_bg_color"):
                    child.md_bg_color = (0, 0, 0, 0)  # Reset background
            button.md_bg_color = (0.2, 0.8, 0.2, 0.3)  # Green highlight for selection

    def _switch_provider(self):
        """Switch to selected provider/model."""
        try:
            if self.selected_provider_value:
                provider, model = self.selected_provider_value.split(":", 1)

                # Update config
                from src.core.config import config

                config.providers.default_provider = provider

                # Update toolbar title
                provider_info = self._get_provider_model_info()
                for child in self.children:
                    if hasattr(child, "children"):
                        for grandchild in child.children:
                            if hasattr(grandchild, "title") and "Enhanced Chat" in grandchild.title:
                                grandchild.title = f"Enhanced Chat - {provider_info}"
                                break

                Notification.success(f"Switched to {provider.title()}/{model}")
            else:
                Notification.warning("No provider selected")

        except Exception as e:
            logger.error(f"Failed to switch provider: {e}")
            Notification.error("Failed to switch provider")

        self.provider_dialog.dismiss()

    def _confirm_delete_conversation(self, conversation_data):
        """Show confirmation dialog for deleting a single conversation."""
        from kivymd.uix.button import MDRaisedButton

        content = MDBoxLayout(
            orientation="vertical", spacing=dp(10), size_hint_y=None, height=dp(100)
        )

        warning_label = MDLabel(
            text=f"Delete conversation '{conversation_data.get('title', 'Untitled')}'?\nThis action cannot be undone.",
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
                    on_release=lambda x: self._delete_conversation(dialog, conversation_data),
                ),
            ],
        )
        dialog.open()

    def _delete_conversation(self, dialog, conversation_data):
        """Delete a single conversation."""
        dialog.dismiss()

        import threading

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
            conversation_id = conversation_data["id"]

            # Delete from session manager
            success = await self.chat_manager.session_manager.delete_conversation(conversation_id)

            if success:
                # Update UI on main thread
                Clock.schedule_once(
                    lambda dt: self._remove_conversation_from_ui(conversation_data), 0
                )
                logger.info(f"Successfully deleted conversation: {conversation_id}")
            else:
                Clock.schedule_once(
                    lambda dt: Notification.error("Conversation not found or already deleted"), 0
                )

        except Exception as e:
            logger.error(f"Failed to delete conversation: {e}")
            Clock.schedule_once(
                lambda dt: Notification.error(f"Failed to delete conversation: {str(e)}"), 0
            )

    def _remove_conversation_from_ui(self, conversation_data):
        """Remove conversation from UI (runs on main thread)."""
        try:
            conversation_id = conversation_data["id"]

            # Remove from active conversations
            if conversation_id in self.active_conversations:
                del self.active_conversations[conversation_id]

            # Remove from chat list
            for widget in self.chat_list.children[:]:  # Create a copy of the list
                if (
                    hasattr(widget, "conversation_data")
                    and widget.conversation_data["id"] == conversation_id
                ):
                    self.chat_list.remove_widget(widget)
                    break

            # If this was the current conversation, switch to another one or create new
            if self.current_conversation_id == conversation_id:
                if self.active_conversations:
                    # Switch to the first available conversation
                    first_conv_id = list(self.active_conversations.keys())[0]
                    self._select_conversation_wrapper(self.active_conversations[first_conv_id])
                else:
                    # No conversations left, create a new one
                    self.current_conversation_id = None
                    self.messages_layout.clear_widgets()
                    self._create_new_chat_sync()

            Notification.success(
                f"Deleted conversation: {conversation_data.get('title', 'Untitled')}"
            )

        except Exception as e:
            logger.error(f"Failed to remove conversation from UI: {e}")
            Notification.error("Failed to update UI after deleting conversation")
