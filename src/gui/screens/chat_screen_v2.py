"""Enterprise chat screen with conversation management."""

import asyncio
from datetime import datetime
from typing import Any

from kivy.clock import Clock
from kivy.metrics import dp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDFlatButton, MDRaisedButton
from kivymd.uix.card import MDCard
from kivymd.uix.label import MDLabel
from kivymd.uix.list import MDList
from kivymd.uix.navigationdrawer import MDNavigationDrawer
from kivymd.uix.progressbar import MDProgressBar
from kivymd.uix.screen import MDScreen
from kivymd.uix.scrollview import MDScrollView
from kivymd.uix.textfield import MDTextField

from src.core.chat_manager import ChatManager
from src.gui.theme import MessageTheme, UIConstants
from src.gui.utils.notifications import Notification
from src.gui.widgets.conversation_list import ConversationList
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class StreamingMessageCard(MDCard):
    """Message card that supports streaming updates."""

    def __init__(self, message_data: dict[str, Any], is_user: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "vertical"
        self.padding = UIConstants.PADDING_MEDIUM
        self.spacing = UIConstants.SPACING_SMALL
        self.elevation = UIConstants.ELEVATION_CARD
        self.size_hint_x = MessageTheme.MAX_WIDTH_RATIO
        self.adaptive_height = True
        self.radius = [UIConstants.RADIUS_MEDIUM]

        self.message_id = message_data.get("id")
        self.is_user = is_user

        # Enhanced styling based on sender
        if is_user:
            self.md_bg_color = MessageTheme.USER_BG
            self.line_color = MessageTheme.USER_BORDER
            self.pos_hint = {"right": 1}
        else:
            self.md_bg_color = MessageTheme.ASSISTANT_BG
            self.line_color = MessageTheme.ASSISTANT_BORDER
            self.pos_hint = {"left": 1}

        # Header with metadata
        header = MDBoxLayout(adaptive_height=True, spacing=dp(10))

        # Role and model
        role_label = MDLabel(
            text=message_data.get("role", "user").title(),
            theme_text_color="Secondary",
            font_style="Caption",
            adaptive_height=True,
        )
        header.add_widget(role_label)

        if message_data.get("model"):
            # Use a simple label instead of MDChip to avoid icon issues
            model_label = MDLabel(
                text=f"[{message_data['model']}]",
                theme_text_color="Custom",
                text_color=(0.5, 0.7, 1.0, 1.0),  # Light blue
                font_style="Caption",
                adaptive_height=True,
            )
            header.add_widget(model_label)

        # Timestamp
        if message_data.get("created_at"):
            timestamp = datetime.fromisoformat(message_data["created_at"])
            time_label = MDLabel(
                text=timestamp.strftime("%H:%M"),
                theme_text_color="Hint",
                font_style="Caption",
                adaptive_height=True,
            )
            header.add_widget(time_label)

        header.add_widget(MDBoxLayout())  # Spacer

        # Cost and tokens
        if message_data.get("tokens"):
            tokens_label = MDLabel(
                text=f"{message_data['tokens']} tokens",
                theme_text_color="Hint",
                font_style="Caption",
                adaptive_height=True,
            )
            header.add_widget(tokens_label)

        self.add_widget(header)

        # Message content
        self.content_label = MDLabel(
            text=message_data.get("content", ""), adaptive_height=True, theme_text_color="Primary"
        )
        self.add_widget(self.content_label)

        # Actions
        if not is_user:
            actions = MDBoxLayout(adaptive_height=True, spacing=dp(5), padding=[0, dp(5), 0, 0])

            # Reaction buttons with text
            reactions = [
                ("Like", "thumbs-up"),
                ("Dislike", "thumbs-down"),
                ("Target", "bullseye"),
                ("Idea", "lightbulb"),
            ]
            for reaction_text, reaction_id in reactions:
                btn = MDFlatButton(
                    text=reaction_text, on_release=lambda x, r=reaction_text: self._add_reaction(r)
                )
                actions.add_widget(btn)

            actions.add_widget(MDBoxLayout())  # Spacer

            # Copy button
            copy_btn = MDFlatButton(text="Copy", on_release=self._copy_content)
            actions.add_widget(copy_btn)

            # Branch button
            branch_btn = MDFlatButton(text="Branch", on_release=self._branch_conversation)
            actions.add_widget(branch_btn)

            self.add_widget(actions)

    def update_content(self, new_content: str):
        """Update message content (for streaming)."""
        self.content_label.text = new_content

    def _add_reaction(self, reaction: str):
        """Add a reaction to the message."""
        logger.info(f"Adding reaction {reaction} to message {self.message_id}")

        # Update button to show reaction was added
        for widget in self.children:
            if hasattr(widget, "children"):
                for child in widget.children:
                    if isinstance(child, MDFlatButton) and child.text == reaction:
                        # Highlight the selected reaction
                        child.md_bg_color = (0.2, 0.6, 1.0, 0.3)
                        Notification.success(f"Added {reaction} reaction")

    def _copy_content(self, *args):
        """Copy message content to clipboard."""
        try:
            from kivy.core.clipboard import Clipboard

            Clipboard.copy(self.content_label.text)
            Notification.success("Copied to clipboard")
        except Exception as e:
            logger.error(f"Failed to copy to clipboard: {e}")
            Notification.error("Failed to copy message")

    def _branch_conversation(self, *args):
        """Branch conversation from this message."""
        try:
            logger.info(f"Branching from message {self.message_id}")
            # Signal to parent to create new branch
            chat_screen = None
            widget = self.parent
            # Navigate up the widget tree to find the chat screen
            while widget and not isinstance(widget, EnterpriseCharScreen):
                widget = widget.parent
            if widget:
                widget._create_branch(self.message_id)
        except Exception as e:
            logger.error(f"Failed to branch conversation: {e}")
            Notification.error("Failed to branch conversation")


class EnterpriseCharScreen(MDScreen):
    """Enterprise chat interface with full session management."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.chat_manager = ChatManager()
        self.current_conversation_id = None
        self.streaming_message = None
        self.build_ui()

        # Load conversations on start (simplified for now)
        Clock.schedule_once(lambda dt: self._load_conversations_sync(), 0.1)

        # Initialize provider/model selection
        Clock.schedule_once(lambda dt: self._initialize_provider_selection(), 0.2)

    def build_ui(self):
        """Build the enterprise chat interface."""
        # Main layout with navigation drawer
        main_layout = MDBoxLayout()

        # Navigation drawer for conversations
        self.nav_drawer = MDNavigationDrawer(radius=(0, dp(16), dp(16), 0), width=dp(320))

        drawer_content = MDBoxLayout(orientation="vertical")

        # Drawer header
        drawer_header = MDCard(
            orientation="vertical",
            padding=dp(20),
            md_bg_color=(0.15, 0.15, 0.15, 1),
            adaptive_height=True,
        )

        drawer_title = MDLabel(text="Conversations", font_style="H5", adaptive_height=True)
        drawer_header.add_widget(drawer_title)

        # Conversation stats
        self.stats_label = MDLabel(
            text="0 conversations • $0.00 total",
            theme_text_color="Secondary",
            font_style="Caption",
            adaptive_height=True,
        )
        drawer_header.add_widget(self.stats_label)

        drawer_content.add_widget(drawer_header)

        # Conversation list
        self.conversation_list = ConversationList(
            on_conversation_select=self._on_conversation_select,
            on_new_conversation=self._on_new_conversation,
        )
        drawer_content.add_widget(self.conversation_list)

        self.nav_drawer.add_widget(drawer_content)

        # Main chat area
        chat_layout = MDBoxLayout(orientation="vertical")

        # Top bar - custom toolbar to avoid icon issues
        toolbar_layout = MDBoxLayout(
            orientation="horizontal",
            size_hint_y=None,
            height=dp(56),
            md_bg_color=(0.12, 0.12, 0.12, 1),
            padding=[dp(16), 0, dp(16), 0],
        )

        # Menu button
        menu_btn = MDFlatButton(
            text="Menu",
            on_release=lambda x: self.nav_drawer.set_state("open"),
            pos_hint={"center_y": 0.5},
        )
        toolbar_layout.add_widget(menu_btn)

        # Title
        self.toolbar_title = MDLabel(
            text="New Conversation",
            font_style="H6",
            theme_text_color="Primary",
            pos_hint={"center_y": 0.5},
        )
        toolbar_layout.add_widget(self.toolbar_title)

        # Spacer
        toolbar_layout.add_widget(MDBoxLayout())

        # Action buttons
        settings_btn = MDFlatButton(text="Settings", on_release=lambda x: self.go_to_settings())
        memory_btn = MDFlatButton(text="Memory", on_release=lambda x: self.go_to_memory())
        export_btn = MDFlatButton(text="Export", on_release=lambda x: self._export_conversation())
        analytics_btn = MDFlatButton(text="Stats", on_release=lambda x: self._show_analytics())

        toolbar_layout.add_widget(settings_btn)
        toolbar_layout.add_widget(memory_btn)
        toolbar_layout.add_widget(export_btn)
        toolbar_layout.add_widget(analytics_btn)

        # Model selector bar
        model_bar = MDBoxLayout(size_hint_y=None, height=dp(60), padding=dp(10), spacing=dp(10))

        self.provider_button = MDRaisedButton(
            text="Provider: Ollama", on_release=self._show_provider_menu
        )

        self.model_button = MDRaisedButton(text="Model: llama3.2", on_release=self._show_model_menu)

        # Token counter
        self.token_label = MDLabel(text="Tokens: 0", theme_text_color="Secondary")

        # Cost indicator
        self.cost_label = MDLabel(text="Cost: $0.0000", theme_text_color="Secondary")

        model_bar.add_widget(self.provider_button)
        model_bar.add_widget(self.model_button)
        model_bar.add_widget(MDBoxLayout())  # Spacer
        model_bar.add_widget(self.token_label)
        model_bar.add_widget(self.cost_label)

        # Messages area
        self.messages_scroll = MDScrollView()
        self.messages_container = MDList(spacing=dp(10), padding=dp(10))
        self.messages_scroll.add_widget(self.messages_container)

        # Streaming indicator
        self.streaming_indicator = MDProgressBar(size_hint_y=None, height=dp(4), opacity=0)

        # Input area
        input_layout = MDBoxLayout(size_hint_y=None, height=dp(80), padding=dp(10), spacing=dp(10))

        # Attach button
        attach_btn = MDFlatButton(text="File", on_release=self._attach_file)

        # Message input
        self.message_input = MDTextField(
            hint_text="Type your message... (Shift+Enter for new line)",
            multiline=True,
            max_height=dp(120),
        )

        # Voice input button
        voice_btn = MDFlatButton(text="Voice", on_release=self._voice_input)

        # Send button
        self.send_button = MDRaisedButton(text="Send", on_release=self._send_message)

        input_layout.add_widget(attach_btn)
        input_layout.add_widget(self.message_input)
        input_layout.add_widget(voice_btn)
        input_layout.add_widget(self.send_button)

        # Add all to chat layout
        chat_layout.add_widget(toolbar_layout)
        chat_layout.add_widget(model_bar)
        chat_layout.add_widget(self.messages_scroll)
        chat_layout.add_widget(self.streaming_indicator)
        chat_layout.add_widget(input_layout)

        # Add to main layout
        main_layout.add_widget(self.nav_drawer)
        main_layout.add_widget(chat_layout)

        self.add_widget(main_layout)

    def _load_conversations_sync(self):
        """Load conversation list (simplified sync version)."""
        try:
            # For now, just show placeholder data
            self.conversation_list.set_conversations([])
            self.stats_label.text = "0 conversations • $0.00 total"
            logger.info("Conversation list initialized")

        except Exception as e:
            logger.error(f"Failed to load conversations: {e}")
            Notification.error("Failed to load conversations")

    def _initialize_provider_selection(self):
        """Initialize provider and model selection."""
        try:
            available_providers = self.chat_manager.get_available_providers()
            if available_providers:
                # Set first available provider as default
                default_provider = available_providers[0]
                self.chat_manager.current_provider = default_provider
                self.provider_button.text = f"Provider: {default_provider.title()}"

                # Load models for default provider
                self._refresh_models_for_provider(default_provider)
            else:
                self.provider_button.text = "Provider: None"
                self.model_button.text = "Model: Configure providers"

        except Exception as e:
            logger.error(f"Failed to initialize provider selection: {e}")

    async def _load_conversations(self):
        """Load conversation list."""
        try:
            self.conversation_list.show_loading()
            conversations = await self.chat_manager.get_conversations()
            self.conversation_list.set_conversations(conversations)

            # Update stats
            total_cost = sum(c.get("total_cost", 0) for c in conversations)
            self.stats_label.text = f"{len(conversations)} conversations • ${total_cost:.2f} total"

        except Exception as e:
            logger.error(f"Failed to load conversations: {e}")
            Notification.error("Failed to load conversations")

    def _on_conversation_select(self, conversation_id: str):
        """Handle conversation selection."""
        asyncio.create_task(self._load_conversation(conversation_id))
        self.nav_drawer.set_state("close")

    async def _load_conversation(self, conversation_id: str):
        """Load a conversation."""
        try:
            # Load session
            await self.chat_manager.load_session(conversation_id)
            self.current_conversation_id = conversation_id

            # Update UI
            self.toolbar_title.text = self.chat_manager.current_session.title

            # Load messages
            messages = await self.chat_manager.current_session.get_messages()
            self.messages_container.clear_widgets()

            for message in messages:
                self._add_message_to_ui(message.to_dict(), is_user=message.role == "user")

            # Scroll to bottom
            Clock.schedule_once(lambda dt: self._scroll_to_bottom(), 0.1)

        except Exception as e:
            logger.error(f"Failed to load conversation: {e}")
            Notification.error("Failed to load conversation")

    def _on_new_conversation(self):
        """Create a new conversation."""
        try:
            # Clear current conversation
            self.current_conversation_id = None
            self.messages_container.clear_widgets()
            self.toolbar_title.text = "New Conversation"

            # Reset provider/model selection to show current config
            available_providers = self.chat_manager.get_available_providers()
            if available_providers:
                self.provider_button.text = f"Provider: {available_providers[0].title()}"
                self.chat_manager.current_provider = available_providers[0]
                # Refresh models for the provider
                self._refresh_models_for_provider(available_providers[0])
            else:
                self.provider_button.text = "Provider: None"
                self.model_button.text = "Model: Configure providers first"

            # Reset counters
            self.token_label.text = "Tokens: 0"
            self.cost_label.text = "Cost: $0.0000"

            # Add welcome message
            welcome_msg = {
                "content": "Hello! I'm ready to help you. What would you like to talk about?",
                "role": "assistant",
                "created_at": datetime.now().isoformat(),
                "model": "system",
            }
            self._add_message_to_ui(welcome_msg, is_user=False)

            # Show success notification
            Notification.success("New conversation started")

            # Focus on input
            self.message_input.focus = True

            # Close navigation drawer
            self.nav_drawer.set_state("close")

        except Exception as e:
            logger.error(f"Error creating new conversation: {e}")
            Notification.error("Failed to create new conversation")

    async def _create_new_conversation(self):
        """Create a new conversation session."""
        try:
            await self.chat_manager.create_session()
            self.current_conversation_id = self.chat_manager.current_session.id

            # Clear messages
            self.messages_container.clear_widgets()

            # Update UI
            self.toolbar_title.text = "New Conversation"
            self.token_label.text = "Tokens: 0"
            self.cost_label.text = "Cost: $0.0000"

            # Add welcome message
            self._add_message_to_ui(
                {
                    "content": "Welcome! I'm ready to assist you. How can I help today?",
                    "role": "assistant",
                    "created_at": datetime.now().isoformat(),
                },
                is_user=False,
            )

            # Refresh conversation list
            await self._load_conversations()

            self.nav_drawer.set_state("close")

        except Exception as e:
            logger.error(f"Failed to create conversation: {e}")
            Notification.error("Failed to create conversation")

    def _send_message(self, *args):
        """Send a message."""
        message = self.message_input.text.strip()
        if not message:
            return

        # Clear input
        self.message_input.text = ""

        # Send message synchronously for now (avoid event loop issues)
        self._sync_send_message(message)

    def _sync_send_message(self, message: str):
        """Send message synchronously."""
        try:
            # Disable send button
            self.send_button.disabled = True

            # Add user message to UI
            user_msg_data = {
                "content": message,
                "role": "user",
                "created_at": datetime.now().isoformat(),
            }
            self._add_message_to_ui(user_msg_data, is_user=True)

            # Show progress indicator
            self.streaming_indicator.opacity = 1

            # Check if we have any providers available
            available_providers = self.chat_manager.get_available_providers()
            if available_providers:
                # Use real AI provider
                Clock.schedule_once(lambda dt: self._send_real_ai_message(message), 0.1)
            else:
                # Fall back to demo mode
                Clock.schedule_once(lambda dt: self._simulate_ai_response(message), 1.0)

        except Exception as e:
            logger.error(f"Error sending message: {e}")
            Notification.error("Failed to send message")
            self.send_button.disabled = False
            self.streaming_indicator.opacity = 0

    def _simulate_ai_response(self, user_message: str):
        """Simulate an AI response for demonstration."""
        try:
            # Create a demo response
            demo_responses = [
                f"I understand you said: '{user_message}'. This is a demo response from Neuromancer's AI system.",
                "I'm currently in demo mode. To enable full AI functionality, please configure your LLM providers in settings.",
                "This is a placeholder response. The real AI conversation system will be available once providers are properly configured.",
                f"You asked about: '{user_message}'. In the full version, I would process this through the configured LLM providers.",
                "Demo mode: Your message has been received. Configure Ollama, OpenAI, or LM Studio to enable real AI conversations.",
            ]

            import random

            response_text = random.choice(demo_responses)

            # Add AI response to UI
            ai_msg_data = {
                "content": response_text,
                "role": "assistant",
                "created_at": datetime.now().isoformat(),
                "model": "demo-mode",
            }
            self._add_message_to_ui(ai_msg_data, is_user=False)

            # Re-enable send button and hide progress
            self.send_button.disabled = False
            self.streaming_indicator.opacity = 0

            # Scroll to bottom
            Clock.schedule_once(lambda dt: self._scroll_to_bottom(), 0.1)

            Notification.success("Response generated")

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            Notification.error("Failed to generate response")
            self.send_button.disabled = False
            self.streaming_indicator.opacity = 0

    def _send_real_ai_message(self, user_message: str):
        """Send message to real AI provider."""

        def run_ai_request():
            try:
                # Get the first available provider
                available_providers = self.chat_manager.get_available_providers()
                if not available_providers:
                    Clock.schedule_once(lambda dt: self._simulate_ai_response(user_message), 0)
                    return

                provider_name = available_providers[0]  # Use first available provider

                # For now, use synchronous Ollama provider to avoid event loop issues
                if provider_name == "ollama":
                    from src.core.config import config_manager
                    from src.providers import Message
                    from src.providers.ollama_sync import OllamaSyncProvider

                    host = config_manager.get("providers.ollama_host", "http://localhost:11434")
                    provider = OllamaSyncProvider(host=host)

                    # Get current model
                    current_model = self.chat_manager.current_model or "llama3.2:latest"

                    try:
                        # Send message
                        response = provider.complete(
                            messages=[Message(role="user", content=user_message)],
                            model=current_model,
                            temperature=0.7,
                        )

                        # Add AI response to UI
                        ai_msg_data = {
                            "content": response.content,
                            "role": "assistant",
                            "created_at": datetime.now().isoformat(),
                            "model": response.model or current_model,
                            "tokens": (
                                response.usage.get("total_tokens") if response.usage else None
                            ),
                        }

                        Clock.schedule_once(lambda dt: self._add_ai_response(ai_msg_data), 0)

                    except Exception as e:
                        logger.error(f"Failed to get AI response: {e}")
                        error_msg = str(e)
                        Clock.schedule_once(lambda dt: self._handle_ai_error(error_msg), 0)
                else:
                    # For other providers, fall back to demo mode for now
                    Clock.schedule_once(lambda dt: self._simulate_ai_response(user_message), 0)

            except Exception as e:
                logger.error(f"Error in AI request thread: {e}")
                error_msg = str(e)
                Clock.schedule_once(lambda dt: self._handle_ai_error(error_msg), 0)

        import threading

        thread = threading.Thread(target=run_ai_request)
        thread.daemon = True
        thread.start()

    def _add_ai_response(self, ai_msg_data):
        """Add AI response to UI."""
        try:
            self._add_message_to_ui(ai_msg_data, is_user=False)
            self.send_button.disabled = False
            self.streaming_indicator.opacity = 0
            Clock.schedule_once(lambda dt: self._scroll_to_bottom(), 0.1)
            Notification.success("AI response received")
        except Exception as e:
            logger.error(f"Error adding AI response: {e}")
            self._handle_ai_error(str(e))

    def _handle_ai_error(self, error_msg):
        """Handle AI response errors."""
        try:
            # Add error message
            error_response = {
                "content": f"Sorry, I encountered an error: {error_msg}\n\nPlease check your provider configuration in settings.",
                "role": "assistant",
                "created_at": datetime.now().isoformat(),
                "model": "error",
            }
            self._add_message_to_ui(error_response, is_user=False)

            self.send_button.disabled = False
            self.streaming_indicator.opacity = 0
            Notification.error("AI response failed")
        except Exception as e:
            logger.error(f"Error handling AI error: {e}")
            self.send_button.disabled = False
            self.streaming_indicator.opacity = 0

    async def _async_send_message(self, message: str):
        """Send message asynchronously."""
        try:
            # Disable send button
            self.send_button.disabled = True

            # Add user message to UI
            user_msg_data = {
                "content": message,
                "role": "user",
                "created_at": datetime.now().isoformat(),
            }
            self._add_message_to_ui(user_msg_data, is_user=True)

            # Show streaming indicator
            self.streaming_indicator.opacity = 1

            # Create streaming message card
            assistant_msg_data = {
                "content": "",
                "role": "assistant",
                "created_at": datetime.now().isoformat(),
                "model": self.chat_manager.current_model,
            }
            self.streaming_message = StreamingMessageCard(
                message_data=assistant_msg_data, is_user=False
            )
            self.messages_container.add_widget(self.streaming_message)

            # Send message with streaming
            full_response = ""
            async for chunk in self.chat_manager.send_message(content=message, stream=True):
                full_response += chunk
                self.streaming_message.update_content(full_response)
                self._scroll_to_bottom()

            # Update token and cost display
            if self.chat_manager.current_session:
                stats = await self.chat_manager.get_conversation_stats(
                    self.chat_manager.current_session.id
                )
                conv = stats.get("conversation", {})
                self.token_label.text = f"Tokens: {conv.get('total_tokens', 0)}"
                self.cost_label.text = f"Cost: ${conv.get('total_cost', 0):.4f}"

        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            Notification.error("Failed to send message")

        finally:
            # Re-enable send button
            self.send_button.disabled = False
            self.streaming_indicator.opacity = 0
            self.streaming_message = None

    def _add_message_to_ui(self, message_data: dict[str, Any], is_user: bool):
        """Add a message to the UI."""
        try:
            msg_card = StreamingMessageCard(message_data=message_data, is_user=is_user)
            self.messages_container.add_widget(msg_card)

            # Auto-scroll to bottom
            Clock.schedule_once(lambda dt: self._scroll_to_bottom(), 0.1)

            # Save message to database if session exists
            if self.current_conversation_id:
                self._save_message_to_db(message_data, is_user)

        except Exception as e:
            logger.error(f"Error adding message to UI: {e}")
            Notification.error("Failed to display message")

    def _save_message_to_db(self, message_data: dict[str, Any], is_user: bool):
        """Save message to database."""
        try:
            # This would be async in real implementation
            # For now, just log that we would save it
            logger.debug(f"Would save message: {message_data.get('content', '')[:50]}...")
        except Exception as e:
            logger.error(f"Error saving message to database: {e}")

    def _scroll_to_bottom(self):
        """Scroll to bottom of messages."""
        self.messages_scroll.scroll_y = 0

    def _show_provider_menu(self, button):
        """Show provider selection menu."""
        try:
            available_providers = self.chat_manager.get_available_providers()
            if not available_providers:
                Notification.warning(
                    "No providers configured. Go to Settings > LLM Providers to configure."
                )
                return

            # Create simple dialog with provider list
            from kivymd.uix.dialog import MDDialog
            from kivymd.uix.list import OneLineListItem

            items = []
            for provider in available_providers:
                item = OneLineListItem(
                    text=provider.title(), on_release=lambda x, p=provider: self._select_provider(p)
                )
                items.append(item)

            self.provider_dialog = MDDialog(
                title="Select Provider",
                type="simple",
                items=items,
                size_hint=(0.8, None),
                height=dp(200),
            )
            self.provider_dialog.open()

        except Exception as e:
            logger.error(f"Error showing provider menu: {e}")
            Notification.error("Failed to show provider menu")

    def _select_provider(self, provider_name):
        """Select a provider."""
        try:
            # Update current provider
            self.chat_manager.current_provider = provider_name
            self.provider_button.text = f"Provider: {provider_name.title()}"

            # Reset model selection when provider changes
            self.model_button.text = "Model: Loading..."

            # Close dialog
            if hasattr(self, "provider_dialog"):
                self.provider_dialog.dismiss()

            # Refresh available models for this provider
            Clock.schedule_once(lambda dt: self._refresh_models_for_provider(provider_name), 0.1)

            Notification.success(f"Provider changed to {provider_name.title()}")

        except Exception as e:
            logger.error(f"Error selecting provider: {e}")
            Notification.error("Failed to change provider")

    def _refresh_models_for_provider(self, provider_name):
        """Refresh models for selected provider."""

        def fetch_models():
            try:
                models = []

                if provider_name == "ollama":
                    from src.core.config import config_manager
                    from src.providers.ollama_sync import OllamaSyncProvider

                    host = config_manager.get("providers.ollama_host", "http://localhost:11434")
                    provider = OllamaSyncProvider(host=host)
                    models = provider.list_models()

                elif provider_name == "openai":
                    from src.core.config import config_manager

                    api_key = config_manager.get("providers.openai_api_key")
                    base_url = config_manager.get("providers.openai_base_url")
                    organization = config_manager.get("providers.openai_organization")

                    import openai

                    client = openai.OpenAI(
                        api_key=api_key,
                        base_url=base_url if base_url else None,
                        organization=organization if organization else None,
                    )
                    models_response = client.models.list()
                    models = [model.id for model in models_response.data]

                elif provider_name == "lmstudio":
                    from src.core.config import config_manager

                    host = config_manager.get("providers.lmstudio_host", "http://localhost:1234")

                    import requests

                    response = requests.get(f"{host}/v1/models", timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        models = [model.get("id", "Unknown") for model in data.get("data", [])]

                # Update UI on main thread
                if models:
                    default_model = models[0]
                    Clock.schedule_once(
                        lambda dt: self._update_model_selection(models, default_model), 0
                    )
                else:
                    Clock.schedule_once(lambda dt: self._update_model_selection([], None), 0)

            except Exception as e:
                logger.error(f"Error fetching models for {provider_name}: {e}")
                Clock.schedule_once(
                    lambda dt: setattr(self.model_button, "text", "Model: Error"), 0
                )

        import threading

        thread = threading.Thread(target=fetch_models)
        thread.daemon = True
        thread.start()

    def _update_model_selection(self, models, default_model):
        """Update model selection UI."""
        try:
            self.available_models = models
            if default_model:
                self.chat_manager.current_model = default_model
                self.model_button.text = f"Model: {default_model}"
            else:
                self.model_button.text = "Model: None available"

        except Exception as e:
            logger.error(f"Error updating model selection: {e}")

    def _show_model_menu(self, button):
        """Show model selection menu."""
        try:
            if not hasattr(self, "available_models") or not self.available_models:
                Notification.warning("No models available for current provider")
                return

            # Create dialog with model list
            from kivymd.uix.dialog import MDDialog
            from kivymd.uix.list import OneLineListItem

            items = []
            for model in self.available_models:
                item = OneLineListItem(
                    text=model, on_release=lambda x, m=model: self._select_model(m)
                )
                items.append(item)

            self.model_dialog = MDDialog(
                title="Select Model",
                type="simple",
                items=items,
                size_hint=(0.8, None),
                height=dp(min(300, len(self.available_models) * 50 + 100)),
            )
            self.model_dialog.open()

        except Exception as e:
            logger.error(f"Error showing model menu: {e}")
            Notification.error("Failed to show model menu")

    def _select_model(self, model_name):
        """Select a model."""
        try:
            self.chat_manager.current_model = model_name
            self.model_button.text = f"Model: {model_name}"

            # Close dialog
            if hasattr(self, "model_dialog"):
                self.model_dialog.dismiss()

            Notification.success(f"Model changed to {model_name}")

        except Exception as e:
            logger.error(f"Error selecting model: {e}")
            Notification.error("Failed to change model")

    def _create_branch(self, message_id):
        """Create a new conversation branch from a specific message."""
        try:
            # Find the message to branch from
            messages_up_to_branch = []
            found_branch_point = False

            for child in reversed(self.messages_container.children):
                if hasattr(child, "message_id"):
                    messages_up_to_branch.append(child)
                    if child.message_id == message_id:
                        found_branch_point = True
                        break

            if not found_branch_point:
                Notification.warning("Could not find message to branch from")
                return

            # Clear current conversation
            self.messages_container.clear_widgets()

            # Add system message about branching
            branch_msg = {
                "content": "[BRANCH] Conversation branched! Previous messages preserved above.",
                "role": "system",
                "created_at": datetime.now().isoformat(),
                "model": "system",
            }
            self._add_message_to_ui(branch_msg, is_user=False)

            # Re-add messages up to branch point
            for msg_widget in messages_up_to_branch:
                if hasattr(msg_widget, "content_label"):
                    msg_data = {
                        "content": msg_widget.content_label.text,
                        "role": "user" if msg_widget.is_user else "assistant",
                        "created_at": datetime.now().isoformat(),
                    }
                    self._add_message_to_ui(msg_data, is_user=msg_widget.is_user)

            # Update title
            self.toolbar_title.text = "Branched Conversation"

            Notification.success("Created new conversation branch!")

        except Exception as e:
            logger.error(f"Failed to create branch: {e}")
            Notification.error("Failed to create branch")

    def _attach_file(self, *args):
        """Attach a file to the message."""
        try:
            from kivymd.uix.filemanager import MDFileManager

            # Create file picker dialog
            def file_selected(path):
                self.file_manager.close()
                if path and Path(path).exists():
                    self._process_attached_file(path)
                else:
                    Notification.warning("No file selected")

            def file_manager_exit():
                self.file_manager.close()

            self.file_manager = MDFileManager(
                exit_manager=file_manager_exit, select_path=file_selected, preview=True
            )

            # Open in home directory
            self.file_manager.show(str(Path.home()))

        except Exception as e:
            logger.error(f"Failed to open file picker: {e}")
            Notification.error("File picker not available")

    def _process_attached_file(self, file_path):
        """Process an attached file."""
        try:
            file_path = Path(file_path)

            # Check file size (limit to 10MB for demo)
            if file_path.stat().st_size > 10 * 1024 * 1024:
                Notification.error("File too large (max 10MB)")
                return

            # Add file attachment message
            attachment_msg = {
                "content": f"[FILE] Attached: {file_path.name} ({file_path.stat().st_size} bytes)",
                "role": "user",
                "created_at": datetime.now().isoformat(),
                "file_attachment": str(file_path),
            }
            self._add_message_to_ui(attachment_msg, is_user=True)

            # For text files, could read and process content
            if file_path.suffix.lower() in [".txt", ".md", ".py", ".json"]:
                try:
                    content = file_path.read_text(encoding="utf-8")[:2000]  # Limit content
                    content_msg = {
                        "content": f"File content preview:\n```\n{content}\n```",
                        "role": "user",
                        "created_at": datetime.now().isoformat(),
                    }
                    self._add_message_to_ui(content_msg, is_user=True)
                except Exception as e:
                    logger.error(f"Failed to read file content: {e}")

            Notification.success(f"Attached {file_path.name}")

        except Exception as e:
            logger.error(f"Failed to process attached file: {e}")
            Notification.error("Failed to process file")

    def _voice_input(self, *args):
        """Start voice input."""
        try:
            # Check if microphone permissions and libraries are available
            try:
                import pyaudio
                import speech_recognition as sr
            except ImportError:
                Notification.warning(
                    "Voice input requires 'speech_recognition' and 'pyaudio' packages"
                )
                return

            def start_recording():
                try:
                    # Initialize recognizer
                    recognizer = sr.Recognizer()
                    microphone = sr.Microphone()

                    # Show recording indicator
                    self.voice_recording_indicator = MDLabel(
                        text="[REC] Recording... (Click to stop)",
                        theme_text_color="Custom",
                        text_color=(1.0, 0.2, 0.2, 1.0),
                        pos_hint={"center_x": 0.5, "center_y": 0.9},
                        size_hint=(None, None),
                    )
                    self.add_widget(self.voice_recording_indicator)

                    # Record audio
                    with microphone as source:
                        recognizer.adjust_for_ambient_noise(source)

                    Clock.schedule_once(lambda dt: Notification.info("Recording started"), 0)

                    # Record for up to 10 seconds
                    with microphone as source:
                        audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)

                    # Remove recording indicator
                    if hasattr(self, "voice_recording_indicator"):
                        self.remove_widget(self.voice_recording_indicator)

                    Clock.schedule_once(lambda dt: Notification.info("Processing speech..."), 0)

                    # Recognize speech
                    text = recognizer.recognize_google(audio)

                    # Add to message input
                    Clock.schedule_once(lambda dt: self._add_voice_text(text), 0)

                except sr.WaitTimeoutError:
                    Clock.schedule_once(lambda dt: Notification.warning("No speech detected"), 0)
                except sr.UnknownValueError:
                    Clock.schedule_once(
                        lambda dt: Notification.error("Could not understand speech"), 0
                    )
                except sr.RequestError:
                    Clock.schedule_once(
                        lambda dt: Notification.error(f"Speech recognition error: {e}"), 0
                    )
                except Exception:
                    Clock.schedule_once(lambda dt: Notification.error(f"Voice input error: {e}"), 0)
                finally:
                    # Clean up recording indicator
                    if hasattr(self, "voice_recording_indicator"):
                        Clock.schedule_once(
                            lambda dt: self.remove_widget(self.voice_recording_indicator), 0
                        )

            # Start recording in background thread
            import threading

            thread = threading.Thread(target=start_recording)
            thread.daemon = True
            thread.start()

        except Exception as e:
            logger.error(f"Failed to start voice input: {e}")
            Notification.error("Voice input not available")

    def _add_voice_text(self, text):
        """Add voice-recognized text to input field."""
        try:
            current_text = self.message_input.text
            if current_text:
                self.message_input.text = f"{current_text} {text}"
            else:
                self.message_input.text = text

            Notification.success(f"Voice input: '{text}'")

        except Exception as e:
            logger.error(f"Failed to add voice text: {e}")

    def _export_conversation(self):
        """Export current conversation."""
        try:
            if not self.messages_container.children:
                Notification.warning("No messages to export")
                return

            # Collect all messages
            messages = []
            for child in reversed(self.messages_container.children):
                if hasattr(child, "content_label"):
                    messages.append(
                        {
                            "content": child.content_label.text,
                            "is_user": child.is_user,
                            "timestamp": getattr(child, "timestamp", datetime.now().isoformat()),
                        }
                    )

            # Create export content
            export_content = "# Neuromancer Conversation Export\n"
            export_content += f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            export_content += f"Provider: {self.chat_manager.current_provider or 'Unknown'}\n"
            export_content += f"Model: {self.chat_manager.current_model or 'Unknown'}\n\n"

            for msg in messages:
                role = "User" if msg["is_user"] else "Assistant"
                export_content += f"## {role}\n{msg['content']}\n\n"

            # Save to file
            from pathlib import Path

            export_dir = Path.home() / "Downloads"
            export_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_file = export_dir / f"neuromancer_chat_{timestamp}.md"

            export_file.write_text(export_content, encoding="utf-8")

            Notification.success(f"Exported to {export_file.name}")

        except Exception as e:
            logger.error(f"Failed to export conversation: {e}")
            Notification.error("Failed to export conversation")

    async def _async_export_conversation(self):
        """Export conversation asynchronously."""
        try:
            export_data = await self.chat_manager.export_conversation(
                self.current_conversation_id, format="markdown"
            )

            # TODO: Save to file or share
            Notification.success("Conversation exported")

        except Exception as e:
            logger.error(f"Failed to export conversation: {e}")
            Notification.error("Failed to export conversation")

    def _show_analytics(self):
        """Show conversation analytics."""
        try:
            # Analyze current conversation
            total_messages = len(self.messages_container.children)
            user_messages = 0
            assistant_messages = 0
            total_chars = 0

            for child in self.messages_container.children:
                if hasattr(child, "content_label") and hasattr(child, "is_user"):
                    content = child.content_label.text
                    total_chars += len(content)

                    if child.is_user:
                        user_messages += 1
                    else:
                        assistant_messages += 1

            # Create analytics dialog
            from kivymd.uix.boxlayout import MDBoxLayout
            from kivymd.uix.button import MDFlatButton
            from kivymd.uix.dialog import MDDialog
            from kivymd.uix.label import MDLabel

            content = MDBoxLayout(
                orientation="vertical", spacing=dp(10), size_hint_y=None, height=dp(200)
            )

            stats = [
                "CONVERSATION ANALYTICS",
                "",
                f"Total Messages: {total_messages}",
                f"Your Messages: {user_messages}",
                f"AI Responses: {assistant_messages}",
                f"Total Characters: {total_chars:,}",
                f"Average Message Length: {total_chars // max(total_messages, 1)} chars",
                f"Provider: {self.chat_manager.current_provider or 'None'}",
                f"Model: {self.chat_manager.current_model or 'None'}",
            ]

            for stat in stats:
                label = MDLabel(
                    text=stat,
                    theme_text_color="Primary" if "ANALYTICS" in stat else "Secondary",
                    font_style="H6" if "ANALYTICS" in stat else "Body1",
                    adaptive_height=True,
                )
                content.add_widget(label)

            self.analytics_dialog = MDDialog(
                title="Conversation Analytics",
                type="custom",
                content_cls=content,
                buttons=[
                    MDFlatButton(text="Close", on_release=lambda x: self.analytics_dialog.dismiss())
                ],
                size_hint=(0.8, None),
                height=dp(400),
            )
            self.analytics_dialog.open()

        except Exception as e:
            logger.error(f"Failed to show analytics: {e}")
            Notification.error("Failed to load analytics")

    async def _load_analytics(self):
        """Load and display analytics."""
        try:
            stats = await self.chat_manager.get_conversation_stats(self.current_conversation_id)

            # TODO: Display in a nice dialog
            logger.info(f"Conversation stats: {stats}")

        except Exception as e:
            logger.error(f"Failed to load analytics: {e}")

    def go_to_settings(self):
        """Navigate to settings."""
        self.manager.current = "settings"

    def go_to_memory(self):
        """Navigate to memory management."""
        self.manager.current = "memory"
