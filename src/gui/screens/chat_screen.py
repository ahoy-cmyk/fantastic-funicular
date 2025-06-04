"""Chat screen for interacting with LLM providers."""

from kivy.clock import Clock
from kivy.metrics import dp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDIconButton, MDRaisedButton
from kivymd.uix.card import MDCard
from kivymd.uix.label import MDLabel
from kivymd.uix.list import MDList
from kivymd.uix.screen import MDScreen
from kivymd.uix.scrollview import MDScrollView
from kivymd.uix.textfield import MDTextField

from src.core.chat_manager import ChatManager
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class MessageCard(MDCard):
    """Custom card widget for displaying messages."""

    def __init__(self, message: str, is_user: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "vertical"
        self.padding = dp(10)
        self.spacing = dp(5)
        self.elevation = 1
        self.md_bg_color = (0.2, 0.2, 0.2, 1) if is_user else (0.1, 0.1, 0.1, 1)
        self.pos_hint = {"right": 1} if is_user else {"left": 1}
        self.size_hint_x = 0.8
        self.adaptive_height = True

        # Add message content
        label = MDLabel(text=message, adaptive_height=True, theme_text_color="Secondary")
        self.add_widget(label)


class ChatScreen(MDScreen):
    """Main chat interface screen."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.chat_manager = ChatManager()
        self.build_ui()

    def build_ui(self):
        """Build the chat interface."""
        # Main layout
        main_layout = MDBoxLayout(orientation="vertical", padding=dp(10), spacing=dp(10))

        # Header
        header = MDBoxLayout(size_hint_y=None, height=dp(60), spacing=dp(10))

        # Provider selector
        provider_button = MDRaisedButton(
            text="Provider: Ollama", on_release=self.show_provider_menu
        )

        # Model selector
        model_button = MDRaisedButton(text="Model: llama3.2", on_release=self.show_model_menu)

        # Settings button
        settings_button = MDIconButton(icon="cog", on_release=self.go_to_settings)

        # Memory button
        memory_button = MDIconButton(icon="memory", on_release=self.go_to_memory)

        header.add_widget(provider_button)
        header.add_widget(model_button)
        header.add_widget(MDBoxLayout())  # Spacer
        header.add_widget(memory_button)
        header.add_widget(settings_button)

        # Chat messages area
        self.messages_scroll = MDScrollView()
        self.messages_list = MDList(spacing=dp(10), padding=dp(10))
        self.messages_scroll.add_widget(self.messages_list)

        # Input area
        input_layout = MDBoxLayout(size_hint_y=None, height=dp(60), spacing=dp(10))

        self.message_input = MDTextField(
            hint_text="Type your message...", multiline=False, on_text_validate=self.send_message
        )

        send_button = MDIconButton(icon="send", on_release=self.send_message)

        input_layout.add_widget(self.message_input)
        input_layout.add_widget(send_button)

        # Add all to main layout
        main_layout.add_widget(header)
        main_layout.add_widget(self.messages_scroll)
        main_layout.add_widget(input_layout)

        self.add_widget(main_layout)

        # Add welcome message
        self.add_message("Welcome to Neuromancer! I'm ready to assist you.", is_user=False)

    def send_message(self, *args):
        """Send a message to the chat."""
        message = self.message_input.text.strip()
        if not message:
            return

        # Add user message
        self.add_message(message, is_user=True)

        # Clear input
        self.message_input.text = ""

        # Get AI response (async)
        Clock.schedule_once(lambda dt: self.get_ai_response(message), 0.1)

    def add_message(self, message: str, is_user: bool = True):
        """Add a message to the chat display."""
        msg_card = MessageCard(message=message, is_user=is_user)
        self.messages_list.add_widget(msg_card)

        # Scroll to bottom
        Clock.schedule_once(lambda dt: self.scroll_to_bottom(), 0.1)

    def scroll_to_bottom(self):
        """Scroll chat to the bottom."""
        self.messages_scroll.scroll_y = 0

    def get_ai_response(self, message: str):
        """Get response from AI provider."""
        try:
            # TODO: Implement actual AI response
            response = f"I received your message: '{message}'. LLM integration coming soon!"
            self.add_message(response, is_user=False)
        except Exception as e:
            logger.error(f"Error getting AI response: {e}")
            self.add_message("Sorry, I encountered an error. Please try again.", is_user=False)

    def show_provider_menu(self, button):
        """Show provider selection menu."""
        # TODO: Implement provider selection
        logger.info("Provider menu requested")

    def show_model_menu(self, button):
        """Show model selection menu."""
        # TODO: Implement model selection
        logger.info("Model menu requested")

    def go_to_settings(self, *args):
        """Navigate to settings screen."""
        self.manager.current = "settings"

    def go_to_memory(self, *args):
        """Navigate to memory screen."""
        self.manager.current = "memory"
