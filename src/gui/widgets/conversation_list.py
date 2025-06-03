"""Conversation list widget with search and filtering."""

from collections.abc import Callable
from datetime import datetime
from typing import Any

from kivy.metrics import dp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDFlatButton, MDRaisedButton
from kivymd.uix.chip import MDChip
from kivymd.uix.label import MDLabel
from kivymd.uix.list import MDList, ThreeLineAvatarIconListItem
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.scrollview import MDScrollView
from kivymd.uix.spinner import MDSpinner as MDCircularProgressIndicator
from kivymd.uix.textfield import MDTextField

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ConversationItem(ThreeLineAvatarIconListItem):
    """Individual conversation list item."""

    def __init__(
        self,
        conversation_data: dict[str, Any],
        on_select: Callable | None = None,
        on_delete: Callable | None = None,
        on_archive: Callable | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.conversation_id = conversation_data["id"]
        self.conversation_data = conversation_data
        self.on_select = on_select
        self.on_delete = on_delete
        self.on_archive = on_archive

        # Set text
        self.text = conversation_data.get("title", "Untitled")

        # Format secondary text
        created = datetime.fromisoformat(conversation_data["created_at"])
        self.secondary_text = (
            f"{conversation_data.get('message_count', 0)} messages • {created:%b %d, %Y}"
        )

        # Format tertiary text with model and cost
        model = conversation_data.get("model", "Unknown")
        cost = conversation_data.get("total_cost", 0)
        self.tertiary_text = f"{model} • ${cost:.4f}"

        # Add status indicator
        status_texts = {"active": "[Active]", "archived": "[Archived]", "deleted": "[Deleted]"}
        status_text = status_texts.get(conversation_data.get("status", "active"), "[Active]")

        # Add status label
        status_label = MDLabel(
            text=status_text,
            size_hint=(None, None),
            size=(dp(60), dp(40)),
            pos_hint={"center_y": 0.5},
            theme_text_color="Secondary",
            font_style="Caption",
        )
        self.add_widget(status_label)

        # Add action buttons
        action_box = MDBoxLayout(adaptive_width=True, spacing=dp(5))

        if on_archive:
            archive_btn = MDFlatButton(text="Archive", on_release=lambda x: self._on_archive())
            action_box.add_widget(archive_btn)

        if on_delete:
            delete_btn = MDFlatButton(text="Delete", on_release=lambda x: self._on_delete())
            action_box.add_widget(delete_btn)

        # Add action box directly
        self.add_widget(action_box)

        # Add tags as chips
        if conversation_data.get("tags"):
            self._add_tags(conversation_data["tags"])

    def _add_tags(self, tags: list[str]):
        """Add tag chips to the item."""
        # TODO: Implement tag display
        pass

    def on_touch_down(self, touch):
        """Handle touch events."""
        if self.collide_point(*touch.pos):
            if self.on_select:
                self.on_select(self.conversation_id)
            return True
        return super().on_touch_down(touch)

    def _on_archive(self):
        """Handle archive action."""
        if self.on_archive:
            self.on_archive(self.conversation_id)

    def _on_delete(self):
        """Handle delete action."""
        if self.on_delete:
            self.on_delete(self.conversation_id)


class ConversationList(MDBoxLayout):
    """Conversation list with search and filtering."""

    def __init__(
        self,
        on_conversation_select: Callable | None = None,
        on_new_conversation: Callable | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.orientation = "vertical"
        self.spacing = dp(10)

        self.on_conversation_select = on_conversation_select
        self.on_new_conversation = on_new_conversation

        self.conversations: list[dict[str, Any]] = []
        self.filtered_conversations: list[dict[str, Any]] = []
        self.current_filter = "active"
        self.search_query = ""

        self.build_ui()

    def build_ui(self):
        """Build the conversation list UI."""
        # Search and filter bar
        search_bar = MDBoxLayout(
            size_hint_y=None, height=dp(60), spacing=dp(10), padding=[dp(10), 0]
        )

        # Search field
        self.search_field = MDTextField(
            hint_text="Search conversations...", on_text=self._on_search_text
        )

        # Filter button
        self.filter_button = MDFlatButton(text="Filter", on_release=self._show_filter_menu)

        # New conversation button
        new_btn = MDRaisedButton(
            text="New Chat",
            on_release=lambda x: self.on_new_conversation() if self.on_new_conversation else None,
        )

        search_bar.add_widget(self.search_field)
        search_bar.add_widget(self.filter_button)
        search_bar.add_widget(new_btn)

        # Filter chips
        self.filter_chips = MDBoxLayout(adaptive_height=True, spacing=dp(5), padding=[dp(10), 0])
        self._update_filter_chips()

        # Conversation list
        self.scroll_view = MDScrollView()
        self.conversation_list = MDList()
        self.scroll_view.add_widget(self.conversation_list)

        # Loading indicator
        self.loading_indicator = MDCircularProgressIndicator(
            size_hint=(None, None),
            size=(dp(48), dp(48)),
            pos_hint={"center_x": 0.5, "center_y": 0.5},
        )

        # Empty state
        self.empty_state = MDLabel(
            text="No conversations found", halign="center", theme_text_color="Hint"
        )

        # Add to layout
        self.add_widget(search_bar)
        self.add_widget(self.filter_chips)
        self.add_widget(self.scroll_view)

    def _update_filter_chips(self):
        """Update filter chip display."""
        self.filter_chips.clear_widgets()

        filters = ["active", "archived", "all"]
        for filter_name in filters:
            chip = MDChip(
                text=filter_name.title(),
                active=self.current_filter == filter_name,
                on_active=lambda w, v, f=filter_name: self._on_filter_change(f) if v else None,
            )
            self.filter_chips.add_widget(chip)

    def _on_search_text(self, widget, text):
        """Handle search text changes."""
        self.search_query = text.lower()
        self._apply_filters()

    def _on_filter_change(self, filter_name: str):
        """Handle filter changes."""
        self.current_filter = filter_name
        self._update_filter_chips()
        self._apply_filters()

    def _show_filter_menu(self, widget):
        """Show advanced filter menu."""
        menu_items = [
            {"text": "Sort by Date", "on_release": lambda: self._sort_conversations("date")},
            {"text": "Sort by Title", "on_release": lambda: self._sort_conversations("title")},
            {"text": "Sort by Cost", "on_release": lambda: self._sort_conversations("cost")},
            {
                "text": "Sort by Messages",
                "on_release": lambda: self._sort_conversations("messages"),
            },
        ]

        self.filter_menu = MDDropdownMenu(caller=widget, items=menu_items, width_mult=4)
        self.filter_menu.open()

    def _sort_conversations(self, sort_by: str):
        """Sort conversations by specified criteria."""
        if sort_by == "date":
            self.filtered_conversations.sort(key=lambda c: c.get("updated_at", ""), reverse=True)
        elif sort_by == "title":
            self.filtered_conversations.sort(key=lambda c: c.get("title", "").lower())
        elif sort_by == "cost":
            self.filtered_conversations.sort(key=lambda c: c.get("total_cost", 0), reverse=True)
        elif sort_by == "messages":
            self.filtered_conversations.sort(key=lambda c: c.get("message_count", 0), reverse=True)

        self._update_display()
        self.filter_menu.dismiss()

    def _apply_filters(self):
        """Apply search and filter criteria."""
        self.filtered_conversations = []

        for conv in self.conversations:
            # Filter by status
            if self.current_filter != "all":
                if conv.get("status") != self.current_filter:
                    continue

            # Filter by search query
            if self.search_query:
                title = conv.get("title", "").lower()
                desc = conv.get("description", "").lower()
                if self.search_query not in title and self.search_query not in desc:
                    continue

            self.filtered_conversations.append(conv)

        self._update_display()

    def _update_display(self):
        """Update the conversation list display."""
        self.conversation_list.clear_widgets()

        if not self.filtered_conversations:
            self.conversation_list.add_widget(self.empty_state)
            return

        for conv in self.filtered_conversations:
            item = ConversationItem(
                conversation_data=conv,
                on_select=self._on_conversation_select,
                on_delete=self._on_conversation_delete,
                on_archive=self._on_conversation_archive,
            )
            self.conversation_list.add_widget(item)

    def _on_conversation_select(self, conversation_id: str):
        """Handle conversation selection."""
        if self.on_conversation_select:
            self.on_conversation_select(conversation_id)

    def _on_conversation_delete(self, conversation_id: str):
        """Handle conversation deletion."""
        # TODO: Show confirmation dialog
        logger.info(f"Delete conversation: {conversation_id}")

    def _on_conversation_archive(self, conversation_id: str):
        """Handle conversation archiving."""
        # TODO: Archive conversation
        logger.info(f"Archive conversation: {conversation_id}")

    def set_conversations(self, conversations: list[dict[str, Any]]):
        """Set the conversation list."""
        self.conversations = conversations
        self._apply_filters()

    def show_loading(self):
        """Show loading indicator."""
        self.conversation_list.clear_widgets()
        self.conversation_list.add_widget(self.loading_indicator)

    def refresh(self):
        """Refresh the conversation list."""
        # This would typically trigger a reload from the chat manager
        pass
