"""Simplified Advanced Memory Management Screen."""

import asyncio
import threading
from typing import Any

from kivy.animation import Animation
from kivy.clock import Clock
from kivy.metrics import dp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDFloatingActionButton, MDIconButton, MDRaisedButton
from kivymd.uix.card import MDCard
from kivymd.uix.chip import MDChip
from kivymd.uix.floatlayout import MDFloatLayout
from kivymd.uix.label import MDLabel
from kivymd.uix.progressbar import MDProgressBar
from kivymd.uix.screen import MDScreen
from kivymd.uix.scrollview import MDScrollView
from kivymd.uix.selectioncontrol import MDCheckbox
from kivymd.uix.textfield import MDTextField
from kivymd.uix.toolbar import MDTopAppBar

from src.gui.utils.notifications import Notification
from src.memory import Memory, MemoryType
from src.memory.safe_operations import create_safe_memory_manager
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class MemoryStatsCard(MDCard):
    """Enhanced stats card with animations and real-time updates."""

    def __init__(self, title: str, value: str, icon: str, color: tuple, trend: str = "", **kwargs):
        super().__init__(**kwargs)
        self.elevation = 8
        self.radius = [dp(16)]
        self.md_bg_color = (0.08, 0.08, 0.10, 1)
        self.size_hint_y = None
        self.height = dp(120)
        self.padding = dp(16)
        self.spacing = dp(8)
        self.orientation = "vertical"

        # Header with icon and trend
        header_layout = MDBoxLayout(
            orientation="horizontal", size_hint_y=None, height=dp(30), spacing=dp(8)
        )

        # Title
        title_label = MDLabel(
            text=title, font_style="Caption", theme_text_color="Secondary", adaptive_height=True
        )
        header_layout.add_widget(title_label)

        # Trend indicator
        if trend:
            trend_label = MDLabel(
                text=trend,
                font_style="Caption",
                theme_text_color="Custom",
                text_color=(0.2, 0.8, 0.2, 1) if trend.startswith("+") else (0.8, 0.2, 0.2, 1),
                size_hint_x=None,
                width=dp(50),
                halign="right",
            )
            header_layout.add_widget(trend_label)

        self.add_widget(header_layout)

        # Value with animation support
        self.value_label = MDLabel(
            text=value, font_style="H4", theme_text_color="Primary", bold=True, adaptive_height=True
        )
        self.add_widget(self.value_label)

        # Progress bar for percentage values
        if "%" in value:
            try:
                percent = float(value.replace("%", ""))
                progress_color = color if percent < 80 else (1.0, 0.6, 0.2, 1.0)
                self.progress_bar = MDProgressBar(
                    value=percent, max=100, color=progress_color, size_hint_y=None, height=dp(4)
                )
                self.add_widget(self.progress_bar)
            except ValueError:
                pass


class MemoryItemCard(MDCard):
    """Individual memory item card with rich functionality."""

    def __init__(self, memory: Memory, on_delete=None, on_edit=None, on_view=None, **kwargs):
        super().__init__(**kwargs)
        self.memory = memory
        self.on_delete = on_delete
        self.on_edit = on_edit
        self.on_view = on_view

        self.elevation = 4
        self.radius = [dp(12)]
        self.md_bg_color = (0.10, 0.10, 0.12, 1)
        self.size_hint_y = None
        self.adaptive_height = True
        self.padding = dp(12)
        self.spacing = dp(8)
        self.orientation = "vertical"

        # Selection state
        self.selected = False

        # Build UI
        self._build_ui()

    def _build_ui(self):
        """Build the memory item UI."""
        # Header with type, importance, and selection
        header = MDBoxLayout(
            orientation="horizontal", size_hint_y=None, height=dp(30), spacing=dp(8)
        )

        # Selection checkbox
        self.selection_checkbox = MDCheckbox(
            size_hint=(None, None), size=(dp(20), dp(20)), on_active=self._on_selection_change
        )
        header.add_widget(self.selection_checkbox)

        # Memory type badge
        type_color = {
            MemoryType.SHORT_TERM: (0.2, 0.6, 1.0, 1),
            MemoryType.LONG_TERM: (0.8, 0.4, 0.2, 1),
            MemoryType.EPISODIC: (0.6, 0.2, 0.8, 1),
            MemoryType.SEMANTIC: (0.2, 0.8, 0.4, 1),
        }

        type_chip = MDChip(
            text=self.memory.memory_type.value.replace("_", " ").title(),
            md_bg_color=type_color.get(self.memory.memory_type, (0.5, 0.5, 0.5, 1)),
            size_hint_x=None,
            height=dp(24),
        )
        header.add_widget(type_chip)

        # Importance indicator
        importance_stars = "★" * int(self.memory.importance * 5)
        importance_label = MDLabel(
            text=importance_stars,
            theme_text_color="Custom",
            text_color=(1.0, 0.8, 0.2, 1),
            size_hint_x=None,
            width=dp(60),
            font_style="Caption",
        )
        header.add_widget(importance_label)

        # Spacer
        header.add_widget(MDBoxLayout())

        # Actions
        actions_layout = MDBoxLayout(
            orientation="horizontal", size_hint_x=None, width=dp(120), spacing=dp(4)
        )

        view_btn = MDIconButton(
            icon="eye",
            theme_icon_color="Primary",
            on_release=lambda x: self._on_view() if self.on_view else None,
        )

        edit_btn = MDIconButton(
            icon="pencil",
            theme_icon_color="Primary",
            on_release=lambda x: self._on_edit() if self.on_edit else None,
        )

        delete_btn = MDIconButton(
            icon="delete",
            theme_icon_color="Error",
            on_release=lambda x: self._on_delete() if self.on_delete else None,
        )

        actions_layout.add_widget(view_btn)
        actions_layout.add_widget(edit_btn)
        actions_layout.add_widget(delete_btn)

        header.add_widget(actions_layout)
        self.add_widget(header)

        # Content preview
        content_preview = (
            self.memory.content[:100] + "..."
            if len(self.memory.content) > 100
            else self.memory.content
        )
        content_label = MDLabel(
            text=content_preview,
            theme_text_color="Primary",
            adaptive_height=True,
            text_size=(None, None),
        )
        self.add_widget(content_label)

        # Metadata footer
        footer = MDBoxLayout(
            orientation="horizontal", size_hint_y=None, height=dp(20), spacing=dp(16)
        )

        # Created date
        created_label = MDLabel(
            text=f"Created: {self.memory.created_at.strftime('%m/%d %H:%M')}",
            font_style="Caption",
            theme_text_color="Secondary",
            size_hint_x=None,
            width=dp(120),
        )
        footer.add_widget(created_label)

        # Last accessed
        accessed_label = MDLabel(
            text=f"Accessed: {self.memory.accessed_at.strftime('%m/%d %H:%M')}",
            font_style="Caption",
            theme_text_color="Secondary",
            size_hint_x=None,
            width=dp(120),
        )
        footer.add_widget(accessed_label)

        # ID (truncated)
        id_label = MDLabel(
            text=f"ID: {self.memory.id[:8]}...",
            font_style="Caption",
            theme_text_color="Hint",
            halign="right",
        )
        footer.add_widget(id_label)

        self.add_widget(footer)

    def _on_selection_change(self, checkbox, value):
        """Handle selection state change."""
        self.selected = value
        if value:
            self.md_bg_color = (0.15, 0.15, 0.20, 1)
        else:
            self.md_bg_color = (0.10, 0.10, 0.12, 1)

    def _on_view(self):
        """Handle view action."""
        if self.on_view:
            self.on_view(self.memory)

    def _on_edit(self):
        """Handle edit action."""
        if self.on_edit:
            self.on_edit(self.memory)

    def _on_delete(self):
        """Handle delete action."""
        if self.on_delete:
            self.on_delete(self.memory)


class SimpleMemoryScreen(MDScreen):
    """Simplified advanced memory management interface."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "advanced_memory"

        # Initialize memory manager
        self.safe_memory = create_safe_memory_manager(self._memory_error_callback)

        # State management
        self.memories = []
        self.filtered_memories = []
        self.selected_memories = []
        self.current_filter = {}
        self.search_query = ""
        self.stats_cache = {}
        self.is_loading = False

        # Build UI
        self.build_ui()

        # Load initial data
        Clock.schedule_once(self._load_initial_data, 0.1)

    def build_ui(self):
        """Build the memory management interface."""
        main_layout = MDBoxLayout(orientation="vertical")

        # Enhanced toolbar with actions
        self.toolbar = MDTopAppBar(
            title="Memory Management", elevation=4, md_bg_color=(0.05, 0.05, 0.08, 1)
        )

        self.toolbar.left_action_items = [
            ["arrow-left", lambda x: self._go_back()],
            ["refresh", lambda x: self._refresh_all()],
        ]

        self.toolbar.right_action_items = [
            ["magnify", lambda x: self._toggle_search()],
            ["plus", lambda x: self._show_add_memory_dialog()],
        ]

        main_layout.add_widget(self.toolbar)

        # Quick stats bar
        self.stats_layout = MDBoxLayout(
            orientation="horizontal",
            size_hint_y=None,
            height=dp(140),
            padding=dp(16),
            spacing=dp(16),
        )

        # Stats cards will be populated dynamically
        self.stats_cards = {}
        main_layout.add_widget(self.stats_layout)

        # Search bar (initially hidden)
        self.search_layout = MDBoxLayout(
            orientation="horizontal",
            size_hint_y=None,
            height=0,
            opacity=0,
            padding=dp(16),
            spacing=dp(8),
        )

        self.search_field = MDTextField(
            hint_text="Search memories by content, type, or metadata...",
            on_text=self._on_search_text_change,
        )

        clear_search_btn = MDIconButton(icon="close", on_release=lambda x: self._clear_search())

        self.search_layout.add_widget(self.search_field)
        self.search_layout.add_widget(clear_search_btn)

        main_layout.add_widget(self.search_layout)

        # Content area
        content_layout = MDBoxLayout(orientation="vertical")

        # Filter and bulk action bar
        action_bar = MDBoxLayout(
            orientation="horizontal", size_hint_y=None, height=dp(50), padding=dp(8), spacing=dp(8)
        )

        # Select all checkbox
        self.select_all_checkbox = MDCheckbox(
            size_hint=(None, None), size=(dp(20), dp(20)), on_active=self._on_select_all
        )
        action_bar.add_widget(self.select_all_checkbox)

        select_all_label = MDLabel(
            text="Select All",
            size_hint_x=None,
            width=dp(80),
            theme_text_color="Secondary",
            font_style="Caption",
        )
        action_bar.add_widget(select_all_label)

        # Spacer
        action_bar.add_widget(MDBoxLayout())

        # Bulk actions
        bulk_delete_btn = MDRaisedButton(
            text="Delete Selected",
            md_bg_color=(0.8, 0.2, 0.2, 1),
            on_release=self._bulk_delete_selected,
            size_hint_x=None,
            width=dp(140),
        )
        action_bar.add_widget(bulk_delete_btn)

        content_layout.add_widget(action_bar)

        # Memories list
        self.memories_scroll = MDScrollView()
        self.memories_list = MDBoxLayout(
            orientation="vertical", spacing=dp(8), padding=dp(8), adaptive_height=True
        )
        self.memories_scroll.add_widget(self.memories_list)
        content_layout.add_widget(self.memories_scroll)

        main_layout.add_widget(content_layout)

        # Floating action button for adding new memories
        self.fab = MDFloatingActionButton(
            icon="plus",
            pos_hint={"center_x": 0.9, "center_y": 0.1},
            elevation=8,
            on_release=self._show_add_memory_dialog,
        )

        # Main container
        float_layout = MDFloatLayout()
        float_layout.add_widget(main_layout)
        float_layout.add_widget(self.fab)

        self.add_widget(float_layout)

    def _memory_error_callback(self, operation: str, error: str):
        """Handle memory operation errors."""
        logger.error(f"Memory operation failed - {operation}: {error}")
        Notification.error(f"Memory error: {operation}")

    def _load_initial_data(self, dt):
        """Load initial memory data."""
        if self.is_loading:
            return

        self.is_loading = True

        def run_async():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._load_memories_async())
            except Exception as e:
                logger.error(f"Failed to load initial data: {e}")
                Clock.schedule_once(lambda dt: Notification.error("Failed to load memory data"), 0)
            finally:
                loop.close()
                Clock.schedule_once(lambda dt: setattr(self, "is_loading", False), 0)

        thread = threading.Thread(target=run_async)
        thread.daemon = True
        thread.start()

    async def _load_memories_async(self):
        """Load memories asynchronously."""
        try:
            # Load all memories by searching with empty query
            all_memories = []
            for memory_type in MemoryType:
                memories = await self.safe_memory.safe_recall(
                    query="", memory_types=[memory_type], limit=1000, threshold=0.0
                )
                all_memories.extend(memories)

            # Load stats
            stats = await self.safe_memory.safe_get_stats()

            # Update UI on main thread
            Clock.schedule_once(lambda dt: self._update_memories_ui(all_memories, stats), 0)

        except Exception as e:
            logger.error(f"Failed to load memories: {e}")
            Clock.schedule_once(lambda dt: Notification.error("Failed to load memories"), 0)

    def _update_memories_ui(self, memories: list[Memory], stats: dict[str, Any]):
        """Update the memories UI with loaded data."""
        try:
            self.memories = memories
            self.filtered_memories = memories.copy()
            self.stats_cache = stats

            # Update stats cards
            self._update_stats_cards(stats)

            # Update memories list
            self._refresh_memories_list()

            logger.info(f"Loaded {len(memories)} memories")

        except Exception as e:
            logger.error(f"Failed to update memories UI: {e}")

    def _update_stats_cards(self, stats: dict[str, Any]):
        """Update the statistics cards."""
        try:
            # Clear existing cards
            self.stats_layout.clear_widgets()
            self.stats_cards.clear()

            # Total memories
            total_card = MemoryStatsCard(
                title="Total Memories",
                value=str(stats.get("total", 0)),
                icon="database",
                color=(0.2, 0.6, 1.0, 1),
                trend="+12 today",
            )
            self.stats_cards["total"] = total_card
            self.stats_layout.add_widget(total_card)

            # Average importance
            avg_importance = stats.get("avg_importance", 0.0)
            importance_card = MemoryStatsCard(
                title="Avg Importance",
                value=f"{avg_importance:.2f}",
                icon="star",
                color=(1.0, 0.8, 0.2, 1),
                trend="+0.05",
            )
            self.stats_cards["importance"] = importance_card
            self.stats_layout.add_widget(importance_card)

            # Memory health (simulated)
            health_percentage = min(95, 80 + (avg_importance * 15))
            health_card = MemoryStatsCard(
                title="Memory Health",
                value=f"{health_percentage:.0f}%",
                icon="heart",
                color=(0.2, 0.8, 0.4, 1),
                trend="+2%",
            )
            self.stats_cards["health"] = health_card
            self.stats_layout.add_widget(health_card)

        except Exception as e:
            logger.error(f"Failed to update stats cards: {e}")

    def _refresh_memories_list(self):
        """Refresh the memories list display."""
        try:
            # Clear current list
            self.memories_list.clear_widgets()

            # Add memory cards
            for memory in self.filtered_memories:
                memory_card = MemoryItemCard(
                    memory=memory,
                    on_delete=self._confirm_delete_memory,
                    on_edit=self._edit_memory,
                    on_view=self._view_memory,
                )
                self.memories_list.add_widget(memory_card)

        except Exception as e:
            logger.error(f"Failed to refresh memories list: {e}")

    # UI Event Handlers
    def _go_back(self):
        """Navigate back to previous screen."""
        self.manager.current = "enhanced_chat"

    def _refresh_all(self):
        """Refresh all data."""
        if not self.is_loading:
            self._load_initial_data(None)
            Notification.info("Refreshing memory data...")

    def _toggle_search(self):
        """Toggle search bar visibility."""
        if self.search_layout.height == 0:
            # Show search
            anim = Animation(height=dp(60), opacity=1, duration=0.3)
            anim.start(self.search_layout)
            self.search_field.focus = True
        else:
            # Hide search
            anim = Animation(height=0, opacity=0, duration=0.3)
            anim.start(self.search_layout)
            self._clear_search()

    def _on_search_text_change(self, instance, text):
        """Handle search text changes."""
        self.search_query = text.strip().lower()
        # Debounce search
        Clock.unschedule(self._perform_search)
        if self.search_query:
            Clock.schedule_once(lambda dt: self._perform_search(), 0.5)
        else:
            self._clear_search()

    def _perform_search(self):
        """Perform memory search."""
        if not self.search_query:
            return

        try:
            self.filtered_memories = [
                memory
                for memory in self.memories
                if (
                    self.search_query in memory.content.lower()
                    or self.search_query in memory.memory_type.value.lower()
                    or any(self.search_query in str(v).lower() for v in memory.metadata.values())
                )
            ]

            self._refresh_memories_list()
            Notification.info(f"Found {len(self.filtered_memories)} matching memories")

        except Exception as e:
            logger.error(f"Search failed: {e}")
            Notification.error("Search failed")

    def _clear_search(self):
        """Clear search and show all memories."""
        self.search_query = ""
        self.search_field.text = ""
        self.filtered_memories = self.memories.copy()
        self._refresh_memories_list()

    def _on_select_all(self, checkbox, value):
        """Handle select all checkbox."""
        try:
            for widget in self.memories_list.children:
                if hasattr(widget, "selection_checkbox"):
                    widget.selection_checkbox.active = value
        except Exception as e:
            logger.error(f"Failed to handle select all: {e}")

    def _show_add_memory_dialog(self, *args):
        """Show dialog to manually add a new memory."""
        Notification.info("Manual memory creation coming soon!")

    def _bulk_delete_selected(self, *args):
        """Delete all selected memories."""
        Notification.info("Bulk deletion coming soon!")

    def _confirm_delete_memory(self, memory: Memory):
        """Show confirmation dialog for deleting a memory."""
        Notification.info(f"Delete memory: {memory.id[:8]}... (coming soon!)")

    def _edit_memory(self, memory: Memory):
        """Edit a memory."""
        Notification.info(f"Edit memory: {memory.id[:8]}... (coming soon!)")

    def _view_memory(self, memory: Memory):
        """View memory details."""
        Notification.info(f"View memory: {memory.id[:8]}... (coming soon!)")
