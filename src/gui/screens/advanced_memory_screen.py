"""Advanced Memory Management Screen with comprehensive functionality."""

import asyncio
import threading
from datetime import datetime
from typing import Any

from kivy.animation import Animation
from kivy.clock import Clock
from kivy.metrics import dp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDFloatingActionButton, MDIconButton, MDRaisedButton
from kivymd.uix.card import MDCard
from kivymd.uix.chip import MDChip
from kivymd.uix.dialog import MDDialog
from kivymd.uix.floatlayout import MDFloatLayout
from kivymd.uix.label import MDLabel
from kivymd.uix.progressbar import MDProgressBar
from kivymd.uix.screen import MDScreen
from kivymd.uix.scrollview import MDScrollView
from kivymd.uix.selectioncontrol import MDCheckbox
from kivymd.uix.slider import MDSlider
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

        # Icon
        icon_label = MDLabel(
            text=f"[size=20][font=Icons]{icon}[/font][/size]",
            markup=True,
            theme_text_color="Custom",
            text_color=color,
            size_hint_x=None,
            width=dp(30),
        )
        header_layout.add_widget(icon_label)

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

    def animate_value_change(self, new_value: str):
        """Animate value changes."""
        # Scale animation
        anim = Animation(scale_x=1.1, scale_y=1.1, duration=0.1) + Animation(
            scale_x=1.0, scale_y=1.0, duration=0.1
        )

        def update_value(dt):
            self.value_label.text = new_value

        Clock.schedule_once(update_value, 0.05)
        anim.start(self.value_label)


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


class AdvancedMemoryScreen(MDScreen):
    """Advanced memory management interface with comprehensive functionality."""

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
        """Build the advanced memory management interface."""
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
            ["plus-circle", lambda x: self._show_add_memory_dialog()],
            ["export", lambda x: self._show_export_dialog()],
            ["database-cog", lambda x: self._optimize_database()],
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

        # Enhanced search bar (initially hidden)
        self.search_layout = MDBoxLayout(
            orientation="vertical",
            size_hint_y=None,
            height=0,
            opacity=0,
            padding=dp(16),
            spacing=dp(8),
        )

        # Search input row
        search_input_layout = MDBoxLayout(
            orientation="horizontal",
            size_hint_y=None,
            height=dp(40),
            spacing=dp(8),
        )

        self.search_field = MDTextField(
            hint_text="Search memories by content, type, or metadata...",
            on_text=self._on_search_text_change,
        )

        search_btn = MDIconButton(icon="magnify", on_release=lambda x: self._perform_search())
        filter_btn = MDIconButton(icon="filter", on_release=lambda x: self._show_search_filters())
        clear_search_btn = MDIconButton(icon="close", on_release=lambda x: self._clear_search())

        search_input_layout.add_widget(self.search_field)
        search_input_layout.add_widget(search_btn)
        search_input_layout.add_widget(filter_btn)
        search_input_layout.add_widget(clear_search_btn)

        self.search_layout.add_widget(search_input_layout)

        # Quick search suggestions
        self.suggestions_layout = MDBoxLayout(
            orientation="horizontal",
            size_hint_y=None,
            height=dp(30),
            spacing=dp(8),
        )

        # Add some common search suggestions
        suggestions = [
            "important memories",
            "recent conversations",
            "user preferences",
            "code examples",
        ]
        for suggestion in suggestions:
            chip = MDChip(
                text=suggestion,
                size_hint=(None, None),
                height=dp(25),
                on_release=lambda x, s=suggestion: self._apply_suggestion(s),
            )
            self.suggestions_layout.add_widget(chip)

        self.search_layout.add_widget(self.suggestions_layout)
        main_layout.add_widget(self.search_layout)

        # Content area (just use the memories tab for now)
        content_layout = MDBoxLayout(orientation="vertical")

        # All memories content
        self._build_memories_content(content_layout)

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

    def _build_memories_content(self, parent_layout):
        """Build the memories list tab."""
        layout = MDBoxLayout(orientation="vertical", padding=dp(8))

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

        bulk_export_btn = MDRaisedButton(
            text="Export Selected",
            md_bg_color=(0.2, 0.6, 0.8, 1),
            on_release=self._bulk_export_selected,
            size_hint_x=None,
            width=dp(140),
        )
        action_bar.add_widget(bulk_export_btn)

        layout.add_widget(action_bar)

        # Memories list
        self.memories_scroll = MDScrollView()
        self.memories_list = MDBoxLayout(
            orientation="vertical", spacing=dp(8), padding=dp(8), adaptive_height=True
        )
        self.memories_scroll.add_widget(self.memories_list)
        layout.add_widget(self.memories_scroll)

        parent_layout.add_widget(layout)

    def _build_analytics_tab(self, tab):
        """Build the analytics and visualization tab."""
        layout = MDBoxLayout(orientation="vertical", padding=dp(16), spacing=dp(16))

        # Memory distribution chart placeholder
        chart_card = MDCard(
            orientation="vertical",
            size_hint_y=None,
            height=dp(300),
            elevation=4,
            radius=[dp(12)],
            md_bg_color=(0.08, 0.08, 0.10, 1),
            padding=dp(16),
        )

        chart_title = MDLabel(
            text="Memory Distribution by Type",
            font_style="H6",
            theme_text_color="Primary",
            size_hint_y=None,
            height=dp(30),
        )
        chart_card.add_widget(chart_title)

        # Placeholder for chart (would integrate with plotting library)
        chart_placeholder = MDLabel(
            text="[Chart visualization would go here]\n\nMemory types distribution:\n• Short-term: 45%\n• Long-term: 30%\n• Episodic: 15%\n• Semantic: 10%",
            theme_text_color="Secondary",
            halign="center",
        )
        chart_card.add_widget(chart_placeholder)

        layout.add_widget(chart_card)

        # Usage trends
        trends_card = MDCard(
            orientation="vertical",
            size_hint_y=None,
            height=dp(200),
            elevation=4,
            radius=[dp(12)],
            md_bg_color=(0.08, 0.08, 0.10, 1),
            padding=dp(16),
        )

        trends_title = MDLabel(
            text="Memory Usage Trends",
            font_style="H6",
            theme_text_color="Primary",
            size_hint_y=None,
            height=dp(30),
        )
        trends_card.add_widget(trends_title)

        trends_content = MDLabel(
            text="• Average daily memory creation: 15 items\n• Most active time: 14:00-16:00\n• Memory retention rate: 89%\n• Average importance score: 0.65",
            theme_text_color="Secondary",
        )
        trends_card.add_widget(trends_content)

        layout.add_widget(trends_card)

        tab.add_widget(layout)

    def _build_settings_tab(self, tab):
        """Build the settings and configuration tab."""
        layout = MDBoxLayout(orientation="vertical", padding=dp(16), spacing=dp(16))

        # Memory retention settings
        retention_card = MDCard(
            orientation="vertical",
            size_hint_y=None,
            height=dp(200),
            elevation=4,
            radius=[dp(12)],
            md_bg_color=(0.08, 0.08, 0.10, 1),
            padding=dp(16),
        )

        retention_title = MDLabel(
            text="Memory Retention Settings",
            font_style="H6",
            theme_text_color="Primary",
            size_hint_y=None,
            height=dp(30),
        )
        retention_card.add_widget(retention_title)

        # Auto-cleanup toggle
        cleanup_layout = MDBoxLayout(
            orientation="horizontal", size_hint_y=None, height=dp(40), spacing=dp(16)
        )

        cleanup_label = MDLabel(text="Auto-cleanup old memories", theme_text_color="Primary")

        self.cleanup_checkbox = MDCheckbox(
            active=True, size_hint=(None, None), size=(dp(24), dp(24))
        )

        cleanup_layout.add_widget(cleanup_label)
        cleanup_layout.add_widget(self.cleanup_checkbox)
        retention_card.add_widget(cleanup_layout)

        # Importance threshold
        threshold_layout = MDBoxLayout(
            orientation="vertical", size_hint_y=None, height=dp(80), spacing=dp(8)
        )

        threshold_label = MDLabel(
            text="Minimum importance threshold: 0.3",
            theme_text_color="Primary",
            size_hint_y=None,
            height=dp(30),
        )

        self.threshold_slider = MDSlider(
            min=0.0, max=1.0, value=0.3, step=0.1, size_hint_y=None, height=dp(40)
        )

        threshold_layout.add_widget(threshold_label)
        threshold_layout.add_widget(self.threshold_slider)
        retention_card.add_widget(threshold_layout)

        layout.add_widget(retention_card)

        # Performance settings
        performance_card = MDCard(
            orientation="vertical",
            size_hint_y=None,
            height=dp(150),
            elevation=4,
            radius=[dp(12)],
            md_bg_color=(0.08, 0.08, 0.10, 1),
            padding=dp(16),
        )

        performance_title = MDLabel(
            text="Performance Settings",
            font_style="H6",
            theme_text_color="Primary",
            size_hint_y=None,
            height=dp(30),
        )
        performance_card.add_widget(performance_title)

        # Cache settings
        cache_layout = MDBoxLayout(
            orientation="horizontal", size_hint_y=None, height=dp(40), spacing=dp(16)
        )

        cache_label = MDLabel(text="Enable memory caching", theme_text_color="Primary")

        self.cache_checkbox = MDCheckbox(active=True, size_hint=(None, None), size=(dp(24), dp(24)))

        cache_layout.add_widget(cache_label)
        cache_layout.add_widget(self.cache_checkbox)
        performance_card.add_widget(cache_layout)

        # Optimization button
        optimize_btn = MDRaisedButton(
            text="Optimize Memory Database",
            md_bg_color=(0.2, 0.8, 0.4, 1),
            on_release=self._optimize_database,
            size_hint_y=None,
            height=dp(40),
        )
        performance_card.add_widget(optimize_btn)

        layout.add_widget(performance_card)

        tab.add_widget(layout)

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
            # Load all memories by searching with wildcard query
            all_memories = []
            for memory_type in MemoryType:
                memories = await self.safe_memory.safe_recall(
                    query="*", memory_types=[memory_type], limit=1000, threshold=0.0
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

            # Storage usage (simulated)
            storage_mb = (stats.get("total", 0) * 0.05) + 2.3  # Rough estimate
            storage_card = MemoryStatsCard(
                title="Storage Used",
                value=f"{storage_mb:.1f} MB",
                icon="harddisk",
                color=(0.8, 0.4, 0.2, 1),
                trend="+0.2 MB",
            )
            self.stats_cards["storage"] = storage_card
            self.stats_layout.add_widget(storage_card)

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

            # Update select all checkbox state
            self._update_select_all_state()

        except Exception as e:
            logger.error(f"Failed to refresh memories list: {e}")

    def _update_select_all_state(self):
        """Update select all checkbox state."""
        try:
            if not self.filtered_memories:
                self.select_all_checkbox.active = False
                return

            # Count selected items
            selected_count = sum(
                1
                for widget in self.memories_list.children
                if hasattr(widget, "selected") and widget.selected
            )

            if selected_count == 0:
                self.select_all_checkbox.active = False
            elif selected_count == len(self.filtered_memories):
                self.select_all_checkbox.active = True
            else:
                # Partial selection - could show indeterminate state
                self.select_all_checkbox.active = False

        except Exception as e:
            logger.error(f"Failed to update select all state: {e}")

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
            anim = Animation(height=dp(100), opacity=1, duration=0.3)
            anim.start(self.search_layout)
            self.search_field.focus = True
        else:
            # Hide search
            anim = Animation(height=0, opacity=0, duration=0.3)
            anim.start(self.search_layout)
            self._clear_search()

    def _apply_suggestion(self, suggestion: str):
        """Apply a search suggestion."""
        self.search_field.text = suggestion
        self.search_query = suggestion.lower()
        self._perform_search()

    def _show_search_filters(self):
        """Show advanced search filters dialog."""
        from kivymd.uix.button import MDRaisedButton
        from kivymd.uix.selectioncontrol import MDCheckbox

        content = MDBoxLayout(
            orientation="vertical", spacing=dp(15), size_hint_y=None, height=dp(300)
        )

        content.add_widget(
            MDLabel(text="Search Filters", font_style="H6", size_hint_y=None, height=dp(30))
        )

        # Memory type filters
        type_label = MDLabel(
            text="Memory Types:", size_hint_y=None, height=dp(25), theme_text_color="Primary"
        )
        content.add_widget(type_label)

        self.filter_checkboxes = {}
        for memory_type in MemoryType:
            type_layout = MDBoxLayout(
                orientation="horizontal", size_hint_y=None, height=dp(30), spacing=dp(10)
            )
            checkbox = MDCheckbox(active=True, size_hint=(None, None), size=(dp(20), dp(20)))
            label = MDLabel(
                text=memory_type.value.replace("_", " ").title(), theme_text_color="Primary"
            )

            type_layout.add_widget(checkbox)
            type_layout.add_widget(label)
            content.add_widget(type_layout)

            self.filter_checkboxes[memory_type] = checkbox

        # Importance range
        importance_label = MDLabel(
            text="Minimum Importance:", size_hint_y=None, height=dp(25), theme_text_color="Primary"
        )
        content.add_widget(importance_label)

        self.importance_filter_slider = MDSlider(
            min=0.0, max=1.0, value=0.0, step=0.1, size_hint_y=None, height=dp(40)
        )
        content.add_widget(self.importance_filter_slider)

        dialog = MDDialog(
            title="Search Filters",
            type="custom",
            content_cls=content,
            size_hint=(0.8, None),
            height=dp(400),
            buttons=[
                MDRaisedButton(
                    text="Clear Filters", on_release=lambda x: self._clear_filters(dialog)
                ),
                MDRaisedButton(text="Apply", on_release=lambda x: self._apply_filters(dialog)),
            ],
        )
        dialog.open()

    def _clear_filters(self, dialog):
        """Clear all search filters."""
        for checkbox in self.filter_checkboxes.values():
            checkbox.active = True
        self.importance_filter_slider.value = 0.0

    def _apply_filters(self, dialog):
        """Apply search filters."""
        dialog.dismiss()

        # Get selected memory types
        selected_types = [
            memory_type
            for memory_type, checkbox in self.filter_checkboxes.items()
            if checkbox.active
        ]

        min_importance = self.importance_filter_slider.value

        # Filter memories
        filtered = []
        for memory in self.filtered_memories:
            if memory.memory_type in selected_types and memory.importance >= min_importance:
                filtered.append(memory)

        self.filtered_memories = filtered
        self._refresh_memories_list()

        Notification.info(f"Applied filters: {len(filtered)} memories match criteria")

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

        def run_async_search():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._search_memories_async())
            except Exception as e:
                logger.error(f"Search failed: {e}")
                Clock.schedule_once(lambda dt: Notification.error("Search failed"), 0)
            finally:
                loop.close()

        thread = threading.Thread(target=run_async_search)
        thread.daemon = True
        thread.start()

    async def _search_memories_async(self):
        """Perform async memory search with semantic similarity."""
        try:
            # Use semantic search with the memory system
            semantic_results = await self.safe_memory.safe_recall(
                query=self.search_query,
                memory_types=None,  # Search all types
                threshold=0.3,  # Lower threshold for better recall
                limit=50,
            )

            # Also perform text-based filtering on existing memories
            text_filtered = [
                memory
                for memory in self.memories
                if (
                    self.search_query.lower() in memory.content.lower()
                    or self.search_query.lower() in memory.memory_type.value.lower()
                    or any(
                        self.search_query.lower() in str(v).lower()
                        for v in memory.metadata.values()
                    )
                )
            ]

            # Combine results, avoiding duplicates
            combined_results = semantic_results.copy()
            for memory in text_filtered:
                if not any(m.id == memory.id for m in combined_results):
                    combined_results.append(memory)

            # Update UI on main thread
            Clock.schedule_once(lambda dt: self._update_search_results(combined_results), 0)

        except Exception as e:
            logger.error(f"Async search failed: {e}")
            Clock.schedule_once(lambda dt: Notification.error("Search failed"), 0)

    def _update_search_results(self, results):
        """Update UI with search results."""
        try:
            self.filtered_memories = results
            self._refresh_memories_list()

            if len(results) > 0:
                Notification.success(f"Found {len(results)} matching memories")
            else:
                Notification.info("No memories found matching your search")

        except Exception as e:
            logger.error(f"Failed to update search results: {e}")
            Notification.error("Failed to update search results")

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

    def _show_filter_dialog(self):
        """Show advanced filter dialog."""
        # TODO: Implement advanced filtering
        Notification.info("Advanced filtering coming soon!")

    def _show_export_dialog(self):
        """Show export options dialog."""
        # TODO: Implement export functionality
        Notification.info("Export functionality coming soon!")

    def _show_settings_dialog(self):
        """Show memory settings dialog."""
        # TODO: Implement settings dialog
        Notification.info("Settings dialog coming soon!")

    def _show_add_memory_dialog(self, *args):
        """Show dialog to manually add a new memory."""
        from kivymd.uix.button import MDRaisedButton
        from kivymd.uix.slider import MDSlider

        content = MDBoxLayout(
            orientation="vertical", spacing=dp(15), size_hint_y=None, height=dp(450)
        )

        # Content editor
        content_label = MDLabel(
            text="Memory Content:", theme_text_color="Primary", size_hint_y=None, height=dp(30)
        )
        content.add_widget(content_label)

        self.new_content_field = MDTextField(
            hint_text="Enter the content you want to remember...",
            multiline=True,
            size_hint_y=None,
            height=dp(120),
        )
        content.add_widget(self.new_content_field)

        # Memory type selector
        type_label = MDLabel(
            text="Memory Type:", theme_text_color="Primary", size_hint_y=None, height=dp(30)
        )
        content.add_widget(type_label)

        type_layout = MDBoxLayout(
            orientation="horizontal", size_hint_y=None, height=dp(40), spacing=dp(10)
        )

        self.new_type_buttons = []
        for mem_type in MemoryType:
            btn = MDRaisedButton(
                text=mem_type.value.replace("_", " ").title(),
                size_hint_x=0.25,
                md_bg_color=(
                    (0.2, 0.6, 0.8, 1) if mem_type == MemoryType.LONG_TERM else (0.3, 0.3, 0.3, 1)
                ),
                on_release=lambda x, t=mem_type: self._select_new_type(t),
            )
            self.new_type_buttons.append((btn, mem_type))
            type_layout.add_widget(btn)

        content.add_widget(type_layout)

        # Importance slider
        importance_label = MDLabel(
            text="Importance: 0.7", theme_text_color="Primary", size_hint_y=None, height=dp(30)
        )
        content.add_widget(importance_label)

        self.new_importance_slider = MDSlider(
            min=0.0, max=1.0, value=0.7, step=0.1, size_hint_y=None, height=dp(40)
        )
        self.new_importance_slider.bind(
            value=lambda x, v: setattr(importance_label, "text", f"Importance: {v:.1f}")
        )
        content.add_widget(self.new_importance_slider)

        # Tags input
        tags_label = MDLabel(
            text="Tags (comma-separated):",
            theme_text_color="Primary",
            size_hint_y=None,
            height=dp(30),
        )
        content.add_widget(tags_label)

        self.new_tags_field = MDTextField(
            hint_text="e.g., important, work, research", size_hint_y=None, height=dp(40)
        )
        content.add_widget(self.new_tags_field)

        # Description field
        desc_label = MDLabel(
            text="Description (optional):",
            theme_text_color="Primary",
            size_hint_y=None,
            height=dp(30),
        )
        content.add_widget(desc_label)

        self.new_desc_field = MDTextField(
            hint_text="Brief description of this memory", size_hint_y=None, height=dp(40)
        )
        content.add_widget(self.new_desc_field)

        self.selected_new_type = MemoryType.LONG_TERM

        dialog = MDDialog(
            title="Create New Memory",
            type="custom",
            content_cls=content,
            size_hint=(0.9, None),
            height=dp(550),
            buttons=[
                MDRaisedButton(text="Cancel", on_release=lambda x: dialog.dismiss()),
                MDRaisedButton(
                    text="Create Memory",
                    md_bg_color=(0.2, 0.8, 0.4, 1),
                    on_release=lambda x: self._create_new_memory(dialog),
                ),
            ],
        )
        dialog.open()

    def _bulk_delete_selected(self, *args):
        """Delete all selected memories."""
        selected_memories = [
            widget.memory
            for widget in self.memories_list.children
            if hasattr(widget, "selected") and widget.selected
        ]

        if not selected_memories:
            Notification.warning("No memories selected")
            return

        from kivymd.uix.button import MDRaisedButton

        content = MDBoxLayout(
            orientation="vertical", spacing=dp(10), size_hint_y=None, height=dp(120)
        )

        warning_label = MDLabel(
            text=f"Delete {len(selected_memories)} selected memories?\nThis action cannot be undone.",
            theme_text_color="Primary",
            adaptive_height=True,
            halign="center",
        )
        content.add_widget(warning_label)

        dialog = MDDialog(
            title="Bulk Delete Memories",
            type="custom",
            content_cls=content,
            buttons=[
                MDRaisedButton(text="Cancel", on_release=lambda x: dialog.dismiss()),
                MDRaisedButton(
                    text=f"Delete {len(selected_memories)}",
                    md_bg_color=(0.8, 0.2, 0.2, 1),
                    on_release=lambda x: self._bulk_delete_confirmed(dialog, selected_memories),
                ),
            ],
        )
        dialog.open()

    def _bulk_export_selected(self, *args):
        """Export all selected memories."""
        selected_memories = [
            widget.memory
            for widget in self.memories_list.children
            if hasattr(widget, "selected") and widget.selected
        ]

        if not selected_memories:
            Notification.warning("No memories selected")
            return

        from kivymd.uix.button import MDRaisedButton
        from kivymd.uix.selectioncontrol import MDCheckbox

        content = MDBoxLayout(
            orientation="vertical", spacing=dp(15), size_hint_y=None, height=dp(250)
        )

        info_label = MDLabel(
            text=f"Export {len(selected_memories)} selected memories",
            theme_text_color="Primary",
            size_hint_y=None,
            height=dp(30),
        )
        content.add_widget(info_label)

        # Export format selection
        format_label = MDLabel(
            text="Export Format:", theme_text_color="Primary", size_hint_y=None, height=dp(30)
        )
        content.add_widget(format_label)

        format_layout = MDBoxLayout(
            orientation="horizontal", size_hint_y=None, height=dp(40), spacing=dp(10)
        )

        self.export_format_buttons = []
        formats = [("JSON", "json"), ("CSV", "csv"), ("Markdown", "md")]

        for fmt_name, fmt_ext in formats:
            btn = MDRaisedButton(
                text=fmt_name,
                size_hint_x=0.33,
                md_bg_color=(0.2, 0.6, 0.8, 1) if fmt_ext == "json" else (0.3, 0.3, 0.3, 1),
                on_release=lambda x, f=fmt_ext: self._select_export_format(f),
            )
            self.export_format_buttons.append((btn, fmt_ext))
            format_layout.add_widget(btn)

        content.add_widget(format_layout)

        # Export options
        options_layout = MDBoxLayout(
            orientation="vertical", size_hint_y=None, height=dp(100), spacing=dp(10)
        )

        # Include metadata
        meta_layout = MDBoxLayout(orientation="horizontal", size_hint_y=None, height=dp(30))

        self.include_metadata_checkbox = MDCheckbox(
            active=True, size_hint=(None, None), size=(dp(20), dp(20))
        )

        meta_label = MDLabel(text="Include metadata", theme_text_color="Primary")

        meta_layout.add_widget(self.include_metadata_checkbox)
        meta_layout.add_widget(meta_label)
        options_layout.add_widget(meta_layout)

        # Include embeddings
        embed_layout = MDBoxLayout(orientation="horizontal", size_hint_y=None, height=dp(30))

        self.include_embeddings_checkbox = MDCheckbox(
            active=False, size_hint=(None, None), size=(dp(20), dp(20))
        )

        embed_label = MDLabel(
            text="Include embeddings (large file size)", theme_text_color="Primary"
        )

        embed_layout.add_widget(self.include_embeddings_checkbox)
        embed_layout.add_widget(embed_label)
        options_layout.add_widget(embed_layout)

        content.add_widget(options_layout)

        self.selected_export_format = "json"

        dialog = MDDialog(
            title="Export Selected Memories",
            type="custom",
            content_cls=content,
            size_hint=(0.9, None),
            height=dp(350),
            buttons=[
                MDRaisedButton(text="Cancel", on_release=lambda x: dialog.dismiss()),
                MDRaisedButton(
                    text="Export",
                    md_bg_color=(0.2, 0.6, 0.8, 1),
                    on_release=lambda x: self._export_memories_confirmed(dialog, selected_memories),
                ),
            ],
        )
        dialog.open()

    def _bulk_delete_confirmed(self, dialog, memories: list[Memory]):
        """Confirm bulk deletion."""
        dialog.dismiss()

        def run_bulk_delete():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._bulk_delete_async(memories))
            except Exception as e:
                logger.error(f"Bulk deletion failed: {e}")
                Clock.schedule_once(lambda dt: Notification.error("Bulk deletion failed"), 0)
            finally:
                loop.close()

        thread = threading.Thread(target=run_bulk_delete)
        thread.daemon = True
        thread.start()

        Notification.info(f"Deleting {len(memories)} memories...")

    async def _bulk_delete_async(self, memories: list[Memory]):
        """Delete multiple memories asynchronously."""
        try:
            deleted_count = 0

            for memory in memories:
                success = await self.safe_memory.safe_forget(memory.id)
                if success:
                    deleted_count += 1
                    # Remove from local lists
                    self.memories = [m for m in self.memories if m.id != memory.id]
                    self.filtered_memories = [
                        m for m in self.filtered_memories if m.id != memory.id
                    ]

            # Update UI
            Clock.schedule_once(lambda dt: self._refresh_memories_list(), 0)
            Clock.schedule_once(
                lambda dt: Notification.success(f"Deleted {deleted_count} memories successfully"), 0
            )

        except Exception as e:
            logger.error(f"Bulk deletion error: {e}")
            Clock.schedule_once(lambda dt: Notification.error("Bulk deletion failed"), 0)

    def _select_export_format(self, format_ext: str):
        """Select export format."""
        self.selected_export_format = format_ext
        # Update button colors
        for btn, btn_format in self.export_format_buttons:
            if btn_format == format_ext:
                btn.md_bg_color = (0.2, 0.6, 0.8, 1)
            else:
                btn.md_bg_color = (0.3, 0.3, 0.3, 1)

    def _export_memories_confirmed(self, dialog, memories: list[Memory]):
        """Confirm memory export."""
        dialog.dismiss()

        include_metadata = self.include_metadata_checkbox.active
        include_embeddings = self.include_embeddings_checkbox.active

        def run_export():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(
                    self._export_memories_async(
                        memories, self.selected_export_format, include_metadata, include_embeddings
                    )
                )
            except Exception as e:
                logger.error(f"Memory export failed: {e}")
                Clock.schedule_once(lambda dt: Notification.error("Memory export failed"), 0)
            finally:
                loop.close()

        thread = threading.Thread(target=run_export)
        thread.daemon = True
        thread.start()

        Notification.info(f"Exporting {len(memories)} memories...")

    async def _export_memories_async(
        self,
        memories: list[Memory],
        format_ext: str,
        include_metadata: bool,
        include_embeddings: bool,
    ):
        """Export memories asynchronously."""
        try:
            from pathlib import Path

            # Create export directory
            export_dir = Path("./exports")
            export_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"memories_export_{timestamp}.{format_ext}"
            filepath = export_dir / filename

            # Prepare data
            export_data = []
            for memory in memories:
                data = {
                    "id": memory.id,
                    "content": memory.content,
                    "memory_type": memory.memory_type.value,
                    "importance": memory.importance,
                    "created_at": memory.created_at.isoformat(),
                    "accessed_at": memory.accessed_at.isoformat(),
                }

                if include_metadata and memory.metadata:
                    data["metadata"] = memory.metadata

                if include_embeddings and memory.embedding:
                    data["embedding"] = memory.embedding

                export_data.append(data)

            # Write file based on format
            if format_ext == "json":
                import json

                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)

            elif format_ext == "csv":
                import csv

                with open(filepath, "w", newline="", encoding="utf-8") as f:
                    if export_data:
                        writer = csv.DictWriter(f, fieldnames=export_data[0].keys())
                        writer.writeheader()
                        writer.writerows(export_data)

            elif format_ext == "md":
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write("# Memory Export\\n\\n")
                    f.write(f"Exported on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
                    f.write(f"Total memories: {len(memories)}\\n\\n")

                    for i, data in enumerate(export_data, 1):
                        f.write(f"## Memory {i}\\n\\n")
                        f.write(f"**ID:** {data['id']}\\n")
                        f.write(f"**Type:** {data['memory_type']}\\n")
                        f.write(f"**Importance:** {data['importance']}\\n")
                        f.write(f"**Created:** {data['created_at']}\\n\\n")
                        f.write(f"**Content:**\\n{data['content']}\\n\\n")

                        if include_metadata and "metadata" in data:
                            f.write("**Metadata:**\\n")
                            for key, value in data["metadata"].items():
                                f.write(f"- {key}: {value}\\n")
                            f.write("\\n")

                        f.write("---\\n\\n")

            Clock.schedule_once(
                lambda dt: Notification.success(f"Exported {len(memories)} memories to {filename}"),
                0,
            )

        except Exception as e:
            logger.error(f"Memory export error: {e}")
            Clock.schedule_once(lambda dt: Notification.error("Memory export failed"), 0)

    def _confirm_delete_memory(self, memory: Memory):
        """Show confirmation dialog for deleting a memory."""
        from kivymd.uix.button import MDRaisedButton

        content = MDBoxLayout(
            orientation="vertical", spacing=dp(10), size_hint_y=None, height=dp(150)
        )

        # Memory preview
        preview_text = memory.content[:100] + "..." if len(memory.content) > 100 else memory.content
        preview_label = MDLabel(
            text=f"Memory: {preview_text}",
            theme_text_color="Primary",
            adaptive_height=True,
            text_size=(dp(300), None),
        )
        content.add_widget(preview_label)

        # Warning
        warning_label = MDLabel(
            text="This action cannot be undone.",
            theme_text_color="Error",
            adaptive_height=True,
            halign="center",
        )
        content.add_widget(warning_label)

        dialog = MDDialog(
            title="Delete Memory?",
            type="custom",
            content_cls=content,
            buttons=[
                MDRaisedButton(text="Cancel", on_release=lambda x: dialog.dismiss()),
                MDRaisedButton(
                    text="Delete",
                    md_bg_color=(0.8, 0.2, 0.2, 1),
                    on_release=lambda x: self._delete_memory_confirmed(dialog, memory),
                ),
            ],
        )
        dialog.open()

    def _edit_memory(self, memory: Memory):
        """Edit a memory."""
        from kivymd.uix.button import MDRaisedButton
        from kivymd.uix.slider import MDSlider

        content = MDBoxLayout(
            orientation="vertical", spacing=dp(15), size_hint_y=None, height=dp(400)
        )

        # Content editor
        content_label = MDLabel(
            text="Memory Content:", theme_text_color="Primary", size_hint_y=None, height=dp(30)
        )
        content.add_widget(content_label)

        self.edit_content_field = MDTextField(
            text=memory.content, multiline=True, size_hint_y=None, height=dp(120)
        )
        content.add_widget(self.edit_content_field)

        # Memory type selector
        type_label = MDLabel(
            text="Memory Type:", theme_text_color="Primary", size_hint_y=None, height=dp(30)
        )
        content.add_widget(type_label)

        type_layout = MDBoxLayout(
            orientation="horizontal", size_hint_y=None, height=dp(40), spacing=dp(10)
        )

        self.edit_type_buttons = []
        for mem_type in MemoryType:
            btn = MDRaisedButton(
                text=mem_type.value.replace("_", " ").title(),
                size_hint_x=0.25,
                md_bg_color=(
                    (0.2, 0.6, 0.8, 1) if mem_type == memory.memory_type else (0.3, 0.3, 0.3, 1)
                ),
                on_release=lambda x, t=mem_type: self._select_edit_type(t),
            )
            self.edit_type_buttons.append((btn, mem_type))
            type_layout.add_widget(btn)

        content.add_widget(type_layout)

        # Importance slider
        importance_label = MDLabel(
            text=f"Importance: {memory.importance:.1f}",
            theme_text_color="Primary",
            size_hint_y=None,
            height=dp(30),
        )
        content.add_widget(importance_label)

        self.edit_importance_slider = MDSlider(
            min=0.0, max=1.0, value=memory.importance, step=0.1, size_hint_y=None, height=dp(40)
        )
        self.edit_importance_slider.bind(
            value=lambda x, v: setattr(importance_label, "text", f"Importance: {v:.1f}")
        )
        content.add_widget(self.edit_importance_slider)

        self.selected_edit_type = memory.memory_type

        dialog = MDDialog(
            title="Edit Memory",
            type="custom",
            content_cls=content,
            size_hint=(0.9, None),
            height=dp(500),
            buttons=[
                MDRaisedButton(text="Cancel", on_release=lambda x: dialog.dismiss()),
                MDRaisedButton(
                    text="Save Changes",
                    md_bg_color=(0.2, 0.8, 0.4, 1),
                    on_release=lambda x: self._save_memory_changes(dialog, memory),
                ),
            ],
        )
        dialog.open()

    def _view_memory(self, memory: Memory):
        """View memory details."""
        from kivymd.uix.button import MDRaisedButton

        content = MDBoxLayout(
            orientation="vertical", spacing=dp(15), size_hint_y=None, height=dp(500)
        )

        # Memory details
        details_scroll = MDScrollView(size_hint_y=None, height=dp(400))
        details_layout = MDBoxLayout(
            orientation="vertical", spacing=dp(10), adaptive_height=True, padding=dp(10)
        )

        # Content
        content_card = MDCard(
            orientation="vertical",
            padding=dp(15),
            spacing=dp(8),
            adaptive_height=True,
            md_bg_color=(0.08, 0.08, 0.10, 1),
            radius=[dp(8)],
        )

        content_title = MDLabel(
            text="Content:",
            font_style="Subtitle1",
            theme_text_color="Primary",
            size_hint_y=None,
            height=dp(30),
        )
        content_card.add_widget(content_title)

        content_text = MDLabel(
            text=memory.content,
            theme_text_color="Primary",
            adaptive_height=True,
            text_size=(dp(350), None),
        )
        content_card.add_widget(content_text)
        details_layout.add_widget(content_card)

        # Metadata
        meta_card = MDCard(
            orientation="vertical",
            padding=dp(15),
            spacing=dp(8),
            adaptive_height=True,
            md_bg_color=(0.08, 0.08, 0.10, 1),
            radius=[dp(8)],
        )

        meta_title = MDLabel(
            text="Metadata:",
            font_style="Subtitle1",
            theme_text_color="Primary",
            size_hint_y=None,
            height=dp(30),
        )
        meta_card.add_widget(meta_title)

        meta_info = [
            f"ID: {memory.id}",
            f"Type: {memory.memory_type.value.replace('_', ' ').title()}",
            f"Importance: {memory.importance:.2f} ({'★' * int(memory.importance * 5)})",
            f"Created: {memory.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Last Accessed: {memory.accessed_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Vector Dimensions: {len(memory.embedding) if memory.embedding else 0}",
        ]

        for info in meta_info:
            info_label = MDLabel(text=info, theme_text_color="Secondary", adaptive_height=True)
            meta_card.add_widget(info_label)

        # Custom metadata
        if memory.metadata:
            meta_card.add_widget(
                MDLabel(
                    text="Custom Metadata:",
                    theme_text_color="Primary",
                    adaptive_height=True,
                    size_hint_y=None,
                    height=dp(25),
                )
            )

            for key, value in memory.metadata.items():
                meta_label = MDLabel(
                    text=f"  {key}: {value}", theme_text_color="Secondary", adaptive_height=True
                )
                meta_card.add_widget(meta_label)

        details_layout.add_widget(meta_card)
        details_scroll.add_widget(details_layout)
        content.add_widget(details_scroll)

        # Action buttons
        actions_layout = MDBoxLayout(
            orientation="horizontal", size_hint_y=None, height=dp(50), spacing=dp(10)
        )

        edit_btn = MDRaisedButton(
            text="Edit",
            md_bg_color=(0.2, 0.6, 0.8, 1),
            on_release=lambda x: self._edit_from_view(dialog, memory),
        )

        delete_btn = MDRaisedButton(
            text="Delete",
            md_bg_color=(0.8, 0.2, 0.2, 1),
            on_release=lambda x: self._delete_from_view(dialog, memory),
        )

        actions_layout.add_widget(edit_btn)
        actions_layout.add_widget(delete_btn)
        actions_layout.add_widget(MDBoxLayout())  # Spacer

        content.add_widget(actions_layout)

        dialog = MDDialog(
            title=f"Memory Details - {memory.id[:8]}...",
            type="custom",
            content_cls=content,
            size_hint=(0.95, None),
            height=dp(600),
            buttons=[MDRaisedButton(text="Close", on_release=lambda x: dialog.dismiss())],
        )
        dialog.open()

    def _optimize_database(self, *args):
        """Optimize the memory database."""

        def run_optimization():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._optimize_database_async())
            except Exception as e:
                logger.error(f"Database optimization failed: {e}")
                Clock.schedule_once(
                    lambda dt: Notification.error("Database optimization failed"), 0
                )
            finally:
                loop.close()

        thread = threading.Thread(target=run_optimization)
        thread.daemon = True
        thread.start()

        Notification.info("Starting database optimization...")

    async def _optimize_database_async(self):
        """Async database optimization."""
        try:
            # Consolidate memories
            await self.safe_memory.manager.consolidate()

            # Clear any cached data
            if hasattr(self.safe_memory.manager, "cache"):
                self.safe_memory.manager.cache.clear()

            # Reload data
            Clock.schedule_once(lambda dt: self._refresh_all(), 0)
            Clock.schedule_once(
                lambda dt: Notification.success("Database optimization completed"), 0
            )

        except Exception as e:
            logger.error(f"Database optimization error: {e}")
            Clock.schedule_once(lambda dt: Notification.error("Database optimization failed"), 0)

    # Helper methods for memory operations
    def _delete_memory_confirmed(self, dialog, memory: Memory):
        """Actually delete the memory after confirmation."""
        dialog.dismiss()

        def run_delete():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._delete_memory_async(memory))
            except Exception as e:
                logger.error(f"Memory deletion failed: {e}")
                Clock.schedule_once(lambda dt: Notification.error("Failed to delete memory"), 0)
            finally:
                loop.close()

        thread = threading.Thread(target=run_delete)
        thread.daemon = True
        thread.start()

    async def _delete_memory_async(self, memory: Memory):
        """Delete memory asynchronously."""
        try:
            success = await self.safe_memory.safe_forget(memory.id)
            if success:
                # Remove from local lists
                self.memories = [m for m in self.memories if m.id != memory.id]
                self.filtered_memories = [m for m in self.filtered_memories if m.id != memory.id]

                # Update UI and stats
                Clock.schedule_once(lambda dt: self._refresh_memories_list(), 0)
                Clock.schedule_once(lambda dt: self._refresh_stats(), 0)
                Clock.schedule_once(
                    lambda dt: Notification.success("Memory deleted successfully"), 0
                )
            else:
                Clock.schedule_once(lambda dt: Notification.error("Failed to delete memory"), 0)

        except Exception as e:
            logger.error(f"Memory deletion error: {e}")
            Clock.schedule_once(lambda dt: Notification.error("Memory deletion failed"), 0)

    def _refresh_stats(self):
        """Refresh memory statistics after changes."""

        def run_stats_refresh():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._refresh_stats_async())
            except Exception as e:
                logger.error(f"Stats refresh failed: {e}")
            finally:
                loop.close()

        thread = threading.Thread(target=run_stats_refresh)
        thread.daemon = True
        thread.start()

    async def _refresh_stats_async(self):
        """Refresh stats asynchronously."""
        try:
            stats = await self.safe_memory.safe_get_stats()
            Clock.schedule_once(lambda dt: self._update_stats_cards(stats), 0)
        except Exception as e:
            logger.error(f"Failed to refresh stats: {e}")

    def _select_edit_type(self, memory_type: MemoryType):
        """Select memory type for editing."""
        self.selected_edit_type = memory_type
        # Update button colors
        for btn, btn_type in self.edit_type_buttons:
            if btn_type == memory_type:
                btn.md_bg_color = (0.2, 0.6, 0.8, 1)
            else:
                btn.md_bg_color = (0.3, 0.3, 0.3, 1)

    def _save_memory_changes(self, dialog, memory: Memory):
        """Save changes to memory."""
        dialog.dismiss()

        new_content = self.edit_content_field.text.strip()
        new_importance = self.edit_importance_slider.value
        new_type = self.selected_edit_type

        if not new_content:
            Notification.error("Memory content cannot be empty")
            return

        def run_update():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(
                    self._update_memory_async(memory, new_content, new_type, new_importance)
                )
            except Exception as e:
                logger.error(f"Memory update failed: {e}")
                Clock.schedule_once(lambda dt: Notification.error("Failed to update memory"), 0)
            finally:
                loop.close()

        thread = threading.Thread(target=run_update)
        thread.daemon = True
        thread.start()

    async def _update_memory_async(
        self, memory: Memory, new_content: str, new_type: MemoryType, new_importance: float
    ):
        """Update memory asynchronously."""
        try:
            # Update memory object
            memory.content = new_content
            memory.memory_type = new_type
            memory.importance = new_importance
            memory.accessed_at = datetime.now()

            # Regenerate embedding if content changed
            if new_content != memory.content:
                from src.utils.embeddings import EmbeddingGenerator

                embedding_gen = EmbeddingGenerator()
                memory.embedding = await embedding_gen.generate(new_content)

            # Update in store
            success = await self.safe_memory.manager.store.update(memory)

            if success:
                # Update local lists
                for i, m in enumerate(self.memories):
                    if m.id == memory.id:
                        self.memories[i] = memory
                        break

                for i, m in enumerate(self.filtered_memories):
                    if m.id == memory.id:
                        self.filtered_memories[i] = memory
                        break

                # Update UI
                Clock.schedule_once(lambda dt: self._refresh_memories_list(), 0)
                Clock.schedule_once(
                    lambda dt: Notification.success("Memory updated successfully"), 0
                )
            else:
                Clock.schedule_once(lambda dt: Notification.error("Failed to update memory"), 0)

        except Exception as e:
            logger.error(f"Memory update error: {e}")
            Clock.schedule_once(lambda dt: Notification.error("Memory update failed"), 0)

    def _edit_from_view(self, view_dialog, memory: Memory):
        """Edit memory from view dialog."""
        view_dialog.dismiss()
        self._edit_memory(memory)

    def _delete_from_view(self, view_dialog, memory: Memory):
        """Delete memory from view dialog."""
        view_dialog.dismiss()
        self._confirm_delete_memory(memory)

    def _select_new_type(self, memory_type: MemoryType):
        """Select memory type for new memory."""
        self.selected_new_type = memory_type
        # Update button colors
        for btn, btn_type in self.new_type_buttons:
            if btn_type == memory_type:
                btn.md_bg_color = (0.2, 0.6, 0.8, 1)
            else:
                btn.md_bg_color = (0.3, 0.3, 0.3, 1)

    def _create_new_memory(self, dialog):
        """Create a new memory."""
        dialog.dismiss()

        content = self.new_content_field.text.strip()
        importance = self.new_importance_slider.value
        memory_type = self.selected_new_type
        tags = [tag.strip() for tag in self.new_tags_field.text.split(",") if tag.strip()]
        description = self.new_desc_field.text.strip()

        if not content:
            Notification.error("Memory content cannot be empty")
            return

        def run_create():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(
                    self._create_memory_async(content, memory_type, importance, tags, description)
                )
            except Exception as e:
                logger.error(f"Memory creation failed: {e}")
                Clock.schedule_once(lambda dt: Notification.error("Failed to create memory"), 0)
            finally:
                loop.close()

        thread = threading.Thread(target=run_create)
        thread.daemon = True
        thread.start()

    async def _create_memory_async(
        self, content: str, memory_type: MemoryType, importance: float, tags: list, description: str
    ):
        """Create memory asynchronously."""
        try:
            # Prepare metadata
            metadata = {
                "created_manually": True,
                "created_at": datetime.now().isoformat(),
                "source": "manual_entry",
            }

            if tags:
                metadata["tags"] = tags

            if description:
                metadata["description"] = description

            # Create memory
            memory_id = await self.safe_memory.safe_remember(
                content=content, memory_type=memory_type, importance=importance, metadata=metadata
            )

            if memory_id:
                # Refresh the memory list to include the new one
                await self._load_memories_async()
                Clock.schedule_once(
                    lambda dt: Notification.success(f"Memory created: {memory_id[:8]}..."), 0
                )
            else:
                Clock.schedule_once(lambda dt: Notification.error("Failed to create memory"), 0)

        except Exception as e:
            logger.error(f"Memory creation error: {e}")
            Clock.schedule_once(lambda dt: Notification.error("Memory creation failed"), 0)
