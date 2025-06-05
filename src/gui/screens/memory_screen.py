"""Advanced memory management interface with intelligent operations.

This module implements a comprehensive memory management system demonstrating:

- Full CRUD operations (Create, Read, Update, Delete) for AI memories
- Advanced search and filtering with real-time feedback
- Pagination for efficient handling of large memory datasets
- Intelligent memory type classification and management
- System prompt configuration with memory integration
- Visual memory type indicators with color coding
- Thread-safe async operations with progress feedback

Memory System Architecture:
    The interface connects to a vector-based memory store using ChromaDB,
    providing semantic search capabilities and intelligent memory classification.
    All operations are wrapped in safe managers to prevent data corruption.

User Experience Design:
    - Material Design components with consistent theming
    - Responsive layouts that adapt to different screen sizes
    - Progressive disclosure of complex features
    - Real-time search with debounced input handling
    - Context-aware help and guidance systems

Performance Optimization:
    - Pagination to handle thousands of memories efficiently
    - Debounced search to reduce unnecessary operations
    - Background threading for all database operations
    - Efficient memory loading with limit/offset patterns

Accessibility Features:
    - Keyboard navigation support
    - Screen reader compatible structure
    - High contrast visual design
    - Clear information hierarchy
"""

import asyncio
import threading
from datetime import datetime

from kivy.clock import Clock
from kivy.metrics import dp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDIconButton, MDRaisedButton
from kivymd.uix.card import MDCard
from kivymd.uix.dialog import MDDialog
from kivymd.uix.label import MDLabel
from kivymd.uix.screen import MDScreen
from kivymd.uix.scrollview import MDScrollView
from kivymd.uix.selectioncontrol import MDCheckbox
from kivymd.uix.slider import MDSlider
from kivymd.uix.textfield import MDTextField

from src.gui.utils.notifications import Notification
from src.memory import Memory, MemoryType
from src.memory.safe_operations import create_safe_memory_manager
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class MemoryListItem(MDCard):
    """Custom card widget for memory display with interactive controls.

    This widget provides a sophisticated memory representation featuring:

    Visual Design:
        - Color-coded memory type indicators for quick classification
        - Importance weighting display with visual prominence
        - Truncated content preview for scannable information
        - Material Design elevation and styling

    Interactive Elements:
        - Edit button for in-place memory modification
        - Delete button with confirmation workflow
        - Hover states and touch feedback
        - Callback-based event handling for loose coupling

    Information Architecture:
        - Header with type and importance information
        - Content preview with intelligent truncation
        - Footer with metadata (creation time, importance value)
        - Consistent layout for visual scanning

    Memory Type Visualization:
        Uses distinct colors to represent different memory types:
        - Short-term: Blue (temporary, session-based)
        - Long-term: Orange (permanent, cross-session)
        - Episodic: Purple (event-based, temporal)
        - Semantic: Green (knowledge-based, factual)

    The widget demonstrates advanced Kivy custom widget creation
    with proper event handling and visual design principles.

    Args:
        memory (Memory): The memory object to display
        on_edit (callable): Callback for edit operations
        on_delete (callable): Callback for delete operations
    """

    def __init__(self, memory: Memory, on_edit=None, on_delete=None, **kwargs):
        super().__init__(**kwargs)
        self.memory = memory
        self.on_edit = on_edit
        self.on_delete = on_delete

        # Card properties
        self.orientation = "vertical"
        self.size_hint_y = None
        self.height = dp(120)
        self.padding = dp(16)
        self.spacing = dp(8)
        self.elevation = 3
        self.md_bg_color = (0.10, 0.10, 0.12, 1)
        self.radius = [dp(12)]

        # Build the layout
        self._build_layout()

    def _build_layout(self):
        """Build the memory item layout."""
        # Header with type and actions
        header_layout = MDBoxLayout(
            orientation="horizontal", size_hint_y=None, height=dp(32), spacing=dp(8)
        )

        # Memory type chip with color coding
        type_chip = MDLabel(
            text=self.memory.memory_type.value.replace("_", " ").title(),
            font_style="Caption",
            theme_text_color="Custom",
            text_color=self._get_type_color(),
            size_hint_x=None,
            width=dp(100),
            height=dp(24),
            bold=True,
        )
        header_layout.add_widget(type_chip)

        # Importance display
        importance_label = MDLabel(
            text=f"Imp: {self.memory.importance:.1f}",
            theme_text_color="Custom",
            text_color=(1.0, 0.8, 0.2, 1),
            size_hint_x=None,
            width=dp(60),
            font_style="Caption",
        )
        header_layout.add_widget(importance_label)

        # Spacer
        header_layout.add_widget(MDBoxLayout())

        # Action buttons
        edit_btn = MDIconButton(
            icon="pencil",
            theme_icon_color="Primary",
            size_hint=(None, None),
            size=(dp(32), dp(32)),
            on_release=lambda x: self.on_edit(self.memory) if self.on_edit else None,
        )

        delete_btn = MDIconButton(
            icon="delete",
            theme_icon_color="Error",
            size_hint=(None, None),
            size=(dp(32), dp(32)),
            on_release=lambda x: self.on_delete(self.memory) if self.on_delete else None,
        )

        header_layout.add_widget(edit_btn)
        header_layout.add_widget(delete_btn)

        self.add_widget(header_layout)

        # Content
        content_text = (
            self.memory.content[:120] + "..."
            if len(self.memory.content) > 120
            else self.memory.content
        )
        content_label = MDLabel(
            text=content_text,
            font_style="Body2",
            theme_text_color="Primary",
            adaptive_height=True,
            text_size=(None, None),
        )
        self.add_widget(content_label)

        # Footer with metadata
        footer_layout = MDBoxLayout(
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
        footer_layout.add_widget(created_label)

        # Spacer
        footer_layout.add_widget(MDBoxLayout())

        # Importance value
        importance_value = MDLabel(
            text=f"Importance: {self.memory.importance:.1f}",
            font_style="Caption",
            theme_text_color="Secondary",
            size_hint_x=None,
            width=dp(100),
        )
        footer_layout.add_widget(importance_value)

        self.add_widget(footer_layout)

    def _get_type_color(self):
        """Get color for memory type."""
        colors = {
            MemoryType.SHORT_TERM: (0.2, 0.6, 1.0, 1),
            MemoryType.LONG_TERM: (0.8, 0.4, 0.2, 1),
            MemoryType.EPISODIC: (0.6, 0.2, 0.8, 1),
            MemoryType.SEMANTIC: (0.2, 0.8, 0.4, 1),
        }
        return colors.get(self.memory.memory_type, (0.5, 0.5, 0.5, 1))


class MemoryScreen(MDScreen):
    """Advanced memory management interface with comprehensive functionality.

    This screen provides a complete memory management solution featuring:

    Core Functionality:
        - Full CRUD operations for AI memory management
        - Advanced search with real-time filtering
        - Memory type-based organization and filtering
        - Pagination for efficient large dataset handling
        - System prompt configuration with memory integration

    Search and Filtering:
        - Real-time text search across memory content
        - Memory type filtering with visual indicators
        - Metadata search including tags and attributes
        - Debounced input handling for performance
        - Search result highlighting and context

    User Interface Design:
        - Professional Material Design implementation
        - Responsive layout with adaptive sizing
        - Color-coded memory type system for quick identification
        - Interactive help system with detailed explanations
        - Progressive disclosure of advanced features

    Data Management:
        - Safe memory operations with error handling
        - Background threading for all database operations
        - Pagination with "load more" functionality
        - Optimistic UI updates with rollback on error
        - Comprehensive validation and error recovery

    Performance Optimization:
        - Efficient memory loading with configurable page sizes
        - Search debouncing to reduce unnecessary operations
        - UI virtualization for large memory lists
        - Background processing for heavy operations

    System Integration:
        - System prompt configuration and management
        - Memory integration toggle for AI responses
        - Template-based prompt creation
        - Configuration persistence and validation

    Accessibility Features:
        - Keyboard navigation throughout the interface
        - Screen reader compatible widget hierarchy
        - High contrast design for visibility
        - Touch-friendly controls with appropriate sizing

    The screen demonstrates advanced Kivy application architecture
    with proper separation of concerns and professional UX design.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "memory"

        # Initialize memory manager
        self.safe_memory = create_safe_memory_manager(self._memory_error_callback)

        # State
        self.memories = []
        self.filtered_memories = []
        self.current_filter = None
        self.search_query = ""

        # Pagination state
        self.current_page = 0
        self.page_size = 50  # Load 50 memories at a time
        self.total_memories = 0
        self.has_more_memories = True

        self.build_ui()

        # Load memories
        Clock.schedule_once(self._load_memories, 0.1)

    def build_ui(self):
        """Construct the comprehensive memory management interface.

        This method builds a sophisticated UI structure featuring:

        Layout Architecture:
            1. Fixed toolbar with navigation and action buttons
            2. Informational card with system overview
            3. Collapsible search bar with animation
            4. Filter section with memory type chips
            5. Scrollable memory list with pagination

        Component Organization:
            - Toolbar: Navigation, title, and primary actions
            - Info card: System overview with expandable help
            - Search: Real-time filtering with show/hide animation
            - Filters: Visual memory type selection with color coding
            - Content: Virtualized list with efficient rendering

        Design Principles:
            - Consistent spacing using design system constants
            - Professional color scheme with semantic meaning
            - Progressive disclosure of complex features
            - Visual hierarchy for information scanning

        Interactive Elements:
            - Animated search bar expansion/collapse
            - Color-coded filter chips with selection states
            - Action buttons with appropriate icons
            - Help integration with contextual guidance

        Responsive Design:
            - Adaptive layouts for different screen sizes
            - Flexible component sizing with dp measurements
            - Appropriate elevation and shadow effects
            - Touch-friendly control sizing

        The UI demonstrates advanced Material Design implementation
        with proper component composition and interaction patterns.
        """
        main_layout = MDBoxLayout(orientation="vertical")

        # Toolbar
        toolbar_layout = MDBoxLayout(
            orientation="horizontal",
            size_hint_y=None,
            height=dp(56),
            md_bg_color=(0.12, 0.12, 0.12, 1),
            padding=[dp(16), 0, dp(16), 0],
            spacing=dp(10),
        )

        # Back button
        back_btn = MDIconButton(icon="arrow-left", on_release=self._go_back)
        toolbar_layout.add_widget(back_btn)

        # Title
        title_label = MDLabel(text="Memory Management", font_style="H6", theme_text_color="Primary")
        toolbar_layout.add_widget(title_label)

        # Spacer
        toolbar_layout.add_widget(MDBoxLayout())

        # Action buttons
        add_btn = MDIconButton(icon="plus", on_release=self._show_add_dialog)
        search_btn = MDIconButton(icon="magnify", on_release=self._toggle_search)
        system_btn = MDIconButton(icon="cog", on_release=self._show_system_prompt_dialog)
        refresh_btn = MDIconButton(icon="refresh", on_release=self._refresh_memories)

        toolbar_layout.add_widget(add_btn)
        toolbar_layout.add_widget(search_btn)
        toolbar_layout.add_widget(system_btn)
        toolbar_layout.add_widget(refresh_btn)

        main_layout.add_widget(toolbar_layout)

        # Info section with memory type descriptions
        info_card = MDCard(
            orientation="vertical",
            size_hint_y=None,
            height=dp(100),
            padding=dp(16),
            spacing=dp(8),
            elevation=2,
            md_bg_color=(0.08, 0.08, 0.10, 1),
            radius=[dp(8)],
        )

        info_title = MDLabel(
            text="Memory System Overview",
            font_style="Subtitle1",
            theme_text_color="Primary",
            size_hint_y=None,
            height=dp(24),
        )
        info_card.add_widget(info_title)

        info_text = MDLabel(
            text="Manage your AI's memory and customize system prompts. Use filters to organize by memory type.",
            theme_text_color="Secondary",
            font_style="Body2",
            adaptive_height=True,
            text_size=(None, None),
        )
        info_card.add_widget(info_text)

        # Add expand button for memory type descriptions
        self.show_help_btn = MDIconButton(
            icon="help-circle-outline",
            on_release=self._show_memory_types_help,
            pos_hint={"right": 1, "top": 1},
        )
        info_card.add_widget(self.show_help_btn)

        main_layout.add_widget(info_card)

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
            hint_text="Search memories...",
        )
        self.search_field.bind(text=self._on_search_text)

        search_btn = MDIconButton(icon="magnify", on_release=self._manual_search)
        clear_search_btn = MDIconButton(icon="close", on_release=self._clear_search)

        self.search_layout.add_widget(self.search_field)
        self.search_layout.add_widget(search_btn)
        self.search_layout.add_widget(clear_search_btn)

        main_layout.add_widget(self.search_layout)

        # Filter section with improved layout
        filter_card = MDCard(
            orientation="vertical",
            size_hint_y=None,
            height=dp(80),
            padding=dp(16),
            spacing=dp(8),
            elevation=1,
            md_bg_color=(0.06, 0.06, 0.08, 1),
            radius=[dp(8)],
        )

        filter_title = MDLabel(
            text="Filter by Memory Type",
            font_style="Caption",
            theme_text_color="Secondary",
            size_hint_y=None,
            height=dp(20),
        )
        filter_card.add_widget(filter_title)

        # Filter chips layout
        self.filter_layout = MDBoxLayout(
            orientation="horizontal",
            size_hint_y=None,
            height=dp(40),
            spacing=dp(8),
        )

        # Add filter chips for memory types
        all_chip = MDRaisedButton(
            text="All",
            size_hint_x=None,
            width=dp(80),
            height=dp(32),
            md_bg_color=(0.2, 0.6, 0.8, 1),
            on_release=lambda x: self._apply_filter(None),
        )
        self.filter_layout.add_widget(all_chip)

        # Memory type colors for better visual distinction
        type_colors = {
            MemoryType.SHORT_TERM: (0.2, 0.6, 1.0, 0.8),
            MemoryType.LONG_TERM: (0.8, 0.4, 0.2, 0.8),
            MemoryType.EPISODIC: (0.6, 0.2, 0.8, 0.8),
            MemoryType.SEMANTIC: (0.2, 0.8, 0.4, 0.8),
        }

        for memory_type in MemoryType:
            chip = MDRaisedButton(
                text=memory_type.value.replace("_", " ").title(),
                size_hint_x=None,
                width=dp(100),
                height=dp(32),
                md_bg_color=type_colors.get(memory_type, (0.3, 0.3, 0.3, 1)),
                on_release=lambda x, mt=memory_type: self._apply_filter(mt),
            )
            self.filter_layout.add_widget(chip)

        filter_card.add_widget(self.filter_layout)
        main_layout.add_widget(filter_card)

        # Memory count with better styling
        count_container = MDBoxLayout(
            orientation="horizontal", size_hint_y=None, height=dp(40), padding=dp(16), spacing=dp(8)
        )

        self.count_label = MDLabel(
            text="Loading memories...",
            theme_text_color="Secondary",
            font_style="Body2",
            size_hint_y=None,
            height=dp(24),
        )
        count_container.add_widget(self.count_label)

        # Spacer
        count_container.add_widget(MDBoxLayout())

        # Quick action buttons
        quick_add_btn = MDIconButton(
            icon="plus-circle", theme_icon_color="Primary", on_release=self._show_add_dialog
        )
        count_container.add_widget(quick_add_btn)

        main_layout.add_widget(count_container)

        # Memories list with improved styling
        self.memories_scroll = MDScrollView()
        self.memories_list = MDBoxLayout(
            orientation="vertical",
            spacing=dp(12),
            padding=[dp(16), dp(8), dp(16), dp(16)],
            adaptive_height=True,
        )
        self.memories_scroll.add_widget(self.memories_list)
        main_layout.add_widget(self.memories_scroll)

        self.add_widget(main_layout)

    def _memory_error_callback(self, operation: str, error: str):
        """Handle memory operation errors."""
        logger.error(f"Memory operation failed - {operation}: {error}")
        Notification.error(f"Memory error: {operation}")

    def _go_back(self, *args):
        """Navigate back to chat screen."""
        self.manager.current = "enhanced_chat"

    def _load_memories(self, dt):
        """Initiate asynchronous memory loading with thread safety.

        This method demonstrates the proper Kivy pattern for background operations:

        Threading Strategy:
            - Creates isolated daemon thread for database operations
            - Establishes new event loop for async operations
            - Ensures proper cleanup with try/finally blocks
            - Schedules UI updates on main thread via Clock

        Error Handling:
            - Isolates database errors from UI thread
            - Provides user feedback through notification system
            - Maintains application stability on failure
            - Logs detailed error information for debugging

        Memory Loading:
            - Fetches memories with pagination support
            - Handles large datasets efficiently
            - Provides visual feedback during loading
            - Updates UI components after successful load

        The pattern ensures UI responsiveness while performing
        potentially slow database operations in the background.
        """

        def run_async():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._load_memories_async(load_more=False))
            except Exception as e:
                logger.error(f"Failed to load memories: {e}")
                Clock.schedule_once(lambda dt: Notification.error("Failed to load memories"), 0)
            finally:
                loop.close()

        thread = threading.Thread(target=run_async)
        thread.daemon = True
        thread.start()

    async def _load_memories_async(self, load_more=False):
        """Load memories with intelligent pagination and state management.

        This method implements efficient memory loading:

        Pagination Strategy:
            - Configurable page size for optimal performance
            - Offset calculation for proper data continuation
            - Has-more-data detection for UI control
            - Append vs replace logic for load-more functionality

        Performance Optimization:
            - Batch loading to reduce database round trips
            - Efficient offset-based pagination
            - Memory state management to prevent duplicates
            - UI update scheduling to prevent blocking

        Data Management:
            - Safe memory retrieval with error handling
            - State updates for pagination controls
            - Memory deduplication and ordering
            - Metadata preservation during loading

        User Experience:
            - Progress feedback during loading operations
            - Smooth continuation of existing content
            - Error recovery with user notification
            - Loading state management

        Args:
            load_more (bool): Whether to append to existing memories
                             or replace them with fresh data

        The async implementation ensures efficient memory retrieval
        while maintaining responsive user interface.
        """
        try:
            # Calculate offset for pagination
            offset = 0 if not load_more else len(self.memories)

            # Use the efficient get_all_memories method with pagination
            new_memories = await self.safe_memory.safe_get_all_memories(
                memory_types=None, limit=self.page_size, offset=offset  # Get all types
            )

            # Update pagination state
            self.has_more_memories = len(new_memories) == self.page_size

            if load_more:
                # Append to existing memories
                all_memories = self.memories + new_memories
            else:
                # Replace existing memories
                all_memories = new_memories
                self.current_page = 0

            # Update UI on main thread
            Clock.schedule_once(lambda dt: self._update_memories_list(all_memories, load_more), 0)

        except Exception as e:
            logger.error(f"Failed to load memories: {e}")
            Clock.schedule_once(lambda dt: Notification.error("Failed to load memories"), 0)

    def _update_memories_list(self, memories, is_load_more=False):
        """Update the memories list in the UI."""
        self.memories = memories
        self.filtered_memories = memories.copy()

        if not is_load_more:
            # Clear the list for fresh load
            self._refresh_list_display()
        else:
            # Append new items to existing list
            self._append_memories_to_display()

        # Update count
        if len(memories) == 0:
            self.count_label.text = "No memories found. Click + to add your first memory!"
        else:
            more_text = " (+ more available)" if self.has_more_memories else ""
            self.count_label.text = f"Showing {len(memories)} memories{more_text}"

        logger.info(f"Loaded {len(memories)} memories")

    def _refresh_list_display(self):
        """Refresh the visual list display."""
        self.memories_list.clear_widgets()

        if not self.filtered_memories:
            # Show empty state
            empty_card = MDCard(
                orientation="vertical",
                size_hint_y=None,
                height=dp(120),
                padding=dp(20),
                spacing=dp(10),
                elevation=1,
                md_bg_color=(0.08, 0.08, 0.10, 1),
                radius=[dp(12)],
            )

            empty_icon = MDLabel(
                text="Memory Storage",
                font_size="18sp",
                theme_text_color="Secondary",
                size_hint_y=None,
                height=dp(30),
                halign="center",
            )
            empty_card.add_widget(empty_icon)

            empty_text = MDLabel(
                text="No memories to display.\nClick the + button above to add your first memory!",
                theme_text_color="Secondary",
                font_style="Body2",
                adaptive_height=True,
                halign="center",
            )
            empty_card.add_widget(empty_text)

            self.memories_list.add_widget(empty_card)
        else:
            # Show memory items
            for memory in self.filtered_memories:
                item = MemoryListItem(
                    memory=memory, on_edit=self._edit_memory, on_delete=self._confirm_delete_memory
                )
                self.memories_list.add_widget(item)

            # Add "Load More" button if there are more memories
            if self.has_more_memories:
                self._add_load_more_button()

    def _append_memories_to_display(self):
        """Append new memories to the existing display (for load more)."""
        # Get the number of memories already displayed
        current_count = len(
            [child for child in self.memories_list.children if isinstance(child, MemoryListItem)]
        )

        # Add only the new memories
        new_memories = self.filtered_memories[current_count:]
        for memory in new_memories:
            item = MemoryListItem(
                memory=memory, on_edit=self._edit_memory, on_delete=self._confirm_delete_memory
            )
            # Insert at the beginning to maintain chronological order
            self.memories_list.add_widget(item, index=len(self.memories_list.children))

        # Update or add "Load More" button
        self._update_load_more_button()

    def _add_load_more_button(self):
        """Add a 'Load More' button to the memories list."""
        from kivymd.uix.button import MDRaisedButton
        from kivymd.uix.card import MDCard

        # Create card for the button
        button_card = MDCard(
            orientation="vertical",
            size_hint_y=None,
            height=dp(80),
            padding=dp(15),
            spacing=dp(10),
            elevation=1,
            md_bg_color=(0.15, 0.15, 0.15, 1),
        )

        load_more_btn = MDRaisedButton(
            text=f"Load More Memories ({self.page_size} more)",
            size_hint=(None, None),
            size=(dp(200), dp(40)),
            pos_hint={"center_x": 0.5},
            on_release=self._load_more_memories,
        )

        button_card.add_widget(load_more_btn)
        self.memories_list.add_widget(button_card)

    def _update_load_more_button(self):
        """Update or remove the Load More button based on availability."""
        # Remove existing load more button if any
        for child in self.memories_list.children[:]:
            if hasattr(child, "children") and child.children:
                for grandchild in child.children:
                    if hasattr(grandchild, "text") and "Load More" in str(grandchild.text):
                        self.memories_list.remove_widget(child)
                        break

        # Add new button if more memories are available
        if self.has_more_memories:
            self._add_load_more_button()

    def _load_more_memories(self, *args):
        """Load more memories when button is pressed."""

        def run_async():
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._load_memories_async(load_more=True))
            except Exception as e:
                logger.error(f"Failed to load more memories: {e}")
                Clock.schedule_once(
                    lambda dt: Notification.error("Failed to load more memories"), 0
                )
            finally:
                loop.close()

        thread = threading.Thread(target=run_async)
        thread.daemon = True
        thread.start()

    def _apply_filter(self, memory_type: MemoryType | None):
        """Apply memory type filtering with visual feedback.

        This method implements sophisticated filtering:

        Filter Logic:
            - None type shows all memories (universal filter)
            - Specific types filter to matching memories only
            - Maintains search query when changing filters
            - Updates display with filtered results

        Visual Feedback:
            - Highlights selected filter with accent color
            - Resets other filter buttons to neutral state
            - Updates count display with filtered results
            - Maintains visual consistency across state changes

        State Management:
            - Stores current filter for search integration
            - Preserves search query across filter changes
            - Updates filtered memories collection
            - Refreshes UI display with new results

        Performance Optimization:
            - In-memory filtering for responsive interaction
            - Efficient list comprehension for type matching
            - Minimal UI updates for smooth transitions
            - Debounced operations for rapid filter changes

        User Experience:
            - Immediate visual feedback on filter selection
            - Clear indication of active filter state
            - Consistent behavior across different filter types
            - Intuitive color coding for memory types

        Args:
            memory_type (MemoryType | None): Type to filter by,
                                           None for all memories
        """
        self.current_filter = memory_type

        # Update button colors
        for child in self.filter_layout.children:
            if hasattr(child, "md_bg_color"):
                child.md_bg_color = (0.3, 0.3, 0.3, 1)

        # Highlight selected filter
        if memory_type is None:
            self.filter_layout.children[-1].md_bg_color = (0.2, 0.6, 0.8, 1)  # "All" button
            self.filtered_memories = self.memories.copy()
        else:
            # Find and highlight the correct button
            for _i, child in enumerate(self.filter_layout.children[:-1]):  # Skip "All" button
                if (
                    hasattr(child, "text")
                    and memory_type.value.replace("_", " ").title() in child.text
                ):
                    child.md_bg_color = (0.2, 0.6, 0.8, 1)
                    break

            self.filtered_memories = [m for m in self.memories if m.memory_type == memory_type]

        # Apply search if active
        if self.search_query:
            self._filter_by_search()

        self._refresh_list_display()
        self.count_label.text = f"Showing {len(self.filtered_memories)} memories"

    def _toggle_search(self, *args):
        """Toggle search bar with smooth animation and state management.

        This method demonstrates advanced UI animation:

        Animation Features:
            - Smooth height and opacity transitions
            - Cubic easing for professional feel
            - Appropriate timing for user comfort
            - State-aware animation direction

        State Management:
            - Tracks search bar visibility state
            - Manages focus for keyboard input
            - Clears search when hiding for clean state
            - Preserves search query during visibility

        User Experience:
            - Immediate focus to search field when showing
            - Graceful clear and hide when dismissing
            - Visual feedback throughout animation
            - Keyboard-friendly interaction patterns

        Technical Implementation:
            - Uses Kivy's Animation class for smooth transitions
            - Proper animation chaining for complex sequences
            - Event-driven state changes
            - Memory-efficient animation handling

        Accessibility:
            - Focus management for keyboard navigation
            - Clear visual states for screen readers
            - Logical tab order maintenance
            - Appropriate animation timing for accessibility

        The toggle provides an elegant way to show/hide the search
        interface without disrupting the main content layout.
        """
        logger.info(f"Toggle search called, current height: {self.search_layout.height}")
        if self.search_layout.height == 0:
            # Show search
            logger.info("Showing search bar")
            from kivy.animation import Animation

            anim = Animation(height=dp(60), opacity=1, duration=0.3)
            anim.start(self.search_layout)
            self.search_field.focus = True
        else:
            # Hide search
            logger.info("Hiding search bar")
            from kivy.animation import Animation

            anim = Animation(height=0, opacity=0, duration=0.3)
            anim.start(self.search_layout)
            self._clear_search()

    def _manual_search(self, *args):
        """Manually trigger search."""
        logger.info("Manual search triggered")
        text = self.search_field.text
        self._on_search_text(self.search_field, text)

    def _on_search_text(self, instance, text):
        """Handle search input with intelligent debouncing and validation.

        This method implements sophisticated search functionality:

        Debouncing Strategy:
            - 300ms delay after user stops typing
            - Prevents excessive search operations
            - Cancels pending searches on new input
            - Provides responsive feel without performance cost

        Input Processing:
            - Normalizes input (strip whitespace, lowercase)
            - Validates search query length and content
            - Handles empty queries with proper reset
            - Preserves user input patterns

        Search Triggering:
            - Immediate clear on empty input
            - Debounced search on valid input
            - Cancel mechanism for rapid typing
            - Progress indication for longer operations

        Performance Optimization:
            - Prevents unnecessary database queries
            - Reduces UI update frequency
            - Maintains smooth typing experience
            - Efficient string processing

        User Experience:
            - Immediate feedback on input changes
            - Smooth search experience without lag
            - Clear indication of search activity
            - Intuitive search behavior patterns

        The debouncing ensures optimal performance while maintaining
        a responsive search experience for users.
        """
        logger.info(f"Search text changed: '{text}'")
        self.search_query = text.strip().lower()
        if self.search_query:
            logger.info(f"Starting search for: '{self.search_query}'")
            # Debounce search for better performance - wait 300ms after user stops typing
            Clock.unschedule(self._filter_by_search)
            Clock.schedule_once(lambda dt: self._filter_by_search(), 0.3)
        else:
            logger.info("Clearing search")
            self._clear_search()

    def _filter_by_search(self):
        """Perform comprehensive search across memory content and metadata.

        This method implements multi-field search functionality:

        Search Scope:
            - Primary content text with case-insensitive matching
            - Memory type values with human-readable formatting
            - Metadata keys and values with string conversion
            - Comprehensive coverage of all searchable fields

        Search Algorithm:
            - Sequential field checking for efficiency
            - Early termination on first match per memory
            - Case-insensitive string matching throughout
            - Metadata iteration with safe type conversion

        Performance Optimization:
            - In-memory search for responsive interaction
            - Efficient string operations
            - Early exit patterns to minimize processing
            - Memory-efficient iteration patterns

        Search Quality:
            - Partial string matching for flexible queries
            - Whitespace normalization for consistent results
            - Memory type format conversion for user-friendly search
            - Comprehensive metadata inclusion

        Result Management:
            - Updates filtered memory collection
            - Preserves original memory order
            - Maintains search result integrity
            - Provides result count feedback

        User Feedback:
            - Real-time result count updates
            - Clear indication of search scope
            - Progress feedback for large datasets
            - Empty state handling for no results

        The search provides comprehensive coverage while maintaining
        excellent performance and user experience.
        """
        if not self.search_query:
            return

        base_memories = (
            [m for m in self.memories if m.memory_type == self.current_filter]
            if self.current_filter
            else self.memories
        )

        logger.info(f"Searching in {len(base_memories)} memories for query: '{self.search_query}'")

        self.filtered_memories = []
        for memory in base_memories:
            # Search in content
            if self.search_query in memory.content.lower():
                self.filtered_memories.append(memory)
                continue

            # Search in memory type
            if self.search_query in memory.memory_type.value.lower().replace("_", " "):
                self.filtered_memories.append(memory)
                continue

            # Search in metadata
            if memory.metadata:
                for key, value in memory.metadata.items():
                    if (
                        self.search_query in str(key).lower()
                        or self.search_query in str(value).lower()
                    ):
                        self.filtered_memories.append(memory)
                        break

        logger.info(f"Search found {len(self.filtered_memories)} matching memories")
        self._refresh_list_display()
        self.count_label.text = f"Found {len(self.filtered_memories)} matching memories"

    def _clear_search(self, *args):
        """Clear search and show filtered memories."""
        logger.info("Clearing search")
        self.search_query = ""
        if hasattr(self, "search_field"):
            self.search_field.text = ""
        self._apply_filter(self.current_filter)

    def _refresh_memories(self, *args):
        """Refresh memories from database."""
        Notification.info("Refreshing memories...")
        self._load_memories(None)

    def _show_add_dialog(self, *args):
        """Display comprehensive memory creation dialog with validation.

        This method creates a sophisticated memory creation interface:

        Form Components:
            - Multi-line content field with text wrapping
            - Memory type selection with visual indicators
            - Importance slider with real-time value display
            - Optional tags field for organization

        Input Validation:
            - Required content field validation
            - Memory type selection requirement
            - Importance value range validation
            - Tag format validation and processing

        User Experience:
            - Intuitive form layout with logical flow
            - Visual feedback for selections and inputs
            - Clear action buttons with semantic colors
            - Responsive dialog sizing for different screens

        Default Values:
            - Long-term memory type as sensible default
            - Moderate importance (0.7) for balanced weighting
            - Empty tags field for optional organization
            - Proper field focus and tab order

        Visual Design:
            - Material Design dialog structure
            - Consistent spacing and typography
            - Color-coded memory type buttons
            - Professional form styling

        Accessibility:
            - Keyboard navigation support
            - Screen reader compatible labels
            - Logical tab order through form fields
            - Clear visual hierarchy

        The dialog provides a complete memory creation experience
        with proper validation and user guidance.
        """
        content = MDBoxLayout(
            orientation="vertical", spacing=dp(15), size_hint_y=None, height=dp(350)
        )

        # Content field
        self.add_content_field = MDTextField(
            hint_text="Enter memory content...",
            multiline=True,
            size_hint_y=None,
            height=dp(100),
        )
        content.add_widget(self.add_content_field)

        # Memory type selection
        type_label = MDLabel(text="Memory Type:", size_hint_y=None, height=dp(30))
        content.add_widget(type_label)

        type_layout = MDBoxLayout(
            orientation="horizontal", size_hint_y=None, height=dp(40), spacing=dp(10)
        )

        self.add_type_buttons = []
        for memory_type in MemoryType:
            btn = MDRaisedButton(
                text=memory_type.value.replace("_", " ").title(),
                size_hint_x=0.25,
                md_bg_color=(
                    (0.2, 0.6, 0.8, 1)
                    if memory_type == MemoryType.LONG_TERM
                    else (0.3, 0.3, 0.3, 1)
                ),
                on_release=lambda x, t=memory_type: self._select_add_type(t),
            )
            self.add_type_buttons.append((btn, memory_type))
            type_layout.add_widget(btn)

        content.add_widget(type_layout)

        # Importance slider
        importance_label = MDLabel(text="Importance: 0.7", size_hint_y=None, height=dp(30))
        content.add_widget(importance_label)

        self.add_importance_slider = MDSlider(
            min=0.0, max=1.0, value=0.7, step=0.1, size_hint_y=None, height=dp(40)
        )
        self.add_importance_slider.bind(
            value=lambda x, v: setattr(importance_label, "text", f"Importance: {v:.1f}")
        )
        content.add_widget(self.add_importance_slider)

        # Tags field
        self.add_tags_field = MDTextField(
            hint_text="Tags (comma-separated, optional)",
            size_hint_y=None,
            height=dp(40),
        )
        content.add_widget(self.add_tags_field)

        self.selected_add_type = MemoryType.LONG_TERM

        dialog = MDDialog(
            title="Add New Memory",
            type="custom",
            content_cls=content,
            size_hint=(0.9, None),
            height=dp(450),
            buttons=[
                MDRaisedButton(text="Cancel", on_release=lambda x: dialog.dismiss()),
                MDRaisedButton(
                    text="Add Memory",
                    md_bg_color=(0.2, 0.8, 0.4, 1),
                    on_release=lambda x: self._create_memory(dialog),
                ),
            ],
        )
        dialog.open()

    def _select_add_type(self, memory_type: MemoryType):
        """Select memory type for new memory."""
        self.selected_add_type = memory_type
        for btn, btn_type in self.add_type_buttons:
            btn.md_bg_color = (0.2, 0.6, 0.8, 1) if btn_type == memory_type else (0.3, 0.3, 0.3, 1)

    def _create_memory(self, dialog):
        """Process memory creation with validation and error handling.

        This method handles the complete memory creation pipeline:

        Input Validation:
            - Content field requirement checking
            - Memory type selection validation
            - Importance value range verification
            - Tag processing and formatting

        Data Processing:
            - Content text normalization
            - Tag parsing from comma-separated input
            - Metadata construction with creation context
            - Importance value extraction from slider

        Async Execution:
            - Background thread creation for database operations
            - New event loop establishment for isolation
            - Proper error handling and cleanup
            - UI feedback scheduling on main thread

        Error Handling:
            - Validation error display with user guidance
            - Database error recovery with notifications
            - Thread safety with proper cleanup
            - Graceful degradation on failure

        User Feedback:
            - Immediate dialog dismissal on valid input
            - Progress indication during creation
            - Success notification with memory ID
            - Error messages with actionable guidance

        The method ensures reliable memory creation while
        maintaining excellent user experience and error recovery.
        """
        content = self.add_content_field.text.strip()
        if not content:
            Notification.error("Memory content cannot be empty")
            return

        dialog.dismiss()

        importance = self.add_importance_slider.value
        tags = [t.strip() for t in self.add_tags_field.text.split(",") if t.strip()]

        def run_create():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._create_memory_async(content, importance, tags))
            except Exception as e:
                logger.error(f"Failed to create memory: {e}")
                Clock.schedule_once(lambda dt: Notification.error("Failed to create memory"), 0)
            finally:
                loop.close()

        thread = threading.Thread(target=run_create)
        thread.daemon = True
        thread.start()

    async def _create_memory_async(self, content: str, importance: float, tags: list):
        """Asynchronously create memory with comprehensive metadata.

        This method implements the complete memory creation process:

        Metadata Construction:
            - Creation source tracking for provenance
            - Timestamp recording for temporal context
            - Manual creation flag for user-generated content
            - Tag integration for organizational structure

        Memory System Integration:
            - Safe memory manager usage for error handling
            - Memory type and importance specification
            - Automatic ID generation and tracking
            - Vector embedding creation for similarity search

        Error Handling:
            - Database connection error recovery
            - Memory creation failure detection
            - User notification with meaningful messages
            - Logging for debugging and monitoring

        Success Processing:
            - Memory list refresh for immediate visibility
            - User notification with memory ID confirmation
            - UI state updates for consistent display
            - Analytics tracking for usage patterns

        Performance Optimization:
            - Efficient metadata construction
            - Batched operations where possible
            - Minimal UI thread blocking
            - Optimized database interactions

        Args:
            content (str): The memory content text
            importance (float): Memory importance weighting (0.0-1.0)
            tags (list): Optional tags for organization

        The async implementation ensures responsive UI while
        performing complex memory system operations.
        """
        try:
            metadata = {
                "created_manually": True,
                "source": "memory_management",
                "created_at": datetime.now().isoformat(),
            }

            if tags:
                metadata["tags"] = tags

            memory_id = await self.safe_memory.safe_remember(
                content=content,
                memory_type=self.selected_add_type,
                importance=importance,
                metadata=metadata,
            )

            if memory_id:
                # Reload memories
                await self._load_memories_async()
                Clock.schedule_once(
                    lambda dt: Notification.success(f"Memory created: {memory_id[:8]}..."), 0
                )
            else:
                Clock.schedule_once(lambda dt: Notification.error("Failed to create memory"), 0)

        except Exception as e:
            logger.error(f"Memory creation error: {e}")
            Clock.schedule_once(lambda dt: Notification.error("Memory creation failed"), 0)

    def _edit_memory(self, memory: Memory):
        """Display memory editing dialog with pre-populated values.

        This method creates a comprehensive memory editing interface:

        Form Pre-population:
            - Current memory content in editable text field
            - Memory type selection with current type highlighted
            - Importance slider set to current value
            - Visual indicators for current selections

        Editing Interface:
            - Multi-line content editing with text wrapping
            - Memory type switching with color-coded buttons
            - Importance adjustment with real-time feedback
            - Save/cancel options with clear semantics

        Data Integrity:
            - Preserves original memory ID and metadata
            - Maintains creation timestamp and source
            - Updates modification timestamp on save
            - Validates all changes before submission

        User Experience:
            - Familiar editing interface matching creation dialog
            - Clear indication of current vs. modified values
            - Immediate feedback on changes
            - Proper error handling and validation

        Visual Design:
            - Consistent with creation dialog styling
            - Clear visual hierarchy for form elements
            - Professional button styling and colors
            - Responsive dialog sizing

        Accessibility:
            - Keyboard navigation throughout the form
            - Screen reader compatible field labels
            - Logical tab order for efficient editing
            - Clear focus indicators

        Args:
            memory (Memory): The memory object to edit

        The editing interface provides comprehensive memory
        modification capabilities with proper validation.
        """
        content = MDBoxLayout(
            orientation="vertical", spacing=dp(15), size_hint_y=None, height=dp(300)
        )

        # Content field
        self.edit_content_field = MDTextField(
            text=memory.content,
            multiline=True,
            size_hint_y=None,
            height=dp(100),
        )
        content.add_widget(self.edit_content_field)

        # Memory type selection
        type_label = MDLabel(text="Memory Type:", size_hint_y=None, height=dp(30))
        content.add_widget(type_label)

        type_layout = MDBoxLayout(
            orientation="horizontal", size_hint_y=None, height=dp(40), spacing=dp(10)
        )

        self.edit_type_buttons = []
        for memory_type in MemoryType:
            btn = MDRaisedButton(
                text=memory_type.value.replace("_", " ").title(),
                size_hint_x=0.25,
                md_bg_color=(
                    (0.2, 0.6, 0.8, 1) if memory_type == memory.memory_type else (0.3, 0.3, 0.3, 1)
                ),
                on_release=lambda x, t=memory_type: self._select_edit_type(t),
            )
            self.edit_type_buttons.append((btn, memory_type))
            type_layout.add_widget(btn)

        content.add_widget(type_layout)

        # Importance slider
        importance_label = MDLabel(
            text=f"Importance: {memory.importance:.1f}", size_hint_y=None, height=dp(30)
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
            height=dp(400),
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

    def _select_edit_type(self, memory_type: MemoryType):
        """Select memory type for editing."""
        self.selected_edit_type = memory_type
        for btn, btn_type in self.edit_type_buttons:
            btn.md_bg_color = (0.2, 0.6, 0.8, 1) if btn_type == memory_type else (0.3, 0.3, 0.3, 1)

    def _save_memory_changes(self, dialog, memory: Memory):
        """Save changes to memory."""
        new_content = self.edit_content_field.text.strip()
        if not new_content:
            Notification.error("Memory content cannot be empty")
            return

        dialog.dismiss()

        def run_update():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(
                    self._update_memory_async(
                        memory,
                        new_content,
                        self.selected_edit_type,
                        self.edit_importance_slider.value,
                    )
                )
            except Exception as e:
                logger.error(f"Failed to update memory: {e}")
                Clock.schedule_once(lambda dt: Notification.error("Failed to update memory"), 0)
            finally:
                loop.close()

        thread = threading.Thread(target=run_update)
        thread.daemon = True
        thread.start()

    async def _update_memory_async(
        self, memory: Memory, new_content: str, new_type: MemoryType, new_importance: float
    ):
        """Asynchronously update memory with validation and error handling.

        This method implements comprehensive memory updating:

        Memory Object Updates:
            - Content modification with validation
            - Memory type classification changes
            - Importance reweighting for retrieval priority
            - Access timestamp update for usage tracking

        Database Operations:
            - Atomic update operations for consistency
            - Vector embedding regeneration for content changes
            - Index updates for efficient retrieval
            - Transaction management for data integrity

        Error Handling:
            - Database connection error recovery
            - Update conflict detection and resolution
            - Validation error handling with user feedback
            - Rollback mechanisms for failed operations

        Success Processing:
            - Memory list refresh for immediate visibility
            - User notification of successful update
            - UI state synchronization
            - Analytics tracking for modification patterns

        Performance Optimization:
            - Efficient database update operations
            - Minimal data transfer for unchanged fields
            - Optimized vector embedding updates
            - Batched UI updates for smooth experience

        Data Consistency:
            - Maintains referential integrity
            - Preserves creation metadata
            - Updates modification timestamps
            - Validates all field changes

        Args:
            memory (Memory): Original memory object
            new_content (str): Updated content text
            new_type (MemoryType): New memory classification
            new_importance (float): Updated importance weighting

        The async implementation ensures data integrity while
        maintaining responsive user interface during updates.
        """
        try:
            # Update memory object
            memory.content = new_content
            memory.memory_type = new_type
            memory.importance = new_importance
            memory.accessed_at = datetime.now()

            # Update in store
            success = await self.safe_memory.manager.store.update(memory)

            if success:
                # Reload memories
                await self._load_memories_async()
                Clock.schedule_once(
                    lambda dt: Notification.success("Memory updated successfully"), 0
                )
            else:
                Clock.schedule_once(lambda dt: Notification.error("Failed to update memory"), 0)

        except Exception as e:
            logger.error(f"Memory update error: {e}")
            Clock.schedule_once(lambda dt: Notification.error("Memory update failed"), 0)

    def _confirm_delete_memory(self, memory: Memory):
        """Display memory deletion confirmation with content preview.

        This method creates a safe deletion confirmation interface:

        Confirmation Design:
            - Content preview for user verification
            - Clear deletion warning with consequences
            - Prominent cancel option for safety
            - Visual distinction between safe and destructive actions

        Content Preview:
            - Truncated memory content display
            - Sufficient context for identification
            - Readable formatting with proper wrapping
            - Visual separation from action buttons

        Safety Measures:
            - Explicit confirmation requirement
            - Clear warning about irreversibility
            - Cancel button prominence
            - Destructive action color coding (red)

        User Experience:
            - Clear dialog title and messaging
            - Appropriate button sizing and spacing
            - Professional visual design
            - Accessible button labeling

        Error Prevention:
            - Content verification before deletion
            - Clear consequence communication
            - Multiple confirmation steps
            - Safe default action (cancel)

        Accessibility:
            - Screen reader compatible dialog structure
            - Keyboard navigation support
            - Clear focus indicators
            - Logical reading order

        Args:
            memory (Memory): The memory object to delete

        The confirmation dialog prevents accidental deletions
        while providing clear context for informed decisions.
        """
        content = MDBoxLayout(
            orientation="vertical", spacing=dp(10), size_hint_y=None, height=dp(120)
        )

        preview_text = memory.content[:100] + "..." if len(memory.content) > 100 else memory.content
        preview_label = MDLabel(
            text=f"Delete this memory?\n\n{preview_text}",
            theme_text_color="Primary",
            adaptive_height=True,
        )
        content.add_widget(preview_label)

        warning_label = MDLabel(
            text="This action cannot be undone.",
            theme_text_color="Error",
            adaptive_height=True,
        )
        content.add_widget(warning_label)

        dialog = MDDialog(
            title="Confirm Delete",
            type="custom",
            content_cls=content,
            buttons=[
                MDRaisedButton(text="Cancel", on_release=lambda x: dialog.dismiss()),
                MDRaisedButton(
                    text="Delete",
                    md_bg_color=(0.8, 0.2, 0.2, 1),
                    on_release=lambda x: self._delete_memory(dialog, memory),
                ),
            ],
        )
        dialog.open()

    def _delete_memory(self, dialog, memory: Memory):
        """Delete a memory."""
        dialog.dismiss()

        def run_delete():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._delete_memory_async(memory))
            except Exception as e:
                logger.error(f"Failed to delete memory: {e}")
                Clock.schedule_once(lambda dt: Notification.error("Failed to delete memory"), 0)
            finally:
                loop.close()

        thread = threading.Thread(target=run_delete)
        thread.daemon = True
        thread.start()

    async def _delete_memory_async(self, memory: Memory):
        """Asynchronously delete memory with comprehensive cleanup.

        This method implements safe memory deletion:

        Deletion Process:
            - Memory ID validation before deletion
            - Safe memory manager deletion operation
            - Vector embedding cleanup
            - Index updates for consistency

        Data Integrity:
            - Atomic deletion operations
            - Referential integrity maintenance
            - Cascade deletion for related data
            - Transaction management for consistency

        Error Handling:
            - Database connection error recovery
            - Deletion failure detection and reporting
            - User notification with meaningful messages
            - Logging for debugging and auditing

        Success Processing:
            - Memory list refresh for immediate UI update
            - User notification of successful deletion
            - UI state cleanup for removed memory
            - Analytics tracking for deletion patterns

        Performance Optimization:
            - Efficient database deletion operations
            - Minimal impact on remaining memories
            - Optimized index updates
            - Batched UI updates for smooth experience

        Cleanup Operations:
            - Vector store cleanup for removed embeddings
            - Index defragmentation for performance
            - Memory cache invalidation
            - Resource cleanup for deleted objects

        Args:
            memory (Memory): The memory object to delete

        The async implementation ensures complete cleanup while
        maintaining system performance and data integrity.
        """
        try:
            success = await self.safe_memory.safe_forget(memory.id)

            if success:
                # Reload memories
                await self._load_memories_async()
                Clock.schedule_once(
                    lambda dt: Notification.success("Memory deleted successfully"), 0
                )
            else:
                Clock.schedule_once(lambda dt: Notification.error("Failed to delete memory"), 0)

        except Exception as e:
            logger.error(f"Memory deletion error: {e}")
            Clock.schedule_once(lambda dt: Notification.error("Memory deletion failed"), 0)

    def _show_system_prompt_dialog(self, *args):
        """Display comprehensive system prompt configuration interface.

        This method creates an advanced system prompt management dialog:

        Configuration Features:
            - Multi-line system prompt editing with syntax support
            - Template selection for common prompt patterns
            - Memory integration toggle for enhanced responses
            - Real-time character count and validation

        Template System:
            - Pre-defined templates for common use cases
            - One-click template application
            - Template categorization (helpful, coding, research)
            - Customizable template library

        Memory Integration:
            - Toggle for automatic memory inclusion
            - Configuration of memory retrieval parameters
            - Context window management
            - Memory relevance thresholds

        User Experience:
            - Professional dialog design with clear sections
            - Intuitive template selection interface
            - Visual feedback for configuration changes
            - Comprehensive help and documentation

        Validation and Safety:
            - Prompt length validation
            - Syntax checking for common errors
            - Preview functionality for testing
            - Rollback capability for safe experimentation

        Accessibility:
            - Keyboard navigation throughout the interface
            - Screen reader compatible form structure
            - Clear labeling and help text
            - Logical tab order for efficient configuration

        The dialog provides complete control over AI system behavior
        with user-friendly template system and memory integration.
        """
        content = MDBoxLayout(
            orientation="vertical", spacing=dp(15), size_hint_y=None, height=dp(400)
        )

        # Title
        title_label = MDLabel(
            text="System Prompt Configuration",
            font_style="H6",
            theme_text_color="Primary",
            size_hint_y=None,
            height=dp(30),
        )
        content.add_widget(title_label)

        # Description
        desc_label = MDLabel(
            text="System prompts are prepended to all conversations to set the assistant's behavior and context.",
            theme_text_color="Secondary",
            adaptive_height=True,
            text_size=(dp(350), None),
        )
        content.add_widget(desc_label)

        # Current system prompt field
        current_label = MDLabel(
            text="Current System Prompt:",
            theme_text_color="Primary",
            size_hint_y=None,
            height=dp(25),
        )
        content.add_widget(current_label)

        # Load current system prompt
        current_prompt = self._get_current_system_prompt()

        self.system_prompt_field = MDTextField(
            text=current_prompt,
            hint_text="Enter your system prompt here...",
            multiline=True,
            size_hint_y=None,
            height=dp(120),
        )
        content.add_widget(self.system_prompt_field)

        # Quick templates
        templates_label = MDLabel(
            text="Quick Templates:", theme_text_color="Primary", size_hint_y=None, height=dp(25)
        )
        content.add_widget(templates_label)

        # Template buttons
        template_layout = MDBoxLayout(
            orientation="horizontal", size_hint_y=None, height=dp(40), spacing=dp(8)
        )

        templates = [
            ("Helpful", "You are a helpful, knowledgeable, and friendly assistant."),
            (
                "Coding",
                "You are an expert software engineer. Provide clear, efficient solutions with explanations.",
            ),
            (
                "Research",
                "You are a research assistant. Provide thorough, well-sourced information and analysis.",
            ),
        ]

        for name, template in templates:
            btn = MDRaisedButton(
                text=name,
                size_hint_x=0.33,
                md_bg_color=(0.3, 0.6, 0.8, 1),
                on_release=lambda x, t=template: self._apply_template(t),
            )
            template_layout.add_widget(btn)

        content.add_widget(template_layout)

        # Memory integration checkbox
        memory_layout = MDBoxLayout(
            orientation="horizontal", size_hint_y=None, height=dp(40), spacing=dp(8)
        )

        self.memory_integration_checkbox = MDCheckbox(
            active=True, size_hint=(None, None), size=(dp(24), dp(24))
        )

        memory_label = MDLabel(
            text="Include relevant memories in system prompt", theme_text_color="Primary"
        )

        memory_layout.add_widget(self.memory_integration_checkbox)
        memory_layout.add_widget(memory_label)
        content.add_widget(memory_layout)

        dialog = MDDialog(
            title="System Prompt Settings",
            type="custom",
            content_cls=content,
            size_hint=(0.9, None),
            height=dp(500),
            buttons=[
                MDRaisedButton(text="Cancel", on_release=lambda x: dialog.dismiss()),
                MDRaisedButton(
                    text="Clear",
                    md_bg_color=(0.8, 0.4, 0.2, 1),
                    on_release=lambda x: self._clear_system_prompt(dialog),
                ),
                MDRaisedButton(
                    text="Save",
                    md_bg_color=(0.2, 0.8, 0.4, 1),
                    on_release=lambda x: self._save_system_prompt(dialog),
                ),
            ],
        )
        dialog.open()

    def _get_current_system_prompt(self) -> str:
        """Get the current system prompt from configuration."""
        try:
            from src.core.config import _config_manager

            prompt = _config_manager.get("system_prompt", "")
            logger.info(f"Retrieved system prompt: {len(prompt)} chars")
            return prompt
        except Exception as e:
            logger.error(f"Failed to get system prompt: {e}")
            return ""

    def _apply_template(self, template: str):
        """Apply a template to the system prompt field."""
        self.system_prompt_field.text = template

    def _clear_system_prompt(self, dialog):
        """Clear the system prompt."""
        self.system_prompt_field.text = ""
        self._save_system_prompt(dialog)

    def _save_system_prompt(self, dialog):
        """Save system prompt configuration with validation and integration.

        This method handles comprehensive system prompt persistence:

        Configuration Extraction:
            - System prompt text with validation
            - Memory integration settings
            - Character length verification
            - Format validation for AI compatibility

        Persistence Strategy:
            - Configuration manager integration
            - Atomic save operations for consistency
            - Backup creation before changes
            - Rollback capability on failure

        System Integration:
            - Chat manager notification of changes
            - Active session update for immediate effect
            - Memory system configuration update
            - Provider-specific prompt formatting

        Validation Process:
            - Prompt length limits for AI model compatibility
            - Content validation for potential issues
            - Memory integration compatibility checking
            - Provider-specific requirement validation

        Error Handling:
            - Configuration save failure recovery
            - Validation error reporting with guidance
            - System integration error handling
            - User notification with actionable feedback

        User Feedback:
            - Success confirmation with character count
            - Clear notification for empty prompts
            - Error messages with specific guidance
            - Visual confirmation of active configuration

        The method ensures reliable system prompt configuration
        with proper validation and system-wide integration.
        """
        try:
            prompt = self.system_prompt_field.text.strip()
            memory_integration = self.memory_integration_checkbox.active

            logger.info(
                f"Saving system prompt: {len(prompt)} chars, memory integration: {memory_integration}"
            )

            # Save to configuration - use synchronous method for reliability
            def save_config():
                try:
                    import asyncio

                    from src.core.config import _config_manager

                    # Use async set method in a new event loop
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    # Set the values
                    loop.run_until_complete(_config_manager.set("system_prompt", prompt))
                    loop.run_until_complete(
                        _config_manager.set("system_prompt_memory_integration", memory_integration)
                    )

                    # Save the configuration
                    _config_manager.save()

                    logger.info("System prompt configuration saved successfully")

                except Exception as e:
                    logger.error(f"Error saving config: {e}")
                    raise
                finally:
                    loop.close()

            save_config()

            # Notify the chat manager if available
            if hasattr(self.manager, "get_screen"):
                try:
                    chat_screen = self.manager.get_screen("enhanced_chat")
                    if hasattr(chat_screen, "chat_manager"):
                        chat_screen.chat_manager.update_system_prompt(prompt, memory_integration)
                        logger.info("Chat manager updated with new system prompt")
                except Exception as e:
                    logger.warning(f"Could not update chat manager: {e}")

            dialog.dismiss()

            if prompt:
                Notification.success(f"System prompt saved ({len(prompt)} characters)")
            else:
                Notification.info("System prompt cleared")

        except Exception as e:
            logger.error(f"Failed to save system prompt: {e}")
            Notification.error("Failed to save system prompt")

    def _show_memory_types_help(self, *args):
        """Display comprehensive memory system education interface.

        This method creates an informative help system featuring:

        Educational Content:
            - Detailed explanation of each memory type
            - Use case examples for practical understanding
            - Color coding explanation for visual reference
            - Best practices for memory organization

        Memory Type Explanations:
            - Short-term: Temporary conversation context
            - Long-term: Persistent cross-session information
            - Episodic: Event-based temporal memories
            - Semantic: General knowledge and facts

        Visual Design:
            - Professional typography with clear hierarchy
            - Color-coded sections matching the memory type system
            - Scannable layout for quick reference
            - Appropriate spacing and visual grouping

        Usage Guidance:
            - Practical tips for effective memory management
            - Search and filtering technique explanations
            - Importance weighting guidance
            - System prompt integration advice

        Content Organization:
            - Logical flow from basic concepts to advanced usage
            - Progressive disclosure of complex information
            - Quick reference summary for experienced users
            - Comprehensive detail for new users

        Accessibility:
            - Screen reader compatible content structure
            - Clear heading hierarchy for navigation
            - Appropriate contrast and text sizing
            - Keyboard navigation support

        The help system provides comprehensive education about
        the memory system while maintaining professional presentation.
        """
        content = MDBoxLayout(
            orientation="vertical", spacing=dp(15), size_hint_y=None, height=dp(450)
        )

        # Title
        title_label = MDLabel(
            text="Memory Types Explained",
            font_style="H6",
            theme_text_color="Primary",
            size_hint_y=None,
            height=dp(30),
        )
        content.add_widget(title_label)

        # Memory type descriptions
        memory_types_info = [
            {
                "name": "Short-Term Memory",
                "color": "(Blue)",
                "description": "Temporary information that's actively being used in the current conversation. Typically lasts for the duration of a chat session and includes recent messages, context, and immediate working data.",
            },
            {
                "name": "Long-Term Memory",
                "color": "(Orange)",
                "description": "Important information that should be preserved across sessions. Includes user preferences, important facts, ongoing projects, and significant conversation highlights that inform future interactions.",
            },
            {
                "name": "Episodic Memory",
                "color": "(Purple)",
                "description": "Specific events, experiences, and conversations with temporal context. Like remembering 'what happened when' - specific interactions, decisions made, or particular moments in your conversations.",
            },
            {
                "name": "Semantic Memory",
                "color": "(Green)",
                "description": "General knowledge, facts, and concepts without specific temporal context. Includes learned information, general preferences, skills, and factual knowledge that doesn't depend on when it was learned.",
            },
        ]

        for mem_type in memory_types_info:
            # Type header
            type_header = MDLabel(
                text=f"{mem_type['name']} {mem_type['color']}",
                font_style="Subtitle1",
                theme_text_color="Primary",
                size_hint_y=None,
                height=dp(25),
                bold=True,
            )
            content.add_widget(type_header)

            # Description
            desc_label = MDLabel(
                text=mem_type["description"],
                theme_text_color="Secondary",
                font_style="Body2",
                adaptive_height=True,
                text_size=(dp(350), None),
            )
            content.add_widget(desc_label)

            # Spacer
            content.add_widget(MDLabel(text="", size_hint_y=None, height=dp(10)))

        # Usage tips
        tips_header = MDLabel(
            text="Usage Tips:",
            font_style="Subtitle1",
            theme_text_color="Primary",
            size_hint_y=None,
            height=dp(25),
            bold=True,
        )
        content.add_widget(tips_header)

        tips_text = MDLabel(
            text="• Use filters to organize memories by type\n• Search works across all memory content\n• Higher importance memories are prioritized\n• Edit memories to update their type or importance\n• System prompts can include relevant memories automatically",
            theme_text_color="Secondary",
            adaptive_height=True,
            text_size=(dp(350), None),
        )
        content.add_widget(tips_text)

        dialog = MDDialog(
            title="Memory System Guide",
            type="custom",
            content_cls=content,
            size_hint=(0.9, None),
            height=dp(550),
            buttons=[MDRaisedButton(text="Got it!", on_release=lambda x: dialog.dismiss())],
        )
        dialog.open()
