"""Memory management screen with full CRUD operations."""

import asyncio
import threading
from datetime import datetime
from typing import Any

from kivy.clock import Clock
from kivy.metrics import dp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDFlatButton, MDIconButton, MDRaisedButton
from kivymd.uix.card import MDCard
from kivymd.uix.dialog import MDDialog
from kivymd.uix.label import MDLabel
from kivymd.uix.list import MDList, TwoLineListItem
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
    """Card widget for displaying individual memories with action buttons."""

    def __init__(self, memory: Memory, on_edit=None, on_delete=None, **kwargs):
        super().__init__(**kwargs)
        self.memory = memory
        self.on_edit = on_edit
        self.on_delete = on_delete
        
        # Card properties
        self.orientation = "horizontal"
        self.size_hint_y = None
        self.height = dp(80)
        self.padding = dp(12)
        self.spacing = dp(12)
        self.elevation = 2
        self.md_bg_color = (0.10, 0.10, 0.12, 1)
        self.radius = [dp(8)]
        
        # Build the layout
        self._build_layout()

    def _build_layout(self):
        """Build the memory item layout."""
        # Content layout (left side)
        content_layout = MDBoxLayout(orientation="vertical", spacing=dp(4))
        
        # Title
        title = self.memory.content[:60] + "..." if len(self.memory.content) > 60 else self.memory.content
        title_label = MDLabel(
            text=title,
            font_style="Subtitle1",
            theme_text_color="Primary",
            size_hint_y=None,
            height=dp(24),
            text_size=(None, None)
        )
        content_layout.add_widget(title_label)
        
        # Subtitle with memory info
        subtitle = f"{self.memory.memory_type.value.replace('_', ' ').title()} • Importance: {self.memory.importance:.2f} • {self.memory.created_at.strftime('%m/%d %H:%M')}"
        subtitle_label = MDLabel(
            text=subtitle,
            font_style="Caption",
            theme_text_color="Secondary",
            size_hint_y=None,
            height=dp(20),
            text_size=(None, None)
        )
        content_layout.add_widget(subtitle_label)
        
        # Memory type chip
        type_chip = MDLabel(
            text=f"[{self.memory.memory_type.value.upper()}]",
            font_style="Caption",
            theme_text_color="Custom",
            text_color=self._get_type_color(),
            size_hint_y=None,
            height=dp(16),
        )
        content_layout.add_widget(type_chip)
        
        self.add_widget(content_layout)
        
        # Action buttons (right side)
        button_layout = MDBoxLayout(
            orientation="horizontal", 
            size_hint_x=None, 
            width=dp(80), 
            spacing=dp(8)
        )
        
        edit_btn = MDIconButton(
            icon="pencil",
            theme_icon_color="Primary", 
            size_hint=(None, None),
            size=(dp(36), dp(36)),
            on_release=lambda x: self.on_edit(self.memory) if self.on_edit else None
        )
        
        delete_btn = MDIconButton(
            icon="delete",
            theme_icon_color="Error",
            size_hint=(None, None),
            size=(dp(36), dp(36)),
            on_release=lambda x: self.on_delete(self.memory) if self.on_delete else None
        )
        
        button_layout.add_widget(edit_btn)
        button_layout.add_widget(delete_btn)
        
        self.add_widget(button_layout)

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
    """Memory management screen with search, add, edit, delete functionality."""

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
        
        self.build_ui()
        
        # Load memories
        Clock.schedule_once(self._load_memories, 0.1)

    def build_ui(self):
        """Build the memory management interface."""
        main_layout = MDBoxLayout(orientation="vertical")

        # Toolbar
        toolbar_layout = MDBoxLayout(
            orientation="horizontal",
            size_hint_y=None,
            height=dp(56),
            md_bg_color=(0.12, 0.12, 0.12, 1),
            padding=[dp(16), 0, dp(16), 0],
            spacing=dp(10)
        )

        # Back button
        back_btn = MDFlatButton(text="← Back", on_release=self._go_back)
        toolbar_layout.add_widget(back_btn)

        # Title
        title_label = MDLabel(text="Memory Management", font_style="H6", theme_text_color="Primary")
        toolbar_layout.add_widget(title_label)

        # Spacer
        toolbar_layout.add_widget(MDBoxLayout())

        # Action buttons
        add_btn = MDIconButton(icon="plus", on_release=self._show_add_dialog)
        search_btn = MDIconButton(icon="magnify", on_release=self._toggle_search)
        refresh_btn = MDIconButton(icon="refresh", on_release=self._refresh_memories)
        
        toolbar_layout.add_widget(add_btn)
        toolbar_layout.add_widget(search_btn)
        toolbar_layout.add_widget(refresh_btn)

        main_layout.add_widget(toolbar_layout)

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
            on_text=self._on_search_text,
        )
        
        clear_search_btn = MDIconButton(icon="close", on_release=self._clear_search)

        self.search_layout.add_widget(self.search_field)
        self.search_layout.add_widget(clear_search_btn)

        main_layout.add_widget(self.search_layout)

        # Filter chips
        self.filter_layout = MDBoxLayout(
            orientation="horizontal",
            size_hint_y=None,
            height=dp(40),
            padding=dp(16),
            spacing=dp(8),
        )
        
        # Add filter chips for memory types
        all_chip = MDRaisedButton(
            text="All", 
            size_hint_x=None, 
            width=dp(80),
            height=dp(30),
            md_bg_color=(0.2, 0.6, 0.8, 1),
            on_release=lambda x: self._apply_filter(None)
        )
        self.filter_layout.add_widget(all_chip)
        
        for memory_type in MemoryType:
            chip = MDRaisedButton(
                text=memory_type.value.replace("_", " ").title(),
                size_hint_x=None,
                width=dp(100),
                height=dp(30),
                md_bg_color=(0.3, 0.3, 0.3, 1),
                on_release=lambda x, mt=memory_type: self._apply_filter(mt)
            )
            self.filter_layout.add_widget(chip)

        main_layout.add_widget(self.filter_layout)

        # Memory count
        self.count_label = MDLabel(
            text="Loading memories...",
            theme_text_color="Secondary",
            font_style="Caption",
            size_hint_y=None,
            height=dp(30),
            padding=dp(16),
        )
        main_layout.add_widget(self.count_label)

        # Memories list
        self.memories_scroll = MDScrollView()
        self.memories_list = MDBoxLayout(
            orientation="vertical", 
            spacing=dp(8), 
            padding=dp(16), 
            adaptive_height=True
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
        """Load memories asynchronously."""
        def run_async():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._load_memories_async())
            except Exception as e:
                logger.error(f"Failed to load memories: {e}")
                Clock.schedule_once(lambda dt: Notification.error("Failed to load memories"), 0)
            finally:
                loop.close()

        thread = threading.Thread(target=run_async)
        thread.daemon = True
        thread.start()

    async def _load_memories_async(self):
        """Load all memories from the memory system."""
        try:
            all_memories = []
            # Load memories of all types using a wildcard query with very low threshold
            for memory_type in MemoryType:
                memories = await self.safe_memory.safe_recall(
                    query="*", memory_types=[memory_type], limit=1000, threshold=0.0
                )
                all_memories.extend(memories)
            
            # If that doesn't work, try using the manager directly
            if not all_memories:
                try:
                    # Get stats to see if there are any memories at all
                    stats = await self.safe_memory.safe_get_stats()
                    if stats.get("total", 0) > 0:
                        # Fallback: try getting memories with a simple word that might match anything
                        for memory_type in MemoryType:
                            memories = await self.safe_memory.safe_recall(
                                query="the", memory_types=[memory_type], limit=1000, threshold=0.1
                            )
                            all_memories.extend(memories)
                except Exception as e:
                    logger.warning(f"Fallback memory loading failed: {e}")
            
            # Sort by creation date (newest first)
            all_memories.sort(key=lambda m: m.created_at, reverse=True)
            
            # Update UI on main thread
            Clock.schedule_once(lambda dt: self._update_memories_list(all_memories), 0)
            
        except Exception as e:
            logger.error(f"Failed to load memories: {e}")
            Clock.schedule_once(lambda dt: Notification.error("Failed to load memories"), 0)

    def _update_memories_list(self, memories):
        """Update the memories list in the UI."""
        self.memories = memories
        self.filtered_memories = memories.copy()
        self._refresh_list_display()
        
        # Update count
        if len(memories) == 0:
            self.count_label.text = "No memories found. Click + to add your first memory!"
        else:
            self.count_label.text = f"Showing {len(memories)} memories"
        
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
                radius=[dp(12)]
            )
            
            empty_icon = MDLabel(
                text="🧠",
                font_size="48sp",
                theme_text_color="Secondary",
                size_hint_y=None,
                height=dp(60),
                halign="center"
            )
            empty_card.add_widget(empty_icon)
            
            empty_text = MDLabel(
                text="No memories to display.\nClick the + button above to add your first memory!",
                theme_text_color="Secondary",
                font_style="Body2",
                adaptive_height=True,
                halign="center"
            )
            empty_card.add_widget(empty_text)
            
            self.memories_list.add_widget(empty_card)
        else:
            # Show memory items
            for memory in self.filtered_memories:
                item = MemoryListItem(
                    memory=memory,
                    on_edit=self._edit_memory,
                    on_delete=self._confirm_delete_memory
                )
                self.memories_list.add_widget(item)

    def _apply_filter(self, memory_type: MemoryType | None):
        """Apply memory type filter."""
        self.current_filter = memory_type
        
        # Update button colors
        for child in self.filter_layout.children:
            if hasattr(child, 'md_bg_color'):
                child.md_bg_color = (0.3, 0.3, 0.3, 1)
        
        # Highlight selected filter
        if memory_type is None:
            self.filter_layout.children[-1].md_bg_color = (0.2, 0.6, 0.8, 1)  # "All" button
            self.filtered_memories = self.memories.copy()
        else:
            # Find and highlight the correct button
            for i, child in enumerate(self.filter_layout.children[:-1]):  # Skip "All" button
                if hasattr(child, 'text') and memory_type.value.replace("_", " ").title() in child.text:
                    child.md_bg_color = (0.2, 0.6, 0.8, 1)
                    break
            
            self.filtered_memories = [m for m in self.memories if m.memory_type == memory_type]
        
        # Apply search if active
        if self.search_query:
            self._filter_by_search()
        
        self._refresh_list_display()
        self.count_label.text = f"Showing {len(self.filtered_memories)} memories"

    def _toggle_search(self, *args):
        """Toggle search bar visibility."""
        if self.search_layout.height == 0:
            # Show search
            from kivy.animation import Animation
            anim = Animation(height=dp(60), opacity=1, duration=0.3)
            anim.start(self.search_layout)
            self.search_field.focus = True
        else:
            # Hide search
            from kivy.animation import Animation
            anim = Animation(height=0, opacity=0, duration=0.3)
            anim.start(self.search_layout)
            self._clear_search()

    def _on_search_text(self, instance, text):
        """Handle search text changes."""
        self.search_query = text.strip().lower()
        if self.search_query:
            Clock.unschedule(self._filter_by_search)
            Clock.schedule_once(lambda dt: self._filter_by_search(), 0.5)
        else:
            self._clear_search()

    def _filter_by_search(self):
        """Filter memories by search query."""
        if not self.search_query:
            return
            
        base_memories = (
            [m for m in self.memories if m.memory_type == self.current_filter] 
            if self.current_filter else self.memories
        )
        
        self.filtered_memories = [
            memory for memory in base_memories
            if (
                self.search_query in memory.content.lower()
                or self.search_query in memory.memory_type.value.lower()
                or any(self.search_query in str(v).lower() for v in memory.metadata.values())
            )
        ]
        
        self._refresh_list_display()
        self.count_label.text = f"Found {len(self.filtered_memories)} matching memories"

    def _clear_search(self, *args):
        """Clear search and show filtered memories."""
        self.search_query = ""
        self.search_field.text = ""
        self._apply_filter(self.current_filter)

    def _refresh_memories(self, *args):
        """Refresh memories from database."""
        Notification.info("Refreshing memories...")
        self._load_memories(None)

    def _show_add_dialog(self, *args):
        """Show dialog to add new memory."""
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

        type_layout = MDBoxLayout(orientation="horizontal", size_hint_y=None, height=dp(40), spacing=dp(10))
        
        self.add_type_buttons = []
        for memory_type in MemoryType:
            btn = MDRaisedButton(
                text=memory_type.value.replace("_", " ").title(),
                size_hint_x=0.25,
                md_bg_color=(0.2, 0.6, 0.8, 1) if memory_type == MemoryType.LONG_TERM else (0.3, 0.3, 0.3, 1),
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
        """Create a new memory."""
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
        """Create memory asynchronously."""
        try:
            metadata = {
                "created_manually": True,
                "source": "memory_management",
                "created_at": datetime.now().isoformat()
            }
            
            if tags:
                metadata["tags"] = tags

            memory_id = await self.safe_memory.safe_remember(
                content=content,
                memory_type=self.selected_add_type,
                importance=importance,
                metadata=metadata
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
        """Edit an existing memory."""
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

        type_layout = MDBoxLayout(orientation="horizontal", size_hint_y=None, height=dp(40), spacing=dp(10))
        
        self.edit_type_buttons = []
        for memory_type in MemoryType:
            btn = MDRaisedButton(
                text=memory_type.value.replace("_", " ").title(),
                size_hint_x=0.25,
                md_bg_color=(0.2, 0.6, 0.8, 1) if memory_type == memory.memory_type else (0.3, 0.3, 0.3, 1),
                on_release=lambda x, t=memory_type: self._select_edit_type(t),
            )
            self.edit_type_buttons.append((btn, memory_type))
            type_layout.add_widget(btn)
        
        content.add_widget(type_layout)

        # Importance slider
        importance_label = MDLabel(text=f"Importance: {memory.importance:.1f}", size_hint_y=None, height=dp(30))
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
                    self._update_memory_async(memory, new_content, self.selected_edit_type, self.edit_importance_slider.value)
                )
            except Exception as e:
                logger.error(f"Failed to update memory: {e}")
                Clock.schedule_once(lambda dt: Notification.error("Failed to update memory"), 0)
            finally:
                loop.close()

        thread = threading.Thread(target=run_update)
        thread.daemon = True
        thread.start()

    async def _update_memory_async(self, memory: Memory, new_content: str, new_type: MemoryType, new_importance: float):
        """Update memory asynchronously."""
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
        """Show confirmation dialog for deleting memory."""
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
        """Delete memory asynchronously."""
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
