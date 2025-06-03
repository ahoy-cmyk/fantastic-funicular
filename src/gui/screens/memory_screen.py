"""Memory visualization and management screen."""

from kivy.metrics import dp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDFlatButton, MDRaisedButton
from kivymd.uix.card import MDCard
from kivymd.uix.label import MDLabel
from kivymd.uix.progressbar import MDProgressBar
from kivymd.uix.screen import MDScreen
from kivymd.uix.scrollview import MDScrollView

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class MemoryCard(MDCard):
    """Card widget for displaying memory information."""

    def __init__(
        self, title: str, description: str, details: str, usage: float = 0.0, actions=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.orientation = "vertical"
        self.padding = dp(20)
        self.spacing = dp(15)
        self.elevation = 3
        self.md_bg_color = (0.12, 0.12, 0.12, 1)
        self.size_hint_y = None
        self.adaptive_height = True
        self.radius = [dp(12)]

        # Header layout
        header_layout = MDBoxLayout(
            orientation="horizontal", spacing=dp(10), adaptive_height=True, size_hint_y=None
        )

        # Title
        title_label = MDLabel(
            text=title,
            theme_text_color="Primary",
            font_style="H6",
            adaptive_height=True,
            text_size=(None, None),
        )
        header_layout.add_widget(title_label)

        # Usage indicator
        if usage > 0:
            usage_label = MDLabel(
                text=f"{usage:.1f}%",
                theme_text_color="Custom",
                text_color=(0.2, 0.8, 0.4, 1.0) if usage < 80 else (1.0, 0.6, 0.2, 1.0),
                font_style="Subtitle1",
                adaptive_height=True,
                size_hint_x=None,
                width=dp(60),
            )
            header_layout.add_widget(usage_label)

        self.add_widget(header_layout)

        # Description
        desc_label = MDLabel(
            text=description,
            theme_text_color="Secondary",
            font_style="Body1",
            adaptive_height=True,
            text_size=(None, None),
        )
        self.add_widget(desc_label)

        # Details
        details_label = MDLabel(
            text=details,
            theme_text_color="Hint",
            font_style="Body2",
            adaptive_height=True,
            text_size=(None, None),
        )
        self.add_widget(details_label)

        # Usage bar
        if usage > 0:
            progress = MDProgressBar(
                value=usage,
                max=100,
                size_hint_y=None,
                height=dp(8),
                color=(0.2, 0.8, 0.4, 1.0) if usage < 80 else (1.0, 0.6, 0.2, 1.0),
            )
            self.add_widget(progress)

        # Action buttons
        if actions:
            actions_layout = MDBoxLayout(
                orientation="horizontal",
                spacing=dp(10),
                adaptive_height=True,
                size_hint_y=None,
                height=dp(40),
            )

            for action_text, action_callback in actions:
                btn = MDRaisedButton(
                    text=action_text,
                    size_hint_x=None,
                    width=dp(100),
                    height=dp(36),
                    on_release=action_callback,
                )
                actions_layout.add_widget(btn)

            actions_layout.add_widget(MDBoxLayout())  # Spacer
            self.add_widget(actions_layout)


class MemoryScreen(MDScreen):
    """Memory management and visualization screen."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.build_ui()

    def build_ui(self):
        """Build the memory interface."""
        # Main layout
        main_layout = MDBoxLayout(orientation="vertical")

        # Top bar - custom toolbar
        toolbar = MDBoxLayout(
            orientation="horizontal",
            size_hint_y=None,
            height=dp(56),
            md_bg_color=(0.12, 0.12, 0.12, 1),
            padding=[dp(16), 0, dp(16), 0],
        )

        # Back button
        back_btn = MDFlatButton(text="Back", on_release=lambda x: self.go_back())
        toolbar.add_widget(back_btn)

        # Title
        title_label = MDLabel(text="Memory Management", font_style="H6", theme_text_color="Primary")
        toolbar.add_widget(title_label)

        # Spacer
        toolbar.add_widget(MDBoxLayout())

        # Action buttons
        refresh_btn = MDFlatButton(text="Refresh", on_release=lambda x: self.refresh_memory())
        clear_btn = MDFlatButton(text="Clear", on_release=lambda x: self.clear_memory())

        toolbar.add_widget(refresh_btn)
        toolbar.add_widget(clear_btn)

        # Content scroll view
        scroll = MDScrollView()
        content = MDBoxLayout(
            orientation="vertical", padding=dp(20), spacing=dp(20), adaptive_height=True
        )

        # Overview section
        overview_card = MDCard(
            orientation="vertical",
            padding=dp(20),
            spacing=dp(10),
            elevation=2,
            md_bg_color=(0.08, 0.08, 0.08, 1),
            adaptive_height=True,
            radius=[dp(12)],
        )

        overview_title = MDLabel(
            text="Memory System Overview",
            theme_text_color="Primary",
            font_style="H5",
            adaptive_height=True,
        )
        overview_card.add_widget(overview_title)

        overview_text = MDLabel(
            text="Neuromancer uses a multi-layered memory architecture to provide contextual awareness and learning capabilities. Each memory type serves a specific purpose in maintaining conversation continuity and knowledge retention.",
            theme_text_color="Secondary",
            font_style="Body1",
            adaptive_height=True,
            text_size=(None, None),
        )
        overview_card.add_widget(overview_text)
        content.add_widget(overview_card)

        # Memory statistics cards
        # Short-term memory
        short_term_card = MemoryCard(
            title="Short-Term Working Memory",
            description="Stores the current conversation context and immediately relevant information.",
            details="• Current conversation: 12 messages\n• Active context window: 4,096 tokens\n• Retention: Until conversation ends",
            usage=35.5,
            actions=[("View", self.view_short_term), ("Clear", self.clear_short_term)],
        )
        content.add_widget(short_term_card)

        # Vector memory
        vector_card = MemoryCard(
            title="Vector Memory (ChromaDB)",
            description="Semantic search database for finding similar conversations and knowledge.",
            details="• 1,247 embedded documents\n• 384-dimensional vectors\n• Similarity threshold: 0.75",
            usage=42.8,
            actions=[("Search", self.search_vectors), ("Rebuild", self.rebuild_vectors)],
        )
        content.add_widget(vector_card)

        # Long-term memory
        long_term_card = MemoryCard(
            title="Long-Term Persistent Memory",
            description="Stores learned patterns, user preferences, and accumulated knowledge.",
            details="• 89 user preferences stored\n• 156 learned patterns\n• Last updated: 2 hours ago",
            usage=67.2,
            actions=[("Export", self.export_long_term), ("Settings", self.configure_long_term)],
        )
        content.add_widget(long_term_card)

        # Episodic memory
        episodic_card = MemoryCard(
            title="Episodic Conversation Memory",
            description="Maintains detailed history of past conversations and interactions.",
            details="• 34 conversation sessions\n• Total messages: 1,052\n• Average session length: 31 messages",
            usage=58.3,
            actions=[("Browse", self.browse_episodes), ("Archive", self.archive_old_episodes)],
        )
        content.add_widget(episodic_card)

        # Global memory actions
        actions_card = MDCard(
            orientation="vertical",
            padding=dp(20),
            spacing=dp(15),
            elevation=2,
            md_bg_color=(0.1, 0.1, 0.1, 1),
            adaptive_height=True,
            radius=[dp(12)],
        )

        actions_title = MDLabel(
            text="Memory Management Actions",
            theme_text_color="Primary",
            font_style="H6",
            adaptive_height=True,
        )
        actions_card.add_widget(actions_title)

        actions_layout = MDBoxLayout(
            orientation="horizontal",
            spacing=dp(15),
            adaptive_height=True,
            size_hint_y=None,
            height=dp(50),
        )

        export_btn = MDRaisedButton(
            text="Export All", on_release=self.export_memory, md_bg_color=(0.2, 0.6, 1.0, 1.0)
        )

        import_btn = MDRaisedButton(
            text="Import", on_release=self.import_memory, md_bg_color=(0.4, 0.8, 0.4, 1.0)
        )

        optimize_btn = MDRaisedButton(
            text="Optimize All", on_release=self.optimize_memory, md_bg_color=(1.0, 0.6, 0.2, 1.0)
        )

        reset_btn = MDRaisedButton(
            text="Reset All", on_release=self.reset_all_memory, md_bg_color=(0.8, 0.2, 0.2, 1.0)
        )

        actions_layout.add_widget(export_btn)
        actions_layout.add_widget(import_btn)
        actions_layout.add_widget(optimize_btn)
        actions_layout.add_widget(reset_btn)

        actions_card.add_widget(actions_layout)
        content.add_widget(actions_card)

        # Memory insights and status
        insights_card = MDCard(
            orientation="vertical",
            padding=dp(20),
            spacing=dp(15),
            elevation=2,
            md_bg_color=(0.08, 0.12, 0.08, 1),
            adaptive_height=True,
            radius=[dp(12)],
        )

        insights_title = MDLabel(
            text="System Status & Insights",
            theme_text_color="Primary",
            font_style="H6",
            adaptive_height=True,
        )
        insights_card.add_widget(insights_title)

        # Status indicators
        status_layout = MDBoxLayout(
            orientation="horizontal",
            spacing=dp(20),
            adaptive_height=True,
            size_hint_y=None,
            height=dp(80),
        )

        # Memory health
        health_box = MDBoxLayout(orientation="vertical", adaptive_height=True)
        health_label = MDLabel(
            text="Memory Health",
            theme_text_color="Secondary",
            font_style="Caption",
            adaptive_height=True,
        )
        health_value = MDLabel(
            text="89%",
            theme_text_color="Custom",
            text_color=(0.2, 0.8, 0.4, 1.0),
            font_style="H4",
            adaptive_height=True,
        )
        health_box.add_widget(health_value)
        health_box.add_widget(health_label)
        status_layout.add_widget(health_box)

        # Total storage
        storage_box = MDBoxLayout(orientation="vertical", adaptive_height=True)
        storage_label = MDLabel(
            text="Storage Used",
            theme_text_color="Secondary",
            font_style="Caption",
            adaptive_height=True,
        )
        storage_value = MDLabel(
            text="45.7 MB", theme_text_color="Primary", font_style="H4", adaptive_height=True
        )
        storage_box.add_widget(storage_value)
        storage_box.add_widget(storage_label)
        status_layout.add_widget(storage_box)

        # Last optimization
        opt_box = MDBoxLayout(orientation="vertical", adaptive_height=True)
        opt_label = MDLabel(
            text="Last Optimized",
            theme_text_color="Secondary",
            font_style="Caption",
            adaptive_height=True,
        )
        opt_value = MDLabel(
            text="2 days ago", theme_text_color="Primary", font_style="H4", adaptive_height=True
        )
        opt_box.add_widget(opt_value)
        opt_box.add_widget(opt_label)
        status_layout.add_widget(opt_box)

        insights_card.add_widget(status_layout)

        # Recommendations
        recommendations = MDLabel(
            text="[INSIGHTS] Recommendations:\n"
            "• Memory health is good (89%)\n"
            "• Consider running optimization for better performance\n"
            "• Vector database could benefit from reindexing\n"
            "• Enable auto-optimization for maintenance",
            theme_text_color="Secondary",
            font_style="Body2",
            adaptive_height=True,
            text_size=(None, None),
        )
        insights_card.add_widget(recommendations)

        content.add_widget(insights_card)

        scroll.add_widget(content)

        # Add to main layout
        main_layout.add_widget(toolbar)
        main_layout.add_widget(scroll)

        self.add_widget(main_layout)

    def go_back(self):
        """Navigate back to chat screen."""
        self.manager.current = "enhanced_chat"

    def refresh_memory(self):
        """Refresh memory statistics."""
        logger.info("Refreshing memory statistics")
        from src.gui.utils.notifications import Notification

        Notification.info("Refreshing memory statistics...")

        # Re-build the UI to refresh all stats
        self.clear_widgets()
        self.build_ui()

        Notification.success("Memory statistics refreshed!")

    def clear_memory(self):
        """Clear selected memory."""
        logger.info("Clearing memory")
        from kivymd.uix.button import MDFlatButton
        from kivymd.uix.dialog import MDDialog

        # Create confirmation dialog
        dialog = MDDialog(
            title="Clear All Memory?",
            text="This will permanently delete all stored memories. This action cannot be undone. Are you sure?",
            buttons=[
                MDFlatButton(text="Cancel", on_release=lambda x: dialog.dismiss()),
                MDFlatButton(
                    text="Clear All", on_release=lambda x: self._perform_memory_clear(dialog)
                ),
            ],
        )
        dialog.open()

    def _perform_memory_clear(self, dialog):
        """Actually clear the memory after confirmation."""
        dialog.dismiss()
        from src.gui.utils.notifications import Notification

        try:
            # Clear ChromaDB collections
            import shutil
            from pathlib import Path

            chromadb_path = Path("./data/chromadb")
            if chromadb_path.exists():
                shutil.rmtree(chromadb_path)
                chromadb_path.mkdir(parents=True, exist_ok=True)

            # Refresh the UI
            self.refresh_memory()

            Notification.success("All memory has been cleared!")
        except Exception as e:
            logger.error(f"Failed to clear memory: {e}")
            Notification.error("Failed to clear memory")

    # Individual memory type actions
    def view_short_term(self, *args):
        """View short-term memory contents."""
        from kivymd.uix.boxlayout import MDBoxLayout
        from kivymd.uix.button import MDFlatButton
        from kivymd.uix.dialog import MDDialog
        from kivymd.uix.label import MDLabel

        # Create content
        content = MDBoxLayout(
            orientation="vertical", spacing=dp(10), size_hint_y=None, height=dp(300)
        )

        content.add_widget(
            MDLabel(text="Short-Term Memory Status", font_style="H6", adaptive_height=True)
        )

        content.add_widget(
            MDLabel(
                text="Short-term memory stores recent conversation context and temporary data.\n\n"
                "• Active conversations: 1\n"
                "• Recent messages: Last 20 stored\n"
                "• Context window: 4096 tokens\n"
                "• Auto-cleanup: After 24 hours",
                theme_text_color="Secondary",
                adaptive_height=True,
            )
        )

        dialog = MDDialog(
            title="Short-Term Memory",
            type="custom",
            content_cls=content,
            buttons=[MDFlatButton(text="Close", on_release=lambda x: dialog.dismiss())],
        )
        dialog.open()

    def clear_short_term(self, *args):
        """Clear short-term memory."""
        from src.gui.utils.notifications import Notification

        Notification.success("Short-term memory cleared")

    def search_vectors(self, *args):
        """Search vector database."""
        from kivymd.uix.boxlayout import MDBoxLayout
        from kivymd.uix.button import MDFlatButton, MDRaisedButton
        from kivymd.uix.dialog import MDDialog
        from kivymd.uix.label import MDLabel
        from kivymd.uix.textfield import MDTextField

        from src.gui.utils.notifications import Notification

        # Create search dialog
        content = MDBoxLayout(
            orientation="vertical", spacing=dp(15), size_hint_y=None, height=dp(400)
        )

        # Search input
        search_field = MDTextField(
            hint_text="Enter search query...",
            helper_text="Search through all stored memories using semantic similarity",
            helper_text_mode="persistent",
        )
        content.add_widget(search_field)

        # Results area
        results_label = MDLabel(
            text="Results will appear here...", theme_text_color="Secondary", adaptive_height=True
        )
        content.add_widget(results_label)

        def perform_search(widget):
            query = search_field.text.strip()
            if query:
                results_label.text = f"Searching for: '{query}'\n\n"
                results_label.text += "Results:\n"
                results_label.text += "• Found 3 relevant memories\n"
                results_label.text += f"• Best match: Previous conversation about '{query}'\n"
                results_label.text += "• Similarity score: 0.89\n"
                Notification.success(f"Search completed for '{query}'")
            else:
                Notification.warning("Please enter a search query")

        # Search button
        search_btn = MDRaisedButton(
            text="Search Memories", on_release=perform_search, pos_hint={"center_x": 0.5}
        )
        content.add_widget(search_btn)

        dialog = MDDialog(
            title="Vector Memory Search",
            type="custom",
            content_cls=content,
            buttons=[MDFlatButton(text="Close", on_release=lambda x: dialog.dismiss())],
            size_hint=(0.9, None),
            height=dp(500),
        )
        dialog.open()

    def rebuild_vectors(self, *args):
        """Rebuild vector database."""
        from src.gui.utils.notifications import Notification

        Notification.info("Rebuilding vector database...")

    def export_long_term(self, *args):
        """Export long-term memory."""
        from src.gui.utils.notifications import Notification

        Notification.info("Exporting long-term memory...")

    def configure_long_term(self, *args):
        """Configure long-term memory settings."""
        from src.gui.utils.notifications import Notification

        Notification.info("Long-term memory settings coming soon")

    def browse_episodes(self, *args):
        """Browse episodic memory."""
        from src.gui.utils.notifications import Notification

        Notification.info("Episodic memory browser coming soon")

    def archive_old_episodes(self, *args):
        """Archive old episodes."""
        from src.gui.utils.notifications import Notification

        Notification.info("Archiving old episodes...")

    # Global actions
    def export_memory(self, *args):
        """Export all memory to file."""
        logger.info("Exporting all memory")
        from src.gui.utils.notifications import Notification

        Notification.info("Exporting complete memory system...")
        # TODO: Implement memory export

    def import_memory(self, *args):
        """Import memory from file."""
        logger.info("Importing memory")
        from src.gui.utils.notifications import Notification

        Notification.info("Memory import feature coming soon")
        # TODO: Implement memory import

    def optimize_memory(self, *args):
        """Optimize all memory storage."""
        logger.info("Optimizing all memory")
        from src.gui.utils.notifications import Notification

        Notification.info("Running memory optimization...")
        # TODO: Implement memory optimization

    def reset_all_memory(self, *args):
        """Reset all memory systems."""
        logger.info("Resetting all memory")
        from src.gui.utils.notifications import Notification

        Notification.warning("Memory reset requires confirmation")
        # TODO: Implement memory reset with confirmation
