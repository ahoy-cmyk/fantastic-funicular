"""Advanced settings screen with comprehensive configuration options."""

import asyncio

from kivy.clock import Clock
from kivy.metrics import dp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDIconButton, MDRaisedButton
from kivymd.uix.card import MDCard
from kivymd.uix.dialog import MDDialog
from kivymd.uix.filemanager import MDFileManager
from kivymd.uix.label import MDLabel
from kivymd.uix.list import MDList
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.screen import MDScreen
from kivymd.uix.scrollview import MDScrollView
from kivymd.uix.selectioncontrol import MDSwitch
from kivymd.uix.slider import MDSlider
from kivymd.uix.snackbar import Snackbar
from kivymd.uix.tab import MDTabs, MDTabsBase
from kivymd.uix.textfield import MDTextField
from kivymd.uix.toolbar import MDTopAppBar

from src.config import ConfigSection
from src.core.config import config, config_manager
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ConfigOptionCard(MDCard):
    """Card widget for a configuration option."""

    def __init__(
        self, title: str, description: str, config_path: str, widget_type: str = "text", **kwargs
    ):
        super().__init__(**kwargs)
        self.orientation = "vertical"
        self.padding = dp(15)
        self.spacing = dp(10)
        self.elevation = 1
        self.md_bg_color = (0.1, 0.1, 0.1, 1)
        self.size_hint_y = None
        self.adaptive_height = True

        self.config_path = config_path
        self.widget_type = widget_type

        # Header
        header = MDBoxLayout(adaptive_height=True)

        # Title and description
        text_box = MDBoxLayout(orientation="vertical", adaptive_height=True)

        title_label = MDLabel(
            text=title, theme_text_color="Primary", font_style="Subtitle1", adaptive_height=True
        )
        text_box.add_widget(title_label)

        desc_label = MDLabel(
            text=description,
            theme_text_color="Secondary",
            font_style="Caption",
            adaptive_height=True,
        )
        text_box.add_widget(desc_label)

        header.add_widget(text_box)

        # Add appropriate widget based on type
        self.value_widget = self._create_value_widget()
        if self.value_widget:
            header.add_widget(MDBoxLayout())  # Spacer
            header.add_widget(self.value_widget)

        self.add_widget(header)

        # Load current value
        Clock.schedule_once(lambda dt: self._load_value(), 0.1)

    def _create_value_widget(self):
        """Create the appropriate widget for the value type."""
        current_value = config_manager.get(self.config_path)

        if self.widget_type == "text":
            widget = MDTextField(
                text=str(current_value or ""),
                hint_text="Enter value",
                on_text_validate=self._on_value_change,
            )
        elif self.widget_type == "number":
            widget = MDTextField(
                text=str(current_value or ""),
                hint_text="Enter number",
                input_filter="float",
                on_text_validate=self._on_value_change,
            )
        elif self.widget_type == "bool":
            widget = MDSwitch(active=bool(current_value), on_active=self._on_switch_change)
        elif self.widget_type == "slider":
            # For sliders, we need min/max from schema
            widget = MDSlider(
                value=float(current_value or 0), min=0, max=1, on_touch_up=self._on_slider_change
            )
        else:
            widget = None

        return widget

    def _load_value(self):
        """Load current configuration value."""
        value = config_manager.get(self.config_path)

        if self.widget_type in ["text", "number"] and hasattr(self.value_widget, "text"):
            self.value_widget.text = str(value or "")
        elif self.widget_type == "bool" and hasattr(self.value_widget, "active"):
            self.value_widget.active = bool(value)
        elif self.widget_type == "slider" and hasattr(self.value_widget, "value"):
            self.value_widget.value = float(value or 0)

    def _on_value_change(self, widget):
        """Handle value changes."""
        value = widget.text

        if self.widget_type == "number":
            try:
                value = float(value) if "." in value else int(value)
            except ValueError:
                Snackbar(text="Invalid number format").open()
                return

        asyncio.create_task(self._update_config(value))

    def _on_switch_change(self, widget, value):
        """Handle switch changes."""
        asyncio.create_task(self._update_config(value))

    def _on_slider_change(self, widget, touch):
        """Handle slider changes."""
        if widget.collide_point(*touch.pos):
            asyncio.create_task(self._update_config(widget.value))

    async def _update_config(self, value):
        """Update configuration value."""
        try:
            success = await config_manager.set(self.config_path, value)
            if success:
                Snackbar(text=f"Updated {self.config_path}").open()
            else:
                Snackbar(text="Failed to update configuration").open()
        except Exception as e:
            logger.error(f"Error updating config: {e}")
            Snackbar(text=f"Error: {str(e)}").open()


class ConfigSectionTab(MDBoxLayout, MDTabsBase):
    """Tab content for a configuration section."""

    def __init__(self, section: ConfigSection, **kwargs):
        super().__init__(**kwargs)
        self.section = section
        self.orientation = "vertical"
        self.build_section()

    def build_section(self):
        """Build the section UI based on configuration schema."""
        scroll = MDScrollView()
        content = MDList(spacing=dp(10), padding=dp(10))

        # Get configuration options for this section
        options = self._get_section_options()

        for option in options:
            card = ConfigOptionCard(**option)
            content.add_widget(card)

        scroll.add_widget(content)
        self.add_widget(scroll)

    def _get_section_options(self):
        """Get configuration options for the current section."""
        # Define options for each section
        section_configs = {
            ConfigSection.GENERAL: [
                {
                    "title": "Log Level",
                    "description": "Set application logging verbosity",
                    "config_path": "general.log_level",
                    "widget_type": "dropdown",
                },
                {
                    "title": "Auto Save",
                    "description": "Automatically save conversations",
                    "config_path": "general.auto_save",
                    "widget_type": "bool",
                },
                {
                    "title": "Auto Save Interval",
                    "description": "Seconds between auto-saves",
                    "config_path": "general.auto_save_interval",
                    "widget_type": "number",
                },
                {
                    "title": "Check Updates",
                    "description": "Check for updates on startup",
                    "config_path": "general.check_updates",
                    "widget_type": "bool",
                },
                {
                    "title": "Language",
                    "description": "UI language code (e.g., en, es, fr)",
                    "config_path": "general.language",
                    "widget_type": "text",
                },
            ],
            ConfigSection.PROVIDERS: [
                {
                    "title": "Default Provider",
                    "description": "Default LLM provider to use",
                    "config_path": "providers.default_provider",
                    "widget_type": "dropdown",
                },
                {
                    "title": "Temperature",
                    "description": "Response creativity (0.0 - 2.0)",
                    "config_path": "providers.temperature",
                    "widget_type": "slider",
                },
                {
                    "title": "Ollama Host",
                    "description": "Ollama server URL",
                    "config_path": "providers.ollama_host",
                    "widget_type": "text",
                },
                {
                    "title": "OpenAI API Key",
                    "description": "Your OpenAI API key",
                    "config_path": "providers.openai_api_key",
                    "widget_type": "text",
                },
                {
                    "title": "Daily Cost Limit",
                    "description": "Maximum daily spending in USD",
                    "config_path": "providers.daily_cost_limit",
                    "widget_type": "number",
                },
            ],
            ConfigSection.MEMORY: [
                {
                    "title": "Memory Enabled",
                    "description": "Enable the memory system",
                    "config_path": "memory.enabled",
                    "widget_type": "bool",
                },
                {
                    "title": "Memory Strategy",
                    "description": "Memory management strategy",
                    "config_path": "memory.strategy",
                    "widget_type": "dropdown",
                },
                {
                    "title": "Short-term Duration",
                    "description": "Hours to keep short-term memories",
                    "config_path": "memory.short_term_duration_hours",
                    "widget_type": "number",
                },
                {
                    "title": "Importance Threshold",
                    "description": "Minimum importance for long-term storage",
                    "config_path": "memory.importance_threshold",
                    "widget_type": "slider",
                },
                {
                    "title": "Max Memories",
                    "description": "Maximum total memories to store",
                    "config_path": "memory.max_memories",
                    "widget_type": "number",
                },
            ],
            ConfigSection.UI: [
                {
                    "title": "Theme Mode",
                    "description": "UI theme (Light, Dark, Auto)",
                    "config_path": "ui.theme_mode",
                    "widget_type": "dropdown",
                },
                {
                    "title": "Font Size",
                    "description": "Base font size in points",
                    "config_path": "ui.font_size",
                    "widget_type": "number",
                },
                {
                    "title": "Enable Animations",
                    "description": "Enable UI animations",
                    "config_path": "ui.enable_animations",
                    "widget_type": "bool",
                },
                {
                    "title": "Show Timestamps",
                    "description": "Show message timestamps",
                    "config_path": "ui.show_timestamps",
                    "widget_type": "bool",
                },
                {
                    "title": "Show Token Count",
                    "description": "Display token usage",
                    "config_path": "ui.show_token_count",
                    "widget_type": "bool",
                },
            ],
            ConfigSection.PERFORMANCE: [
                {
                    "title": "Enable Cache",
                    "description": "Cache LLM responses",
                    "config_path": "performance.enable_cache",
                    "widget_type": "bool",
                },
                {
                    "title": "Cache Size",
                    "description": "Cache size in MB",
                    "config_path": "performance.cache_size_mb",
                    "widget_type": "number",
                },
                {
                    "title": "Concurrent Requests",
                    "description": "Maximum parallel LLM requests",
                    "config_path": "performance.max_concurrent_requests",
                    "widget_type": "number",
                },
                {
                    "title": "Enable Profiling",
                    "description": "Enable performance profiling",
                    "config_path": "performance.enable_profiling",
                    "widget_type": "bool",
                },
            ],
            ConfigSection.PRIVACY: [
                {
                    "title": "Store Conversations",
                    "description": "Save conversation history",
                    "config_path": "privacy.store_conversations",
                    "widget_type": "bool",
                },
                {
                    "title": "Encrypt Storage",
                    "description": "Encrypt local data storage",
                    "config_path": "privacy.encrypt_storage",
                    "widget_type": "bool",
                },
                {
                    "title": "Disable Telemetry",
                    "description": "Disable all usage tracking",
                    "config_path": "privacy.disable_telemetry",
                    "widget_type": "bool",
                },
                {
                    "title": "Local Only Mode",
                    "description": "Disable all network features",
                    "config_path": "privacy.local_only_mode",
                    "widget_type": "bool",
                },
            ],
            ConfigSection.EXPERIMENTAL: [
                {
                    "title": "Enable Experimental",
                    "description": "Enable experimental features",
                    "config_path": "experimental.enable_experimental",
                    "widget_type": "bool",
                },
                {
                    "title": "Voice Input",
                    "description": "Enable voice input support",
                    "config_path": "experimental.voice_input",
                    "widget_type": "bool",
                },
                {
                    "title": "Multi-Agent Mode",
                    "description": "Enable multi-agent conversations",
                    "config_path": "experimental.multi_agent",
                    "widget_type": "bool",
                },
                {
                    "title": "Continuous Learning",
                    "description": "Enable continuous learning from interactions",
                    "config_path": "experimental.continuous_learning",
                    "widget_type": "bool",
                },
            ],
        }

        return section_configs.get(self.section, [])


class AdvancedSettingsScreen(MDScreen):
    """Advanced settings screen with tabs for each configuration section."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.file_manager = None
        self.build_ui()

    def build_ui(self):
        """Build the advanced settings interface."""
        # Main layout
        main_layout = MDBoxLayout(orientation="vertical")

        # Top bar
        toolbar = MDTopAppBar(
            title="Advanced Settings",
            left_action_items=[["arrow-left", lambda x: self.go_back()]],
            right_action_items=[
                ["content-save", lambda x: self.save_config()],
                ["import", lambda x: self.import_config()],
                ["export", lambda x: self.export_config()],
                ["refresh", lambda x: self.reload_config()],
            ],
            elevation=0,
        )

        # Profile selector
        profile_layout = MDBoxLayout(
            size_hint_y=None, height=dp(60), padding=dp(10), spacing=dp(10)
        )

        profile_button = MDRaisedButton(
            text=f"Profile: {config.profile}", on_release=self.show_profile_menu
        )

        save_profile_button = MDIconButton(
            icon="content-save-outline", on_release=self.save_profile
        )

        profile_layout.add_widget(profile_button)
        profile_layout.add_widget(save_profile_button)
        profile_layout.add_widget(MDBoxLayout())  # Spacer

        # Tabs for each section
        self.tabs = MDTabs()

        for section in ConfigSection:
            tab = ConfigSectionTab(section)
            tab.title = section.value.title()
            self.tabs.add_widget(tab)

        # Add to main layout
        main_layout.add_widget(toolbar)
        main_layout.add_widget(profile_layout)
        main_layout.add_widget(self.tabs)

        self.add_widget(main_layout)

    def go_back(self):
        """Navigate back to settings screen."""
        self.manager.current = "settings"

    def save_config(self):
        """Save current configuration."""
        config_manager.save()
        Snackbar(text="Configuration saved").open()

    def reload_config(self):
        """Reload configuration from disk."""
        asyncio.create_task(config_manager.reload())
        Snackbar(text="Configuration reloaded").open()

        # Refresh UI
        self.clear_widgets()
        self.build_ui()

    def show_profile_menu(self, button):
        """Show profile selection menu."""
        profiles = config_manager.list_profiles()

        menu_items = [
            {"text": profile, "on_release": lambda x=profile: self.load_profile(x)}
            for profile in profiles
        ]

        self.profile_menu = MDDropdownMenu(caller=button, items=menu_items, width_mult=4)
        self.profile_menu.open()

    def load_profile(self, profile_name: str):
        """Load a configuration profile."""
        if config_manager.load_profile(profile_name):
            Snackbar(text=f"Loaded profile: {profile_name}").open()
            self.reload_config()
        else:
            Snackbar(text=f"Failed to load profile: {profile_name}").open()

        self.profile_menu.dismiss()

    def save_profile(self, *args):
        """Save current configuration as a profile."""
        # TODO: Show dialog to enter profile name
        dialog = MDDialog(
            title="Save Profile",
            type="custom",
            content_cls=MDTextField(hint_text="Profile name", text=""),
            buttons=[
                MDRaisedButton(text="CANCEL", on_release=lambda x: dialog.dismiss()),
                MDRaisedButton(
                    text="SAVE",
                    on_release=lambda x: self._save_profile_with_name(
                        dialog.content_cls.text, dialog
                    ),
                ),
            ],
        )
        dialog.open()

    def _save_profile_with_name(self, name: str, dialog):
        """Save profile with given name."""
        if name:
            config_manager.save_profile(name)
            Snackbar(text=f"Saved profile: {name}").open()
        dialog.dismiss()

    def import_config(self):
        """Import configuration from file."""
        if not self.file_manager:
            self.file_manager = MDFileManager(
                exit_manager=self._exit_file_manager,
                select_path=self._import_config_file,
                ext=[".json", ".yaml", ".toml"],
            )
        self.file_manager.show("/")

    def export_config(self):
        """Export configuration to file."""
        # TODO: Show file save dialog
        from pathlib import Path

        export_path = Path.home() / "neuromancer_config.json"
        config_manager.export_config(export_path)
        Snackbar(text=f"Exported to: {export_path}").open()

    def _import_config_file(self, path):
        """Import selected configuration file."""
        from pathlib import Path

        if config_manager.import_config(Path(path)):
            Snackbar(text="Configuration imported successfully").open()
            self.reload_config()
        else:
            Snackbar(text="Failed to import configuration").open()
        self._exit_file_manager()

    def _exit_file_manager(self, *args):
        """Close file manager."""
        self.file_manager.close()
