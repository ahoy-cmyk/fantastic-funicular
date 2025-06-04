"""Settings screen for configuring the application."""

from datetime import datetime

from kivy.metrics import dp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDFlatButton, MDRaisedButton
from kivymd.uix.dialog import MDDialog
from kivymd.uix.label import MDLabel
from kivymd.uix.list import MDList, OneLineListItem
from kivymd.uix.screen import MDScreen
from kivymd.uix.scrollview import MDScrollView

from src.gui.utils.error_handling import CrashPrevention, safe_dialog_operation
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class SettingsListItem(OneLineListItem):
    """Custom list item for settings."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class SettingsScreen(MDScreen):
    """Settings configuration screen."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.build_ui()

    def build_ui(self):
        """Build the settings interface."""
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
        title_label = MDLabel(
            text="Settings", font_style="H6", theme_text_color="Primary", pos_hint={"center_y": 0.5}
        )
        toolbar.add_widget(title_label)

        # Spacer
        toolbar.add_widget(MDBoxLayout())

        # Settings list
        scroll = MDScrollView()
        settings_list = MDList()

        # Provider settings
        settings_list.add_widget(
            SettingsListItem(text="LLM Providers", on_release=self.open_provider_settings)
        )

        # Memory management (redirect to advanced screen)
        settings_list.add_widget(
            SettingsListItem(text="Memory Management", on_release=self.open_memory_management)
        )

        # MCP settings
        settings_list.add_widget(
            SettingsListItem(text="MCP Servers", on_release=self.open_mcp_settings)
        )

        # API Keys
        settings_list.add_widget(
            SettingsListItem(text="API Keys", on_release=self.open_api_settings)
        )

        # Appearance
        settings_list.add_widget(
            SettingsListItem(text="Appearance", on_release=self.open_appearance_settings)
        )

        # Advanced
        settings_list.add_widget(
            SettingsListItem(text="Advanced Settings", on_release=self.open_advanced_settings)
        )

        # Configuration Profiles
        settings_list.add_widget(
            SettingsListItem(text="Configuration Profiles", on_release=self.open_profiles)
        )

        # Import/Export
        settings_list.add_widget(
            SettingsListItem(text="Import/Export Settings", on_release=self.open_import_export)
        )

        # About
        settings_list.add_widget(SettingsListItem(text="About", on_release=self.open_about))

        scroll.add_widget(settings_list)

        # Add to main layout
        main_layout.add_widget(toolbar)
        main_layout.add_widget(scroll)

        self.add_widget(main_layout)

    def go_back(self):
        """Navigate back to chat screen."""
        self.manager.current = "enhanced_chat"

    def open_provider_settings(self, *args):
        """Open LLM provider settings."""
        logger.info("Opening provider settings")
        self.manager.current = "provider_config"

    def open_memory_management(self, *args):
        """Open advanced memory management."""
        logger.info("Opening advanced memory management")
        self.manager.current = "advanced_memory"

    def open_mcp_settings(self, *args):
        """Open MCP server settings."""
        from kivymd.uix.boxlayout import MDBoxLayout
        from kivymd.uix.button import MDFlatButton
        from kivymd.uix.dialog import MDDialog
        from kivymd.uix.label import MDLabel

        # Create MCP configuration dialog
        content = MDBoxLayout(
            orientation="vertical", spacing=dp(10), size_hint_y=None, height=dp(200)
        )

        content.add_widget(
            MDLabel(
                text="MCP (Model Context Protocol) Settings", font_style="H6", adaptive_height=True
            )
        )

        content.add_widget(
            MDLabel(
                text="MCP allows integration with external tools and services.\nCurrently configured servers:",
                theme_text_color="Secondary",
                adaptive_height=True,
            )
        )

        content.add_widget(
            MDLabel(
                text="• No MCP servers configured\n• Click 'Configure' to add servers",
                theme_text_color="Primary",
                adaptive_height=True,
            )
        )

        mcp_dialog = MDDialog(
            title="MCP Server Configuration",
            type="custom",
            content_cls=content,
            buttons=[
                MDFlatButton(text="Cancel", on_release=lambda x: mcp_dialog.dismiss()),
                MDFlatButton(
                    text="Configure", on_release=lambda x: self._configure_mcp(mcp_dialog)
                ),
            ],
        )
        mcp_dialog.open()

    def _configure_mcp(self, dialog):
        """Configure MCP servers."""
        dialog.dismiss()
        from src.gui.utils.notifications import Notification

        Notification.info("MCP configuration would open here - feature can be added as needed")

    def open_api_settings(self, *args):
        """Open API key settings."""
        # API settings are now handled in provider configuration
        logger.info("Redirecting to provider settings for API configuration")
        self.manager.current = "provider_config"

    @safe_dialog_operation
    def open_appearance_settings(self, *args):
        """Open appearance settings."""
        from kivymd.uix.boxlayout import MDBoxLayout
        from kivymd.uix.button import MDFlatButton
        from kivymd.uix.dialog import MDDialog
        from kivymd.uix.label import MDLabel

        content = MDBoxLayout(
            orientation="vertical", spacing=dp(15), size_hint_y=None, height=dp(250)
        )

        content.add_widget(
            MDLabel(text="Appearance Settings", font_style="H6", adaptive_height=True)
        )

        # Theme switch
        theme_layout = MDBoxLayout(orientation="horizontal", spacing=dp(10), adaptive_height=True)
        theme_layout.add_widget(MDLabel(text="Dark Theme", adaptive_height=True))
        theme_switch = MDFlatButton(
            text="Dark Theme: ON",
            theme_text_color="Primary",
            on_release=lambda x: self.show_info_popup(
                "Theme Settings",
                "Theme switching will be available in a future update. Currently using optimized dark theme.",
            ),
        )
        theme_layout.add_widget(theme_switch)
        content.add_widget(theme_layout)

        content.add_widget(
            MDLabel(
                text="Current theme: Material Design 3 Dark\nThe interface uses a professional dark theme optimized for long coding sessions.",
                theme_text_color="Secondary",
                adaptive_height=True,
            )
        )

        appearance_dialog = MDDialog(
            title="Appearance Settings",
            type="custom",
            content_cls=content,
            buttons=[MDFlatButton(text="Close", on_release=lambda x: appearance_dialog.dismiss())],
        )
        appearance_dialog.open()

    def open_advanced_settings(self, *args):
        """Open advanced settings."""
        from kivymd.uix.boxlayout import MDBoxLayout
        from kivymd.uix.button import MDFlatButton
        from kivymd.uix.dialog import MDDialog
        from kivymd.uix.label import MDLabel

        content = MDBoxLayout(
            orientation="vertical", spacing=dp(10), size_hint_y=None, height=dp(200)
        )

        content.add_widget(MDLabel(text="Advanced Settings", font_style="H6", adaptive_height=True))

        settings_info = [
            "Configuration file location:",
            "   ~/.config/Neuromancer/config.json",
            "",
            "Data directory:",
            "   ~/.neuromancer/",
            "",
            "Logs directory:",
            "   ~/.neuromancer/logs/",
        ]

        for info in settings_info:
            content.add_widget(
                MDLabel(
                    text=info,
                    theme_text_color="Secondary" if info.startswith("   ") else "Primary",
                    font_style="Caption" if info.startswith("   ") else "Body2",
                    adaptive_height=True,
                )
            )

        advanced_dialog = MDDialog(
            title="Advanced Settings",
            type="custom",
            content_cls=content,
            buttons=[MDFlatButton(text="Close", on_release=lambda x: advanced_dialog.dismiss())],
        )
        advanced_dialog.open()

    def open_about(self, *args):
        """Open about dialog."""
        from kivymd.uix.boxlayout import MDBoxLayout
        from kivymd.uix.button import MDFlatButton
        from kivymd.uix.dialog import MDDialog
        from kivymd.uix.label import MDLabel

        from src import __version__

        content = MDBoxLayout(
            orientation="vertical", spacing=dp(15), size_hint_y=None, height=dp(300)
        )

        content.add_widget(
            MDLabel(text="NEUROMANCER", font_style="H4", adaptive_height=True, halign="center")
        )

        about_info = [
            f"Version: {__version__}",
            "",
            "An advanced AI assistant with:",
            "• Multi-provider LLM support",
            "• Intelligent memory management",
            "• MCP protocol integration",
            "• Enterprise-grade features",
            "",
            "Built with Python, Kivy, and KivyMD",
            "",
            "Created with Claude Code",
        ]

        for info in about_info:
            content.add_widget(
                MDLabel(
                    text=info,
                    theme_text_color=(
                        "Primary" if info and not info.startswith("•") else "Secondary"
                    ),
                    font_style="Subtitle1" if info.startswith("Version") else "Body2",
                    adaptive_height=True,
                    halign="center" if not info.startswith("•") else "left",
                )
            )

        about_dialog = MDDialog(
            title="About Neuromancer",
            type="custom",
            content_cls=content,
            buttons=[MDFlatButton(text="Close", on_release=lambda x: about_dialog.dismiss())],
        )
        about_dialog.open()

    def open_profiles(self, *args):
        """Open configuration profiles."""
        from kivymd.uix.boxlayout import MDBoxLayout
        from kivymd.uix.button import MDFlatButton
        from kivymd.uix.dialog import MDDialog
        from kivymd.uix.label import MDLabel

        content = MDBoxLayout(
            orientation="vertical", spacing=dp(10), size_hint_y=None, height=dp(200)
        )

        content.add_widget(
            MDLabel(text="Configuration Profiles", font_style="H6", adaptive_height=True)
        )

        content.add_widget(
            MDLabel(
                text="Profiles allow you to save and switch between different configurations for different use cases.",
                theme_text_color="Secondary",
                adaptive_height=True,
            )
        )

        content.add_widget(
            MDLabel(
                text="Available profiles:\n• Default (current)\n• No custom profiles created",
                theme_text_color="Primary",
                adaptive_height=True,
            )
        )

        profiles_dialog = MDDialog(
            title="Configuration Profiles",
            type="custom",
            content_cls=content,
            buttons=[MDFlatButton(text="Close", on_release=lambda x: profiles_dialog.dismiss())],
        )
        profiles_dialog.open()

    def open_import_export(self, *args):
        """Open import/export settings."""

        from kivymd.uix.boxlayout import MDBoxLayout
        from kivymd.uix.button import MDFlatButton
        from kivymd.uix.dialog import MDDialog
        from kivymd.uix.label import MDLabel

        content = MDBoxLayout(
            orientation="vertical", spacing=dp(15), size_hint_y=None, height=dp(250)
        )

        content.add_widget(
            MDLabel(text="Import/Export Settings", font_style="H6", adaptive_height=True)
        )

        content.add_widget(
            MDLabel(
                text="Export your configuration and conversations for backup or sharing.",
                theme_text_color="Secondary",
                adaptive_height=True,
            )
        )

        export_dialog = MDDialog(
            title="Import/Export",
            type="custom",
            content_cls=content,
            buttons=[
                MDFlatButton(text="Cancel", on_release=lambda x: export_dialog.dismiss()),
                MDFlatButton(
                    text="Export Config", on_release=lambda x: self._export_config(export_dialog)
                ),
                MDFlatButton(
                    text="Export All", on_release=lambda x: self._export_all(export_dialog)
                ),
            ],
        )
        export_dialog.open()

    def _export_config(self, dialog):
        """Export configuration."""
        try:
            from pathlib import Path

            from src.core.config import config_manager

            export_dir = Path.home() / "Downloads"
            export_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_file = export_dir / f"neuromancer_config_{timestamp}.json"

            # Export configuration
            config_manager.save(export_file)

            dialog.dismiss()
            from src.gui.utils.notifications import Notification

            Notification.success(f"Configuration exported to {export_file.name}")

        except Exception as e:
            logger.error(f"Failed to export config: {e}")
            from src.gui.utils.notifications import Notification

            Notification.error("Failed to export configuration")

    def _export_all(self, dialog):
        """Export everything."""
        try:
            import zipfile
            from pathlib import Path

            export_dir = Path.home() / "Downloads"
            export_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_file = export_dir / f"neuromancer_backup_{timestamp}.zip"

            with zipfile.ZipFile(zip_file, "w") as zf:
                # Add config file if it exists
                config_file = Path.home() / ".config" / "Neuromancer" / "config.json"
                if config_file.exists():
                    zf.write(config_file, "config.json")

                # Could add database, logs, etc.

            dialog.dismiss()
            from src.gui.utils.notifications import Notification

            Notification.success(f"Full backup exported to {zip_file.name}")

        except Exception as e:
            logger.error(f"Failed to export backup: {e}")
            from src.gui.utils.notifications import Notification

            Notification.error("Failed to create backup")

    @safe_dialog_operation
    def show_info_popup(self, title, message):
        """Show an information popup dialog."""
        # Use crash prevention utilities for safer dialog creation
        dialog = CrashPrevention.safe_dialog_creation(
            MDDialog,
            title=title,
            text=message,
            buttons=[
                CrashPrevention.safe_button_creation(
                    MDRaisedButton,
                    text="OK",
                    on_release=lambda x: CrashPrevention.safe_widget_method(dialog, "dismiss"),
                )
            ],
        )

        if dialog:
            CrashPrevention.safe_widget_method(dialog, "open")
        else:
            # Fallback to notification if dialog creation fails
            from src.gui.utils.notifications import Notification

            Notification.info(f"{title}: {message}")
