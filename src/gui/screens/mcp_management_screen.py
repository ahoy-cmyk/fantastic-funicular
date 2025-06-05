"""MCP (Model Context Protocol) management screen."""

from typing import Any

from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDFlatButton, MDRaisedButton
from kivymd.uix.card import MDCard
from kivymd.uix.dialog import MDDialog
from kivymd.uix.label import MDLabel
from kivymd.uix.list import (
    IconLeftWidget,
    MDList,
    ThreeLineIconListItem,
)
from kivymd.uix.screen import MDScreen
from kivymd.uix.scrollview import MDScrollView
from kivymd.uix.textfield import MDTextField
from kivymd.uix.toolbar import MDTopAppBar

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ServerAddDialog(MDBoxLayout):
    """Dialog content for adding an MCP server."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "vertical"
        self.spacing = "12dp"
        self.size_hint_y = None
        self.height = "380dp"  # Increased height for better spacing

        # Server name field
        self.name_field = MDTextField(
            hint_text="Server Name",
            helper_text="Unique name for this server",
            helper_text_mode="on_focus",
            required=True,
        )
        self.add_widget(self.name_field)

        # Server type selector
        from kivymd.uix.boxlayout import MDBoxLayout as TypeLayout
        from kivymd.uix.selectioncontrol import MDCheckbox

        type_layout = TypeLayout(
            orientation="horizontal",
            spacing="20dp",
            size_hint_y=None,
            height="40dp",
        )

        # WebSocket mode checkbox
        ws_layout = TypeLayout(orientation="horizontal", spacing="5dp", size_hint_x=0.5)
        self.websocket_checkbox = MDCheckbox(
            group="server_type",
            active=True,
            size_hint=(None, None),
            size=("20dp", "20dp"),
        )
        ws_layout.add_widget(self.websocket_checkbox)
        ws_layout.add_widget(MDLabel(text="WebSocket", size_hint_y=None, height="40dp"))
        type_layout.add_widget(ws_layout)

        # Subprocess mode checkbox
        sub_layout = TypeLayout(orientation="horizontal", spacing="5dp", size_hint_x=0.5)
        self.subprocess_checkbox = MDCheckbox(
            group="server_type",
            active=False,
            size_hint=(None, None),
            size=("20dp", "20dp"),
        )
        sub_layout.add_widget(self.subprocess_checkbox)
        sub_layout.add_widget(MDLabel(text="Subprocess", size_hint_y=None, height="40dp"))
        type_layout.add_widget(sub_layout)

        self.add_widget(type_layout)

        # Server URL field (for WebSocket)
        self.url_field = MDTextField(
            hint_text="Server URL",
            helper_text="WebSocket URL (ws://, wss://, http://, https://)",
            helper_text_mode="on_focus",
        )
        self.add_widget(self.url_field)

        # Command field (for subprocess)
        self.command_field = MDTextField(
            hint_text="Command",
            helper_text="Executable command (e.g., 'npx')",
            helper_text_mode="on_focus",
            disabled=True,
        )
        self.add_widget(self.command_field)

        # Args field (for subprocess)
        self.args_field = MDTextField(
            hint_text="Arguments",
            helper_text="Space-separated arguments (e.g., '-y mcp-remote https://...')",
            helper_text_mode="on_focus",
            multiline=True,
            disabled=True,
        )
        self.add_widget(self.args_field)

        # Bind checkbox events
        self.websocket_checkbox.bind(active=self._on_websocket_selected)
        self.subprocess_checkbox.bind(active=self._on_subprocess_selected)

        # SSL verification toggle (for HTTPS/WSS)
        CheckBoxLayout = MDBoxLayout

        ssl_layout = CheckBoxLayout(
            orientation="horizontal",
            spacing="10dp",
            size_hint_y=None,
            height="30dp",
        )

        self.ssl_verify_checkbox = MDCheckbox(
            active=True,
            size_hint=(None, None),
            size=("20dp", "20dp"),
        )
        ssl_layout.add_widget(self.ssl_verify_checkbox)

        ssl_layout.add_widget(
            MDLabel(
                text="Verify SSL certificates (uncheck for self-signed)",
                size_hint_y=None,
                height="30dp",
                theme_text_color="Hint",
            )
        )

        self.add_widget(ssl_layout)

    def _on_websocket_selected(self, checkbox, value):
        """Handle WebSocket mode selection."""
        if value:
            self.url_field.disabled = False
            self.command_field.disabled = True
            self.args_field.disabled = True
            self.ssl_verify_checkbox.disabled = False

    def _on_subprocess_selected(self, checkbox, value):
        """Handle subprocess mode selection."""
        if value:
            self.url_field.disabled = True
            self.command_field.disabled = False
            self.args_field.disabled = False
            self.ssl_verify_checkbox.disabled = True


class ToolDetailDialog(MDBoxLayout):
    """Dialog content for showing tool details."""

    def __init__(self, tool, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "vertical"
        self.spacing = "12dp"
        self.size_hint_y = None
        self.height = "300dp"

        # Tool name
        self.add_widget(
            MDLabel(
                text=f"[b]Tool:[/b] {tool.name}",
                markup=True,
                size_hint_y=None,
                height="30dp",
            )
        )

        # Description
        self.add_widget(
            MDLabel(
                text=f"[b]Description:[/b] {tool.description}",
                markup=True,
                size_hint_y=None,
                height="60dp",
            )
        )

        # Server
        self.add_widget(
            MDLabel(
                text=f"[b]Server:[/b] {tool.server}",
                markup=True,
                size_hint_y=None,
                height="30dp",
            )
        )

        # Tool type
        self.add_widget(
            MDLabel(
                text=f"[b]Type:[/b] {tool.tool_type.value}",
                markup=True,
                size_hint_y=None,
                height="30dp",
            )
        )

        # Parameters
        if tool.parameters:
            params_text = "[b]Parameters:[/b]\n"
            for name, details in tool.parameters.items():
                params_text += f"  • {name}: {details}\n"

            scroll = MDScrollView(size_hint_y=None, height="150dp")
            params_label = MDLabel(
                text=params_text,
                markup=True,
                size_hint_y=None,
            )
            params_label.bind(texture_size=params_label.setter("size"))
            scroll.add_widget(params_label)
            self.add_widget(scroll)


class MCPManagementScreen(MDScreen):
    """Screen for managing MCP servers and tools."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "mcp_management"
        self.chat_manager = None
        self.server_dialog = None
        self.tool_dialog = None
        self._build_ui()

    def _build_ui(self):
        """Build the UI components."""
        layout = MDBoxLayout(orientation="vertical")

        # Top app bar
        self.toolbar = MDTopAppBar(
            title="MCP Management",
            left_action_items=[["arrow-left", lambda x: self.go_back()]],
            right_action_items=[
                ["plus", lambda x: self.show_add_server_dialog()],
                ["refresh", lambda x: self.refresh_servers()],
            ],
            elevation=2,
        )
        layout.add_widget(self.toolbar)

        # Main content area
        content_layout = MDBoxLayout(
            orientation="horizontal",
            spacing="10dp",
            padding="10dp",
        )

        # Servers panel (left)
        servers_card = MDCard(
            orientation="vertical",
            size_hint_x=0.4,
            elevation=2,
            padding="10dp",
        )

        servers_card.add_widget(
            MDLabel(
                text="[b]MCP Servers[/b]",
                markup=True,
                size_hint_y=None,
                height="40dp",
            )
        )

        self.servers_scroll = MDScrollView()
        self.servers_list = MDList()
        self.servers_scroll.add_widget(self.servers_list)
        servers_card.add_widget(self.servers_scroll)

        content_layout.add_widget(servers_card)

        # Tools panel (right)
        tools_card = MDCard(
            orientation="vertical",
            size_hint_x=0.6,
            elevation=2,
            padding="10dp",
        )

        tools_card.add_widget(
            MDLabel(
                text="[b]Available Tools[/b]",
                markup=True,
                size_hint_y=None,
                height="40dp",
            )
        )

        self.tools_scroll = MDScrollView()
        self.tools_list = MDList()
        self.tools_scroll.add_widget(self.tools_list)
        tools_card.add_widget(self.tools_scroll)

        content_layout.add_widget(tools_card)

        layout.add_widget(content_layout)
        self.add_widget(layout)

    def set_chat_manager(self, chat_manager):
        """Set the chat manager instance."""
        self.chat_manager = chat_manager

    def on_enter(self):
        """Called when entering the screen."""
        super().on_enter()
        if self.chat_manager:
            self.refresh_servers()

    def refresh_servers(self):
        """Refresh the server and tool lists."""
        if not self.chat_manager:
            return

        # Use a simpler approach with Clock.schedule_once
        from kivy.clock import Clock
        Clock.schedule_once(lambda dt: self._start_refresh_task(), 0)

    def _start_refresh_task(self):
        """Start the refresh task safely."""
        import asyncio
        import threading

        def run_refresh():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._refresh_servers_async())
                loop.close()
            except Exception as e:
                logger.error(f"Error in refresh task: {e}")

        thread = threading.Thread(target=run_refresh, daemon=True)
        thread.start()

    async def _refresh_servers_async(self):
        """Async refresh of servers and tools."""
        try:
            logger.info("Starting MCP screen refresh...")
            
            # Get servers
            servers = await self.chat_manager.list_mcp_servers()
            logger.info(f"Found {len(servers)} MCP servers: {[s.get('name', 'unnamed') for s in servers]}")

            # Update UI on main thread
            from kivy.clock import Clock

            Clock.schedule_once(lambda dt: self._update_servers_ui(servers), 0)

            # Get tools
            tools = await self.chat_manager.list_mcp_tools()
            logger.info(f"Found {len(tools)} MCP tools")
            Clock.schedule_once(lambda dt: self._update_tools_ui(tools), 0)

        except Exception as e:
            logger.error(f"Error refreshing MCP data: {e}")
            import traceback
            logger.error(f"Refresh error traceback: {traceback.format_exc()}")

    def _update_servers_ui(self, servers):
        """Update the servers list UI."""
        self.servers_list.clear_widgets()

        if not servers:
            self.servers_list.add_widget(
                MDLabel(
                    text="No MCP servers configured",
                    halign="center",
                    theme_text_color="Hint",
                )
            )
            return

        for server in servers:
            # Format server info better based on type
            if server.get("type") == "subprocess":
                # For subprocess servers, show command instead of URL
                server_info = f"Command: {server['url']}"
                if len(server_info) > 60:
                    server_info = server_info[:57] + "..."
            else:
                # For WebSocket servers, show URL
                server_info = server["url"]

            # Create server item with better formatting
            item = ThreeLineIconListItem(
                text=f"[b]{server['name']}[/b]",
                secondary_text=server_info,
                tertiary_text=f"Type: {server.get('type', 'websocket').title()} | Tools: {server['tool_count']} | Status: {'Connected' if server['connected'] else 'Disconnected'}",
            )

            # Status icon with better colors
            icon = IconLeftWidget(
                icon="check-circle" if server["connected"] else "alert-circle",
                theme_icon_color="Custom",
                icon_color=(
                    [0.2, 0.8, 0.2, 1] if server["connected"] else [0.8, 0.2, 0.2, 1]
                ),  # Green or Red
            )
            item.add_widget(icon)

            # Skip the problematic right widget for now - we can add it back later if needed
            # actions = IconRightWidget(icon="dots-vertical")
            # actions.bind(on_release=lambda x, s=server: self.show_server_actions(s))
            # item.add_widget(actions)

            self.servers_list.add_widget(item)

    def _update_tools_ui(self, tools):
        """Update the tools list UI."""
        self.tools_list.clear_widgets()

        if not tools:
            self.tools_list.add_widget(
                MDLabel(
                    text="No tools available",
                    halign="center",
                    theme_text_color="Hint",
                )
            )
            return

        for tool in tools:
            # Create tool item with better formatting
            server_prefix = tool.name.split(":")[0] if ":" in tool.name else "unknown"
            tool_name = tool.name.split(":")[1] if ":" in tool.name else tool.name

            # Truncate long descriptions
            description = tool.description
            if len(description) > 80:
                description = description[:77] + "..."

            item = ThreeLineIconListItem(
                text=f"[b]{tool_name}[/b]",
                secondary_text=description,
                tertiary_text=f"Server: [color=3498db]{server_prefix}[/color] | Type: {tool.tool_type.value.title()}",
                on_release=lambda x, t=tool: self.show_tool_details(t),
            )

            # Tool icon with color coding
            icon = IconLeftWidget(
                icon=self._get_tool_icon(tool.tool_type.value),
                theme_icon_color="Custom",
                icon_color=[0.2, 0.6, 0.9, 1],  # Nice blue color
            )
            item.add_widget(icon)

            self.tools_list.add_widget(item)

    def _get_tool_icon(self, tool_type: str) -> str:
        """Get icon for tool type."""
        icons = {
            "function": "cog",
            "retrieval": "database-search",
            "generation": "creation-outline",
            "action": "play-circle-outline",
        }
        return icons.get(tool_type, "tools")

    def show_add_server_dialog(self):
        """Show dialog to add a new server."""
        content = ServerAddDialog()

        self.server_dialog = MDDialog(
            title="Add MCP Server",
            type="custom",
            content_cls=content,
            buttons=[
                MDFlatButton(
                    text="CANCEL",
                    on_release=lambda x: self.server_dialog.dismiss(),
                ),
                MDRaisedButton(
                    text="ADD",
                    on_release=lambda x: self.add_server_from_dialog(content),
                ),
            ],
        )
        self.server_dialog.open()

    def add_server_from_dialog(self, content):
        """Add a new MCP server from dialog content."""
        name = content.name_field.text.strip()

        if not name:
            return

        if self.server_dialog:
            self.server_dialog.dismiss()

        # Determine server type
        if content.websocket_checkbox.active:
            # WebSocket server
            url = content.url_field.text.strip()
            if not url:
                return

            ssl_verify = content.ssl_verify_checkbox.active
            self.add_websocket_server(name, url, ssl_verify)

        elif content.subprocess_checkbox.active:
            # Subprocess server
            command = content.command_field.text.strip()
            args_text = content.args_field.text.strip()

            if not command or not args_text:
                return

            args = args_text.split()
            self.add_subprocess_server(name, command, args)

    def add_websocket_server(self, name: str, url: str, ssl_verify: bool = True):
        """Add a WebSocket MCP server."""
        # Use threading for async operation
        import asyncio
        import threading

        # Create SSL config based on UI settings
        from urllib.parse import urlparse

        parsed_url = urlparse(url)
        ssl_config = None

        if parsed_url.scheme in ["wss", "https"]:
            ssl_config = {"verify": ssl_verify, "allow_self_signed": not ssl_verify}

        async def add_server_task():
            await self._add_websocket_server_async(name, url, ssl_config)

        def run_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(add_server_task())
            loop.close()

        thread = threading.Thread(target=run_async)
        thread.daemon = True
        thread.start()

    def add_subprocess_server(self, name: str, command: str, args: list[str]):
        """Add a subprocess MCP server."""
        import asyncio
        import threading

        async def add_server_task():
            await self._add_subprocess_server_async(name, command, args)

        def run_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(add_server_task())
            loop.close()

        thread = threading.Thread(target=run_async)
        thread.daemon = True
        thread.start()

    async def _add_websocket_server_async(self, name: str, url: str, ssl_config: dict = None):
        """Async add WebSocket server."""
        try:
            success = await self.chat_manager.add_mcp_server(
                name, server_url=url, ssl_config=ssl_config
            )

            if success:
                logger.info(f"Added MCP WebSocket server: {name}")
                # Refresh the lists
                self.refresh_servers()
            else:
                logger.error(f"Failed to add MCP WebSocket server: {name}")

        except Exception as e:
            logger.error(f"Error adding MCP WebSocket server: {e}")

    async def _add_subprocess_server_async(self, name: str, command: str, args: list[str]):
        """Async add subprocess server."""
        try:
            success = await self.chat_manager.add_mcp_server(name, command=command, args=args)

            if success:
                logger.info(f"Added MCP subprocess server: {name}")
                # Refresh the lists
                self.refresh_servers()
            else:
                logger.error(f"Failed to add MCP subprocess server: {name}")

        except Exception as e:
            logger.error(f"Error adding MCP subprocess server: {e}")

    def show_server_actions(self, server: dict[str, Any]):
        """Show server action menu."""
        # TODO: Implement server actions (remove, reconnect, etc.)
        logger.info(f"Server actions for: {server['name']}")

    def show_tool_details(self, tool):
        """Show detailed information about a tool."""
        content = ToolDetailDialog(tool)

        self.tool_dialog = MDDialog(
            title="Tool Details",
            type="custom",
            content_cls=content,
            buttons=[
                MDRaisedButton(
                    text="CLOSE",
                    on_release=lambda x: self.tool_dialog.dismiss(),
                ),
            ],
        )
        self.tool_dialog.open()

    def go_back(self):
        """Go back to previous screen."""
        self.manager.current = "settings"
