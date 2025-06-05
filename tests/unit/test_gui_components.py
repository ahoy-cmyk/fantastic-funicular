"""Tests for GUI component functionality."""

import os
from unittest.mock import MagicMock, patch

import pytest

# Set environment variables to prevent GUI initialization during tests
os.environ["KIVY_NO_CONSOLELOG"] = "1"
os.environ["KIVY_LOG_LEVEL"] = "error"

from src.gui.screens.mcp_management_screen import MCPManagementScreen, ServerAddDialog


@pytest.mark.unit
class TestServerAddDialog:
    """Test MCP server add dialog functionality."""

    def test_dialog_initialization(self):
        """Test dialog initialization."""
        dialog = ServerAddDialog()

        assert dialog.name_field is not None
        assert dialog.url_field is not None
        assert dialog.command_field is not None
        assert dialog.args_field is not None
        assert dialog.websocket_checkbox is not None
        assert dialog.subprocess_checkbox is not None
        assert dialog.ssl_verify_checkbox is not None

    def test_websocket_mode_selection(self):
        """Test WebSocket mode selection."""
        dialog = ServerAddDialog()

        # Simulate WebSocket checkbox activation
        dialog._on_websocket_selected(dialog.websocket_checkbox, True)

        assert dialog.url_field.disabled is False
        assert dialog.command_field.disabled is True
        assert dialog.args_field.disabled is True
        assert dialog.ssl_verify_checkbox.disabled is False

    def test_subprocess_mode_selection(self):
        """Test subprocess mode selection."""
        dialog = ServerAddDialog()

        # Simulate subprocess checkbox activation
        dialog._on_subprocess_selected(dialog.subprocess_checkbox, True)

        assert dialog.url_field.disabled is True
        assert dialog.command_field.disabled is False
        assert dialog.args_field.disabled is False
        assert dialog.ssl_verify_checkbox.disabled is True


@pytest.mark.unit
class TestMCPManagementScreen:
    """Test MCP management screen functionality."""

    @pytest.fixture
    def mcp_screen(self):
        """Create MCP management screen for testing."""
        return MCPManagementScreen()

    def test_screen_initialization(self, mcp_screen):
        """Test screen initialization."""
        assert mcp_screen.name == "mcp_management"
        assert mcp_screen.chat_manager is None
        assert mcp_screen.servers_list is not None
        assert mcp_screen.tools_list is not None

    def test_set_chat_manager(self, mcp_screen):
        """Test setting chat manager."""
        mock_chat_manager = MagicMock()
        mcp_screen.set_chat_manager(mock_chat_manager)

        assert mcp_screen.chat_manager is mock_chat_manager

    def test_tool_icon_mapping(self, mcp_screen):
        """Test tool type to icon mapping."""
        assert mcp_screen._get_tool_icon("function") == "cog"
        assert mcp_screen._get_tool_icon("retrieval") == "database-search"
        assert mcp_screen._get_tool_icon("generation") == "creation-outline"
        assert mcp_screen._get_tool_icon("action") == "play-circle-outline"
        assert mcp_screen._get_tool_icon("unknown") == "tools"

    @patch("threading.Thread")
    @patch("asyncio.new_event_loop")
    def test_refresh_servers(self, mock_loop, mock_thread, mcp_screen):
        """Test server refresh functionality."""
        mock_chat_manager = MagicMock()
        mcp_screen.set_chat_manager(mock_chat_manager)

        mcp_screen.refresh_servers()

        # Should create a thread for async operation
        mock_thread.assert_called_once()
        mock_loop.assert_called_once()

    def test_update_servers_ui_empty(self, mcp_screen):
        """Test UI update with no servers."""
        mcp_screen._update_servers_ui([])

        # Should clear the list and add "no servers" message
        assert mcp_screen.servers_list.children  # Should have at least the "no servers" message

    def test_update_servers_ui_with_data(self, mcp_screen):
        """Test UI update with server data."""
        server_data = [
            {
                "name": "test_server",
                "url": "ws://localhost:8080",
                "type": "websocket",
                "connected": True,
                "tool_count": 5,
            },
            {
                "name": "subprocess_server",
                "url": "npx -y mcp-remote https://example.com",
                "type": "subprocess",
                "connected": False,
                "tool_count": 2,
            },
        ]

        mcp_screen._update_servers_ui(server_data)

        # Should have server items in the list
        assert len(mcp_screen.servers_list.children) == len(server_data)

    def test_update_tools_ui_empty(self, mcp_screen):
        """Test tools UI update with no tools."""
        mcp_screen._update_tools_ui([])

        # Should show "no tools available" message
        assert mcp_screen.tools_list.children

    def test_update_tools_ui_with_data(self, mcp_screen, sample_mcp_tools):
        """Test tools UI update with tool data."""
        mcp_screen._update_tools_ui(sample_mcp_tools)

        # Should have tool items in the list
        assert len(mcp_screen.tools_list.children) == len(sample_mcp_tools)

    @patch("threading.Thread")
    @patch("asyncio.new_event_loop")
    def test_add_websocket_server(self, mock_loop, mock_thread, mcp_screen):
        """Test adding WebSocket server."""
        mock_chat_manager = MagicMock()
        mcp_screen.set_chat_manager(mock_chat_manager)

        mcp_screen.add_websocket_server("test_server", "ws://localhost:8080", True)

        # Should create thread for async operation
        mock_thread.assert_called_once()

    @patch("threading.Thread")
    @patch("asyncio.new_event_loop")
    def test_add_subprocess_server(self, mock_loop, mock_thread, mcp_screen):
        """Test adding subprocess server."""
        mock_chat_manager = MagicMock()
        mcp_screen.set_chat_manager(mock_chat_manager)

        mcp_screen.add_subprocess_server("subprocess_server", "npx", ["-y", "mcp-remote"])

        # Should create thread for async operation
        mock_thread.assert_called_once()

    def test_show_add_server_dialog(self, mcp_screen):
        """Test showing add server dialog."""
        with patch("src.gui.screens.mcp_management_screen.MDDialog") as mock_dialog:
            mcp_screen.show_add_server_dialog()

            # Should create and open dialog
            mock_dialog.assert_called_once()
            mock_dialog.return_value.open.assert_called_once()

    def test_add_server_from_dialog_websocket(self, mcp_screen):
        """Test adding server from dialog - WebSocket mode."""
        mock_content = MagicMock()
        mock_content.name_field.text = "test_server"
        mock_content.websocket_checkbox.active = True
        mock_content.subprocess_checkbox.active = False
        mock_content.url_field.text = "ws://localhost:8080"
        mock_content.ssl_verify_checkbox.active = True

        mcp_screen.server_dialog = MagicMock()

        with patch.object(mcp_screen, "add_websocket_server") as mock_add_ws:
            mcp_screen.add_server_from_dialog(mock_content)

            mock_add_ws.assert_called_once_with("test_server", "ws://localhost:8080", True)
            mcp_screen.server_dialog.dismiss.assert_called_once()

    def test_add_server_from_dialog_subprocess(self, mcp_screen):
        """Test adding server from dialog - subprocess mode."""
        mock_content = MagicMock()
        mock_content.name_field.text = "subprocess_server"
        mock_content.websocket_checkbox.active = False
        mock_content.subprocess_checkbox.active = True
        mock_content.command_field.text = "npx"
        mock_content.args_field.text = "-y mcp-remote https://example.com"

        mcp_screen.server_dialog = MagicMock()

        with patch.object(mcp_screen, "add_subprocess_server") as mock_add_sub:
            mcp_screen.add_server_from_dialog(mock_content)

            mock_add_sub.assert_called_once_with(
                "subprocess_server", "npx", ["-y", "mcp-remote", "https://example.com"]
            )
            mcp_screen.server_dialog.dismiss.assert_called_once()

    def test_show_tool_details(self, mcp_screen, sample_mcp_tools):
        """Test showing tool details dialog."""
        tool = sample_mcp_tools[0]

        with patch("src.gui.screens.mcp_management_screen.MDDialog") as mock_dialog:
            mcp_screen.show_tool_details(tool)

            # Should create and open dialog
            mock_dialog.assert_called_once()
            mock_dialog.return_value.open.assert_called_once()

    def test_go_back(self, mcp_screen):
        """Test navigation back to settings."""
        mock_manager = MagicMock()
        mcp_screen.manager = mock_manager

        mcp_screen.go_back()

        assert mock_manager.current == "settings"
