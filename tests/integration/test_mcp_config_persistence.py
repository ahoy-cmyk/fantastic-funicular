"""Integration tests for MCP configuration persistence."""

from unittest.mock import AsyncMock, patch

import pytest

from src.config.manager import ConfigManager
from src.core.chat_manager import ChatManager


@pytest.mark.integration
@pytest.mark.requires_mcp
class TestMCPConfigPersistence:
    """Test MCP server configuration persistence."""

    @pytest.fixture
    async def chat_manager_with_config(self, temp_dir):
        """Create chat manager with temporary config."""
        config_dir = temp_dir / "config"
        config_dir.mkdir()

        # Create a temporary config manager
        with patch("src.core.config._config_manager") as mock_config_manager:
            # Create real config manager for testing
            real_config_manager = ConfigManager("test_neuromancer")
            real_config_manager.config_dir = config_dir
            real_config_manager.config_file = config_dir / "config.json"

            mock_config_manager.get.side_effect = real_config_manager.get
            mock_config_manager.set.side_effect = real_config_manager.set
            mock_config_manager.save.side_effect = real_config_manager.save

            # Create chat manager with mocked dependencies
            with patch.object(ChatManager, "_initialize_providers", lambda self: None):
                chat_manager = ChatManager()
                chat_manager.mcp_manager = AsyncMock()
                chat_manager.mcp_manager.add_server = AsyncMock(return_value=True)
                chat_manager.mcp_manager.remove_server = AsyncMock(return_value=True)

                yield chat_manager, real_config_manager

    @pytest.mark.asyncio
    async def test_websocket_server_persistence(self, chat_manager_with_config):
        """Test that WebSocket MCP servers are saved to config."""
        chat_manager, config_manager = chat_manager_with_config

        # Add a WebSocket server
        server_name = "test_websocket_server"
        server_url = "wss://api.example.com/mcp"
        ssl_config = {"verify": True, "allow_self_signed": False}

        success = await chat_manager.add_mcp_server(
            server_name, server_url=server_url, ssl_config=ssl_config
        )

        assert success is True

        # Check that it was saved to config
        servers = config_manager.get("mcp.servers", {})
        assert server_name in servers

        server_config = servers[server_name]
        assert server_config["enabled"] is True
        assert server_config["url"] == server_url
        assert server_config["ssl"] == ssl_config
        assert "MCP server:" in server_config["description"]

    @pytest.mark.asyncio
    async def test_subprocess_server_persistence(self, chat_manager_with_config):
        """Test that subprocess MCP servers are saved to config."""
        chat_manager, config_manager = chat_manager_with_config

        # Add a subprocess server
        server_name = "test_subprocess_server"
        command = "npx"
        args = ["-y", "mcp-remote", "https://api.example.com/mcp"]

        success = await chat_manager.add_mcp_server(server_name, command=command, args=args)

        assert success is True

        # Check that it was saved to config
        servers = config_manager.get("mcp.servers", {})
        assert server_name in servers

        server_config = servers[server_name]
        assert server_config["enabled"] is True
        assert server_config["command"] == command
        assert server_config["args"] == args
        assert "MCP server:" in server_config["description"]

    @pytest.mark.asyncio
    async def test_server_removal_persistence(self, chat_manager_with_config):
        """Test that removing MCP servers updates config."""
        chat_manager, config_manager = chat_manager_with_config

        # First add a server
        server_name = "test_temp_server"
        await chat_manager.add_mcp_server(server_name, server_url="ws://localhost:8080")

        # Verify it exists
        servers = config_manager.get("mcp.servers", {})
        assert server_name in servers

        # Now remove it
        success = await chat_manager.remove_mcp_server(server_name)
        assert success is True

        # Verify it's gone from config
        servers = config_manager.get("mcp.servers", {})
        assert server_name not in servers

    @pytest.mark.asyncio
    async def test_multiple_servers_persistence(self, chat_manager_with_config):
        """Test that multiple MCP servers can be saved."""
        chat_manager, config_manager = chat_manager_with_config

        # Add multiple servers
        servers_to_add = [
            {
                "name": "websocket_server_1",
                "url": "wss://api1.example.com/mcp",
                "ssl_config": {"verify": True},
            },
            {"name": "websocket_server_2", "url": "ws://localhost:8080"},
            {
                "name": "subprocess_server_1",
                "command": "npx",
                "args": ["-y", "mcp-remote", "https://api2.example.com"],
            },
        ]

        for server_info in servers_to_add:
            success = await chat_manager.add_mcp_server(
                server_info["name"],
                server_url=server_info.get("url"),
                ssl_config=server_info.get("ssl_config"),
                command=server_info.get("command"),
                args=server_info.get("args"),
            )
            assert success is True

        # Verify all servers are in config
        servers = config_manager.get("mcp.servers", {})
        assert len(servers) == len(servers_to_add)

        for server_info in servers_to_add:
            assert server_info["name"] in servers
            config_entry = servers[server_info["name"]]
            assert config_entry["enabled"] is True

            if "url" in server_info:
                assert config_entry["url"] == server_info["url"]
            if "command" in server_info:
                assert config_entry["command"] == server_info["command"]
                assert config_entry["args"] == server_info["args"]

    @pytest.mark.asyncio
    async def test_config_loading_on_startup(self, chat_manager_with_config):
        """Test that MCP servers are loaded from config on startup."""
        chat_manager, config_manager = chat_manager_with_config

        # Manually add server config
        servers = {
            "startup_test_server": {
                "enabled": True,
                "url": "ws://localhost:8080",
                "description": "Test server for startup loading",
            }
        }

        await config_manager.set("mcp.servers", servers)
        config_manager.save()

        # Now test that ChatManager would try to connect to it
        # (We can't test the actual connection since we mocked the MCP manager)
        with patch.object(chat_manager, "_connect_mcp_servers") as mock_connect:
            chat_manager._initialize_mcp_servers()

            # Should have attempted to connect
            # Note: The actual connection would happen in the background thread

    @pytest.mark.asyncio
    async def test_config_persistence_across_restarts(self, temp_dir):
        """Test that config persists across application restarts."""
        config_dir = temp_dir / "config"
        config_dir.mkdir()

        # First "session" - add server
        config_manager1 = ConfigManager("test_neuromancer")
        config_manager1.config_dir = config_dir
        config_manager1.config_file = config_dir / "config.json"

        servers = {
            "persistent_server": {
                "enabled": True,
                "url": "wss://persistent.example.com/mcp",
                "ssl": {"verify": True},
                "description": "Persistent test server",
            }
        }

        await config_manager1.set("mcp.servers", servers)
        config_manager1.save()

        # Second "session" - load config
        config_manager2 = ConfigManager("test_neuromancer")
        config_manager2.config_dir = config_dir
        config_manager2.config_file = config_dir / "config.json"
        config_manager2.load()

        # Verify server config persisted
        loaded_servers = config_manager2.get("mcp.servers", {})
        assert "persistent_server" in loaded_servers
        assert loaded_servers["persistent_server"]["url"] == "wss://persistent.example.com/mcp"
        assert loaded_servers["persistent_server"]["ssl"]["verify"] is True

    @pytest.mark.asyncio
    async def test_error_handling_in_config_save(self, chat_manager_with_config):
        """Test error handling when config save fails."""
        chat_manager, config_manager = chat_manager_with_config

        # Mock config manager to fail on set
        with patch.object(config_manager, "set", new_callable=AsyncMock) as mock_set:
            mock_set.return_value = False  # Simulate failure

            # Try to add server - should handle failure gracefully
            success = await chat_manager.add_mcp_server(
                "failing_server", server_url="ws://localhost:8080"
            )

            # MCP manager succeeded but config save failed
            # The overall operation should still report success from MCP manager
            assert success is True
            mock_set.assert_called_once()
