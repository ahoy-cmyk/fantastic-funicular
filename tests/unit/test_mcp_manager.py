"""Tests for MCP manager and client functionality."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.mcp import MCPResponse, MCPTool, ToolType
from src.mcp.client import MCPClient
from src.mcp.manager import MCPManager
from src.mcp.subprocess_client import MCPSubprocessClient


@pytest.mark.unit
class TestMCPManager:
    """Test MCP manager functionality."""

    @pytest.fixture
    def mcp_manager(self):
        """Create MCP manager for testing."""
        return MCPManager()

    @pytest.mark.asyncio
    async def test_add_websocket_server(self, mcp_manager):
        """Test adding a WebSocket MCP server."""
        with patch("src.mcp.manager.MCPClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock(return_value=True)
            mock_client.list_tools = AsyncMock(return_value=[])
            mock_client_class.return_value = mock_client

            result = await mcp_manager.add_server("test_server", server_url="ws://localhost:8080")

            assert result is True
            assert "test_server" in mcp_manager.servers
            mock_client.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_subprocess_server(self, mcp_manager):
        """Test adding a subprocess MCP server."""
        with patch("src.mcp.manager.MCPSubprocessClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock(return_value=True)
            mock_client.list_tools = AsyncMock(return_value=[])
            mock_client_class.return_value = mock_client

            result = await mcp_manager.add_server(
                "test_subprocess", command="npx", args=["-y", "mcp-remote", "https://example.com"]
            )

            assert result is True
            assert "test_subprocess" in mcp_manager.servers

    @pytest.mark.asyncio
    async def test_add_server_failure(self, mcp_manager):
        """Test handling server connection failure."""
        with patch("src.mcp.manager.MCPClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.connect = AsyncMock(return_value=False)
            mock_client_class.return_value = mock_client

            result = await mcp_manager.add_server(
                "failing_server", server_url="ws://localhost:8080"
            )

            assert result is False
            assert "failing_server" not in mcp_manager.servers

    @pytest.mark.asyncio
    async def test_remove_server(self, mcp_manager):
        """Test removing an MCP server."""
        # Add a mock server first
        mock_server = AsyncMock()
        mock_server.disconnect = AsyncMock(return_value=True)
        mcp_manager.servers["test_server"] = mock_server
        mcp_manager.tools_cache["test_server:tool1"] = MagicMock()

        result = await mcp_manager.remove_server("test_server")

        assert result is True
        assert "test_server" not in mcp_manager.servers
        assert "test_server:tool1" not in mcp_manager.tools_cache
        mock_server.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_servers(self, mcp_manager):
        """Test listing all servers."""
        # Add mock servers
        mock_server1 = AsyncMock()
        mock_server1.health_check = AsyncMock(return_value=True)
        mock_server1.list_tools = AsyncMock(return_value=[MagicMock(), MagicMock()])
        mock_server1.server_url = "ws://localhost:8080"

        mock_server2 = AsyncMock()
        mock_server2.health_check = AsyncMock(return_value=False)
        mock_server2.list_tools = AsyncMock(return_value=[])
        mock_server2.command = "npx"
        mock_server2.args = ["-y", "mcp-remote", "url"]

        mcp_manager.servers["ws_server"] = mock_server1
        mcp_manager.servers["subprocess_server"] = mock_server2

        servers = await mcp_manager.list_servers()

        assert len(servers) == 2
        assert servers[0]["name"] == "ws_server"
        assert servers[0]["connected"] is True
        assert servers[0]["tool_count"] == 2
        assert servers[0]["type"] == "websocket"

        assert servers[1]["name"] == "subprocess_server"
        assert servers[1]["connected"] is False
        assert servers[1]["type"] == "subprocess"

    @pytest.mark.asyncio
    async def test_execute_tool(self, mcp_manager):
        """Test executing an MCP tool."""
        # Setup mock server with tool
        mock_server = AsyncMock()
        mock_response = MCPResponse(success=True, result="Tool executed successfully", error=None)
        mock_server.execute_tool = AsyncMock(return_value=mock_response)

        mcp_manager.servers["test_server"] = mock_server

        result = await mcp_manager.execute_tool("test_server:test_tool", {"param1": "value1"})

        assert result.success is True
        assert result.result == "Tool executed successfully"
        assert result.metadata["server"] == "test_server"
        mock_server.execute_tool.assert_called_once_with("test_tool", {"param1": "value1"})

    @pytest.mark.asyncio
    async def test_execute_tool_invalid_format(self, mcp_manager):
        """Test executing tool with invalid name format."""
        result = await mcp_manager.execute_tool("invalid_tool_name", {})

        assert result.success is False
        assert "Tool name must be in format 'server:tool'" in result.error

    @pytest.mark.asyncio
    async def test_health_check_all(self, mcp_manager):
        """Test health checking all servers."""
        mock_server1 = AsyncMock()
        mock_server1.health_check = AsyncMock(return_value=True)

        mock_server2 = AsyncMock()
        mock_server2.health_check = AsyncMock(return_value=False)

        mcp_manager.servers["server1"] = mock_server1
        mcp_manager.servers["server2"] = mock_server2

        health_status = await mcp_manager.health_check_all()

        assert health_status["server1"] is True
        assert health_status["server2"] is False

    @pytest.mark.asyncio
    async def test_reconnect_failed(self, mcp_manager):
        """Test reconnecting failed servers."""
        mock_server1 = AsyncMock()
        mock_server1.health_check = AsyncMock(return_value=True)
        mock_server1.connect = AsyncMock(return_value=True)

        mock_server2 = AsyncMock()
        mock_server2.health_check = AsyncMock(return_value=False)
        mock_server2.connect = AsyncMock(return_value=True)

        mcp_manager.servers["healthy_server"] = mock_server1
        mcp_manager.servers["failed_server"] = mock_server2

        results = await mcp_manager.reconnect_failed()

        assert results["healthy_server"] is True
        assert results["failed_server"] is True
        mock_server1.connect.assert_not_called()  # Healthy server shouldn't reconnect
        mock_server2.connect.assert_called_once()  # Failed server should reconnect


@pytest.mark.unit
class TestMCPClient:
    """Test MCP WebSocket client functionality."""

    @pytest.fixture
    def mcp_client(self):
        """Create MCP client for testing."""
        return MCPClient("ws://localhost:8080", name="test_client")

    @pytest.mark.asyncio
    async def test_url_conversion(self, mcp_client):
        """Test HTTP to WebSocket URL conversion."""
        # Test HTTPS to WSS conversion
        https_client = MCPClient("https://example.com/mcp", name="test")
        assert hasattr(https_client, "server_url")

        # Test HTTP to WS conversion
        http_client = MCPClient("http://example.com/mcp", name="test")
        assert hasattr(http_client, "server_url")

    @pytest.mark.asyncio
    async def test_connect_success(self, mcp_client):
        """Test successful connection."""
        with patch("websockets.connect") as mock_connect:
            mock_websocket = AsyncMock()
            mock_connect.return_value = mock_websocket

            # Mock handshake response
            mock_websocket.recv = AsyncMock(
                return_value=json.dumps(
                    {
                        "type": "handshake_response",
                        "success": True,
                        "server_name": "Test Server",
                        "version": "1.0",
                    }
                )
            )

            with patch.object(mcp_client, "_load_tools", new_callable=AsyncMock):
                result = await mcp_client.connect()

            assert result is True
            assert mcp_client._connected is True

    @pytest.mark.asyncio
    async def test_connect_failure(self, mcp_client):
        """Test connection failure."""
        with patch("websockets.connect") as mock_connect:
            mock_connect.side_effect = Exception("Connection failed")

            result = await mcp_client.connect()

            assert result is False
            assert mcp_client._connected is False

    @pytest.mark.asyncio
    async def test_execute_tool(self, mcp_client):
        """Test tool execution."""
        # Setup connected client
        mcp_client._connected = True
        mcp_client.websocket = AsyncMock()

        # Add a mock tool
        mock_tool = MCPTool(
            name="test_tool",
            description="Test tool",
            parameters={},
            tool_type=ToolType.FUNCTION,
            server="test_server",
        )
        mcp_client.tools["test_tool"] = mock_tool

        # Mock tool response
        mcp_client.websocket.recv = AsyncMock(
            return_value=json.dumps(
                {"type": "tool_response", "success": True, "result": "Tool executed", "error": None}
            )
        )

        response = await mcp_client.execute_tool("test_tool", {"param": "value"})

        assert response.success is True
        assert response.result == "Tool executed"

    @pytest.mark.asyncio
    async def test_health_check(self, mcp_client):
        """Test health check."""
        mcp_client._connected = True
        mcp_client.websocket = AsyncMock()

        # Mock pong response
        mcp_client.websocket.recv = AsyncMock(return_value=json.dumps({"type": "pong"}))

        result = await mcp_client.health_check()
        assert result is True

        # Test when not connected
        mcp_client._connected = False
        result = await mcp_client.health_check()
        assert result is False


@pytest.mark.unit
class TestMCPSubprocessClient:
    """Test MCP subprocess client functionality."""

    @pytest.fixture
    def subprocess_client(self):
        """Create MCP subprocess client for testing."""
        return MCPSubprocessClient("npx", ["-y", "mcp-remote", "https://example.com"])

    @pytest.mark.asyncio
    async def test_connect_success(self, subprocess_client):
        """Test successful subprocess connection."""
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            mock_process.returncode = None
            mock_process.stderr = AsyncMock()
            mock_process.stderr.read = AsyncMock(return_value=b"")
            mock_exec.return_value = mock_process

            # Mock handshake response
            async def mock_receive():
                return {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {"serverInfo": {"name": "Test MCP Server"}},
                }

            with patch.object(
                subprocess_client,
                "_receive_message",
                new_callable=AsyncMock,
                return_value=mock_receive(),
            ):
                with patch.object(subprocess_client, "_load_tools", new_callable=AsyncMock):
                    result = await subprocess_client.connect()

            assert result is True
            assert subprocess_client._connected is True

    @pytest.mark.asyncio
    async def test_execute_tool(self, subprocess_client):
        """Test tool execution via subprocess."""
        # Setup connected client
        subprocess_client._connected = True
        subprocess_client.process = AsyncMock()

        # Add a mock tool
        mock_tool = MCPTool(
            name="test_tool",
            description="Test tool",
            parameters={},
            tool_type=ToolType.FUNCTION,
            server="subprocess",
        )
        subprocess_client.tools["test_tool"] = mock_tool

        # Mock tool response
        async def mock_receive():
            return {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {"content": "Tool executed successfully", "toolCallId": "call_123"},
            }

        with patch.object(
            subprocess_client,
            "_receive_message",
            new_callable=AsyncMock,
            return_value=mock_receive(),
        ):
            response = await subprocess_client.execute_tool("test_tool", {"param": "value"})

        assert response.success is True
        assert response.result == "Tool executed successfully"

    @pytest.mark.asyncio
    async def test_disconnect(self, subprocess_client):
        """Test subprocess disconnection."""
        mock_process = AsyncMock()
        mock_process.wait = AsyncMock()
        subprocess_client.process = mock_process
        subprocess_client._connected = True

        result = await subprocess_client.disconnect()

        assert result is True
        assert subprocess_client._connected is False
        mock_process.terminate.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check(self, subprocess_client):
        """Test subprocess health check."""
        # Test when connected and process running
        mock_process = AsyncMock()
        mock_process.returncode = None
        subprocess_client.process = mock_process
        subprocess_client._connected = True

        result = await subprocess_client.health_check()
        assert result is True

        # Test when process has terminated
        mock_process.returncode = 1
        result = await subprocess_client.health_check()
        assert result is False

        # Test when not connected
        subprocess_client._connected = False
        result = await subprocess_client.health_check()
        assert result is False
