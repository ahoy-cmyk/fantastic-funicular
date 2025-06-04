"""MCP client implementation for connecting to MCP servers."""

import asyncio
import json
from typing import Any

import websockets

from src.mcp import MCPResponse, MCPServer, MCPTool, ToolType
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class MCPClient(MCPServer):
    """WebSocket-based MCP client implementation."""

    def __init__(self, server_url: str, name: str = "neuromancer"):
        """Initialize MCP client.

        Args:
            server_url: WebSocket URL of MCP server
            name: Client identifier
        """
        self.server_url = server_url
        self.name = name
        self.websocket = None
        self.tools: dict[str, MCPTool] = {}
        self._connected = False

    async def connect(self) -> bool:
        """Connect to the MCP server."""
        try:
            logger.info(f"Connecting to MCP server: {self.server_url}")

            # Parse URL and connect
            self.websocket = await websockets.connect(self.server_url)

            # Send handshake
            await self._send_message({"type": "handshake", "client": self.name, "version": "1.0"})

            # Wait for handshake response
            response = await self._receive_message()

            if response.get("type") == "handshake_response" and response.get("success"):
                self._connected = True
                logger.info(f"Connected to MCP server: {response.get('server_name', 'Unknown')}")

                # Load available tools
                await self._load_tools()
                return True
            else:
                logger.error(f"MCP handshake failed: {response}")
                return False

        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            return False

    async def disconnect(self) -> bool:
        """Disconnect from the MCP server."""
        try:
            if self.websocket:
                await self.websocket.close()

            self._connected = False
            self.tools.clear()
            logger.info("Disconnected from MCP server")
            return True

        except Exception as e:
            logger.error(f"Error disconnecting from MCP server: {e}")
            return False

    async def list_tools(self) -> list[MCPTool]:
        """List available tools from the server."""
        if not self._connected:
            logger.warning("Not connected to MCP server")
            return []

        return list(self.tools.values())

    async def execute_tool(self, tool_name: str, parameters: dict[str, Any]) -> MCPResponse:
        """Execute a tool on the server."""
        try:
            if not self._connected:
                return MCPResponse(success=False, result=None, error="Not connected to MCP server")

            if tool_name not in self.tools:
                return MCPResponse(
                    success=False, result=None, error=f"Tool '{tool_name}' not found"
                )

            # Send tool execution request
            await self._send_message(
                {"type": "execute_tool", "tool": tool_name, "parameters": parameters}
            )

            # Wait for response
            response = await self._receive_message()

            if response.get("type") == "tool_response":
                return MCPResponse(
                    success=response.get("success", False),
                    result=response.get("result"),
                    error=response.get("error"),
                    metadata=response.get("metadata"),
                )
            else:
                return MCPResponse(
                    success=False,
                    result=None,
                    error=f"Unexpected response type: {response.get('type')}",
                )

        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}': {e}")
            return MCPResponse(success=False, result=None, error=str(e))

    async def health_check(self) -> bool:
        """Check if the server is healthy."""
        try:
            if not self._connected:
                return False

            # Send ping
            await self._send_message({"type": "ping"})

            # Wait for pong with timeout
            response = await asyncio.wait_for(self._receive_message(), timeout=5.0)

            return response.get("type") == "pong"

        except Exception:
            return False

    async def _load_tools(self):
        """Load available tools from the server."""
        try:
            # Request tool list
            await self._send_message({"type": "list_tools"})

            # Receive tool list
            response = await self._receive_message()

            if response.get("type") == "tools_list":
                self.tools.clear()

                for tool_data in response.get("tools", []):
                    tool = MCPTool(
                        name=tool_data["name"],
                        description=tool_data["description"],
                        parameters=tool_data.get("parameters", {}),
                        tool_type=ToolType(tool_data.get("type", "function")),
                        server=self.server_url,
                    )
                    self.tools[tool.name] = tool

                logger.info(f"Loaded {len(self.tools)} tools from MCP server")

        except Exception as e:
            logger.error(f"Failed to load tools: {e}")

    async def _send_message(self, message: dict[str, Any]):
        """Send a message to the server."""
        if self.websocket:
            await self.websocket.send(json.dumps(message))

    async def _receive_message(self) -> dict[str, Any]:
        """Receive a message from the server."""
        if self.websocket:
            message = await self.websocket.recv()
            return json.loads(message)
        return {}
