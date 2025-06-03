"""MCP connection and tool management."""

import asyncio
from typing import Any

from src.mcp import MCPResponse, MCPServer, MCPTool
from src.mcp.client import MCPClient
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class MCPManager:
    """Manages multiple MCP server connections and tool execution."""

    def __init__(self):
        """Initialize MCP manager."""
        self.servers: dict[str, MCPServer] = {}
        self.tools_cache: dict[str, MCPTool] = {}
        self._lock = asyncio.Lock()

    async def add_server(self, name: str, server_url: str) -> bool:
        """Add and connect to an MCP server.

        Args:
            name: Unique name for the server
            server_url: WebSocket URL of the server

        Returns:
            True if successfully connected
        """
        async with self._lock:
            try:
                if name in self.servers:
                    logger.warning(f"Server '{name}' already exists")
                    return False

                # Create client
                client = MCPClient(server_url, name=f"neuromancer-{name}")

                # Connect to server
                if await client.connect():
                    self.servers[name] = client

                    # Cache tools
                    tools = await client.list_tools()
                    for tool in tools:
                        # Prefix tool name with server name to avoid conflicts
                        cached_name = f"{name}:{tool.name}"
                        self.tools_cache[cached_name] = tool

                    logger.info(f"Added MCP server '{name}' with {len(tools)} tools")
                    return True
                else:
                    logger.error(f"Failed to connect to server '{name}'")
                    return False

            except Exception as e:
                logger.error(f"Error adding server '{name}': {e}")
                return False

    async def remove_server(self, name: str) -> bool:
        """Disconnect and remove an MCP server.

        Args:
            name: Name of the server to remove

        Returns:
            True if successfully removed
        """
        async with self._lock:
            try:
                if name not in self.servers:
                    logger.warning(f"Server '{name}' not found")
                    return False

                # Disconnect
                server = self.servers[name]
                await server.disconnect()

                # Remove from cache
                del self.servers[name]

                # Remove tools from cache
                self.tools_cache = {
                    k: v for k, v in self.tools_cache.items() if not k.startswith(f"{name}:")
                }

                logger.info(f"Removed MCP server '{name}'")
                return True

            except Exception as e:
                logger.error(f"Error removing server '{name}': {e}")
                return False

    async def list_servers(self) -> list[dict[str, Any]]:
        """List all connected MCP servers.

        Returns:
            List of server information
        """
        servers = []

        for name, server in self.servers.items():
            health = await server.health_check()
            tools = await server.list_tools()

            servers.append(
                {
                    "name": name,
                    "url": getattr(server, "server_url", "Unknown"),
                    "connected": health,
                    "tool_count": len(tools),
                }
            )

        return servers

    async def list_all_tools(self) -> list[MCPTool]:
        """List all available tools from all servers.

        Returns:
            List of all tools with server prefixes
        """
        return list(self.tools_cache.values())

    async def execute_tool(self, tool_name: str, parameters: dict[str, Any]) -> MCPResponse:
        """Execute a tool on the appropriate server.

        Args:
            tool_name: Name of the tool (format: "server:tool")
            parameters: Tool parameters

        Returns:
            Tool execution response
        """
        try:
            # Parse server and tool name
            if ":" not in tool_name:
                return MCPResponse(
                    success=False, result=None, error="Tool name must be in format 'server:tool'"
                )

            server_name, actual_tool_name = tool_name.split(":", 1)

            # Get server
            if server_name not in self.servers:
                return MCPResponse(
                    success=False, result=None, error=f"Server '{server_name}' not found"
                )

            server = self.servers[server_name]

            # Execute tool
            response = await server.execute_tool(actual_tool_name, parameters)

            # Add server info to metadata
            if response.metadata is None:
                response.metadata = {}
            response.metadata["server"] = server_name

            return response

        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}': {e}")
            return MCPResponse(success=False, result=None, error=str(e))

    async def health_check_all(self) -> dict[str, bool]:
        """Check health of all servers.

        Returns:
            Dictionary of server health statuses
        """
        health_status = {}

        for name, server in self.servers.items():
            health_status[name] = await server.health_check()

        return health_status

    async def reconnect_failed(self) -> dict[str, bool]:
        """Attempt to reconnect to failed servers.

        Returns:
            Dictionary of reconnection results
        """
        results = {}

        for name, server in self.servers.items():
            if not await server.health_check():
                logger.info(f"Attempting to reconnect to server '{name}'")
                results[name] = await server.connect()
            else:
                results[name] = True

        return results
