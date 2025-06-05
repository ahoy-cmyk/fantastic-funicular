"""MCP connection and tool management."""

import asyncio
from typing import Any

from src.mcp import MCPResponse, MCPServer, MCPTool
from src.mcp.client import MCPClient
from src.mcp.subprocess_client import MCPSubprocessClient
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class MCPManager:
    """Manages multiple MCP server connections and tool execution."""

    def __init__(self):
        """Initialize MCP manager."""
        self.servers: dict[str, MCPServer] = {}
        self.tools_cache: dict[str, MCPTool] = {}
        self._lock = asyncio.Lock()

    async def add_server(
        self,
        name: str,
        server_url: str = None,
        ssl_config: dict[str, Any] = None,
        command: str = None,
        args: list[str] = None,
    ) -> bool:
        """Add and connect to an MCP server.

        Args:
            name: Unique name for the server
            server_url: WebSocket URL of the server (for WebSocket MCP servers)
            ssl_config: Optional SSL configuration for secure connections
            command: Command to execute (for subprocess MCP servers)
            args: Arguments for the command (for subprocess MCP servers)

        Returns:
            True if successfully connected
        """
        async with self._lock:
            try:
                if name in self.servers:
                    logger.warning(f"Server '{name}' already exists")
                    return False

                # Determine client type
                if server_url:
                    # WebSocket MCP server
                    client = MCPClient(
                        server_url, name=f"neuromancer-{name}", ssl_config=ssl_config
                    )
                elif command and args:
                    # Subprocess MCP server
                    client = MCPSubprocessClient(command, args, name=f"neuromancer-{name}")
                else:
                    logger.error(
                        f"Must provide either server_url or command+args for server '{name}'"
                    )
                    return False

                # Connect to server
                if await client.connect():
                    self.servers[name] = client

                    # Cache tools
                    tools = await client.list_tools()
                    logger.debug(f"Server '{name}' returned {len(tools)} tools for caching")
                    for tool in tools:
                        # Prefix tool name with server name to avoid conflicts
                        cached_name = f"{name}:{tool.name}"
                        self.tools_cache[cached_name] = tool
                        logger.debug(f"Cached tool: {cached_name} - {tool.description}")

                    logger.info(f"Added MCP server '{name}' with {len(tools)} tools")
                    logger.debug(f"Total tools in cache now: {len(self.tools_cache)}")
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

            # Get server info based on type
            if hasattr(server, "server_url"):
                server_info = getattr(server, "server_url", "Unknown")
                server_type = "websocket"
            elif hasattr(server, "command"):
                server_info = f"{server.command} {' '.join(server.args)}"
                server_type = "subprocess"
            else:
                server_info = "Unknown"
                server_type = "unknown"

            servers.append(
                {
                    "name": name,
                    "url": server_info,
                    "type": server_type,
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
        logger.debug(f"Listing tools from cache. Cache has {len(self.tools_cache)} tools:")
        for tool_name, tool in self.tools_cache.items():
            logger.debug(f"  Tool: {tool_name} - {tool.description}")
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

            # Execute tool with timeout
            logger.debug(f"Executing tool '{actual_tool_name}' on server '{server_name}' with params: {parameters}")
            try:
                response = await asyncio.wait_for(
                    server.execute_tool(actual_tool_name, parameters),
                    timeout=30.0  # 30 second timeout
                )
                logger.debug(f"Got response from server: success={response.success}, error={response.error}")
            except asyncio.TimeoutError:
                logger.error(f"Tool execution timed out for {server_name}:{actual_tool_name}")
                return MCPResponse(
                    success=False, result=None, error="Tool execution timed out"
                )

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
