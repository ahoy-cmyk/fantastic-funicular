"""MCP client implementation for connecting to MCP servers."""

import asyncio
import json
import ssl
from typing import Any
from urllib.parse import urlparse

import websockets

from src.mcp import MCPResponse, MCPServer, MCPTool, ToolType
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class MCPClient(MCPServer):
    """WebSocket-based MCP client implementation."""

    def __init__(
        self, server_url: str, name: str = "neuromancer", ssl_config: dict[str, Any] = None
    ):
        """Initialize MCP client.

        Args:
            server_url: WebSocket URL of MCP server (ws://, wss://, http://, https://)
            name: Client identifier
            ssl_config: Optional SSL configuration dict with keys:
                - verify: bool - Verify SSL certificates (default: True)
                - ca_bundle: str - Path to CA bundle file
                - allow_self_signed: bool - Allow self-signed certs (default: False)
        """
        self.server_url = server_url
        self.name = name
        self.websocket = None
        self.tools: dict[str, MCPTool] = {}
        self._connected = False
        self.ssl_config = ssl_config or {}

    async def connect(self) -> bool:
        """Connect to the MCP server with SSL/TLS support."""
        try:
            logger.info(f"Connecting to MCP server: {self.server_url}")

            # Parse URL and convert HTTP(S) schemes to WebSocket equivalents
            parsed_url = urlparse(self.server_url)
            original_scheme = parsed_url.scheme

            # Convert HTTP schemes to WebSocket schemes
            if parsed_url.scheme == "http":
                # Convert http:// to ws://
                websocket_url = self.server_url.replace("http://", "ws://", 1)
                use_ssl = False
                logger.info(f"Converted HTTP URL to WebSocket: {websocket_url}")
            elif parsed_url.scheme == "https":
                # Convert https:// to wss://
                websocket_url = self.server_url.replace("https://", "wss://", 1)
                use_ssl = True
                logger.info(f"Converted HTTPS URL to WebSocket Secure: {websocket_url}")
            elif parsed_url.scheme == "ws":
                websocket_url = self.server_url
                use_ssl = False
            elif parsed_url.scheme == "wss":
                websocket_url = self.server_url
                use_ssl = True
            else:
                raise ValueError(
                    f"Unsupported URL scheme: {parsed_url.scheme}. Use http, https, ws, or wss."
                )

            # Update the URL for connection
            self.websocket_url = websocket_url

            # Configure SSL context for secure connections
            ssl_context = None
            if use_ssl:
                ssl_context = ssl.create_default_context()

                # Apply SSL configuration
                if self.ssl_config.get("ca_bundle"):
                    ssl_context.load_verify_locations(self.ssl_config["ca_bundle"])
                    logger.info(f"Using custom CA bundle: {self.ssl_config['ca_bundle']}")

                if not self.ssl_config.get("verify", True):
                    ssl_context.check_hostname = False
                    ssl_context.verify_mode = ssl.CERT_NONE
                    logger.warning(
                        "SSL certificate verification disabled - connection may be insecure!"
                    )
                elif self.ssl_config.get("allow_self_signed", False):
                    ssl_context.check_hostname = False
                    ssl_context.verify_mode = ssl.CERT_OPTIONAL
                    logger.warning("Allowing self-signed certificates - use only for development!")

                logger.info("Using SSL/TLS for secure connection")

            # Connect with optional SSL context
            extra_headers = {
                "User-Agent": f"Neuromancer-MCP-Client/{self.name}",
                "X-Client-Version": "1.0",
            }

            try:
                # Try with extra_headers first (newer websockets versions)
                self.websocket = await websockets.connect(
                    websocket_url,
                    ssl=ssl_context,
                    extra_headers=extra_headers,
                    ping_interval=20,  # Send ping every 20 seconds
                    ping_timeout=10,  # Wait 10 seconds for pong
                    close_timeout=10,  # Wait 10 seconds for close
                )
            except TypeError as e:
                if "extra_headers" in str(e):
                    # Fallback for older websockets versions without extra_headers
                    logger.info("Using legacy websockets connection (no extra headers)")
                    self.websocket = await websockets.connect(
                        websocket_url,
                        ssl=ssl_context,
                        ping_interval=20,  # Send ping every 20 seconds
                        ping_timeout=10,  # Wait 10 seconds for pong
                        close_timeout=10,  # Wait 10 seconds for close
                    )
                else:
                    raise

            # Send handshake
            await self._send_message(
                {
                    "type": "handshake",
                    "client": self.name,
                    "version": "1.0",
                    "capabilities": ["tools", "streaming", "batch"],
                }
            )

            # Wait for handshake response
            response = await self._receive_message()

            if response.get("type") == "handshake_response" and response.get("success"):
                self._connected = True
                server_name = response.get("server_name", "Unknown")
                server_version = response.get("version", "Unknown")
                logger.info(f"Connected to MCP server: {server_name} (v{server_version})")

                # Load available tools
                await self._load_tools()
                return True
            else:
                logger.error(f"MCP handshake failed: {response}")
                return False

        except websockets.exceptions.InvalidURI as e:
            logger.error(f"Invalid WebSocket URL: {e}")
            logger.info(
                f"Original URL: {self.server_url}, Converted URL: {websocket_url if 'websocket_url' in locals() else 'N/A'}"
            )
            return False
        except ValueError as e:
            logger.error(f"URL conversion error: {e}")
            return False
        except websockets.exceptions.InvalidStatusCode as e:
            logger.error(f"Server rejected connection (HTTP {e.status_code}): {e}")
            return False
        except ssl.SSLError as e:
            logger.error(f"SSL/TLS error: {e}")
            logger.info(
                "Tip: For self-signed certificates, you may need to add the certificate to your trust store"
            )
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
