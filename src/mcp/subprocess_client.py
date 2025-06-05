"""MCP client for subprocess-based servers (like mcp-remote)."""

import asyncio
import json
import subprocess
from typing import Any, Optional

from src.mcp import MCPResponse, MCPServer, MCPTool, ToolType
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class MCPSubprocessClient(MCPServer):
    """MCP client that communicates via subprocess STDIO."""

    def __init__(self, command: str, args: list[str], name: str = "neuromancer"):
        """Initialize MCP subprocess client.

        Args:
            command: Command to execute (e.g., "npx")
            args: Arguments for the command (e.g., ["-y", "mcp-remote", "url"])
            name: Client identifier
        """
        self.command = command
        self.args = args
        self.name = name
        self.process: Optional[asyncio.subprocess.Process] = None
        self.tools: dict[str, MCPTool] = {}
        self._connected = False

    async def connect(self) -> bool:
        """Connect to the MCP server via subprocess."""
        try:
            logger.info(f"Starting MCP subprocess: {self.command} {' '.join(self.args)}")

            # Start subprocess
            self.process = await asyncio.create_subprocess_exec(
                self.command,
                *self.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            # Give it time to initialize
            await asyncio.sleep(2)

            # Check if process is still running
            if self.process.returncode is not None:
                stderr = await self.process.stderr.read()
                logger.error(f"Subprocess failed to start: {stderr.decode()}")
                return False

            # Send handshake
            await self._send_message({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": self.name,
                        "version": "1.0.0"
                    }
                }
            })

            # Wait for handshake response
            response = await self._receive_message()

            if response and response.get("result"):
                self._connected = True
                server_info = response.get("result", {}).get("serverInfo", {})
                server_name = server_info.get("name", "Unknown")
                logger.info(f"Connected to MCP server: {server_name}")

                # Load available tools
                await self._load_tools()
                return True
            else:
                logger.error(f"MCP handshake failed: {response}")
                return False

        except Exception as e:
            logger.error(f"Failed to connect to MCP subprocess: {e}")
            return False

    async def disconnect(self) -> bool:
        """Disconnect from the MCP server."""
        try:
            if self.process:
                self.process.terminate()
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    self.process.kill()
                    await self.process.wait()

            self._connected = False
            self.tools.clear()
            logger.info("Disconnected from MCP subprocess")
            return True

        except Exception as e:
            logger.error(f"Error disconnecting from MCP subprocess: {e}")
            return False

    async def list_tools(self) -> list[MCPTool]:
        """List available tools from the server."""
        if not self._connected:
            logger.warning("Not connected to MCP subprocess")
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
            await self._send_message({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": parameters
                }
            })

            # Wait for response
            response = await self._receive_message()

            if response and "result" in response:
                result = response["result"]
                return MCPResponse(
                    success=True,
                    result=result.get("content", []),
                    error=None,
                    metadata={"toolCallId": result.get("toolCallId")}
                )
            elif response and "error" in response:
                error = response["error"]
                return MCPResponse(
                    success=False,
                    result=None,
                    error=f"{error.get('code', 'Unknown')}: {error.get('message', 'Unknown error')}"
                )
            else:
                return MCPResponse(
                    success=False,
                    result=None,
                    error="Invalid response from MCP server"
                )

        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}': {e}")
            return MCPResponse(success=False, result=None, error=str(e))

    async def health_check(self) -> bool:
        """Check if the server is healthy."""
        try:
            if not self._connected or not self.process:
                return False

            # Check if process is still running
            return self.process.returncode is None

        except Exception:
            return False

    async def _load_tools(self):
        """Load available tools from the server."""
        try:
            # Request tool list
            await self._send_message({
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/list",
                "params": {}
            })

            # Receive tool list
            response = await self._receive_message()

            if response and "result" in response:
                self.tools.clear()

                for tool_data in response["result"].get("tools", []):
                    tool = MCPTool(
                        name=tool_data["name"],
                        description=tool_data["description"],
                        parameters=tool_data.get("inputSchema", {}).get("properties", {}),
                        tool_type=ToolType.FUNCTION,  # Assume function type
                        server=f"{self.command} subprocess"
                    )
                    self.tools[tool.name] = tool

                logger.info(f"Loaded {len(self.tools)} tools from MCP subprocess")

        except Exception as e:
            logger.error(f"Failed to load tools: {e}")

    async def _send_message(self, message: dict[str, Any]):
        """Send a message to the subprocess."""
        if self.process and self.process.stdin:
            message_str = json.dumps(message) + "\n"
            self.process.stdin.write(message_str.encode())
            await self.process.stdin.drain()

    async def _receive_message(self) -> dict[str, Any]:
        """Receive a message from the subprocess."""
        if self.process and self.process.stdout:
            try:
                line = await asyncio.wait_for(
                    self.process.stdout.readline(), 
                    timeout=10.0
                )
                if line:
                    line_str = line.decode().strip()
                    if line_str:
                        return json.loads(line_str)
            except (asyncio.TimeoutError, json.JSONDecodeError) as e:
                logger.error(f"Error receiving message: {e}")
        return {}