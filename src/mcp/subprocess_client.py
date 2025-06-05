"""MCP client for subprocess-based servers (like mcp-remote)."""

import asyncio
import concurrent.futures
import json
from typing import Any

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
        self.process: asyncio.subprocess.Process | None = None
        self.tools: dict[str, MCPTool] = {}
        self._connected = False
        self._execution_lock = None  # Will be created with the event loop
        self._request_id = 0

        # Dedicated thread and event loop for subprocess operations
        self._thread_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"mcp-{name}"
        )
        self._subprocess_loop = None
        self._loop_thread = None

    async def connect(self) -> bool:
        """Connect to the MCP server via subprocess."""
        try:
            logger.info(f"Starting MCP subprocess: {self.command} {' '.join(self.args)}")

            # Run connection in dedicated thread to avoid event loop conflicts
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(self._thread_executor, self._connect_sync)
            return result

        except Exception as e:
            logger.error(f"Failed to connect to MCP subprocess: {e}")
            return False

    def _connect_sync(self) -> bool:
        """Synchronous connection in dedicated thread."""
        # Create new event loop for this thread
        self._subprocess_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._subprocess_loop)

        try:
            return self._subprocess_loop.run_until_complete(self._connect_impl())
        except Exception as e:
            logger.error(f"Connection failed in subprocess thread: {e}")
            return False

    async def _connect_impl(self) -> bool:
        """Internal connection implementation."""
        try:
            # Create lock with current event loop
            if self._execution_lock is None:
                self._execution_lock = asyncio.Lock()

            # Start subprocess
            self.process = await asyncio.create_subprocess_exec(
                self.command,
                *self.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Give it time to initialize
            await asyncio.sleep(2)

            # Check if process is still running
            if self.process.returncode is not None:
                stderr = await self.process.stderr.read()
                logger.error(f"Subprocess failed to start: {stderr.decode()}")
                return False

            # Send handshake
            await self._send_message_impl(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": self.name, "version": "1.0.0"},
                    },
                }
            )

            # Wait for handshake response
            response = await self._receive_message_impl()

            if response and response.get("result"):
                self._connected = True
                server_info = response.get("result", {}).get("serverInfo", {})
                server_name = server_info.get("name", "Unknown")
                logger.info(f"Connected to MCP server: {server_name}")

                # Load available tools
                await self._load_tools_impl()
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

            # Clean up thread executor and subprocess loop
            if self._thread_executor:
                self._thread_executor.shutdown(wait=False)

            if self._subprocess_loop and not self._subprocess_loop.is_closed():
                try:
                    self._subprocess_loop.call_soon_threadsafe(self._subprocess_loop.stop)
                except Exception:
                    pass  # Loop might already be stopped

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

            # Run tool execution in dedicated subprocess thread
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                self._thread_executor, lambda: self._execute_tool_sync(tool_name, parameters)
            )
            return result

        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}': {e}")
            return MCPResponse(success=False, result=None, error=str(e))

    def _execute_tool_sync(self, tool_name: str, parameters: dict[str, Any]) -> MCPResponse:
        """Synchronous tool execution in dedicated thread."""
        if not self._subprocess_loop:
            logger.error("Subprocess loop not initialized")
            return MCPResponse(success=False, result=None, error="Subprocess loop not available")

        # Ensure we're running in the subprocess loop
        if asyncio.get_event_loop() != self._subprocess_loop:
            # Run in the correct loop
            future = asyncio.run_coroutine_threadsafe(
                self._execute_tool_impl(tool_name, parameters), self._subprocess_loop
            )
            return future.result(timeout=30.0)  # 30 second timeout
        else:
            # Already in the right loop
            return self._subprocess_loop.run_until_complete(
                self._execute_tool_impl(tool_name, parameters)
            )

    async def _execute_tool_impl(self, tool_name: str, parameters: dict[str, Any]) -> MCPResponse:
        """Internal tool execution implementation."""
        # Ensure lock exists
        if self._execution_lock is None:
            self._execution_lock = asyncio.Lock()

        async with self._execution_lock:  # Prevent concurrent executions
            try:
                # Send tool execution request with unique ID
                self._request_id += 1
                request_id = self._request_id

                await self._send_message_impl(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "method": "tools/call",
                        "params": {"name": tool_name, "arguments": parameters},
                    }
                )

                # Wait for response
                response = await self._receive_message_impl()
                logger.debug(f"Tool execution response: {response}")

                if response and "result" in response:
                    result = response["result"]
                    return MCPResponse(
                        success=True,
                        result=result.get("content", []),
                        error=None,
                        metadata={"toolCallId": result.get("toolCallId")},
                    )
                elif response and "error" in response:
                    error = response["error"]
                    error_msg = (
                        f"{error.get('code', 'Unknown')}: {error.get('message', 'Unknown error')}"
                    )
                    return MCPResponse(success=False, result=None, error=error_msg)
                else:
                    return MCPResponse(
                        success=False, result=None, error="Invalid response from MCP server"
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

    async def _load_tools_impl(self):
        """Load available tools from the server."""
        try:
            # Request tool list with unique ID
            self._request_id += 1
            await self._send_message_impl(
                {"jsonrpc": "2.0", "id": self._request_id, "method": "tools/list", "params": {}}
            )

            # Receive tool list
            response = await self._receive_message_impl()

            if response and "result" in response:
                self.tools.clear()

                for tool_data in response["result"].get("tools", []):
                    tool = MCPTool(
                        name=tool_data["name"],
                        description=tool_data["description"],
                        parameters=tool_data.get("inputSchema", {}).get("properties", {}),
                        tool_type=ToolType.FUNCTION,  # Assume function type
                        server=f"{self.command} subprocess",
                    )
                    self.tools[tool.name] = tool

                logger.info(f"Loaded {len(self.tools)} tools from MCP subprocess")

        except Exception as e:
            logger.error(f"Failed to load tools: {e}")

    async def _send_message_impl(self, message: dict[str, Any]):
        """Send a message to the subprocess (subprocess loop version)."""
        if self.process and self.process.stdin:
            try:
                message_str = json.dumps(message) + "\n"
                self.process.stdin.write(message_str.encode())
                await self.process.stdin.drain()
                logger.debug(f"Sent message: {message.get('method', 'unknown')}")

            except (RuntimeError, OSError) as e:
                # Handle event loop issues and broken pipes
                logger.warning(f"I/O issue in send: {e}")
                raise

    async def _receive_message_impl(self) -> dict[str, Any]:
        """Receive a message from the subprocess (subprocess loop version)."""
        if self.process and self.process.stdout:
            try:
                logger.debug("Waiting for MCP response...")

                line = await asyncio.wait_for(self.process.stdout.readline(), timeout=15.0)

                if line:
                    line_str = line.decode().strip()
                    logger.debug(f"Received line: {line_str[:100]}...")
                    if line_str:
                        return json.loads(line_str)

            except asyncio.TimeoutError:
                logger.error("Timeout waiting for MCP response (15s)")
                # Check if process is still alive
                if self.process:
                    logger.error(f"Process returncode: {self.process.returncode}")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON from MCP server: {e}")
                logger.error(f"Raw response: {line_str}")
            except (RuntimeError, OSError) as e:
                logger.error(f"I/O error receiving message: {e}")
            except Exception as e:
                logger.error(f"Unexpected error receiving message: {e}")
        return {}
