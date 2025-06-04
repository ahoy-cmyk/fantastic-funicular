"""Model Context Protocol (MCP) integration for Neuromancer."""

from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol


class ToolType(Enum):
    """Types of MCP tools."""

    FUNCTION = "function"
    RETRIEVAL = "retrieval"
    ACTION = "action"


@dataclass
class MCPTool:
    """Represents an MCP tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    tool_type: ToolType = ToolType.FUNCTION
    server: str | None = None


@dataclass
class MCPResponse:
    """Response from MCP tool execution."""

    success: bool
    result: Any
    error: str | None = None
    metadata: dict[str, Any] | None = None


class MCPServer(Protocol):
    """Protocol for MCP server implementations."""

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the MCP server."""
        ...

    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from the MCP server."""
        ...

    @abstractmethod
    async def list_tools(self) -> list[MCPTool]:
        """List available tools from the server."""
        ...

    @abstractmethod
    async def execute_tool(self, tool_name: str, parameters: dict[str, Any]) -> MCPResponse:
        """Execute a tool on the server."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the server is healthy."""
        ...
