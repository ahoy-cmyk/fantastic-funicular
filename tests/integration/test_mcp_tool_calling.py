"""Integration tests for automatic MCP tool calling from chat."""

from unittest.mock import AsyncMock

import pytest

from src.mcp import MCPResponse, MCPTool, ToolType


@pytest.mark.integration
@pytest.mark.requires_mcp
class TestMCPToolCalling:
    """Test automatic MCP tool calling from AI responses."""

    @pytest.fixture
    async def chat_manager_with_tools(self, chat_manager):
        """Chat manager with mock MCP tools."""
        # Mock MCP tools
        mock_tools = [
            MCPTool(
                name="search:web_search",
                description="Search the web for information",
                parameters={"query": {"type": "string"}, "limit": {"type": "integer"}},
                tool_type=ToolType.RETRIEVAL,
                server="search",
            ),
            MCPTool(
                name="calc:calculate",
                description="Perform mathematical calculations",
                parameters={"expression": {"type": "string"}},
                tool_type=ToolType.FUNCTION,
                server="calc",
            ),
        ]

        # Mock MCP manager methods
        chat_manager.mcp_manager.list_all_tools = AsyncMock(return_value=mock_tools)
        chat_manager.mcp_manager.execute_tool = AsyncMock()

        return chat_manager

    @pytest.mark.asyncio
    async def test_tool_call_parsing(self, chat_manager_with_tools):
        """Test parsing tool calls from AI response."""
        response_with_tool_call = """
I'll search for that information for you.

```tool_call
tool_name: search:web_search
parameters:
  query: "latest AI developments"
  limit: 5
```

Let me get the latest information about AI developments.
        """

        # Mock successful tool execution
        mock_result = MCPResponse(
            success=True, result="Found 5 articles about latest AI developments...", error=None
        )
        chat_manager_with_tools.mcp_manager.execute_tool.return_value = mock_result

        # Parse and execute tool calls
        modified_response = await chat_manager_with_tools._parse_and_execute_tool_calls(
            response_with_tool_call
        )

        # Verify tool was called
        chat_manager_with_tools.mcp_manager.execute_tool.assert_called_once_with(
            "search:web_search", {"query": "latest AI developments", "limit": 5}
        )

        # Verify response was modified
        assert "Tool Result (search:web_search)" in modified_response
        assert "Found 5 articles about latest AI developments" in modified_response
        assert "```tool_call" not in modified_response  # Tool call block should be replaced

    @pytest.mark.asyncio
    async def test_multiple_tool_calls(self, chat_manager_with_tools):
        """Test handling multiple tool calls in one response."""
        response_with_multiple_tools = """
I'll help you with both calculations and searching.

```tool_call
tool_name: calc:calculate
parameters:
  expression: "2 + 2"
```

```tool_call
tool_name: search:web_search
parameters:
  query: "mathematics tutorials"
  limit: 3
```

Here are the results from both operations.
        """

        # Mock tool execution results
        calc_result = MCPResponse(success=True, result="4", error=None)
        search_result = MCPResponse(success=True, result="Found 3 math tutorials...", error=None)

        chat_manager_with_tools.mcp_manager.execute_tool.side_effect = [calc_result, search_result]

        # Parse and execute tool calls
        modified_response = await chat_manager_with_tools._parse_and_execute_tool_calls(
            response_with_multiple_tools
        )

        # Verify both tools were called
        assert chat_manager_with_tools.mcp_manager.execute_tool.call_count == 2

        # Verify response contains both results
        assert "Tool Result (calc:calculate)" in modified_response
        assert "Tool Result (search:web_search)" in modified_response
        assert "4" in modified_response
        assert "Found 3 math tutorials" in modified_response

    @pytest.mark.asyncio
    async def test_tool_call_error_handling(self, chat_manager_with_tools):
        """Test handling of tool execution errors."""
        response_with_failing_tool = """
Let me calculate that for you.

```tool_call
tool_name: calc:calculate
parameters:
  expression: "invalid expression"
```
        """

        # Mock tool execution failure
        error_result = MCPResponse(
            success=False, result=None, error="Invalid mathematical expression"
        )
        chat_manager_with_tools.mcp_manager.execute_tool.return_value = error_result

        # Parse and execute tool calls
        modified_response = await chat_manager_with_tools._parse_and_execute_tool_calls(
            response_with_failing_tool
        )

        # Verify error is handled gracefully
        assert "Tool Error (calc:calculate)" in modified_response
        assert "Invalid mathematical expression" in modified_response
        assert "```tool_call" not in modified_response

    @pytest.mark.asyncio
    async def test_invalid_tool_call_format(self, chat_manager_with_tools):
        """Test handling of malformed tool calls."""
        response_with_invalid_tool = """
Here's an invalid tool call:

```tool_call
invalid yaml format
no proper structure
```
        """

        # Parse and execute tool calls
        modified_response = await chat_manager_with_tools._parse_and_execute_tool_calls(
            response_with_invalid_tool
        )

        # Verify error handling for invalid format
        assert "Tool Execution Error" in modified_response
        assert "```tool_call" not in modified_response

    @pytest.mark.asyncio
    async def test_tool_calls_in_streaming(self, chat_manager_with_tools):
        """Test tool calling in streaming responses."""

        # Mock streaming generator that includes tool calls
        async def mock_stream():
            chunks = [
                "I'll search for that. ",
                "```tool_call\n",
                "tool_name: search:web_search\n",
                "parameters:\n",
                "  query: test\n",
                "```\n",
                "Done with search.",
            ]
            for chunk in chunks:
                yield chunk

        # Mock successful tool execution
        mock_result = MCPResponse(success=True, result="Search completed", error=None)
        chat_manager_with_tools.mcp_manager.execute_tool.return_value = mock_result

        # Test streaming with tool calls
        accumulated_chunks = []
        async for chunk in chat_manager_with_tools._handle_streaming_with_tool_calls(
            mock_stream(), "test_message_id"
        ):
            accumulated_chunks.append(chunk)

        full_response = "".join(accumulated_chunks)

        # Should contain original response plus tool results
        assert "I'll search for that" in full_response
        assert "Done with search" in full_response
        assert "Tool Result (search:web_search)" in full_response
        assert "Search completed" in full_response

    @pytest.mark.asyncio
    async def test_no_tool_calls_in_response(self, chat_manager_with_tools):
        """Test that normal responses without tool calls are unchanged."""
        normal_response = "This is a normal response without any tool calls."

        # Parse response (should be unchanged)
        modified_response = await chat_manager_with_tools._parse_and_execute_tool_calls(
            normal_response
        )

        # Verify no changes were made
        assert modified_response == normal_response
        chat_manager_with_tools.mcp_manager.execute_tool.assert_not_called()

    @pytest.mark.asyncio
    async def test_tool_list_in_system_prompt(self, chat_manager_with_tools):
        """Test that available tools are included in system prompt."""
        # Build context messages
        messages = await chat_manager_with_tools._build_context_messages("test query")

        system_message = messages[0]
        assert system_message.role == "system"

        # Should include tool information
        assert "Available tools (2)" in system_message.content
        assert "search:web_search" in system_message.content
        assert "calc:calculate" in system_message.content
        assert "tool_call" in system_message.content  # Should include format instructions

    @pytest.mark.asyncio
    async def test_tool_parameters_validation(self, chat_manager_with_tools):
        """Test that tool parameters are properly passed through."""
        response_with_complex_params = """
```tool_call
tool_name: search:web_search
parameters:
  query: "complex search query with spaces"
  limit: 10
  filters:
    - recent
    - relevant
```
        """

        # Mock tool execution
        mock_result = MCPResponse(success=True, result="Search result", error=None)
        chat_manager_with_tools.mcp_manager.execute_tool.return_value = mock_result

        # Parse and execute
        await chat_manager_with_tools._parse_and_execute_tool_calls(response_with_complex_params)

        # Verify parameters were passed correctly
        call_args = chat_manager_with_tools.mcp_manager.execute_tool.call_args
        assert call_args[0][0] == "search:web_search"  # tool_name

        params = call_args[0][1]  # parameters
        assert params["query"] == "complex search query with spaces"
        assert params["limit"] == 10
        assert params["filters"] == ["recent", "relevant"]
