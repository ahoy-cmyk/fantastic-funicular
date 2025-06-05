# MCP Tool Usage Guide

This document explains how to use MCP (Model Context Protocol) tools in Neuromancer.

## Overview

Neuromancer now supports automatic execution of MCP tools during chat conversations. When the AI decides to use a tool, it formats a tool call in its response, and the system automatically executes it and includes the results.

## How It Works

1. **Tool Discovery**: Available MCP tools are automatically included in the AI's system prompt
2. **Tool Calling**: The AI can request tool execution using a specific format
3. **Automatic Execution**: The system parses tool calls and executes them automatically
4. **Result Integration**: Tool results are seamlessly integrated into the conversation

## Tool Call Format

The AI uses this format to call tools:

```tool_call
tool_name: server:tool_name
parameters:
  param1: value1
  param2: value2
```

## Example Usage

### User Input:
"Search for the latest news about artificial intelligence"

### AI Response:
"I'll search for the latest AI news for you.

```tool_call
tool_name: search:web_search
parameters:
  query: "latest artificial intelligence news"
  limit: 5
```

Based on the search results, here are the latest developments in AI..."

### Actual Output:
"I'll search for the latest AI news for you.

**Tool Result (search:web_search):**
Found 5 recent articles about AI developments including breakthrough in LLM training...

Based on the search results, here are the latest developments in AI..."

## Supported Server Types

### WebSocket Servers
- Direct connection to MCP-compatible services
- Real-time tool execution
- HTTPS/WSS support with SSL configuration

### Subprocess Servers
- Use `mcp-remote` to bridge HTTP-based APIs
- Example: `npx -y mcp-remote https://api.example.com/mcp`
- Automatic process management

## Setting Up MCP Servers

1. **Via GUI**: Use the MCP Management screen in Settings
2. **Via Configuration**: Add servers to the config file
3. **Programmatically**: Use the ChatManager API

### Example Configuration:
```json
{
  "mcp": {
    "enabled": true,
    "auto_connect": true,
    "servers": {
      "search_server": {
        "enabled": true,
        "url": "wss://api.search.com/mcp",
        "description": "Web search capabilities"
      },
      "exa_server": {
        "enabled": true,
        "command": "npx",
        "args": ["-y", "mcp-remote", "https://mcp.exa.ai/mcp?exaApiKey=YOUR_KEY"],
        "description": "Exa search via mcp-remote"
      }
    }
  }
}
```

## Tool Categories

- **Retrieval Tools**: Search, database queries, information lookup
- **Function Tools**: Calculations, data processing, API calls
- **Generation Tools**: Content creation, image generation
- **Action Tools**: File operations, system commands, workflows

## Error Handling

- **Tool Errors**: Displayed inline with error details
- **Network Issues**: Automatic retry with fallback
- **Invalid Calls**: Graceful error messages in chat
- **Missing Tools**: Clear feedback about unavailable tools

## Performance Notes

- Tools execute after the AI completes its response
- Results are streamed back in real-time
- Multiple tools can be called in one response
- Tool execution is asynchronous and non-blocking

## Debugging

Enable debug logging to see:
- Tool discovery and registration
- Tool call parsing and execution
- Parameter validation and results
- Error details and stack traces

Example log output:
```
INFO: Executing MCP tool: search:web_search with parameters: {'query': 'AI news', 'limit': 5}
INFO: Tool execution successful, result length: 1247 characters
```

## Best Practices

1. **Clear Descriptions**: Provide detailed tool descriptions for better AI understanding
2. **Parameter Validation**: Ensure tools validate input parameters properly
3. **Error Messages**: Return helpful error messages for debugging
4. **Performance**: Keep tool execution fast for better user experience
5. **Security**: Validate and sanitize all tool inputs

## Limitations

- Tools must follow MCP protocol specification
- Maximum response size limits apply
- Some tools may require authentication
- Network connectivity required for remote tools

For more details, see the MCP specification at: https://modelcontextprotocol.io/
