#!/usr/bin/env python3
"""Test the YAML parsing fix."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.chat_manager import ChatManager
from src.utils.logger import setup_logger
import logging

logger = setup_logger(__name__)
# Enable debug logging to see the parsing
logging.getLogger("src.core.chat_manager").setLevel(logging.DEBUG)


async def test_yaml_parsing_fix():
    """Test the fixed YAML parsing."""
    print("🔧 Testing YAML parsing fix...")
    
    try:
        chat_manager = ChatManager()
        await chat_manager.connect_pending_mcp_servers()
        await asyncio.sleep(2)
        
        # Test with the problematic format that was generated
        test_response = """I'll search for the latest AI developments.

```tool_call
exa:web_search_exa
parameters:
  query: "AI developments 2024"
  num_results: 3
```

Let me find that information for you."""
        
        print("📝 Testing problematic tool call format...")
        print("Input:", repr(test_response))
        
        # Test the parser
        result = await chat_manager._parse_and_execute_tool_calls(test_response)
        
        print(f"\n📤 Result length: {len(result)} chars")
        
        if "Tool Result" in result:
            print("✅ Tool execution successful!")
            # Show just the start of the tool result
            start = result.find("**Tool Result")
            if start != -1:
                end = result.find("\n", start + 200)
                if end == -1:
                    end = start + 200
                print(f"Tool result preview: {result[start:end]}...")
            return True
        else:
            print("❌ Tool execution failed")
            print(f"Result preview: {result[:300]}...")
            return False
            
    except Exception as e:
        print(f"💥 Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run the test."""
    print("🧪 Testing YAML parsing fix...")
    
    success = await test_yaml_parsing_fix()
    
    print(f"\n📊 Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
    
    if success:
        print("🎉 YAML parsing fix works! Tool calling should now work in the app.")
    else:
        print("🔧 YAML parsing still needs more work.")


if __name__ == "__main__":
    asyncio.run(main())