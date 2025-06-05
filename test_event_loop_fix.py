#!/usr/bin/env python3
"""Test if the event loop issue is fixed."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.chat_manager import ChatManager
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


async def test_tool_execution_with_gui_simulation():
    """Simulate what happens in GUI with tool execution."""
    print("🔧 Testing tool execution with GUI event loop simulation...")
    
    try:
        # Create chat manager
        chat_manager = ChatManager()
        await chat_manager.connect_pending_mcp_servers()
        await asyncio.sleep(3)
        
        # Create session like GUI does
        async with chat_manager.session_manager.create_session("Test Session") as session:
            await chat_manager.load_session(session.conversation.id)
            
            # Test message that should trigger tool usage
            test_message = "search the internet for latest claude news"
            print(f"📝 Testing with: {test_message}")
            
            # Process through RAG pipeline like GUI does
            response_generator = chat_manager.send_message_with_rag(
                content=test_message,
                stream=True
            )
            
            full_response = ""
            tool_executed = False
            error_occurred = False
            
            try:
                async for chunk in response_generator:
                    if isinstance(chunk, str):
                        full_response += chunk
                        if "Tool Result" in chunk:
                            tool_executed = True
                            
            except Exception as e:
                error_occurred = True
                print(f"❌ Error during streaming: {e}")
                
            print(f"\n📄 Response length: {len(full_response)} chars")
            
            if error_occurred:
                print("❌ Event loop error occurred during tool execution")
                return False
            elif tool_executed:
                print("✅ Tool executed successfully!")
                # Show tool result preview
                if "Tool Result" in full_response:
                    start = full_response.find("**Tool Result")
                    end = full_response.find("\n", start + 200)
                    if end == -1:
                        end = start + 200
                    print(f"Tool result preview: {full_response[start:end]}...")
                return True
            elif "```tool_call" in full_response:
                print("⚠️  Tool call generated but not executed")
                return False
            else:
                print("❌ No tool calls generated")
                return False
                
    except Exception as e:
        print(f"💥 Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run the test."""
    print("🧪 Testing event loop handling in tool execution...")
    
    success = await test_tool_execution_with_gui_simulation()
    
    print(f"\n📊 Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
    
    if success:
        print("🎉 Tool execution works with proper event loop handling!")
    else:
        print("🔧 Event loop issues still need to be resolved")
        print("\n💡 The issue is that the GUI and subprocess are using different event loops.")
        print("This is a known issue when integrating asyncio with GUI frameworks like Kivy.")


if __name__ == "__main__":
    asyncio.run(main())