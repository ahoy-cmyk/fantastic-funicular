#!/usr/bin/env python3
"""Debug actual usage scenario to see why tool calling isn't working."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.chat_manager import ChatManager
from src.utils.logger import setup_logger
import logging

logger = setup_logger(__name__)
# Enable ALL debug logging
logging.getLogger().setLevel(logging.DEBUG)


async def test_exact_user_scenario():
    """Simulate exactly what happens when a user asks for current info."""
    print("🔍 Simulating exact user scenario...")
    
    try:
        # Initialize exactly like the real app
        chat_manager = ChatManager()
        await chat_manager.connect_pending_mcp_servers()
        await asyncio.sleep(3)
        
        # Create session like real app
        async with chat_manager.session_manager.create_session("Debug Session") as session:
            await chat_manager.load_session(session.conversation.id)
            
            # Show current system prompt that will be sent
            print("\n📋 Building system prompt...")
            messages = await chat_manager._build_context_messages("test query")
            system_msg = next((m for m in messages if m.role == "system"), None)
            
            if system_msg:
                prompt_preview = system_msg.content
                print(f"System prompt length: {len(prompt_preview)} chars")
                
                # Show tool section specifically
                if "Available tools" in prompt_preview:
                    print("✅ Tools section found in prompt")
                    tool_start = prompt_preview.find("Available tools")
                    tool_section = prompt_preview[tool_start:tool_start+1000]
                    print("🔧 Tool section preview:")
                    print(tool_section)
                else:
                    print("❌ No tools section in prompt!")
                    print("📄 System prompt preview:")
                    print(prompt_preview[:500] + "...")
            
            # Test with different models to see behavior differences
            print(f"\n🤖 Current model: {chat_manager.current_model}")
            
            # Try a search query that SHOULD trigger tools
            queries = [
                "What are the latest developments in AI for 2024?",
                "Search for recent news about Anthropic",
                "I need current information about Claude AI updates",
            ]
            
            for i, query in enumerate(queries):
                print(f"\n📝 Test {i+1}: {query}")
                
                # Process the message and capture everything
                response_generator = chat_manager.send_message_with_rag(
                    content=query,
                    stream=True
                )
                
                full_response = ""
                tool_calls_detected = False
                
                async for chunk in response_generator:
                    if isinstance(chunk, str):
                        full_response += chunk
                        
                        # Check for tool call patterns as they come in
                        if "```tool_call" in chunk:
                            tool_calls_detected = True
                            print(f"🔧 Tool call detected in chunk: {chunk}")
                
                print(f"📄 Final response length: {len(full_response)} chars")
                
                # Analyze the response
                if "Tool Result" in full_response:
                    print("✅ Tool execution successful - found Tool Result section")
                    # Show the tool result
                    start = full_response.find("**Tool Result")
                    if start != -1:
                        end = full_response.find("\n\n", start + 100)
                        if end == -1:
                            end = start + 300
                        print(f"Tool result: {full_response[start:end]}")
                elif "```tool_call" in full_response:
                    print("⚠️  Tool call generated but not executed")
                    # Extract the tool call
                    start = full_response.find("```tool_call")
                    end = full_response.find("```", start + 12)
                    if end != -1:
                        tool_call = full_response[start:end+3]
                        print(f"Generated tool call: {tool_call}")
                        
                        # Manually test the parser
                        print("🔧 Testing parser on this tool call...")
                        parsed_result = await chat_manager._parse_and_execute_tool_calls(full_response)
                        if len(parsed_result) > len(full_response):
                            print("✅ Parser worked - response expanded")
                        else:
                            print("❌ Parser failed - response unchanged")
                else:
                    print("❌ No tool calls generated")
                    print(f"Response preview: {full_response[:200]}...")
                
                # Check if the AI is using training data instead
                if any(word in full_response.lower() for word in ["i don't have access", "i can't browse", "my training data", "as of my last update"]):
                    print("🚫 AI is explicitly refusing to use tools or using training data")
                
                print("-" * 50)
                
                # Only test first query for now
                break
                
    except Exception as e:
        print(f"💥 Error: {e}")
        import traceback
        traceback.print_exc()


async def test_tool_execution_directly():
    """Test if the issue is in tool execution vs detection."""
    print("\n🔧 Testing tool execution directly...")
    
    try:
        chat_manager = ChatManager()
        await chat_manager.connect_pending_mcp_servers()
        await asyncio.sleep(3)
        
        # Test direct tool execution
        print("📡 Testing direct tool execution...")
        response = await chat_manager.execute_mcp_tool(
            "exa:web_search_exa", 
            {"query": "Claude AI 2024", "num_results": 2}
        )
        
        print(f"Direct execution: success={response.success}")
        if response.error:
            print(f"Error: {response.error}")
        if response.result:
            print(f"Result length: {len(str(response.result))} chars")
            return True
        return False
        
    except Exception as e:
        print(f"💥 Direct execution error: {e}")
        return False


async def main():
    """Run comprehensive debugging."""
    print("🧪 Comprehensive tool calling debug...")
    
    # Test direct execution first
    direct_works = await test_tool_execution_directly()
    print(f"Direct tool execution: {'✅' if direct_works else '❌'}")
    
    # Test actual user scenario
    await test_exact_user_scenario()
    
    print("\n🔍 If you're still not seeing tool results:")
    print("1. Check if response contains 'Tool Result' sections")
    print("2. Try with a different model (some follow instructions better)")
    print("3. Look for tool calls that aren't being executed")
    print("4. Check if AI is refusing to use tools")


if __name__ == "__main__":
    asyncio.run(main())