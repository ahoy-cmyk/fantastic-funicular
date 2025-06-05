#!/usr/bin/env python3
"""Create a simple tool test button to verify tool calling works."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.clock import Clock

from src.core.chat_manager import ChatManager
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ToolTestApp(App):
    def build(self):
        # Create layout
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Query input
        self.query_input = TextInput(
            text="Claude AI latest news",
            size_hint_y=0.1,
            multiline=False
        )
        layout.add_widget(self.query_input)
        
        # Test button
        test_btn = Button(
            text="Test Tool Execution",
            size_hint_y=0.1
        )
        test_btn.bind(on_press=self.test_tool)
        layout.add_widget(test_btn)
        
        # Result display
        self.result_text = TextInput(
            text="Results will appear here...",
            multiline=True,
            readonly=True
        )
        layout.add_widget(self.result_text)
        
        # Initialize chat manager in background
        Clock.schedule_once(lambda dt: asyncio.create_task(self.init_chat_manager()), 0.1)
        
        return layout
    
    async def init_chat_manager(self):
        """Initialize chat manager."""
        self.result_text.text = "Initializing MCP servers..."
        
        try:
            self.chat_manager = ChatManager()
            await self.chat_manager.connect_pending_mcp_servers()
            await asyncio.sleep(3)
            
            self.result_text.text = "Ready! Enter a query and click Test Tool Execution"
            
        except Exception as e:
            self.result_text.text = f"Error initializing: {e}"
    
    def test_tool(self, instance):
        """Test tool execution."""
        asyncio.create_task(self._test_tool_async())
    
    async def _test_tool_async(self):
        """Async tool test."""
        query = self.query_input.text
        self.result_text.text = f"Testing tool with query: {query}\n\n"
        
        try:
            # Direct tool execution test
            self.result_text.text += "1. Testing direct tool execution...\n"
            
            result = await self.chat_manager.execute_mcp_tool(
                "exa:web_search_exa",
                {"query": query, "num_results": 3}
            )
            
            if result.success:
                self.result_text.text += f"✅ Direct execution successful!\n"
                self.result_text.text += f"Result preview: {str(result.result)[:200]}...\n\n"
            else:
                self.result_text.text += f"❌ Direct execution failed: {result.error}\n\n"
            
            # Test through chat pipeline
            self.result_text.text += "2. Testing through chat pipeline...\n"
            
            # Create session
            async with self.chat_manager.session_manager.create_session("Test") as session:
                await self.chat_manager.load_session(session.conversation.id)
                
                # Send message
                test_message = f"Use the web search tool to find: {query}"
                response_gen = self.chat_manager.send_message_with_rag(
                    content=test_message,
                    stream=True
                )
                
                full_response = ""
                async for chunk in response_gen:
                    if isinstance(chunk, str):
                        full_response += chunk
                
                self.result_text.text += f"Response length: {len(full_response)} chars\n"
                
                if "Tool Result" in full_response:
                    self.result_text.text += "✅ Tool was executed in chat!\n"
                    # Extract tool result
                    start = full_response.find("**Tool Result")
                    if start != -1:
                        end = full_response.find("\n\n", start)
                        if end == -1 or end - start > 500:
                            end = start + 500
                        self.result_text.text += f"\nTool result:\n{full_response[start:end]}...\n"
                elif "```tool_call" in full_response:
                    self.result_text.text += "⚠️ Tool call generated but not executed\n"
                    # Show the tool call
                    start = full_response.find("```tool_call")
                    end = full_response.find("```", start + 12)
                    if end != -1:
                        self.result_text.text += f"\nGenerated tool call:\n{full_response[start:end+3]}\n"
                else:
                    self.result_text.text += "❌ No tool calls generated\n"
                    self.result_text.text += f"\nResponse preview:\n{full_response[:300]}...\n"
                
        except Exception as e:
            self.result_text.text += f"\n💥 Error: {e}\n"
            import traceback
            self.result_text.text += traceback.format_exc()


if __name__ == "__main__":
    ToolTestApp().run()