#!/usr/bin/env python3
"""Stress test RAG streaming to trigger the error."""

import asyncio
from src.core.chat_manager import ChatManager
from src.memory import MemoryType
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

async def test_rag_stress():
    """Stress test RAG streaming to trigger the error."""
    print("💪 RAG STRESS TEST")
    print("=" * 20)
    
    chat_manager = ChatManager()
    await chat_manager.create_session("Stress Test")
    
    # Store memory
    await chat_manager.memory_manager.remember(
        content="Important: The user's name is STRESS_TESTER. They love debugging.",
        memory_type=MemoryType.LONG_TERM,
        importance=1.0,
        metadata={"type": "personal_info"}
    )
    
    # Create a conversation with multiple back-and-forth messages
    print("📝 Building conversation history...")
    
    messages = [
        "Hello there!",
        "How are you today?", 
        "What's my name?",
        "Can you help me with something?",
        "Tell me about yourself."
    ]
    
    for i, msg in enumerate(messages):
        print(f"  {i+1}. Sending: {msg}")
        try:
            # Mix streaming and non-streaming
            if i % 2 == 0:
                response = await chat_manager.send_message_with_rag(msg, stream=False)
                print(f"     Non-stream response: {response[:50]}...")
            else:
                chunks = []
                async for chunk in chat_manager.send_message_with_rag(msg, stream=True):
                    chunks.append(chunk)
                response = "".join(chunks)
                print(f"     Stream response: {response[:50]}...")
        except Exception as e:
            print(f"     ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print("\n🎯 Final test - asking about name with full history...")
    try:
        chunks = []
        async for chunk in chat_manager.send_message_with_rag("What is my name again?", stream=True):
            chunks.append(chunk)
        response = "".join(chunks)
        print(f"Final response: {response}")
    except Exception as e:
        print(f"❌ FINAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_rag_stress())