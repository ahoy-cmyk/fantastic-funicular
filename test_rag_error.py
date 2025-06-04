#!/usr/bin/env python3
"""Test to reproduce the exact RAG streaming error."""

import asyncio
from src.core.chat_manager import ChatManager
from src.memory import MemoryType
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

async def test_rag_streaming_error():
    """Test to reproduce the RAG streaming error."""
    print("🐛 TESTING RAG STREAMING ERROR")
    print("=" * 35)
    
    chat_manager = ChatManager()
    
    # Create session and add some messages first
    await chat_manager.create_session("Error Test")
    
    # Store some memory
    await chat_manager.memory_manager.remember(
        content="The user's name is TEST_USER for debugging.",
        memory_type=MemoryType.LONG_TERM,
        importance=1.0,
        metadata={"type": "personal_info"}
    )
    
    # Add a regular message first to create session history
    print("📝 Adding regular message to create session history...")
    try:
        response = await chat_manager.send_message_with_rag("Hello", stream=False)
        print(f"✅ Regular message worked: {response[:50]}...")
    except Exception as e:
        print(f"❌ Regular message failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Now try streaming with session history
    print("\n🎬 Testing streaming with session history...")
    try:
        response_chunks = []
        async for chunk in chat_manager.send_message_with_rag("What is my name?", stream=True):
            response_chunks.append(chunk)
            if len(response_chunks) > 5:  # Test first few chunks
                break
        
        response = "".join(response_chunks)
        print(f"✅ Streaming worked: {response}")
        
    except Exception as e:
        print(f"❌ Streaming failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_rag_streaming_error())