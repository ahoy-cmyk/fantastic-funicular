#!/usr/bin/env python3
"""Debug streaming RAG issue."""

import asyncio
from src.core.chat_manager import ChatManager
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

async def test_streaming_debug():
    """Debug the streaming issue."""
    print("🔧 DEBUGGING STREAMING RAG")
    print("=" * 30)
    
    chat_manager = ChatManager()
    await chat_manager.create_session("Debug Test")
    
    # Test regular streaming first
    print("\n📺 Testing regular streaming...")
    try:
        result = chat_manager.send_message("Hello", stream=True)
        print(f"Regular streaming result type: {type(result)}")
        
        if hasattr(result, '__aiter__'):
            print("✅ Regular streaming returns async generator")
            count = 0
            async for chunk in result:
                count += 1
                if count > 3:  # Just test first few chunks
                    break
            print(f"✅ Regular streaming worked ({count} chunks)")
        else:
            print("❌ Regular streaming doesn't return async generator")
            
    except Exception as e:
        print(f"❌ Regular streaming error: {e}")
    
    # Test RAG streaming
    print("\n🤖 Testing RAG streaming...")
    try:
        result = chat_manager.send_message_with_rag("Test RAG streaming", stream=True)
        print(f"RAG streaming result type: {type(result)}")
        
        if hasattr(result, '__aiter__'):
            print("✅ RAG streaming returns async generator")
            count = 0
            async for chunk in result:
                count += 1
                if count > 3:  # Just test first few chunks
                    break
            print(f"✅ RAG streaming worked ({count} chunks)")
        else:
            print("❌ RAG streaming doesn't return async generator")
            print(f"Result: {result}")
            
    except Exception as e:
        print(f"❌ RAG streaming error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_streaming_debug())