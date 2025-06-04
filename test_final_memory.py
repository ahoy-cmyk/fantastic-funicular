#!/usr/bin/env python3
"""Test final memory fix with very clear personal info."""

import asyncio
from datetime import datetime

from src.core.chat_manager import ChatManager
from src.memory import MemoryType
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

async def test_final_memory():
    """Test with very clear personal memory."""
    print("🎯 FINAL MEMORY TEST")
    print("=" * 30)
    
    chat_manager = ChatManager()
    await chat_manager.create_session("Final Memory Test")
    
    # Store very clear personal info
    print("👤 Storing crystal clear personal info...")
    
    clear_personal_info = "The user's name is STEPHAN. He is a software engineer who loves Python programming."
    
    memory_id = await chat_manager.memory_manager.remember(
        content=clear_personal_info,
        memory_type=MemoryType.LONG_TERM,
        importance=1.0,  # Maximum importance
        metadata={
            "type": "personal_info",
            "category": "identity",
            "keywords": ["name", "stephan", "user", "identity"]
        }
    )
    
    print(f"✅ Stored clear personal info: {memory_id}")
    
    # Test direct recall
    print("\n🔍 Testing direct recall...")
    memories = await chat_manager.memory_manager.recall(
        query="What is the user's name?",
        threshold=0.1,  # Very low threshold
        limit=10
    )
    
    print(f"Found {len(memories)} memories:")
    for i, memory in enumerate(memories):
        print(f"  {i+1}. {memory.content[:100]}...")
        if memory.metadata:
            print(f"      Type: {memory.metadata.get('type', 'unknown')}")
    
    # Test RAG enhanced response
    print("\n💬 Testing RAG response...")
    try:
        response = await chat_manager.send_message_with_rag(
            "What is my name?", 
            stream=False
        )
        print(f"Response: {response}")
        
        if "stephan" in response.lower():
            print("✅ SUCCESS! RAG found and used the name!")
        else:
            print("❌ RAG still not using personal info correctly")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_final_memory())