#!/usr/bin/env python3
"""Test the chat-to-memory integration to see why chat doesn't use memories."""

import asyncio
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.chat_manager import ChatManager
from src.memory import MemoryType
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

async def test_chat_memory_integration():
    """Test how chat integrates with memory system."""
    print("💬 CHAT MEMORY INTEGRATION TEST")
    print("=" * 35)
    
    try:
        # Create chat manager
        chat_manager = ChatManager()
        await chat_manager.create_session("Memory Integration Test")
        print("✅ Chat manager created")
        
        # Store some personal info in memory first
        print("\n📝 Storing personal information...")
        memory_id = await chat_manager.memory_manager.remember(
            content="The user's name is STEPHAN. He is a software engineer who loves Python programming.",
            memory_type=MemoryType.LONG_TERM,
            importance=1.0,
            metadata={"type": "personal_info", "category": "identity"}
        )
        print(f"✅ Stored personal info: {memory_id}")
        
        # Test 1: Check if RAG is enabled
        print(f"\n🔍 RAG enabled: {chat_manager.rag_enabled}")
        
        # Test 2: Test direct memory recall
        print("\n🧠 Testing direct memory recall...")
        memories = await chat_manager.memory_manager.recall(
            query="What is my name?",
            memory_types=[MemoryType.LONG_TERM],
            limit=5,
            threshold=0.3
        )
        print(f"Direct recall found {len(memories)} memories:")
        for i, mem in enumerate(memories):
            print(f"  {i+1}. {mem.content[:60]}...")
            if "STEPHAN" in mem.content.upper():
                print("     🎯 FOUND NAME!")
        
        # Test 3: Test RAG system
        print("\n🎯 Testing RAG system...")
        rag_memories = await chat_manager.rag_system.get_relevant_memories("What is my name?")
        print(f"RAG found {len(rag_memories)} memories:")
        for i, mem in enumerate(rag_memories):
            print(f"  {i+1}. {mem.content[:60]}...")
            if "STEPHAN" in mem.content.upper():
                print("     🎯 RAG FOUND NAME!")
        
        # Test 4: Test RAG context building
        print("\n📝 Testing RAG context building...")
        context = await chat_manager.rag_system.retrieve_context("What is my name?")
        print(f"RAG context: {len(context.memories)} memories")
        if context.memories:
            enhanced_prompt = await chat_manager.rag_system.enhance_prompt(
                "What is my name?", 
                context
            )
            print(f"Enhanced prompt contains 'STEPHAN': {'STEPHAN' in enhanced_prompt.upper()}")
            print(f"Enhanced prompt (first 200 chars): {enhanced_prompt[:200]}...")
        
        # Test 5: Test actual chat with RAG (non-streaming first)
        print("\n💬 Testing chat with RAG (non-streaming)...")
        response = await chat_manager.send_message_with_rag("What is my name?", stream=False)
        print(f"Chat response: {response}")
        print(f"Response contains 'STEPHAN': {'STEPHAN' in response.upper()}")
        
        # Test 6: Test streaming chat with RAG  
        print("\n🎬 Testing chat with RAG (streaming)...")
        try:
            chunks = []
            async for chunk in chat_manager.send_message_with_rag("Please tell me my name again", stream=True):
                chunks.append(chunk)
                if len(chunks) > 10:  # Get first few chunks
                    break
            
            streaming_response = "".join(chunks)
            print(f"Streaming response: {streaming_response}")
            print(f"Streaming contains 'STEPHAN': {'STEPHAN' in streaming_response.upper()}")
            
        except Exception as e:
            print(f"❌ Streaming failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test 7: Check if memories are being stored from chat
        print("\n📦 Testing memory storage from chat...")
        
        # Send a message about preferences
        await chat_manager.send_message_with_rag("Remember that I love pizza", stream=False)
        
        # Check if it was stored
        pizza_memories = await chat_manager.memory_manager.recall(
            query="pizza",
            memory_types=None,
            limit=5,
            threshold=0.3
        )
        print(f"Found {len(pizza_memories)} memories about pizza:")
        for mem in pizza_memories:
            if "pizza" in mem.content.lower():
                print(f"  🍕 {mem.content[:60]}...")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_chat_memory_integration())