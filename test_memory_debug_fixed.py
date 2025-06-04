#!/usr/bin/env python3
"""Comprehensive memory debugging test to trace storage and retrieval."""

import asyncio
from src.core.chat_manager import ChatManager
from src.memory import MemoryType
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

async def test_memory_debug():
    """Debug memory storage and retrieval in detail."""
    print("🔍 COMPREHENSIVE MEMORY DEBUG")
    print("=" * 40)
    
    chat_manager = ChatManager()
    await chat_manager.create_session("Memory Debug")
    
    # Test 1: Store personal info explicitly
    print("\n📝 TEST 1: Storing personal information...")
    memory_id = await chat_manager.memory_manager.remember(
        content="The user's name is JOHN SMITH. He is a software engineer from San Francisco.",
        memory_type=MemoryType.LONG_TERM,
        importance=1.0,
        metadata={"type": "personal_info", "category": "identity"}
    )
    print(f"✅ Stored personal info with ID: {memory_id}")
    
    # Test 2: Store another personal fact
    print("\n📝 TEST 2: Storing another personal fact...")
    memory_id2 = await chat_manager.memory_manager.remember(
        content="John Smith loves playing guitar and has been doing it for 10 years.",
        memory_type=MemoryType.LONG_TERM,
        importance=0.9,
        metadata={"type": "personal_info", "category": "hobby"}
    )
    print(f"✅ Stored hobby info with ID: {memory_id2}")
    
    # Test 3: Store some conversation data for comparison
    print("\n📝 TEST 3: Storing conversation data...")
    memory_id3 = await chat_manager.memory_manager.remember(
        content="The user asked about the weather today.",
        memory_type=MemoryType.SHORT_TERM,
        importance=0.3,
        metadata={"type": "conversation", "category": "query"}
    )
    print(f"✅ Stored conversation with ID: {memory_id3}")
    
    # Test 4: Check total memories
    print("\n📊 TEST 4: Checking memory stats...")
    try:
        stats = await chat_manager.memory_manager.get_stats()
        print(f"Total memories: {stats.get('total', 0)}")
        print(f"By type: {stats.get('by_type', {})}")
    except Exception as e:
        print(f"❌ Failed to get stats: {e}")
    
    # Test 5: Direct recall with various queries
    print("\n🔍 TEST 5: Testing direct recall...")
    
    test_queries = [
        "What is my name?",
        "name",
        "JOHN SMITH", 
        "John",
        "user's name",
        "identity",
        "personal information",
        "guitar",
        "hobby",
        "software engineer"
    ]
    
    for query in test_queries:
        print(f"\n  🔎 Query: '{query}'")
        try:
            # Test with different thresholds
            for threshold in [0.2, 0.3, 0.5, 0.7]:
                memories = await chat_manager.memory_manager.recall(
                    query=query,
                    limit=5,
                    memory_types=[MemoryType.LONG_TERM, MemoryType.SHORT_TERM],
                    min_relevance_threshold=threshold
                )
                
                if memories:
                    print(f"    Threshold {threshold}: Found {len(memories)} memories")
                    for i, mem in enumerate(memories):
                        print(f"      {i+1}. Score: {mem.relevance_score:.3f}, Type: {mem.metadata.get('type', 'unknown')}")
                        print(f"         Content: {mem.content[:60]}...")
                    break
            else:
                print(f"    ❌ No memories found at any threshold")
                
        except Exception as e:
            print(f"    ❌ Error: {e}")
    
    # Test 6: Check what's actually in the vector store
    print("\n📦 TEST 6: Checking vector store contents...")
    try:
        # Get all memories directly from vector store
        all_memories = await chat_manager.memory_manager.get_all_memories()
        print(f"Total memories in vector store: {len(all_memories)}")
        
        for i, mem in enumerate(all_memories):
            print(f"  {i+1}. ID: {mem.id[:8]}...")
            print(f"     Type: {mem.memory_type}, Importance: {mem.importance}")
            print(f"     Metadata: {mem.metadata}")
            print(f"     Content: {mem.content[:80]}...")
            print()
            
    except Exception as e:
        print(f"❌ Failed to get all memories: {e}")
    
    # Test 7: Test the RAG system directly
    print("\n🎯 TEST 7: Testing RAG system...")
    try:
        # Get relevant memories through RAG system
        rag_memories = await chat_manager.rag_system.get_relevant_memories("What is my name?")
        print(f"RAG found {len(rag_memories)} memories")
        
        for i, mem in enumerate(rag_memories):
            print(f"  {i+1}. Score: {getattr(mem, 'relevance_score', 'N/A')}")
            print(f"     Content: {mem.content[:60]}...")
            
    except Exception as e:
        print(f"❌ RAG system error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 8: Test send_message_with_rag
    print("\n🚀 TEST 8: Testing send_message_with_rag...")
    try:
        response = await chat_manager.send_message_with_rag("What is my name?", stream=False)
        print(f"Response: {response}")
        
        # Check if it found the name
        if "JOHN SMITH" in response.upper() or "JOHN" in response.upper():
            print("✅ Successfully found the name in response!")
        else:
            print("❌ Name not found in response")
            
    except Exception as e:
        print(f"❌ send_message_with_rag failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_memory_debug())