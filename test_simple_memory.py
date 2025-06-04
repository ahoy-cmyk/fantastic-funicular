#!/usr/bin/env python3
"""Simple memory test without chat manager dependencies."""

import asyncio
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from memory import MemoryType  
from memory.manager import MemoryManager
from utils.logger import setup_logger

logger = setup_logger(__name__)

async def test_simple_memory():
    """Test memory system directly."""
    print("🧠 SIMPLE MEMORY TEST")
    print("=" * 25)
    
    try:
        # Create memory manager directly
        print("📝 Creating memory manager...")
        memory_manager = MemoryManager()
        print("✅ Memory manager created")
        
        # Test 1: Store personal info
        print("\n📝 TEST 1: Storing personal information...")
        memory_id = await memory_manager.remember(
            content="My name is TEST_USER and I work as a software engineer.",
            memory_type=MemoryType.LONG_TERM,
            importance=1.0,
            metadata={"type": "personal_info", "category": "identity"}
        )
        print(f"✅ Stored personal info with ID: {memory_id}")
        
        # Test 2: Store another fact
        print("\n📝 TEST 2: Storing hobby information...")
        memory_id2 = await memory_manager.remember(
            content="TEST_USER loves programming and playing guitar.",
            memory_type=MemoryType.LONG_TERM,
            importance=0.9,
            metadata={"type": "personal_info", "category": "hobby"}
        )
        print(f"✅ Stored hobby info with ID: {memory_id2}")
        
        # Test 3: Get stats
        print("\n📊 TEST 3: Getting memory stats...")
        stats = await memory_manager.get_memory_stats()
        print(f"Total memories: {stats.get('total', 0)}")
        print(f"By type: {stats.get('by_type', {})}")
        
        # Test 4: Test recall with different queries and thresholds
        print("\n🔍 TEST 4: Testing recall...")
        
        test_queries = [
            "What is my name?",
            "name",
            "TEST_USER",
            "personal info",
            "software engineer",
            "guitar"
        ]
        
        for query in test_queries:
            print(f"\n  🔎 Query: '{query}'")
            
            # Test with multiple thresholds
            for threshold in [0.2, 0.3, 0.5, 0.7]:
                memories = await memory_manager.recall(
                    query=query,
                    memory_types=[MemoryType.LONG_TERM, MemoryType.SHORT_TERM],
                    limit=5,
                    threshold=threshold
                )
                
                if memories:
                    print(f"    Threshold {threshold}: Found {len(memories)} memories")
                    for i, mem in enumerate(memories):
                        print(f"      {i+1}. Score: {getattr(mem, 'relevance_score', 'N/A')}, Type: {mem.metadata.get('type', 'unknown')}")
                        print(f"         Content: {mem.content[:60]}...")
                    break
            else:
                print(f"    ❌ No memories found at any threshold")
        
        # Test 5: Get all memories to see what's stored
        print("\n📦 TEST 5: Getting all memories...")
        all_memories = await memory_manager.get_all_memories()
        print(f"Found {len(all_memories)} total memories:")
        
        for i, mem in enumerate(all_memories):
            print(f"  {i+1}. ID: {mem.id[:8]}...")
            print(f"     Type: {mem.memory_type}, Importance: {mem.importance}")
            print(f"     Metadata: {mem.metadata}")
            print(f"     Content: {mem.content}")
            print()
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_simple_memory())