#!/usr/bin/env python3
"""Test the vector search to see what memories are being returned by similarity."""

import asyncio
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.memory import MemoryType
from src.memory.manager import MemoryManager
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

async def test_vector_search():
    """Test vector search to see what's actually being returned."""
    print("🔍 VECTOR SEARCH DEBUG")
    print("=" * 25)
    
    try:
        # Create memory manager
        memory_manager = MemoryManager()
        print("✅ Memory manager created")
        
        # Store clear personal info
        print("\n📝 Storing clear personal information...")
        memory_id = await memory_manager.remember(
            content="STEPHAN is the user's name. He works as a software engineer.",
            memory_type=MemoryType.LONG_TERM,
            importance=1.0,
            metadata={"type": "personal_info", "category": "identity", "priority": "high"}
        )
        print(f"✅ Stored: {memory_id}")
        
        # Test the vector store search directly
        print("\n🔍 Testing vector store search directly...")
        
        # Test with different similarity thresholds
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
            print(f"\n  Threshold {threshold}:")
            
            # Search LONG_TERM memories specifically
            results = await memory_manager.store.search(
                query="What is my name?",
                memory_type=MemoryType.LONG_TERM,
                limit=10,
                threshold=threshold
            )
            
            print(f"    Found {len(results)} LONG_TERM memories:")
            for i, mem in enumerate(results):
                relevance = getattr(mem, 'relevance_score', 'N/A')
                print(f"      {i+1}. Relevance: {relevance}, Importance: {mem.importance}")
                print(f"         Content: {mem.content[:80]}...")
                if "STEPHAN" in mem.content.upper():
                    print("         🎯 CONTAINS NAME!")
                print(f"         Metadata: {mem.metadata.get('type', 'unknown')}")
        
        # Test all memory types
        print(f"\n🌐 Testing search across ALL memory types...")
        all_results = await memory_manager.store.search(
            query="What is my name?",
            memory_type=None,  # All types
            limit=15,
            threshold=0.2
        )
        
        print(f"Found {len(all_results)} memories across all types:")
        personal_info_found = False
        for i, mem in enumerate(all_results):
            relevance = getattr(mem, 'relevance_score', 'N/A')
            mem_type = mem.metadata.get('type', 'unknown')
            print(f"  {i+1}. Type: {mem.memory_type.value}, Meta: {mem_type}")
            print(f"     Relevance: {relevance}, Importance: {mem.importance}")
            print(f"     Content: {mem.content[:80]}...")
            if "STEPHAN" in mem.content.upper():
                print("     🎯 CONTAINS NAME!")
                if mem_type == "personal_info":
                    personal_info_found = True
                    print("     ⭐ THIS IS PERSONAL INFO!")
            print()
        
        if not personal_info_found:
            print("❌ Personal info not found in search results!")
            
            # Let's check if it exists at all
            print("\n🔍 Checking if personal info exists...")
            all_long_term = await memory_manager.get_all_memories(
                memory_type=MemoryType.LONG_TERM,
                limit=50
            )
            
            print(f"Total LONG_TERM memories: {len(all_long_term)}")
            for mem in all_long_term:
                if mem.metadata.get("type") == "personal_info" and "STEPHAN" in mem.content.upper():
                    print(f"✅ Found personal info: {mem.content[:60]}...")
                    print(f"   Metadata: {mem.metadata}")
                    break
            else:
                print("❌ No personal info found in storage!")
        
        # Test the memory manager recall method (which should apply the personal info boost)
        print(f"\n🧠 Testing memory manager recall method...")
        recalled = await memory_manager.recall(
            query="What is my name?",
            memory_types=[MemoryType.LONG_TERM],
            limit=10,
            threshold=0.2
        )
        
        print(f"Memory manager recall found {len(recalled)} memories:")
        for i, mem in enumerate(recalled):
            mem_type = mem.metadata.get('type', 'unknown')
            print(f"  {i+1}. Type: {mem_type}, Importance: {mem.importance}")
            print(f"     Content: {mem.content[:80]}...")
            if "STEPHAN" in mem.content.upper():
                print("     🎯 CONTAINS NAME!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_vector_search())