#!/usr/bin/env python3
"""Test memory ordering and recent memories visibility."""

import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.memory import MemoryType  
from src.memory.safe_operations import create_safe_memory_manager
from utils.logger import setup_logger

logger = setup_logger(__name__)

async def test_memory_order():
    """Test memory ordering and check recent memories."""
    print("📅 MEMORY ORDER TEST")
    print("=" * 25)
    
    try:
        # Create safe memory manager (like the memory screen uses)
        safe_memory = create_safe_memory_manager()
        print("✅ Safe memory manager created")
        
        # Add a test memory with current timestamp
        print("\n📝 Adding test memory with current timestamp...")
        test_memory_id = await safe_memory.safe_remember(
            content="TEST MEMORY: This is a new memory created right now for debugging!",
            memory_type=MemoryType.LONG_TERM,
            importance=1.0,
            metadata={"type": "test", "created_manually": True, "test_time": datetime.now().isoformat()}
        )
        print(f"✅ Added test memory: {test_memory_id}")
        
        # Get recent memories (like the memory screen does)
        print("\n📦 Getting recent memories (first 10)...")
        recent_memories = await safe_memory.safe_get_all_memories(
            memory_types=None,  # All types
            limit=10,
            offset=0
        )
        
        print(f"Found {len(recent_memories)} recent memories:")
        for i, mem in enumerate(recent_memories):
            created_str = mem.created_at.strftime('%Y-%m-%d %H:%M:%S') if mem.created_at else "Unknown"
            print(f"  {i+1}. {created_str} - {mem.memory_type.value}")
            print(f"     Content: {mem.content[:80]}...")
            if "TEST MEMORY" in mem.content:
                print("     🎯 FOUND OUR TEST MEMORY!")
            print()
        
        # Check if our test memory is in the first 10
        test_found = any("TEST MEMORY" in mem.content for mem in recent_memories)
        if test_found:
            print("✅ Test memory appears in recent memories (memory screen should show it)")
        else:
            print("❌ Test memory NOT in recent memories - this is the issue!")
            
            # Let's check the first 50 memories
            print("\n🔍 Checking first 50 memories...")
            more_memories = await safe_memory.safe_get_all_memories(
                memory_types=None,
                limit=50,
                offset=0
            )
            
            test_found_in_50 = False
            for i, mem in enumerate(more_memories):
                if "TEST MEMORY" in mem.content:
                    created_str = mem.created_at.strftime('%Y-%m-%d %H:%M:%S') if mem.created_at else "Unknown"
                    print(f"🎯 Found test memory at position {i+1}: {created_str}")
                    test_found_in_50 = True
                    break
            
            if not test_found_in_50:
                print("❌ Test memory not found in first 50 either - storage issue!")
        
        # Get memory stats
        print("\n📊 Memory statistics...")
        stats = await safe_memory.safe_get_stats()
        print(f"Total memories: {stats.get('total', 0)}")
        print(f"By type: {stats.get('by_type', {})}")
        
        # Test with specific memory type filter
        print("\n🔍 Testing LONG_TERM memory filter...")
        long_term_memories = await safe_memory.safe_get_all_memories(
            memory_types=[MemoryType.LONG_TERM],
            limit=10,
            offset=0
        )
        
        print(f"Found {len(long_term_memories)} long-term memories:")
        for i, mem in enumerate(long_term_memories):
            created_str = mem.created_at.strftime('%Y-%m-%d %H:%M:%S') if mem.created_at else "Unknown"
            print(f"  {i+1}. {created_str}")
            print(f"     Content: {mem.content[:60]}...")
            if "TEST MEMORY" in mem.content:
                print("     🎯 FOUND TEST MEMORY IN LONG_TERM!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_memory_order())