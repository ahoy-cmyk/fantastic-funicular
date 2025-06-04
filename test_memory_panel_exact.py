#!/usr/bin/env python3
"""Test exactly what the memory manager panel does when loading memories."""

import asyncio
import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.memory import MemoryType  
from src.memory.safe_operations import create_safe_memory_manager
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

async def test_memory_panel_exact():
    """Test exactly what the memory manager panel does."""
    print("🖥️ MEMORY PANEL EXACT TEST")
    print("=" * 30)
    
    try:
        # Create safe memory manager exactly like the memory screen does
        def error_callback(operation: str, error: str):
            print(f"❌ Memory error - {operation}: {error}")
        
        safe_memory = create_safe_memory_manager(error_callback)
        print("✅ Safe memory manager created (like memory screen)")
        
        # Add a test memory first
        print("\n📝 Adding test memory...")
        test_content = f"PANEL TEST: Memory created at {datetime.now().isoformat()}"
        test_memory_id = await safe_memory.safe_remember(
            content=test_content,
            memory_type=MemoryType.LONG_TERM,
            importance=0.9,
            metadata={
                "created_manually": True,
                "source": "memory_management", 
                "created_at": datetime.now().isoformat(),
                "test": True
            }
        )
        print(f"✅ Added test memory: {test_memory_id}")
        
        # Test the exact method call the memory screen uses
        print("\n📦 Loading memories exactly like memory screen...")
        print("Calling: safe_memory.safe_get_all_memories(memory_types=None, limit=50, offset=0)")
        
        memories = await safe_memory.safe_get_all_memories(
            memory_types=None,  # Get all types (like memory screen default)
            limit=50,          # Default page size in memory screen
            offset=0           # First page
        )
        
        print(f"✅ Retrieved {len(memories)} memories")
        
        if not memories:
            print("❌ NO MEMORIES RETRIEVED - This is why the panel is empty!")
            
            # Let's try alternative approaches
            print("\n🔍 Trying alternative approaches...")
            
            # Try with specific memory types
            for mem_type in MemoryType:
                type_memories = await safe_memory.safe_get_all_memories(
                    memory_types=[mem_type],
                    limit=10,
                    offset=0
                )
                print(f"  {mem_type.value}: {len(type_memories)} memories")
                if type_memories and mem_type == MemoryType.LONG_TERM:
                    print(f"    Latest: {type_memories[0].content[:50]}...")
            
            return
        
        # Display memories like the memory screen would
        print(f"\n📋 Memories (sorted by creation date, newest first):")
        for i, memory in enumerate(memories[:10]):  # Show first 10
            created_str = memory.created_at.strftime('%Y-%m-%d %H:%M:%S') if memory.created_at else "Unknown"
            content_preview = memory.content[:80] + "..." if len(memory.content) > 80 else memory.content
            
            print(f"\n  {i+1}. [{memory.memory_type.value}] {created_str}")
            print(f"     Importance: {memory.importance}")
            print(f"     Content: {content_preview}")
            print(f"     Metadata: {memory.metadata}")
            
            if "PANEL TEST" in memory.content:
                print("     🎯 FOUND OUR TEST MEMORY!")
            elif "STEPHAN" in memory.content.upper():
                print("     👤 CONTAINS USER NAME!")
        
        # Check if our test memory is in the results
        test_found = any("PANEL TEST" in mem.content for mem in memories)
        name_found = any("STEPHAN" in mem.content.upper() for mem in memories)
        
        print(f"\n📊 Analysis:")
        print(f"  Our test memory found: {test_found}")
        print(f"  User name found: {name_found}")
        print(f"  Total memories: {len(memories)}")
        print(f"  Memory types present: {set(m.memory_type.value for m in memories)}")
        
        # Check ages of memories
        if memories:
            now = datetime.now()
            recent_count = 0
            for mem in memories:
                if mem.created_at:
                    age_hours = (now - mem.created_at).total_seconds() / 3600
                    if age_hours < 24:  # Less than 24 hours old
                        recent_count += 1
            
            print(f"  Recent memories (< 24h): {recent_count}")
            print(f"  Oldest memory: {memories[-1].created_at if memories else 'N/A'}")
            print(f"  Newest memory: {memories[0].created_at if memories else 'N/A'}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_memory_panel_exact())