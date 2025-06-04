#!/usr/bin/env python3
"""Test ChromaDB collections to see what's actually created."""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from memory import MemoryType
from memory.vector_store import VectorMemoryStore
from utils.logger import setup_logger

logger = setup_logger(__name__)

def test_collections():
    """Test what collections are created."""
    print("🗂️ COLLECTIONS TEST")
    print("=" * 20)
    
    try:
        # Create vector store
        store = VectorMemoryStore()
        print("✅ Vector store created")
        
        print("\n📋 Available memory types:")
        for mem_type in MemoryType:
            print(f"  - {mem_type}: {mem_type.value}")
        
        print("\n🗂️ Created collections:")
        for mem_type, collection in store.collections.items():
            print(f"  - {mem_type}: {collection.name} (count: {collection.count()})")
        
        print("\n🔍 Testing collection access:")
        for mem_type in MemoryType:
            if mem_type in store.collections:
                count = store.collections[mem_type].count()
                print(f"  ✅ {mem_type}: {count} memories")
            else:
                print(f"  ❌ {mem_type}: NOT FOUND!")
        
        # Check the actual ChromaDB client collections
        print("\n🗂️ Raw ChromaDB collections:")
        all_collections = store.client.list_collections()
        for col in all_collections:
            print(f"  - {col.name}: {col.count()} items")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_collections()