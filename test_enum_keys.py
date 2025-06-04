#!/usr/bin/env python3
"""Test enum key handling in collections dictionary."""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from memory import MemoryType
from memory.vector_store import VectorMemoryStore

def test_enum_keys():
    """Test enum key handling."""
    print("🔑 ENUM KEYS TEST")
    print("=" * 18)
    
    try:
        # Create vector store
        store = VectorMemoryStore()
        print("✅ Vector store created")
        
        print("\n🔍 Analyzing collections dictionary:")
        print(f"Dictionary type: {type(store.collections)}")
        print(f"Dictionary length: {len(store.collections)}")
        
        print("\n🗝️ Dictionary keys analysis:")
        for key in store.collections.keys():
            print(f"  Key: {key}")
            print(f"  Key type: {type(key)}")
            print(f"  Key repr: {repr(key)}")
            print(f"  Key == MemoryType.LONG_TERM: {key == MemoryType.LONG_TERM}")
            print(f"  Key is MemoryType.LONG_TERM: {key is MemoryType.LONG_TERM}")
            print()
        
        print("🧪 Testing direct lookups:")
        test_key = MemoryType.LONG_TERM
        print(f"Looking for: {test_key} (type: {type(test_key)})")
        
        # Test different ways to check if key exists
        print(f"  test_key in store.collections: {test_key in store.collections}")
        print(f"  str(test_key) in [str(k) for k in store.collections]: {str(test_key) in [str(k) for k in store.collections]}")
        
        # Try direct access
        try:
            collection = store.collections[test_key]
            print(f"  Direct access works: {collection.name}")
        except KeyError as e:
            print(f"  Direct access failed: {e}")
        
        # Test with iteration
        print("\n🔄 Testing with iteration:")
        for key, collection in store.collections.items():
            if key == MemoryType.LONG_TERM:
                print(f"  Found via iteration: {key} -> {collection.name}")
                break
        else:
            print("  Not found via iteration")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enum_keys()