#!/usr/bin/env python3
"""Test enum import paths to find the issue."""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_enum_imports():
    """Test different enum import paths."""
    print("📦 ENUM IMPORT TEST")
    print("=" * 20)
    
    # Test different import paths
    print("🔍 Testing different import paths:")
    
    # Import 1: From memory module
    from memory import MemoryType as MemoryType1
    print(f"1. from memory import MemoryType: {MemoryType1}")
    print(f"   Module: {MemoryType1.__module__}")
    print(f"   LONG_TERM: {MemoryType1.LONG_TERM}")
    
    # Import 2: From src.memory
    from src.memory import MemoryType as MemoryType2
    print(f"\n2. from src.memory import MemoryType: {MemoryType2}")
    print(f"   Module: {MemoryType2.__module__}")
    print(f"   LONG_TERM: {MemoryType2.LONG_TERM}")
    
    # Test equality
    print(f"\n🔗 Equality tests:")
    print(f"MemoryType1 is MemoryType2: {MemoryType1 is MemoryType2}")
    print(f"MemoryType1.LONG_TERM == MemoryType2.LONG_TERM: {MemoryType1.LONG_TERM == MemoryType2.LONG_TERM}")
    print(f"MemoryType1.LONG_TERM is MemoryType2.LONG_TERM: {MemoryType1.LONG_TERM is MemoryType2.LONG_TERM}")
    
    # Check vector store import
    print(f"\n🗂️ Vector store import analysis:")
    from memory.vector_store import VectorMemoryStore
    store = VectorMemoryStore()
    
    # Get a key from the collections dict
    sample_key = next(iter(store.collections.keys()))
    print(f"Sample key from collections: {sample_key}")
    print(f"Sample key module: {sample_key.__class__.__module__}")
    print(f"Sample key type: {type(sample_key)}")
    
    # Compare with our imports
    print(f"\nComparisons with sample key:")
    print(f"sample_key == MemoryType1.LONG_TERM: {sample_key == MemoryType1.LONG_TERM}")
    print(f"sample_key == MemoryType2.LONG_TERM: {sample_key == MemoryType2.LONG_TERM}")
    
    # Test using the same import as vector store
    print(f"\n🎯 Testing with vector store's import:")
    # Let's see what the vector store is importing
    import inspect
    store_source = inspect.getsource(VectorMemoryStore.__init__)
    print("Vector store init source:")
    print(store_source[:200] + "...")

if __name__ == "__main__":
    test_enum_imports()