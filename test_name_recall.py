#!/usr/bin/env python3
"""Test specifically for name recall functionality."""

import asyncio
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.memory import MemoryType  
from src.memory.manager import MemoryManager
from src.core.rag_system import RAGSystem
from utils.logger import setup_logger

logger = setup_logger(__name__)

async def test_name_recall():
    """Test the name recall functionality specifically."""
    print("👤 NAME RECALL TEST")
    print("=" * 20)
    
    try:
        # Create memory manager and RAG system
        print("📝 Creating memory and RAG systems...")
        memory_manager = MemoryManager()
        rag_system = RAGSystem(memory_manager)
        print("✅ Systems created")
        
        # Store personal name information
        print("\n📝 Storing name information...")
        name_memory_id = await memory_manager.remember(
            content="The user's name is STEPHAN and he is a software engineer.",
            memory_type=MemoryType.LONG_TERM,
            importance=1.0,
            metadata={"type": "personal_info", "category": "identity", "entity": "name"}
        )
        print(f"✅ Stored name info: {name_memory_id}")
        
        # Test direct memory recall with different thresholds
        print("\n🔍 Testing direct memory recall...")
        name_queries = [
            "What is my name?",
            "name",
            "my name",
            "STEPHAN",
            "user name"
        ]
        
        for query in name_queries:
            print(f"\n  Query: '{query}'")
            
            # Test with the new default threshold (0.3)
            memories = await memory_manager.recall(
                query=query,
                memory_types=[MemoryType.LONG_TERM],
                limit=5
            )
            
            if memories:
                print(f"    ✅ Found {len(memories)} memories (threshold=0.3)")
                for i, mem in enumerate(memories):
                    print(f"      {i+1}. Type: {mem.metadata.get('type', 'unknown')}")
                    print(f"         Content: {mem.content[:80]}...")
                    if "STEPHAN" in mem.content.upper():
                        print("         🎯 CONTAINS NAME!")
            else:
                print(f"    ❌ No memories found with default threshold")
        
        # Test RAG system
        print("\n🎯 Testing RAG system...")
        rag_memories = await rag_system.get_relevant_memories("What is my name?")
        
        if rag_memories:
            print(f"✅ RAG found {len(rag_memories)} memories")
            for i, mem in enumerate(rag_memories):
                print(f"  {i+1}. Type: {mem.metadata.get('type', 'unknown')}")
                print(f"     Content: {mem.content[:80]}...")
                if "STEPHAN" in mem.content.upper():
                    print("     🎯 RAG FOUND THE NAME!")
        else:
            print("❌ RAG found no memories")
        
        # Test the enhanced prompt building
        print("\n📝 Testing RAG context building...")
        context = await rag_system.retrieve_context("What is my name?")
        
        print(f"Context memories: {len(context.memories)}")
        print(f"Retrieval time: {context.retrieval_time_ms:.1f}ms")
        print(f"Reasoning: {context.reasoning}")
        
        if context.memories:
            enhanced_prompt = await rag_system.enhance_prompt(
                "What is my name?", 
                context
            )
            print(f"\nEnhanced prompt (first 200 chars):")
            print(enhanced_prompt[:200] + "...")
            
            if "STEPHAN" in enhanced_prompt.upper():
                print("🎯 ENHANCED PROMPT CONTAINS THE NAME!")
            else:
                print("❌ Enhanced prompt doesn't contain the name")
        
        print("\n🏁 Test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_name_recall())