#!/usr/bin/env python3
"""Test improved RAG system with better thresholds."""

import asyncio
from datetime import datetime

from src.core.chat_manager import ChatManager
from src.memory import MemoryType
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

async def test_improved_rag():
    """Test RAG with improved thresholds."""
    print("🚀 TESTING IMPROVED RAG SYSTEM")
    print("=" * 40)
    
    # Initialize chat manager
    chat_manager = ChatManager()
    
    # Create a session
    await chat_manager.create_session("RAG Test")
    
    # Test 1: Store personal information
    print("\n👤 Test 1: Storing personal information...")
    personal_info = "My name is STEPHAN and I work as a software engineer. I love programming in Python."
    
    await chat_manager.memory_manager.remember(
        content=personal_info,
        memory_type=MemoryType.LONG_TERM,
        importance=0.9,
        metadata={"type": "personal_info"}
    )
    print("✅ Stored personal information")
    
    # Test 2: Store file content
    print("\n📄 Test 2: Storing file content...")
    file_content = """
# calculator.py
def add(a, b):
    '''Add two numbers together.'''
    return a + b

def multiply(a, b):
    '''Multiply two numbers.'''
    return a * b

def calculate_compound_interest(principal, rate, time):
    '''Calculate compound interest.'''
    return principal * (1 + rate) ** time
"""
    
    await chat_manager.memory_manager.remember(
        content=f"File content from calculator.py:\n\n{file_content}",
        memory_type=MemoryType.SEMANTIC,
        importance=0.8,
        metadata={
            "source": "file_upload",
            "file_name": "calculator.py",
            "file_type": ".py"
        }
    )
    print("✅ Stored file content")
    
    # Test 3: Test RAG retrieval with improved thresholds
    print("\n🔍 Test 3: Testing RAG retrieval...")
    
    queries = [
        "What is my name?",
        "Tell me about the calculator functions",
        "How do I calculate compound interest?",
        "Show me the add function"
    ]
    
    for query in queries:
        print(f"\n📝 Query: {query}")
        
        # Test retrieval context
        rag_context = await chat_manager.rag_system.retrieve_context(query)
        print(f"📋 Retrieved {len(rag_context.memories)} memories")
        
        for memory in rag_context.memories:
            if memory.metadata and memory.metadata.get("source") == "file_upload":
                print(f"  📁 File: {memory.metadata.get('file_name', 'unknown')}")
            elif memory.metadata and memory.metadata.get("type") == "personal_info":
                print(f"  👤 Personal: {memory.content[:50]}...")
            else:
                print(f"  💭 Memory: {memory.content[:50]}...")
    
    # Test 4: Test actual RAG-enhanced responses (non-streaming first)
    print("\n💬 Test 4: Testing RAG-enhanced responses...")
    
    test_queries = [
        "What is my name?",
        "Show me the add function from the uploaded file"
    ]
    
    for query in test_queries:
        print(f"\n🤔 Query: {query}")
        print("🤖 Response: ", end="")
        
        try:
            # Test non-streaming first
            response = await chat_manager.send_message_with_rag(query, stream=False)
            print(response)
            
            # Check if response uses context appropriately
            if "stephan" in response.lower() and "name" in query.lower():
                print("✅ Personal info recalled correctly!")
            elif "add" in response.lower() and "function" in query.lower():
                print("✅ File content recalled correctly!")
            else:
                print("⚠️ Response may not be using RAG context optimally")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Test 5: Test streaming (if working)
    print("\n🎬 Test 5: Testing streaming RAG...")
    try:
        query = "Tell me about my programming work and the calculator file"
        print(f"🤔 Streaming Query: {query}")
        print("🤖 Streaming Response: ", end="", flush=True)
        
        response_parts = []
        async for chunk in chat_manager.send_message_with_rag(query, stream=True):
            print(chunk, end="", flush=True)
            response_parts.append(chunk)
        
        full_response = "".join(response_parts)
        print()
        
        if full_response:
            print("✅ Streaming RAG working!")
        else:
            print("⚠️ Streaming produced empty response")
            
    except Exception as e:
        print(f"❌ Streaming error: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Run the test."""
    try:
        await test_improved_rag()
        print("\n🎉 All tests completed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())