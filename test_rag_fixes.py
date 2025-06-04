#!/usr/bin/env python3
"""Test script to verify RAG fixes and model display."""

import asyncio
import os
import tempfile
from datetime import datetime

from src.core.chat_manager import ChatManager
from src.memory import MemoryType
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

async def test_rag_functionality():
    """Test that RAG properly uses uploaded file content."""
    print("🧪 Testing RAG functionality with uploaded files...")
    
    # Initialize chat manager
    chat_manager = ChatManager()
    
    # Discover models first
    print("🔍 Discovering models...")
    await chat_manager.discover_models()
    
    # Check current model
    current_model = f"{chat_manager.current_provider}:{chat_manager.current_model}"
    print(f"📱 Current model: {current_model}")
    
    # Upload a test file to memory
    print("📤 Uploading test file content...")
    test_content = """
def calculate_area(length, width):
    '''Calculate the area of a rectangle.'''
    return length * width

def calculate_volume(length, width, height):
    '''Calculate the volume of a rectangular prism.'''
    return length * width * height

# Example usage
room_area = calculate_area(12, 10)
print(f"Room area: {room_area} square feet")
"""
    
    # Store file content in memory
    memory_id = await chat_manager.memory_manager.remember(
        content=f"File content from geometry_calculator.py:\n\n{test_content}",
        memory_type=MemoryType.SEMANTIC,
        importance=0.8,
        metadata={
            "source": "file_upload",
            "file_name": "geometry_calculator.py",
            "file_type": ".py",
            "file_size": len(test_content),
            "upload_date": datetime.now().isoformat()
        }
    )
    
    print(f"✅ Stored file in memory: {memory_id}")
    
    # Test RAG retrieval
    print("\n🔍 Testing RAG retrieval...")
    rag_context = await chat_manager.rag_system.retrieve_context(
        query="show me the area calculation function"
    )
    
    print(f"📋 Retrieved {len(rag_context.memories)} memories")
    for i, memory in enumerate(rag_context.memories):
        print(f"  {i+1}. {memory.content[:100]}...")
        if memory.metadata:
            print(f"     📁 Source: {memory.metadata.get('source', 'unknown')}")
            if memory.metadata.get('file_name'):
                print(f"     📄 File: {memory.metadata['file_name']}")
    
    # Test RAG-enhanced messaging
    print("\n💬 Testing RAG-enhanced messaging...")
    
    # Create a session first
    await chat_manager.create_session("RAG Test Session")
    
    # Test question about uploaded file
    query = "Explain the calculate_area function from the uploaded Python file"
    
    print(f"🤔 Query: {query}")
    print("🤖 Response: ", end="", flush=True)
    
    response_parts = []
    try:
        async for chunk in chat_manager.send_message_with_rag(query, stream=True):
            print(chunk, end="", flush=True)
            response_parts.append(chunk)
        
        full_response = "".join(response_parts)
        print("\n")
        
        # Check if response references the uploaded file
        if any(keyword in full_response.lower() for keyword in ["calculate_area", "rectangle", "geometry", "upload", "file"]):
            print("✅ RAG successfully used uploaded file content!")
        else:
            print("❌ RAG did not use uploaded file content")
            print(f"Response: {full_response[:200]}...")
            
    except Exception as e:
        print(f"\n❌ Error during RAG messaging: {e}")
    
    # Test model info display
    print("\n📱 Testing model info display...")
    provider_model_info = f"{chat_manager.current_provider}:{chat_manager.current_model}"
    print(f"Current model info: {provider_model_info}")
    
    # Test if RAG is enabled
    rag_enabled = getattr(chat_manager, 'rag_enabled', False)
    print(f"RAG status: {'Enabled' if rag_enabled else 'Disabled'}")
    
    return True

async def main():
    """Run all tests."""
    print("🚀 TESTING RAG FIXES AND MODEL DISPLAY")
    print("=" * 50)
    
    try:
        await test_rag_functionality()
        print("\n🎉 All tests completed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())