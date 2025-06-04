#!/usr/bin/env python3
"""Test the enhanced prompt to see exactly what the LLM is receiving."""

import asyncio
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.chat_manager import ChatManager
from src.memory import MemoryType
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

async def test_enhanced_prompt():
    """Test what the enhanced prompt actually contains."""
    print("📝 ENHANCED PROMPT TEST")
    print("=" * 25)
    
    try:
        # Create chat manager
        chat_manager = ChatManager()
        await chat_manager.create_session("Prompt Test")
        print("✅ Chat manager created")
        
        # Store personal info
        print("\n📝 Storing personal information...")
        await chat_manager.memory_manager.remember(
            content="The user's name is STEPHAN. He is a software engineer who loves Python.",
            memory_type=MemoryType.LONG_TERM,
            importance=1.0,
            metadata={"type": "personal_info", "category": "identity"}
        )
        print("✅ Personal info stored")
        
        # Get RAG context
        print("\n🎯 Getting RAG context...")
        context = await chat_manager.rag_system.retrieve_context("What is my name?")
        print(f"Context has {len(context.memories)} memories")
        
        # Show the enhanced prompt
        print("\n📋 Enhanced prompt (full):")
        enhanced_prompt = await chat_manager.rag_system.enhance_prompt(
            "What is my name?", 
            context
        )
        
        print("="*80)
        print(enhanced_prompt)
        print("="*80)
        
        # Also test with system prompt
        print("\n🤖 Enhanced prompt with system prompt:")
        system_prompt = "You are a helpful assistant with access to relevant memories about the user."
        
        enhanced_with_system = await chat_manager.rag_system.enhance_prompt(
            "What is my name?", 
            context,
            system_prompt
        )
        
        print("="*80)
        print(enhanced_with_system)
        print("="*80)
        
        # Check what messages are actually sent to the LLM
        print("\n💬 Testing message building...")
        enhanced_messages = await chat_manager._build_rag_enhanced_messages("What is my name?", context)
        
        print(f"Number of messages to LLM: {len(enhanced_messages)}")
        for i, msg in enumerate(enhanced_messages):
            print(f"\nMessage {i+1} ({msg.role}):")
            print("---")
            print(msg.content[:500] + ("..." if len(msg.content) > 500 else ""))
            print("---")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_enhanced_prompt())