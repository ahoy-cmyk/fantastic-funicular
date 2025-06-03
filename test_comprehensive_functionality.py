#!/usr/bin/env python3
"""Comprehensive test of all enhanced chat functionality."""

import asyncio
import sys

sys.path.insert(0, "src")


async def test_conversation_management():
    """Test conversation loading and management."""
    print("🗨️ TESTING CONVERSATION MANAGEMENT")
    print("=" * 50)

    try:
        from src.core.chat_manager import ChatManager
        from src.core.models import MessageRole

        chat_manager = ChatManager()

        # Create and test conversation
        print("Creating conversation...")
        await chat_manager.create_session("Test Chat")
        print(f"✅ Created session: {chat_manager.current_session.id}")

        # Add messages
        print("Adding messages...")
        await chat_manager.current_session.add_message(
            MessageRole.USER, "Hello, my name is Stephan"
        )
        await chat_manager.current_session.add_message(
            MessageRole.ASSISTANT, "Nice to meet you, Stephan!"
        )

        # Get conversation ID
        conversation_id = chat_manager.current_session.conversation.id
        print(f"✅ Added messages to conversation: {conversation_id}")

        # Test loading the conversation
        print("Reloading conversation...")
        await chat_manager.load_session(conversation_id)
        messages = await chat_manager.current_session.get_messages()
        print(f"✅ Reloaded conversation with {len(messages)} messages")

        for msg in messages:
            role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            print(f"  - {role}: {msg.content}")

        return len(messages) == 2

    except Exception as e:
        print(f"❌ Conversation management test failed: {e}")
        return False


async def test_memory_functionality():
    """Test memory storage and recall."""
    print("\n🧠 TESTING MEMORY FUNCTIONALITY")
    print("=" * 50)

    try:
        from src.memory import MemoryType
        from src.memory.safe_operations import create_safe_memory_manager

        def error_callback(op, error):
            print(f"Memory error: {op} - {error}")

        safe_memory = create_safe_memory_manager(error_callback)

        # Store memory about Stephan
        print("Storing memory about Stephan...")
        await safe_memory.safe_remember(
            content="The user's name is Stephan and he is testing the memory system",
            memory_type=MemoryType.LONG_TERM,
            metadata={"test": "comprehensive"},
        )
        print("✅ Memory stored")

        # Test recall with different queries
        print("Testing memory recall...")
        queries = ["Stephan", "user name", "testing memory"]

        for query in queries:
            memories = await safe_memory.safe_recall(query, threshold=0.3, limit=3)
            print(f"Query '{query}': {len(memories)} memories found")
            if memories:
                for mem in memories[:1]:  # Show first memory
                    print(f"  - {mem.content[:60]}...")

        # Test with exact query
        memories = await safe_memory.safe_recall("Stephan", threshold=0.3, limit=5)
        return len(memories) > 0

    except Exception as e:
        print(f"❌ Memory functionality test failed: {e}")
        return False


async def test_provider_detection():
    """Test provider configuration."""
    print("\n⚙️ TESTING PROVIDER DETECTION")
    print("=" * 50)

    try:
        from src.core.chat_manager import ChatManager

        chat_manager = ChatManager()
        providers = chat_manager.get_available_providers()
        print(f"✅ Available providers: {providers}")

        # Test provider information
        for provider in providers:
            available = chat_manager.is_provider_available(provider)
            print(f"  - {provider}: {'Available' if available else 'Not available'}")

        return len(providers) > 0

    except Exception as e:
        print(f"❌ Provider detection test failed: {e}")
        return False


async def test_session_clearing():
    """Test clearing conversations."""
    print("\n🗑️ TESTING SESSION CLEARING")
    print("=" * 50)

    try:
        from src.core.models import MessageRole
        from src.core.session_manager import SessionManager

        session_manager = SessionManager()
        await session_manager.start()

        # Create test conversations
        conversation_ids = []
        for i in range(3):
            async with session_manager.create_session(f"Test Clear {i+1}") as session:
                await session.add_message(MessageRole.USER, f"Test message {i+1}")
                conversation_ids.append(session.conversation.id)

        print(f"✅ Created {len(conversation_ids)} test conversations")

        # List conversations before clear
        before_count = len(await session_manager.list_conversations())
        print(f"Conversations before clear: {before_count}")

        # Clear all conversations
        cleared_count = await session_manager.clear_all_conversations()
        print(f"✅ Cleared {cleared_count} conversations")

        # Verify they're gone
        after_count = len(await session_manager.list_conversations())
        print(f"Conversations after clear: {after_count}")

        await session_manager.stop()
        return after_count == 0

    except Exception as e:
        print(f"❌ Session clearing test failed: {e}")
        return False


async def main():
    """Run all comprehensive tests."""
    print("🚀 COMPREHENSIVE ENHANCED CHAT FUNCTIONALITY TEST")
    print("=" * 60)

    # Run all tests
    conv_test = await test_conversation_management()
    memory_test = await test_memory_functionality()
    provider_test = await test_provider_detection()
    clear_test = await test_session_clearing()

    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE TEST RESULTS")
    print("=" * 60)

    print(f"Conversation Management: {'✅ WORKING' if conv_test else '❌ BROKEN'}")
    print(f"Memory Functionality: {'✅ WORKING' if memory_test else '❌ BROKEN'}")
    print(f"Provider Detection: {'✅ WORKING' if provider_test else '❌ BROKEN'}")
    print(f"Session Clearing: {'✅ WORKING' if clear_test else '❌ BROKEN'}")

    all_working = conv_test and memory_test and provider_test and clear_test

    if all_working:
        print("\n🎉 ALL FUNCTIONALITY WORKING PERFECTLY!")
        print("🚀 Enhanced chat system is ready for production!")
    else:
        print("\n⚠️ SOME FUNCTIONALITY NEEDS ATTENTION!")

    return all_working


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
