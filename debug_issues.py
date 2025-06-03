#!/usr/bin/env python3
"""Debug conversation loading, clearing, and memory issues."""

import asyncio
import sys

sys.path.insert(0, "src")


async def debug_conversation_loading():
    """Debug conversation loading issues."""
    print("🔍 DEBUGGING CONVERSATION LOADING")
    print("=" * 50)

    try:
        from src.core.models import MessageRole
        from src.core.session_manager import SessionManager

        session_manager = SessionManager()
        await session_manager.start()

        # Create a test conversation with messages
        print("Creating test conversation...")
        async with session_manager.create_session("Test Conversation") as session:
            conversation_id = session.conversation.id
            print(f"Created conversation: {conversation_id}")

            # Add test messages
            print("Adding test messages...")
            msg1 = await session.add_message(MessageRole.USER, "Hello, my name is Stephan")
            msg2 = await session.add_message(MessageRole.ASSISTANT, "Nice to meet you, Stephan!")
            print(f"Added messages: {msg1.id}, {msg2.id}")

        # Reload session
        print("Reloading session...")
        async with session_manager.load_session(conversation_id) as session:
            print(f"Reloaded session: {session.id if session else 'None'}")

            if session:
                messages = await session.get_messages()
                print(f"Found {len(messages)} messages after reload:")
                for msg in messages:
                    role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
                    print(f"  - {role}: {msg.content}")

        # Test listing conversations
        print("\nListing all conversations...")
        conversations = await session_manager.list_conversations()
        print(f"Found {len(conversations)} conversations:")
        for conv in conversations:
            print(f"  - {conv.get('id')}: {conv.get('title')}")

        await session_manager.stop()
        return True

    except Exception as e:
        print(f"❌ Conversation loading debug failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def debug_clear_conversations():
    """Debug clear conversations functionality."""
    print("\n🔍 DEBUGGING CLEAR CONVERSATIONS")
    print("=" * 50)

    try:
        from src.core.models import MessageRole
        from src.core.session_manager import SessionManager

        session_manager = SessionManager()
        await session_manager.start()

        # Create multiple test conversations
        print("Creating multiple test conversations...")
        for i in range(3):
            async with session_manager.create_session(f"Test Conversation {i+1}") as session:
                await session.add_message(MessageRole.USER, f"Test message {i+1}")
                print(f"Created conversation {i+1}: {session.conversation.id}")

        # List conversations before clear
        conversations_before = await session_manager.list_conversations()
        print(f"Conversations before clear: {len(conversations_before)}")

        # Clear all conversations
        print("Clearing all conversations...")
        count = await session_manager.clear_all_conversations()
        print(f"Cleared {count} conversations")

        # List conversations after clear
        conversations_after = await session_manager.list_conversations()
        print(f"Conversations after clear: {len(conversations_after)}")

        await session_manager.stop()
        return len(conversations_after) == 0

    except Exception as e:
        print(f"❌ Clear conversations debug failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def debug_memory_persistence():
    """Debug memory persistence issues."""
    print("\n🔍 DEBUGGING MEMORY PERSISTENCE")
    print("=" * 50)

    try:
        from src.memory import MemoryType
        from src.memory.manager import MemoryManager

        memory_manager = MemoryManager()

        # Store a memory
        print("Storing test memory...")
        memory_id = await memory_manager.remember(
            content="My name is Stephan and I am testing the memory system",
            memory_type=MemoryType.LONG_TERM,
            importance=0.8,
            metadata={"test": "debug"},
        )
        print(f"Stored memory: {memory_id}")

        # Recall the memory with lower threshold
        print("Recalling memories with different thresholds...")
        for threshold in [0.9, 0.7, 0.5, 0.3, 0.1]:
            recalled = await memory_manager.recall("Stephan", threshold=threshold, limit=5)
            print(f"Threshold {threshold}: {len(recalled)} memories")
            if recalled:
                for mem in recalled:
                    print(f"  - {mem.content[:50]}...")
                break

        # Test with exact content
        print("Testing recall with exact content...")
        recalled_exact = await memory_manager.recall("My name is Stephan", threshold=0.1, limit=5)
        print(f"Exact content recall: {len(recalled_exact)} memories")
        for mem in recalled_exact:
            print(f"  - {mem.content[:50]}...")

        # Test stats
        print("Getting memory stats...")
        stats = await memory_manager.get_memory_stats()
        print(f"Memory stats: {stats}")

        return len(recalled_exact) > 0

    except Exception as e:
        print(f"❌ Memory persistence debug failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Run all debugging tests."""
    print("🚀 DEBUGGING ENHANCED CHAT ISSUES")
    print("=" * 60)

    conv_test = await debug_conversation_loading()
    clear_test = await debug_clear_conversations()
    memory_test = await debug_memory_persistence()

    print("\n" + "=" * 60)
    print("📊 DEBUG RESULTS")
    print("=" * 60)

    print(f"Conversation Loading: {'✅ WORKING' if conv_test else '❌ BROKEN'}")
    print(f"Clear Conversations: {'✅ WORKING' if clear_test else '❌ BROKEN'}")
    print(f"Memory Persistence: {'✅ WORKING' if memory_test else '❌ BROKEN'}")

    if conv_test and clear_test and memory_test:
        print("\n✅ ALL SYSTEMS WORKING!")
    else:
        print("\n❌ ISSUES FOUND - NEED FIXES!")

    return conv_test and clear_test and memory_test


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
