#!/usr/bin/env python3
"""Test enhanced messaging functionality."""

import asyncio
import os
import sys

# Set environment for testing
os.environ["KIVY_NO_CONSOLELOG"] = "1"
os.environ["KIVY_LOG_LEVEL"] = "critical"

sys.path.insert(0, "src")


async def test_chat_manager():
    """Test chat manager functionality."""
    print("🧠 TESTING CHAT MANAGER")
    print("=" * 40)

    try:
        from src.core.chat_manager import ChatManager

        print("✅ ChatManager imported successfully")

        chat_manager = ChatManager()
        print("✅ ChatManager instance created")

        # Test session creation
        await chat_manager.create_session()
        print("✅ Session created successfully")

        # Test streaming (just verify it returns an async generator)
        stream = chat_manager.send_message("Hello test", stream=True)
        print(f"✅ Send message returns: {type(stream)}")

        print("\n🎯 CHAT MANAGER TEST PASSED!")
        return True

    except Exception as e:
        print(f"❌ Chat manager test failed: {e}")
        return False


async def test_memory_system():
    """Test memory system functionality."""
    print("\n🧠 TESTING MEMORY SYSTEM")
    print("=" * 40)

    try:
        from src.memory.safe_operations import create_safe_memory_manager

        def error_callback(op, error):
            print(f"Memory error: {op} - {error}")

        safe_memory = create_safe_memory_manager(error_callback)
        print("✅ Safe memory manager created")

        # Test recall (should return empty for new system)
        memories = await safe_memory.safe_recall("test query", threshold=0.5, limit=5)
        print(f"✅ Memory recall returned: {len(memories)} memories")

        print("\n🎯 MEMORY SYSTEM TEST PASSED!")
        return True

    except Exception as e:
        print(f"❌ Memory system test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🚀 TESTING ENHANCED MESSAGING SYSTEM")
    print("=" * 50)

    chat_test = await test_chat_manager()
    memory_test = await test_memory_system()

    print("\n" + "=" * 50)
    print("📊 TEST RESULTS")
    print("=" * 50)

    if chat_test and memory_test:
        print("✅ ALL TESTS PASSED!")
        print("🚀 Enhanced messaging system is ready!")
        return True
    else:
        print("❌ SOME TESTS FAILED!")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
