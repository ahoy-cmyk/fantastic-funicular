#!/usr/bin/env python3
"""Debug clear chats functionality specifically."""

import asyncio
import sys

sys.path.insert(0, "src")


async def debug_clear_chats_ui():
    """Debug the clear chats UI flow."""
    print("🗑️ DEBUGGING CLEAR CHATS UI FLOW")
    print("=" * 50)

    try:
        # Import the enhanced chat screen
        from src.core.models import MessageRole
        from src.core.session_manager import SessionManager
        from src.gui.screens.enhanced_chat_screen import EnhancedChatScreen

        # Create session manager and add test conversations
        session_manager = SessionManager()
        await session_manager.start()

        print("Creating test conversations...")
        conversation_ids = []
        for i in range(3):
            async with session_manager.create_session(f"Test Conversation {i+1}") as session:
                await session.add_message(MessageRole.USER, f"Test message {i+1}")
                conversation_ids.append(session.conversation.id)
                print(f"Created conversation {i+1}: {session.conversation.id}")

        # Check conversations before clear
        conversations_before = await session_manager.list_conversations()
        print(f"Conversations before clear: {len(conversations_before)}")

        # Test the clear method directly
        print("Testing clear_all_conversations method...")
        cleared_count = await session_manager.clear_all_conversations()
        print(f"Cleared count returned: {cleared_count}")

        # Check conversations after clear
        conversations_after = await session_manager.list_conversations()
        print(f"Conversations after clear: {len(conversations_after)}")

        # Test if the enhanced chat screen method works
        print("Testing enhanced chat screen clear method...")
        chat_screen = EnhancedChatScreen()

        # Create more test conversations
        for i in range(2):
            async with session_manager.create_session(f"UI Test {i+1}") as session:
                await session.add_message(MessageRole.USER, f"UI test message {i+1}")

        conversations_before_ui = await session_manager.list_conversations()
        print(f"Conversations before UI clear: {len(conversations_before_ui)}")

        # Test the UI clear method
        await chat_screen._clear_all_conversations_async()

        conversations_after_ui = await session_manager.list_conversations()
        print(f"Conversations after UI clear: {len(conversations_after_ui)}")

        await session_manager.stop()

        return len(conversations_after_ui) == 0

    except Exception as e:
        print(f"❌ Clear chats debug failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def debug_conversation_status():
    """Debug conversation status handling."""
    print("\n📋 DEBUGGING CONVERSATION STATUS")
    print("=" * 50)

    try:
        from src.core.models import MessageRole
        from src.core.session_manager import SessionManager

        session_manager = SessionManager()
        await session_manager.start()

        # Create a test conversation
        async with session_manager.create_session("Status Test") as session:
            await session.add_message(MessageRole.USER, "Status test message")
            conversation_id = session.conversation.id
            print(f"Created conversation: {conversation_id}")

        # Check initial status
        conversations = await session_manager.list_conversations()
        test_conv = next((c for c in conversations if c["id"] == conversation_id), None)
        if test_conv:
            print(f"Initial status: {test_conv.get('status', 'unknown')}")

        # Clear and check status
        cleared_count = await session_manager.clear_all_conversations()
        print(f"Clear operation cleared: {cleared_count}")

        # Check status after clear (including deleted ones)
        all_conversations = await session_manager.list_conversations()
        print(f"Active conversations after clear: {len(all_conversations)}")

        # Try to get all conversations including deleted
        from src.core.models import Conversation

        db_session = session_manager.SessionFactory()
        try:
            all_convs_raw = db_session.query(Conversation).all()
            print(f"Total conversations in DB: {len(all_convs_raw)}")
            for conv in all_convs_raw[-3:]:  # Show last 3
                print(f"  - {conv.id}: {conv.status}")
        finally:
            db_session.close()

        await session_manager.stop()
        return True

    except Exception as e:
        print(f"❌ Status debug failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Run clear chats debugging."""
    print("🚀 DEBUGGING CLEAR CHATS FUNCTIONALITY")
    print("=" * 60)

    ui_test = await debug_clear_chats_ui()
    status_test = await debug_conversation_status()

    print("\n" + "=" * 60)
    print("📊 CLEAR CHATS DEBUG RESULTS")
    print("=" * 60)

    print(f"UI Clear Flow: {'✅ WORKING' if ui_test else '❌ BROKEN'}")
    print(f"Status Handling: {'✅ WORKING' if status_test else '❌ BROKEN'}")

    if ui_test and status_test:
        print("\n✅ CLEAR CHATS FUNCTIONALITY IS WORKING!")
    else:
        print("\n❌ CLEAR CHATS HAS ISSUES!")

    return ui_test and status_test


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
