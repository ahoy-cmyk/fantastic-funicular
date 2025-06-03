#!/usr/bin/env python3
"""Simple test of clear functionality without UI components."""

import asyncio
import sys

sys.path.insert(0, "src")


async def test_clear_functionality():
    """Test clear functionality end-to-end."""
    print("🧪 TESTING CLEAR FUNCTIONALITY")
    print("=" * 50)

    try:
        from src.core.models import MessageRole
        from src.core.session_manager import SessionManager

        # Create session manager
        session_manager = SessionManager()
        await session_manager.start()
        print("✅ Session manager started")

        # Create test conversations
        conversation_ids = []
        for i in range(3):
            async with session_manager.create_session(f"Test Chat {i+1}") as session:
                await session.add_message(MessageRole.USER, f"Test message {i+1}")
                conversation_ids.append(session.conversation.id)
                print(f"✅ Created conversation {i+1}: {session.conversation.id}")

        # Verify conversations exist
        conversations_before = await session_manager.list_conversations()
        print(f"✅ Conversations before clear: {len(conversations_before)}")

        # Test clear operation
        cleared_count = await session_manager.clear_all_conversations()
        print(f"✅ Clear operation returned: {cleared_count}")

        # Verify conversations are cleared
        conversations_after = await session_manager.list_conversations()
        print(f"✅ Conversations after clear: {len(conversations_after)}")

        # Check if conversations are marked as deleted vs removed
        from src.core.models import Conversation

        db_session = session_manager.SessionFactory()
        try:
            all_convs = db_session.query(Conversation).all()
            deleted_convs = [c for c in all_convs if str(c.status) == "deleted"]
            print(f"✅ Total conversations in DB: {len(all_convs)}")
            print(f"✅ Deleted conversations: {len(deleted_convs)}")
        finally:
            db_session.close()

        await session_manager.stop()
        print("✅ Session manager stopped")

        # Test results
        success = (
            cleared_count >= 3  # Should have cleared at least our test conversations
            and len(conversations_after) == 0  # No active conversations should remain
        )

        return success

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Run the test."""
    success = await test_clear_functionality()

    print("\n" + "=" * 50)
    print("📊 TEST RESULTS")
    print("=" * 50)

    if success:
        print("✅ CLEAR FUNCTIONALITY IS WORKING!")
        print("The API-level clear operation works correctly.")
        print("If users report issues, it may be a UI interaction problem.")
    else:
        print("❌ CLEAR FUNCTIONALITY HAS ISSUES!")

    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
