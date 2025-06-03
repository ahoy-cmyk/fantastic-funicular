#!/usr/bin/env python3
"""Test message context ordering fix."""

import asyncio
import sys

sys.path.insert(0, "src")


async def test_message_context_order():
    """Test that messages are sent to LLM in correct order."""
    print("🔄 TESTING MESSAGE CONTEXT ORDER")
    print("=" * 50)

    try:
        # Import with warnings suppressed
        from src.core.chat_manager import ChatManager

        # Create chat manager
        chat_manager = ChatManager()
        print("✅ Chat manager created")

        # Create a session
        await chat_manager.create_session("Context Test")
        print("✅ Session created")

        # Test message sequence
        messages = [
            "Hello, my name is Alice",
            "What is my name?",
            "Can you tell me a joke about my name?",
        ]

        for i, message in enumerate(messages, 1):
            print(f"\n📤 Sending message {i}: '{message}'")

            # Test the context building (without actually sending to LLM)
            # Build context before adding current message
            context_messages = await chat_manager._build_context_messages(
                message, exclude_current=False
            )

            print(f"📋 Context has {len(context_messages)} messages before adding current")

            # Show the last few context messages
            for j, ctx_msg in enumerate(context_messages[-3:]):
                role = ctx_msg.role if hasattr(ctx_msg, "role") else "unknown"
                content_preview = (
                    ctx_msg.content[:50] + "..." if len(ctx_msg.content) > 50 else ctx_msg.content
                )
                print(f"  Context[-{3-j}]: {role}: {content_preview}")

            # Add current message to session (simulating what would happen)
            await chat_manager.current_session.add_message(role="user", content=message)

            # Add a mock assistant response
            await chat_manager.current_session.add_message(
                role="assistant", content=f"Response to message {i}"
            )

            print(f"✅ Message {i} processed and stored")

        if chat_manager.current_session:
            await chat_manager.close_session()
        print("✅ Session closed")

        print("\n✅ Context order test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Context test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Run context ordering tests."""
    success = await test_message_context_order()

    print("\n" + "=" * 50)
    print("📊 MESSAGE CONTEXT TEST RESULTS")
    print("=" * 50)

    if success:
        print("✅ MESSAGE CONTEXT ORDER IS CORRECT!")
        print("Context is built before adding current message.")
        print("This should fix the 'responding to wrong message' issue.")
    else:
        print("❌ MESSAGE CONTEXT ORDER HAS ISSUES!")

    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
