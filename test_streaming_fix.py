#!/usr/bin/env python3
"""Test streaming message functionality."""

import asyncio
import sys

sys.path.insert(0, "src")


async def test_streaming_messages():
    """Test multiple consecutive streaming messages."""
    print("🔄 TESTING STREAMING MESSAGE FIX")
    print("=" * 50)

    try:
        # Import with warnings suppressed
        from src.core.chat_manager import ChatManager

        # Create chat manager
        chat_manager = ChatManager()
        print("✅ Chat manager created")

        # Create a session
        await chat_manager.create_session("Streaming Test")
        print("✅ Session created")

        # Test first message (streaming)
        print("\n📤 Testing first streaming message...")
        response_chunks = []
        message_generator = chat_manager.send_message("Hello, how are you?", stream=True)

        # Verify we get an async generator
        if hasattr(message_generator, "__aiter__"):
            print("✅ Got async generator for streaming")

            async for chunk in message_generator:
                response_chunks.append(chunk)
                if len(response_chunks) <= 3:  # Show first few chunks
                    print(f"📦 Chunk {len(response_chunks)}: '{chunk[:20]}...'")
                if len(response_chunks) >= 5:  # Don't wait for full response
                    print("✅ Streaming working, stopping early...")
                    break
        else:
            print("❌ Did not get async generator")
            return False

        print(f"✅ First message: Got {len(response_chunks)} chunks")

        # Test follow-up message (streaming)
        print("\n📤 Testing follow-up streaming message...")
        followup_chunks = []
        followup_generator = chat_manager.send_message("Can you tell me a joke?", stream=True)

        if hasattr(followup_generator, "__aiter__"):
            print("✅ Got async generator for follow-up streaming")

            async for chunk in followup_generator:
                followup_chunks.append(chunk)
                if len(followup_chunks) <= 3:
                    print(f"📦 Follow-up chunk {len(followup_chunks)}: '{chunk[:20]}...'")
                if len(followup_chunks) >= 5:
                    print("✅ Follow-up streaming working, stopping early...")
                    break
        else:
            print("❌ Follow-up did not get async generator")
            return False

        print(f"✅ Follow-up message: Got {len(followup_chunks)} chunks")

        if chat_manager.current_session:
            await chat_manager.close_session()
        print("✅ Chat manager session closed")

        # Success if both messages streamed properly
        success = len(response_chunks) > 0 and len(followup_chunks) > 0
        return success

    except Exception as e:
        print(f"❌ Streaming test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Run streaming tests."""
    success = await test_streaming_messages()

    print("\n" + "=" * 50)
    print("📊 STREAMING TEST RESULTS")
    print("=" * 50)

    if success:
        print("✅ STREAMING FIX IS WORKING!")
        print("Both first messages and follow-ups now stream correctly.")
        print("The async generator issue has been resolved.")
    else:
        print("❌ STREAMING STILL HAS ISSUES!")

    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
