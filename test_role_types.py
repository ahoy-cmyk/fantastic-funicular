#!/usr/bin/env python3
"""Test to check what types roles are in session messages."""

import asyncio
from src.core.chat_manager import ChatManager
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

async def test_role_types():
    """Test to check role types in session messages."""
    print("🔍 TESTING ROLE TYPES")
    print("=" * 25)
    
    chat_manager = ChatManager()
    await chat_manager.create_session("Role Test")
    
    # Add a few messages
    print("📝 Adding messages...")
    await chat_manager.send_message_with_rag("Hello", stream=False)
    await chat_manager.send_message_with_rag("How are you?", stream=False)
    
    # Check the types of roles in session
    print("\n🔍 Checking role types...")
    if chat_manager.current_session:
        messages = await chat_manager.current_session.get_messages()
        
        for i, msg in enumerate(messages):
            print(f"Message {i+1}:")
            print(f"  Role: {msg.role}")
            print(f"  Role type: {type(msg.role)}")
            print(f"  Has .value: {hasattr(msg.role, 'value')}")
            if hasattr(msg.role, 'value'):
                print(f"  Role.value: {msg.role.value}")
            print(f"  str(role): {str(msg.role)}")
            print()

if __name__ == "__main__":
    asyncio.run(test_role_types())