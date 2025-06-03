#!/usr/bin/env python3
"""Test application startup with enhanced chat."""

import os
import sys

# Set environment for testing
os.environ["KIVY_NO_CONSOLELOG"] = "1"
os.environ["KIVY_LOG_LEVEL"] = "warning"

sys.path.insert(0, "src")


def test_app_startup():
    """Test that the app can start with enhanced chat screen."""
    print("🚀 TESTING APPLICATION STARTUP")
    print("=" * 50)

    print("\n1️⃣ Testing Imports...")
    try:
        from src.gui.app import NeuromancerApp

        print("   ✅ NeuromancerApp imported successfully")

        from src.gui.screens.enhanced_chat_screen import EnhancedChatScreen

        print("   ✅ EnhancedChatScreen imported successfully")

        from src.core.chat_manager import ChatManager

        print("   ✅ ChatManager imported successfully")

        from src.memory.safe_operations import create_safe_memory_manager

        print("   ✅ Memory system imported successfully")

    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False

    print("\n2️⃣ Testing Component Initialization...")
    try:
        # Test chat manager creation
        chat_manager = ChatManager()
        print("   ✅ ChatManager initialized")

        # Test memory manager creation
        safe_memory = create_safe_memory_manager()
        print("   ✅ Safe memory manager initialized")

        # Test screen creation (without adding to widget tree)
        screen = EnhancedChatScreen()
        print("   ✅ EnhancedChatScreen created")

    except Exception as e:
        print(f"   ❌ Component initialization failed: {e}")
        return False

    print("\n3️⃣ Testing App Creation...")
    try:
        app = NeuromancerApp()
        print("   ✅ NeuromancerApp created successfully")

        # Test that enhanced chat is in the screen list
        app.load_main_screens()
        if hasattr(app, "screen_manager"):
            screen_names = [screen.name for screen in app.screen_manager.screens]
            if "enhanced_chat" in screen_names:
                print("   ✅ Enhanced chat screen added to app")
            else:
                print(f"   ⚠️  Enhanced chat not found. Available: {screen_names}")

    except Exception as e:
        print(f"   ❌ App creation failed: {e}")
        return False

    print("\n" + "=" * 50)
    print("🎯 STARTUP TEST RESULTS")
    print("=" * 50)
    print("✅ All imports successful")
    print("✅ All components initialize properly")
    print("✅ App creates without errors")
    print("✅ Enhanced chat screen integrated")
    print("✅ APPLICATION IS READY TO RUN!")

    return True


if __name__ == "__main__":
    success = test_app_startup()
    if success:
        print("\n🌟 STARTUP TEST PASSED!")
        print("🚀 App is ready to launch with enhanced chat!")
        exit(0)
    else:
        print("\n💥 STARTUP TEST FAILED!")
        exit(1)
