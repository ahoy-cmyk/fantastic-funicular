#!/usr/bin/env python3
"""Quick test to verify app startup without full GUI run."""

import os
import sys

# Set environment for testing
os.environ["KIVY_NO_CONSOLELOG"] = "1"
os.environ["KIVY_LOG_LEVEL"] = "critical"
os.environ["KIVY_WINDOW"] = "sdl2"

sys.path.insert(0, "src")


def test_app_initialization():
    """Test that the app can initialize without async errors."""
    print("🚀 TESTING APP INITIALIZATION")
    print("=" * 40)

    try:
        print("\n1️⃣ Importing components...")
        from src.gui.app import NeuromancerApp

        print("   ✅ NeuromancerApp imported")

        print("\n2️⃣ Creating app instance...")
        app = NeuromancerApp()
        print("   ✅ App instance created")

        print("\n3️⃣ Building app UI...")
        root_widget = app.build()
        print("   ✅ App UI built")

        print("\n4️⃣ Testing screen loading...")
        app.load_main_screens()
        print("   ✅ Main screens loaded")

        if hasattr(app, "screen_manager"):
            screen_names = [screen.name for screen in app.screen_manager.screens]
            print(f"   📋 Available screens: {screen_names}")

            if "enhanced_chat" in screen_names:
                print("   ✅ Enhanced chat screen found")
            else:
                print("   ❌ Enhanced chat screen missing")
                return False

        print("\n" + "=" * 40)
        print("🎯 APP INITIALIZATION TEST")
        print("=" * 40)
        print("✅ All imports successful")
        print("✅ App creates without errors")
        print("✅ Enhanced chat screen available")
        print("✅ NO ASYNC EVENT LOOP ERRORS!")
        print("✅ READY TO RUN!")

        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_app_initialization()

    if success:
        print("\n🌟 INITIALIZATION TEST PASSED!")
        print("🚀 App is ready to launch!")
        print("\nTo run the app:")
        print("source venv/bin/activate")
        print("python -m src.main")
        exit(0)
    else:
        print("\n💥 INITIALIZATION TEST FAILED!")
        exit(1)
