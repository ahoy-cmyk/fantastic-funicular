#!/usr/bin/env python3
"""Test UI improvements: memory bubble color and conversation slide out."""

import os
import sys

# Set environment for UI testing
os.environ["KIVY_NO_CONSOLELOG"] = "1"
os.environ["KIVY_LOG_LEVEL"] = "critical"
os.environ["KIVY_WINDOW"] = "sdl2"

sys.path.insert(0, "src")


def test_memory_bubble_color():
    """Test memory bubble has bright color."""
    print("🎨 TESTING MEMORY BUBBLE COLOR")
    print("=" * 50)

    try:
        from src.gui.screens.enhanced_chat_screen import MemoryContextCard

        # Create memory context card
        memory_card = MemoryContextCard()
        print("✅ Memory context card created")

        # Check background color
        bg_color = memory_card.md_bg_color
        print(f"✅ Background color: {bg_color}")

        # Verify it's brighter (blue color components should be significant)
        r, g, b, a = bg_color

        if b > 0.5:  # Blue component should be > 0.5 for bright blue
            print("✅ Memory bubble has bright blue color")
            is_bright = True
        else:
            print(f"⚠️ Memory bubble color might not be bright enough (blue={b})")
            is_bright = False

        if a > 0.8:  # High opacity
            print("✅ Memory bubble has high opacity")
            high_opacity = True
        else:
            print(f"⚠️ Memory bubble opacity might be low (alpha={a})")
            high_opacity = False

        return is_bright and high_opacity

    except Exception as e:
        print(f"❌ Memory bubble test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_navigation_drawer_config():
    """Test navigation drawer configuration."""
    print("\n🔄 TESTING NAVIGATION DRAWER CONFIG")
    print("=" * 50)

    try:
        from src.gui.screens.enhanced_chat_screen import EnhancedChatScreen

        # Create enhanced chat screen (without full Kivy app)
        chat_screen = EnhancedChatScreen()
        print("✅ Enhanced chat screen created")

        # Check navigation drawer properties
        nav_drawer = chat_screen.nav_drawer

        # Check key properties for smooth sliding
        if hasattr(nav_drawer, "enable_swiping") and nav_drawer.enable_swiping:
            print("✅ Swiping enabled for drawer")
            swiping_ok = True
        else:
            print("⚠️ Swiping not enabled")
            swiping_ok = False

        if hasattr(nav_drawer, "anchor") and nav_drawer.anchor == "left":
            print("✅ Drawer anchored to left")
            anchor_ok = True
        else:
            print("⚠️ Drawer anchor not set to left")
            anchor_ok = False

        if hasattr(nav_drawer, "drawer_type"):
            print(f"✅ Drawer type: {nav_drawer.drawer_type}")
            type_ok = True
        else:
            print("⚠️ Drawer type not specified")
            type_ok = False

        if hasattr(nav_drawer, "elevation") and nav_drawer.elevation > 0:
            print(f"✅ Drawer elevation: {nav_drawer.elevation}")
            elevation_ok = True
        else:
            print("⚠️ Drawer elevation not set")
            elevation_ok = False

        return swiping_ok and anchor_ok and type_ok and elevation_ok

    except Exception as e:
        print(f"❌ Navigation drawer test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run UI improvement tests."""
    print("🚀 TESTING UI IMPROVEMENTS")
    print("=" * 60)

    memory_test = test_memory_bubble_color()
    drawer_test = test_navigation_drawer_config()

    print("\n" + "=" * 60)
    print("📊 UI IMPROVEMENT TEST RESULTS")
    print("=" * 60)

    print(f"Memory Bubble Color: {'✅ BRIGHT' if memory_test else '❌ NEEDS WORK'}")
    print(f"Navigation Drawer: {'✅ CONFIGURED' if drawer_test else '❌ NEEDS WORK'}")

    if memory_test and drawer_test:
        print("\n✅ ALL UI IMPROVEMENTS ARE WORKING!")
        print("💙 Memory bubble now has bright blue color")
        print("🔄 Navigation drawer properly configured for sliding")
    else:
        print("\n⚠️ SOME UI IMPROVEMENTS NEED ATTENTION!")

    return memory_test and drawer_test


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
