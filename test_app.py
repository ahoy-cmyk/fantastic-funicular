#!/usr/bin/env python3
"""Test script to verify basic app functionality."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    print("Testing imports...")

    # Test basic imports
    from kivy.app import App

    print("✓ Kivy imported successfully")

    from kivymd.app import MDApp

    print("✓ KivyMD imported successfully")

    # Test our modules
    from src import __version__

    print(f"✓ Neuromancer version: {__version__}")

    from src.core.config import config

    print(f"✓ Config loaded: {config.general.app_name}")

    from src.providers import LLMProvider

    print("✓ Providers module imported")

    from src.memory import MemoryManager

    print("✓ Memory module imported")

    # Test simple app
    from kivymd.uix.boxlayout import MDBoxLayout
    from kivymd.uix.button import MDRaisedButton
    from kivymd.uix.screen import MDScreen

    class TestApp(MDApp):
        def build(self):
            self.theme_cls.theme_style = "Dark"

            screen = MDScreen()
            layout = MDBoxLayout(orientation="vertical", padding=20, spacing=20)

            btn = MDRaisedButton(text="Test Button - App is Working!", pos_hint={"center_x": 0.5})

            layout.add_widget(btn)
            screen.add_widget(layout)

            return screen

    if __name__ == "__main__":
        print("\nStarting test app...")
        TestApp().run()

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nMake sure to run: ./scripts/setup.sh")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
