#!/usr/bin/env python3
"""Simplified main entry point for testing."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

if __name__ == "__main__":
    try:
        from kivy.config import Config

        # Configure Kivy before importing other modules
        Config.set("graphics", "width", "1200")
        Config.set("graphics", "height", "800")

        from kivymd.app import MDApp
        from kivymd.uix.boxlayout import MDBoxLayout
        from kivymd.uix.button import MDButton, MDButtonText
        from kivymd.uix.label import MDLabel
        from kivymd.uix.screen import MDScreen

        class SimpleNeuromancerApp(MDApp):
            def build(self):
                self.title = "Neuromancer - AI Assistant"
                self.theme_cls.theme_style = "Dark"
                self.theme_cls.primary_palette = "DeepPurple"

                screen = MDScreen()
                layout = MDBoxLayout(orientation="vertical", padding=20, spacing=20)

                # Title
                title = MDLabel(text="Neuromancer AI Assistant", halign="center", font_style="H3")

                # Status
                status = MDLabel(
                    text="Simplified version running successfully!",
                    halign="center",
                    theme_text_color="Secondary",
                )

                # Button
                btn = MDButton(style="elevated", pos_hint={"center_x": 0.5})
                btn.add_widget(MDButtonText(text="Launch Full App"))
                btn.bind(on_release=lambda x: print("Full app would launch here"))

                layout.add_widget(title)
                layout.add_widget(status)
                layout.add_widget(btn)

                screen.add_widget(layout)
                return screen

        print("Starting simplified Neuromancer app...")
        SimpleNeuromancerApp().run()

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
