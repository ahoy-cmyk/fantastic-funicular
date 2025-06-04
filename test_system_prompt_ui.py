#!/usr/bin/env python3
"""Test script to verify system prompt UI integration."""


from kivy.app import App
from kivy.clock import Clock
from kivy.uix.screenmanager import ScreenManager

from src.gui.screens.memory_screen import MemoryScreen
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class TestSystemPromptApp(App):
    """Test app for system prompt UI."""

    def build(self):
        """Build the test app."""
        # Create screen manager
        self.sm = ScreenManager()

        # Add memory screen
        memory_screen = MemoryScreen(name="memory")
        self.sm.add_widget(memory_screen)

        # Schedule tests
        Clock.schedule_once(self.run_tests, 1)

        return self.sm

    def run_tests(self, dt):
        """Run system prompt UI tests."""
        print("\n=== Testing System Prompt UI ===\n")

        memory_screen = self.sm.get_screen("memory")

        # Test 1: Check if system prompt button exists
        print("Test 1: Checking system prompt button")

        # Find the system button (cog icon)
        system_btn = None
        for widget in memory_screen.walk():
            if hasattr(widget, "icon") and widget.icon == "cog":
                system_btn = widget
                break

        if system_btn:
            print("✓ System prompt button found")

            # Test 2: Simulate clicking the button
            print("\nTest 2: Opening system prompt dialog")
            system_btn.dispatch("on_release")

            # Give dialog time to open
            Clock.schedule_once(lambda dt: self.test_dialog_functionality(), 0.5)
        else:
            print("✗ System prompt button not found")
            self.stop()

    def test_dialog_functionality(self):
        """Test the system prompt dialog functionality."""
        memory_screen = self.sm.get_screen("memory")

        # Check if dialog opened and fields are accessible
        if hasattr(memory_screen, "system_prompt_field"):
            print("✓ System prompt dialog opened successfully")

            # Test 3: Set a test prompt
            print("\nTest 3: Setting test system prompt")
            test_prompt = "Test prompt from UI integration test"
            memory_screen.system_prompt_field.text = test_prompt
            print(f"✓ Set test prompt: '{test_prompt}'")

            # Test 4: Check memory integration checkbox
            if hasattr(memory_screen, "memory_integration_checkbox"):
                print("\nTest 4: Checking memory integration checkbox")
                memory_screen.memory_integration_checkbox.active = False
                print("✓ Memory integration checkbox set to False")

            # Test 5: Save the system prompt
            print("\nTest 5: Saving system prompt")

            # Find save button in dialog
            from kivymd.uix.dialog import MDDialog

            for widget in App.get_running_app().root_window.children:
                if isinstance(widget, MDDialog):
                    for button in widget.buttons:
                        if button.text == "Save":
                            button.dispatch("on_release")
                            print("✓ Save button clicked")

                            # Schedule verification
                            Clock.schedule_once(self.verify_saved_prompt, 1)
                            return

            print("✗ Could not find save button")
            self.stop()
        else:
            print("✗ System prompt dialog did not open properly")
            self.stop()

    def verify_saved_prompt(self, dt):
        """Verify the prompt was saved."""
        print("\nTest 6: Verifying saved prompt")

        try:
            from src.core.config import _config_manager

            saved_prompt = _config_manager.get("system_prompt")
            saved_memory_integration = _config_manager.get("system_prompt_memory_integration")

            if saved_prompt == "Test prompt from UI integration test":
                print(f"✓ System prompt saved correctly: '{saved_prompt}'")
            else:
                print(f"✗ System prompt not saved correctly: '{saved_prompt}'")

            if saved_memory_integration == False:
                print(f"✓ Memory integration saved correctly: {saved_memory_integration}")
            else:
                print(f"✗ Memory integration not saved correctly: {saved_memory_integration}")

            print("\n✓ All UI tests completed!")

        except Exception as e:
            print(f"✗ Error verifying saved prompt: {e}")

        # Stop the app
        Clock.schedule_once(lambda dt: self.stop(), 0.5)


def test_system_prompt_ui():
    """Run the system prompt UI test."""
    print("Starting system prompt UI test...")

    # Set up test environment
    import os

    os.environ["KIVY_LOG_MODE"] = "MIXED"

    # Run the test app
    app = TestSystemPromptApp()
    try:
        app.run()
    except Exception as e:
        print(f"Error running UI test: {e}")
        return False

    return True


if __name__ == "__main__":
    try:
        test_system_prompt_ui()
    except Exception as e:
        print(f"\n✗ UI test failed with error: {e}")
        import traceback

        traceback.print_exc()
