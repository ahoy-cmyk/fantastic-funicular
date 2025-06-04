#!/usr/bin/env python3
"""Script to clean up GUI code - remove redundancy and fix issues."""

import re
import sys
from pathlib import Path

def remove_provider_switcher(file_path):
    """Remove the redundant provider switcher method."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find and remove _show_provider_switcher and related methods
    pattern = r'(\n    def _show_provider_switcher\(self.*?\n(?:(?!^    def ).*\n)*)'
    content = re.sub(pattern, '\n', content, flags=re.MULTILINE | re.DOTALL)
    
    # Remove _select_provider_option
    pattern = r'(\n    def _select_provider_option\(self.*?\n(?:(?!^    def ).*\n)*)'
    content = re.sub(pattern, '\n', content, flags=re.MULTILINE | re.DOTALL)
    
    # Remove _switch_provider  
    pattern = r'(\n    def _switch_provider\(self.*?\n(?:(?!^    def ).*\n)*)'
    content = re.sub(pattern, '\n', content, flags=re.MULTILINE | re.DOTALL)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"✓ Removed provider switcher methods from {file_path}")

def fix_model_management_screen(file_path):
    """Fix issues in model management screen."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Already fixed in previous edits
    print(f"✓ Model management screen already fixed: {file_path}")

def check_redundant_screens():
    """Check for redundant screen files."""
    screens_dir = Path("/Users/sarrington/Workspace/fantastic-funicular/src/gui/screens")
    
    redundant_screens = [
        "chat_screen.py",  # Old version
        "chat_screen_v2.py",  # Old version
        "provider_config_screen.py",  # Redundant with model_management_screen
    ]
    
    for screen in redundant_screens:
        screen_path = screens_dir / screen
        if screen_path.exists():
            print(f"⚠️  Found redundant screen: {screen} - consider removing")

def fix_app_screens(file_path):
    """Remove redundant screens from app.py."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Remove provider_config_screen import if present
    content = re.sub(r'from src\.gui\.screens\.provider_config_screen import .*\n', '', content)
    
    # Remove the screen addition
    content = re.sub(r'.*ProviderConfigScreen.*\n', '', content)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"✓ Cleaned up app.py")

def main():
    """Run all cleanup tasks."""
    print("🧹 Cleaning up GUI code...")
    print("-" * 50)
    
    # Clean enhanced chat screen
    enhanced_chat = "/Users/sarrington/Workspace/fantastic-funicular/src/gui/screens/enhanced_chat_screen.py"
    remove_provider_switcher(enhanced_chat)
    
    # Fix model management
    model_mgmt = "/Users/sarrington/Workspace/fantastic-funicular/src/gui/screens/model_management_screen.py"
    fix_model_management_screen(model_mgmt)
    
    # Clean app.py
    app_file = "/Users/sarrington/Workspace/fantastic-funicular/src/gui/app.py"
    fix_app_screens(app_file)
    
    # Check for redundant screens
    check_redundant_screens()
    
    print("\n✅ Cleanup complete!")
    print("\nRecommendations:")
    print("1. Remove chat_screen.py and chat_screen_v2.py (old versions)")
    print("2. Remove provider_config_screen.py (replaced by model_management_screen.py)")
    print("3. Test the application to ensure everything works smoothly")

if __name__ == "__main__":
    main()