#!/usr/bin/env python3
"""Test script to verify UI components work without MDSwitch issues."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

# Suppress Kivy logs for cleaner output
import os
os.environ['KIVY_LOG_LEVEL'] = 'warning'

from kivy.app import App
from kivy.uix.screenmanager import ScreenManager
from kivymd.app import MDApp

from src.core.chat_manager import ChatManager
from src.core.rag_system import RAGConfig
from src.gui.screens.model_management_screen import RAGConfigCard, ModelManagementScreen
from src.gui.theme import NeuromancerTheme


class TestApp(MDApp):
    """Test app to verify components work."""
    
    def build(self):
        # Apply theme
        NeuromancerTheme.apply_theme(self)
        
        # Create screen manager
        sm = ScreenManager()
        
        # Test RAGConfigCard creation
        try:
            config = RAGConfig()
            rag_card = RAGConfigCard(config)
            print("✓ RAGConfigCard created successfully (no MDSwitch issues)")
        except Exception as e:
            print(f"✗ RAGConfigCard creation failed: {e}")
            return sm
        
        # Test ModelManagementScreen creation
        try:
            chat_manager = ChatManager()
            model_screen = ModelManagementScreen(chat_manager=chat_manager, name="model_management")
            sm.add_widget(model_screen)
            print("✓ ModelManagementScreen created successfully")
        except Exception as e:
            print(f"✗ ModelManagementScreen creation failed: {e}")
            return sm
        
        print("🎉 All UI components created without errors!")
        
        return sm


if __name__ == "__main__":
    print("Testing UI components after MDSwitch fix...")
    print("-" * 50)
    
    try:
        app = TestApp()
        app.run()
    except Exception as e:
        print(f"✗ App failed to run: {e}")
        import traceback
        traceback.print_exc()