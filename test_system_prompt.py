#!/usr/bin/env python3
"""Test script to verify system prompt persistence and functionality."""

import asyncio
import shutil
from pathlib import Path

from src.config.manager import ConfigManager
from src.core.chat_manager import ChatManager
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def test_system_prompt_configuration():
    """Test system prompt configuration end-to-end."""
    print("\n=== Testing System Prompt Configuration ===\n")

    # Create a test config directory
    test_config_dir = Path.home() / ".neuromancer_test"
    if test_config_dir.exists():
        shutil.rmtree(test_config_dir)

    # Create a test configuration manager
    config_manager = ConfigManager(app_name="NeuromancerTest")
    config_manager.config_dir = test_config_dir
    config_manager.config_file = test_config_dir / "config.json"
    test_config_dir.mkdir(parents=True, exist_ok=True)

    # Test 1: Set and save system prompt
    print("Test 1: Setting and saving system prompt")
    test_prompt = "You are a helpful coding assistant specialized in Python development."
    test_memory_integration = True

    try:
        # Set the system prompt
        success = config_manager.set_sync("system_prompt", test_prompt)
        assert success, "Failed to set system_prompt"
        print(f"✓ Set system_prompt: '{test_prompt[:50]}...'")

        # Set memory integration
        success = config_manager.set_sync(
            "system_prompt_memory_integration", test_memory_integration
        )
        assert success, "Failed to set system_prompt_memory_integration"
        print(f"✓ Set memory integration: {test_memory_integration}")

        # Save the configuration
        config_manager.save()
        print("✓ Configuration saved to disk")

    except Exception as e:
        print(f"✗ Error setting system prompt: {e}")
        return False

    # Test 2: Load configuration and verify persistence
    print("\nTest 2: Loading configuration and verifying persistence")

    # Create a new config manager instance to simulate app restart
    config_manager2 = ConfigManager(app_name="NeuromancerTest")
    config_manager2.config_dir = test_config_dir
    config_manager2.config_file = test_config_dir / "config.json"

    # Load the configuration
    loaded = config_manager2.load()
    if loaded:
        print("✓ Configuration loaded from disk")

        # Verify the values
        loaded_prompt = config_manager2.get("system_prompt")
        loaded_memory_integration = config_manager2.get("system_prompt_memory_integration")

        assert (
            loaded_prompt == test_prompt
        ), f"System prompt mismatch: '{loaded_prompt}' != '{test_prompt}'"
        print(f"✓ System prompt persisted correctly: '{loaded_prompt[:50]}...'")

        assert (
            loaded_memory_integration == test_memory_integration
        ), f"Memory integration mismatch: {loaded_memory_integration} != {test_memory_integration}"
        print(f"✓ Memory integration persisted correctly: {loaded_memory_integration}")
    else:
        print("✗ Failed to load configuration")
        return False

    # Test 3: Verify chat manager loads system prompt
    print("\nTest 3: Verifying chat manager loads system prompt")

    try:
        # Temporarily set the global config manager for chat manager
        from src.core import config as core_config

        original_config_manager = core_config._config_manager
        core_config._config_manager = config_manager2

        # Create chat manager
        chat_manager = ChatManager()

        # Verify it loaded the system prompt
        assert (
            chat_manager.system_prompt == test_prompt
        ), f"Chat manager system prompt mismatch: '{chat_manager.system_prompt}' != '{test_prompt}'"
        print(f"✓ Chat manager loaded system prompt: '{chat_manager.system_prompt[:50]}...'")

        assert (
            chat_manager.system_prompt_memory_integration == test_memory_integration
        ), "Chat manager memory integration mismatch"
        print(
            f"✓ Chat manager loaded memory integration: {chat_manager.system_prompt_memory_integration}"
        )

        # Test 4: Verify system prompt is used in context
        print("\nTest 4: Verifying system prompt is used in message context")

        async def test_context():
            # Create a test session
            await chat_manager.create_session("Test Session")

            # Build context messages
            context_messages = await chat_manager._build_context_messages(
                "Test query", exclude_current=True
            )

            # Verify system message includes our prompt
            assert len(context_messages) > 0, "No context messages built"
            system_message = context_messages[0]

            assert (
                system_message.role == "system"
            ), f"First message should be system, got {system_message.role}"
            assert test_prompt in system_message.content, "System prompt not found in context"
            print(f"✓ System prompt included in context: '{system_message.content[:100]}...'")

            # Clean up
            await chat_manager.close_session()

        # Run async test
        asyncio.run(test_context())

        # Restore original config manager
        core_config._config_manager = original_config_manager

    except Exception as e:
        print(f"✗ Error testing chat manager: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test 5: Update system prompt and verify it's applied
    print("\nTest 5: Testing system prompt updates")

    new_prompt = "You are an expert mathematician who loves solving complex problems."

    try:
        # Update through chat manager
        chat_manager.update_system_prompt(new_prompt, False)

        assert chat_manager.system_prompt == new_prompt, "System prompt not updated in chat manager"
        print(f"✓ Chat manager updated with new prompt: '{new_prompt[:50]}...'")

        assert (
            chat_manager.system_prompt_memory_integration == False
        ), "Memory integration not updated"
        print("✓ Memory integration updated to False")

    except Exception as e:
        print(f"✗ Error updating system prompt: {e}")
        return False

    # Clean up test directory
    shutil.rmtree(test_config_dir)
    print("\n✓ All tests passed! System prompt configuration is working correctly.")
    return True


if __name__ == "__main__":
    try:
        success = test_system_prompt_configuration()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
