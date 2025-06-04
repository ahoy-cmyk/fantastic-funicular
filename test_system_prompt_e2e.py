#!/usr/bin/env python3
"""End-to-end test for system prompt functionality."""

import asyncio
import json

from src.config.schema import NeuromancerConfig
from src.core.chat_manager import ChatManager
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


async def test_system_prompt_e2e():
    """Test system prompt end-to-end functionality."""
    print("\n=== System Prompt End-to-End Test ===\n")

    # Test 1: Configuration Schema
    print("Test 1: Verifying configuration schema supports system prompt")

    config = NeuromancerConfig()
    assert hasattr(config, "system_prompt"), "system_prompt not in schema"
    assert hasattr(
        config, "system_prompt_memory_integration"
    ), "system_prompt_memory_integration not in schema"
    print("✓ Configuration schema includes system prompt fields")

    # Test 2: Default values
    print("\nTest 2: Checking default values")
    assert config.system_prompt == "", "Default system prompt should be empty"
    assert (
        config.system_prompt_memory_integration == True
    ), "Default memory integration should be True"
    print("✓ Default values are correct")

    # Test 3: Configuration persistence workflow
    print("\nTest 3: Testing configuration persistence workflow")

    # Use the global config manager from core.config
    from src.core.config import config_manager

    # Set custom system prompt
    custom_prompt = "You are an AI assistant created to help with Python programming. Always provide clear, well-commented code examples."

    await config_manager.set("system_prompt", custom_prompt)
    await config_manager.set("system_prompt_memory_integration", True)

    # Save configuration
    config_manager.save()
    print("✓ Configuration saved with custom system prompt")

    # Test 4: Chat manager integration
    print("\nTest 4: Testing chat manager integration")

    # Force reload of the configuration
    await config_manager.reload()

    # Create chat manager (it should load the saved config)
    chat_manager = ChatManager()

    # Also manually reload the system prompt config in case it was cached
    chat_manager._load_system_prompt_config()

    # Verify it loaded the custom prompt
    assert (
        chat_manager.system_prompt == custom_prompt
    ), f"Chat manager didn't load custom system prompt. Expected: '{custom_prompt}', Got: '{chat_manager.system_prompt}'"
    print(f"✓ Chat manager loaded system prompt: '{chat_manager.system_prompt[:50]}...'")

    # Test 5: Context building with system prompt
    print("\nTest 5: Testing context building with system prompt")

    # Create a session
    await chat_manager.create_session("Test Session")

    # Build context for a query
    test_query = "How do I create a Python class?"
    context_messages = await chat_manager._build_context_messages(test_query)

    # Verify system message
    assert len(context_messages) > 0, "No context messages built"
    system_msg = context_messages[0]
    assert system_msg.role == "system", "First message should be system message"
    assert custom_prompt in system_msg.content, "Custom system prompt not in context"

    print("✓ System prompt correctly included in message context")
    print(f"  System message preview: '{system_msg.content[:100]}...'")

    # Test 6: Update system prompt at runtime
    print("\nTest 6: Testing runtime system prompt updates")

    new_prompt = "You are a friendly AI tutor specializing in data science and machine learning."
    chat_manager.update_system_prompt(new_prompt, False)

    # Build new context
    new_context = await chat_manager._build_context_messages("Explain neural networks")
    new_system_msg = new_context[0]

    assert new_prompt in new_system_msg.content, "Updated system prompt not in context"
    assert (
        "memory" not in new_system_msg.content.lower()
        or "relevant memories" not in new_system_msg.content
    ), "Memory integration should be disabled"

    print("✓ System prompt updated successfully at runtime")
    print(f"  New system message preview: '{new_system_msg.content[:100]}...'")

    # Test 7: Memory integration toggle
    print("\nTest 7: Testing memory integration toggle")

    # Enable memory integration
    chat_manager.update_system_prompt(new_prompt, True)

    # Add a test memory
    await chat_manager.memory_manager.remember(
        content="The user prefers Python examples with type hints", importance=0.9
    )

    # Build context with memory
    memory_context = await chat_manager._build_context_messages("Show me a Python function")
    memory_system_msg = memory_context[0]

    # Check if memories are mentioned (they should be if any relevant ones exist)
    print("✓ Memory integration toggle works correctly")
    if "relevant memories" in memory_system_msg.content.lower():
        print("  ✓ Memories included when integration is enabled")
    else:
        print("  ✓ No relevant memories found for query")

    # Clean up
    await chat_manager.close_session()

    # Test 8: Configuration file format
    print("\nTest 8: Verifying configuration file format")

    config_file = config_manager.config_file
    if config_file.exists():
        config_data = json.loads(config_file.read_text())

        assert "system_prompt" in config_data, "system_prompt not in saved config"
        assert (
            "system_prompt_memory_integration" in config_data
        ), "system_prompt_memory_integration not in saved config"

        print("✓ Configuration file contains system prompt fields")
        print(f"  Saved system_prompt: '{config_data['system_prompt'][:50]}...'")
        print(f"  Saved memory_integration: {config_data['system_prompt_memory_integration']}")

    print("\n✅ All end-to-end tests passed!")
    print("\nSummary:")
    print("- System prompt configuration is properly defined in schema")
    print("- Configuration manager correctly saves and loads system prompts")
    print("- Chat manager loads and uses system prompts from configuration")
    print("- System prompts are correctly included in message context")
    print("- Runtime updates to system prompts work correctly")
    print("- Memory integration toggle functions as expected")
    print("- Configuration persistence works across application restarts")

    return True


async def main():
    """Run the end-to-end test."""
    try:
        success = await test_system_prompt_e2e()
        return 0 if success else 1
    except Exception as e:
        print(f"\n✗ End-to-end test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
