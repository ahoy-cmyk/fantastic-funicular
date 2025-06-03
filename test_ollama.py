#!/usr/bin/env python3
"""Test Ollama connection directly."""

import asyncio
import sys

sys.path.insert(0, "src")

from src.providers import Message
from src.providers.ollama_provider import OllamaProvider


async def test_ollama():
    """Test Ollama provider."""
    try:
        # Initialize provider
        provider = OllamaProvider(host="http://localhost:11434")
        print("✅ Ollama provider initialized")

        # List models
        print("\n📋 Available models:")
        models = await provider.list_models()
        for model in models:
            print(f"   • {model}")

        if not models:
            print("   ❌ No models found. Please install a model with 'ollama pull llama3.2'")
            return

        # Test completion
        print(f"\n🤖 Testing completion with {models[0]}...")
        response = await provider.complete(
            messages=[Message(role="user", content="Say hello in one sentence.")],
            model=models[0],
            temperature=0.7,
        )

        print(f"   Response: {response.content}")
        print(f"   Model: {response.model}")
        print(f"   Tokens: {response.usage}")

        print("\n✅ Ollama test successful!")

    except Exception as e:
        print(f"\n❌ Ollama test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_ollama())
