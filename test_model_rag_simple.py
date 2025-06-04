#!/usr/bin/env python3
"""
Simple test script for model management and RAG functionality without external dependencies.
"""

import asyncio
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.core.model_manager import ModelManager, ModelInfo, ProviderInfo, ModelSelectionStrategy, ModelStatus, ProviderStatus
    from src.core.rag_system import RAGSystem, RAGConfig, RetrievalContext
    from src.memory import Memory, MemoryType
    from src.providers import Message, CompletionResponse
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)


class MockLLMProvider:
    """Mock LLM provider for testing."""
    
    def __init__(self, name: str, models: list[str] = None, healthy: bool = True):
        self.name = name
        self.models = models or ["test-model-1", "test-model-2"]
        self.healthy = healthy
    
    async def complete(self, messages, model, **kwargs):
        """Mock completion."""
        return CompletionResponse(
            content="Test response from mock provider",
            model=model,
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        )
    
    async def stream_complete(self, messages, model, **kwargs):
        """Mock streaming completion."""
        for chunk in ["Test ", "streaming ", "response"]:
            yield chunk
    
    async def list_models(self):
        """Mock model listing."""
        if not self.healthy:
            raise Exception("Provider unavailable")
        return self.models
    
    async def health_check(self):
        """Mock health check."""
        return self.healthy


async def test_model_manager():
    """Test ModelManager functionality."""
    print("\n=== Testing ModelManager ===")
    
    # Create model manager
    model_manager = ModelManager()
    print("✓ ModelManager created")
    
    # Create mock chat manager
    mock_chat_manager = MagicMock()
    mock_chat_manager.providers = {
        "ollama": MockLLMProvider("ollama", ["llama3.2", "mistral"]),
        "openai": MockLLMProvider("openai", ["gpt-4", "gpt-3.5-turbo"]),
        "broken": MockLLMProvider("broken", [], False)
    }
    
    model_manager.set_chat_manager(mock_chat_manager)
    print("✓ Mock chat manager set")
    
    # Test model discovery
    try:
        success = await model_manager.discover_models(force_refresh=True)
        if success:
            print("✓ Model discovery completed successfully")
        else:
            print("✗ Model discovery failed")
            return False
    except Exception as e:
        print(f"✗ Model discovery error: {e}")
        return False
    
    # Test model retrieval
    models = model_manager.get_available_models()
    print(f"✓ Found {len(models)} available models")
    
    for model in models[:3]:  # Show first 3
        print(f"  - {model.full_name} ({model.status.value})")
    
    # Test provider health
    providers = ["ollama", "openai", "broken"]
    for provider_name in providers:
        provider_info = model_manager.get_provider_info(provider_name)
        if provider_info:
            print(f"✓ Provider {provider_name}: {provider_info.status.value}")
        else:
            print(f"✗ Provider {provider_name}: not found")
    
    # Test model selection strategies
    try:
        manual_model = await model_manager.select_best_model(ModelSelectionStrategy.MANUAL)
        if manual_model:
            print(f"✓ Manual selection: {manual_model.full_name}")
        
        fastest_model = await model_manager.select_best_model(ModelSelectionStrategy.FASTEST)
        if fastest_model:
            print(f"✓ Fastest selection: {fastest_model.full_name}")
        
        balanced_model = await model_manager.select_best_model(ModelSelectionStrategy.BALANCED)
        if balanced_model:
            print(f"✓ Balanced selection: {balanced_model.full_name}")
            
    except Exception as e:
        print(f"✗ Model selection error: {e}")
        return False
    
    return True


async def test_rag_system():
    """Test RAG system functionality."""
    print("\n=== Testing RAG System ===")
    
    # Create mock memory manager
    mock_memory_manager = MagicMock()
    
    # Create sample memories
    sample_memories = [
        Memory(
            id="mem1",
            content="Python is a programming language known for its simplicity and readability.",
            memory_type=MemoryType.SEMANTIC,
            importance=0.8,
            created_at=datetime.now()
        ),
        Memory(
            id="mem2", 
            content="Yesterday I helped the user debug a Python script with list comprehensions.",
            memory_type=MemoryType.EPISODIC,
            importance=0.6,
            created_at=datetime.now()
        ),
        Memory(
            id="mem3",
            content="The user prefers detailed explanations with code examples.",
            memory_type=MemoryType.LONG_TERM,
            importance=0.9,
            created_at=datetime.now()
        )
    ]
    
    # Mock the recall method
    async def mock_recall(query, memory_types=None, limit=10, threshold=0.7):
        # Simple keyword matching for demo
        relevant_memories = []
        query_words = query.lower().split()
        
        for memory in sample_memories:
            memory_words = memory.content.lower().split()
            if any(word in memory_words for word in query_words):
                relevant_memories.append(memory)
        
        return relevant_memories[:limit]
    
    mock_memory_manager.recall = mock_recall
    
    # Create RAG system
    rag_config = RAGConfig(
        max_memories=5,
        min_relevance_threshold=0.6,
        cite_sources=True,
        explain_reasoning=True
    )
    
    rag_system = RAGSystem(mock_memory_manager, rag_config)
    print("✓ RAG system created")
    
    # Test retrieval
    try:
        context = await rag_system.retrieve_context("python programming help")
        print(f"✓ Retrieved {len(context.memories)} relevant memories")
        print(f"  Query: {context.query}")
        print(f"  Retrieval time: {context.retrieval_time_ms:.1f}ms")
        print(f"  Reasoning: {context.reasoning}")
        
        if context.memories:
            print("  Top memory:", context.memories[0].content[:60] + "...")
            
    except Exception as e:
        print(f"✗ Retrieval error: {e}")
        return False
    
    # Test prompt enhancement
    try:
        original_prompt = "How do I use list comprehensions in Python?"
        enhanced_prompt = await rag_system.enhance_prompt(original_prompt, context)
        
        print("✓ Prompt enhancement successful")
        print(f"  Original length: {len(original_prompt)} chars")
        print(f"  Enhanced length: {len(enhanced_prompt)} chars")
        
        if len(enhanced_prompt) > len(original_prompt):
            print("  ✓ Prompt was actually enhanced")
        else:
            print("  ! Prompt not enhanced (no relevant context)")
            
    except Exception as e:
        print(f"✗ Prompt enhancement error: {e}")
        return False
    
    # Test RAG response generation
    try:
        mock_provider = MockLLMProvider("test")
        
        response, response_context = await rag_system.generate_rag_response(
            query="What is Python?",
            llm_provider=mock_provider,
            model="test-model"
        )
        
        print("✓ RAG response generation successful")
        print(f"  Response: {response[:60]}...")
        print(f"  Context memories: {len(response_context.memories)}")
        
    except Exception as e:
        print(f"✗ RAG response generation error: {e}")
        return False
    
    # Test configuration
    stats = rag_system.get_stats()
    print(f"✓ RAG stats retrieved: {stats['total_queries']} queries processed")
    
    return True


def test_model_info():
    """Test ModelInfo data structures."""
    print("\n=== Testing ModelInfo ===")
    
    model_info = ModelInfo(
        name="llama3.2",
        provider="ollama",
        display_name="Llama 3.2",
        description="Open source language model",
        status=ModelStatus.AVAILABLE,
        parameters="7B",
        capabilities=["chat", "code", "reasoning"],
        context_length=4096,
        size_bytes=7_000_000_000
    )
    
    print(f"✓ Model: {model_info.full_name}")
    print(f"  Display: {model_info.display_name}")
    print(f"  Available: {model_info.is_available}")
    print(f"  Size: {model_info.size_human}")
    print(f"  Capabilities: {', '.join(model_info.capabilities)}")
    
    return True


def test_provider_info():
    """Test ProviderInfo data structures."""
    print("\n=== Testing ProviderInfo ===")
    
    provider_info = ProviderInfo(
        name="ollama",
        status=ProviderStatus.HEALTHY,
        response_time_ms=150.0,
        available_models=["llama3.2", "mistral", "codellama"],
        last_health_check=datetime.now()
    )
    
    print(f"✓ Provider: {provider_info.name}")
    print(f"  Status: {provider_info.status.value}")
    print(f"  Healthy: {provider_info.is_healthy}")
    print(f"  Response time: {provider_info.response_time_ms}ms")
    print(f"  Models: {len(provider_info.available_models)}")
    
    return True


async def test_integration():
    """Test integration between components."""
    print("\n=== Integration Test ===")
    
    try:
        # Test that model manager can work with RAG system
        model_manager = ModelManager()
        
        # Create a simple mock setup
        mock_chat_manager = MagicMock()
        mock_chat_manager.providers = {
            "test": MockLLMProvider("test", ["integration-model"])
        }
        
        model_manager.set_chat_manager(mock_chat_manager)
        await model_manager.discover_models()
        
        models = model_manager.get_available_models()
        if models:
            selected_model = models[0]
            print(f"✓ Integration test: Selected {selected_model.full_name}")
            
            # Test that we can create a provider instance for the model
            provider = mock_chat_manager.providers[selected_model.provider]
            response = await provider.complete(
                [Message(role="user", content="test")],
                selected_model.name
            )
            print(f"✓ Integration test: Got response from {selected_model.provider}")
            
            return True
        else:
            print("✗ Integration test: No models found")
            return False
            
    except Exception as e:
        print(f"✗ Integration test error: {e}")
        return False


async def main():
    """Run all tests."""
    print("🤖 Model Management and RAG System Test Suite")
    print("=" * 50)
    
    tests = [
        ("Model Info", test_model_info),
        ("Provider Info", test_provider_info),
        ("Model Manager", test_model_manager),
        ("RAG System", test_rag_system), 
        ("Integration", test_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} test...")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            if result:
                print(f"✅ {test_name} test PASSED")
                results.append(True)
            else:
                print(f"❌ {test_name} test FAILED")
                results.append(False)
                
        except Exception as e:
            print(f"💥 {test_name} test ERROR: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Model management and RAG system are working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)