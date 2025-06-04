#!/usr/bin/env python3
"""
Demo script showcasing the new model management and RAG functionality.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.chat_manager import ChatManager
from src.core.model_manager import ModelSelectionStrategy
from src.core.rag_system import RAGConfig


async def demo_model_management():
    """Demonstrate model management features."""
    print("🤖 Neuromancer Model Management Demo")
    print("=" * 50)
    
    # Initialize chat manager (includes model management)
    print("\n📦 Initializing Chat Manager...")
    chat_manager = ChatManager()
    print("✓ Chat manager initialized with model management and RAG")
    
    # Discover available models
    print("\n🔍 Discovering available models...")
    success = await chat_manager.discover_models(force_refresh=True)
    
    if success:
        print("✓ Model discovery completed")
        
        # Show available models
        models = chat_manager.get_available_models()
        print(f"\n📋 Found {len(models)} available models:")
        
        for i, model in enumerate(models[:5]):  # Show first 5
            status_icon = "✅" if model.is_available else "❌"
            print(f"  {i+1}. {status_icon} {model.full_name}")
            print(f"      Provider: {model.provider}")
            if model.parameters:
                print(f"      Parameters: {model.parameters}")
            if model.capabilities:
                print(f"      Capabilities: {', '.join(model.capabilities[:3])}")
            print()
        
        # Show provider health
        print("🏥 Provider Health Status:")
        provider_health = chat_manager.get_provider_health()
        for provider_name, provider_info in provider_health.items():
            if provider_info:
                health_icon = "💚" if provider_info.is_healthy else "💔"
                print(f"  {health_icon} {provider_name}: {provider_info.status.value}")
                if provider_info.response_time_ms:
                    print(f"      Response time: {provider_info.response_time_ms:.0f}ms")
                print(f"      Models available: {len(provider_info.available_models)}")
        
        # Test automatic model selection
        print("\n🎯 Testing automatic model selection strategies:")
        
        strategies = [
            (ModelSelectionStrategy.FASTEST, "Fastest"),
            (ModelSelectionStrategy.BALANCED, "Balanced"),
            (ModelSelectionStrategy.CHEAPEST, "Cheapest")
        ]
        
        for strategy, name in strategies:
            try:
                selected = await chat_manager.select_best_model(strategy)
                if selected:
                    print(f"  {name}: {selected.full_name}")
                else:
                    print(f"  {name}: No suitable model found")
            except Exception as e:
                print(f"  {name}: Error - {e}")
        
        # Test model switching
        if models:
            print(f"\n🔄 Testing model switching...")
            first_model = models[0]
            success = chat_manager.set_model(first_model.name, first_model.provider)
            if success:
                print(f"✓ Switched to {first_model.full_name}")
                print(f"  Current provider: {chat_manager.current_provider}")
                print(f"  Current model: {chat_manager.current_model}")
            else:
                print("✗ Model switching failed")
    
    else:
        print("✗ Model discovery failed")
    
    return chat_manager


async def demo_rag_system(chat_manager):
    """Demonstrate RAG system features."""
    print("\n🧠 RAG System Demo")
    print("=" * 30)
    
    # Configure RAG system
    print("\n⚙️  Configuring RAG system...")
    rag_config = RAGConfig(
        max_memories=10,
        min_relevance_threshold=0.6,
        cite_sources=True,
        explain_reasoning=True,
        max_context_tokens=3000
    )
    
    chat_manager.configure_rag(rag_config)
    chat_manager.enable_rag(True)
    print("✓ RAG system configured and enabled")
    
    # Store some sample memories
    print("\n💾 Storing sample memories...")
    
    sample_memories = [
        "Python is an excellent programming language for beginners and experts alike.",
        "The user prefers detailed code examples with explanations.", 
        "Machine learning frameworks like PyTorch and TensorFlow are popular in Python.",
        "Web development with Django and Flask is a common Python use case.",
        "Data science libraries like pandas and numpy are essential Python tools."
    ]
    
    for memory_content in sample_memories:
        await chat_manager.memory_manager.remember(
            memory_content,
            auto_classify=True
        )
    
    print(f"✓ Stored {len(sample_memories)} sample memories")
    
    # Test RAG retrieval
    print("\n🔍 Testing RAG retrieval...")
    context = await chat_manager.rag_system.retrieve_context(
        "python programming help for beginners"
    )
    
    print(f"✓ Retrieved {len(context.memories)} relevant memories")
    print(f"  Retrieval time: {context.retrieval_time_ms:.1f}ms")
    print(f"  Reasoning: {context.reasoning}")
    
    if context.memories:
        print("  Top memory:", context.memories[0].content[:60] + "...")
    
    # Test prompt enhancement
    print("\n✨ Testing prompt enhancement...")
    original_query = "How do I get started with Python programming?"
    enhanced_prompt = await chat_manager.rag_system.enhance_prompt(
        original_query, context
    )
    
    print(f"  Original query length: {len(original_query)} chars")
    print(f"  Enhanced prompt length: {len(enhanced_prompt)} chars")
    print(f"  Enhancement ratio: {len(enhanced_prompt) / len(original_query):.1f}x")
    
    # Show RAG stats
    stats = chat_manager.get_rag_stats()
    print(f"\n📊 RAG Statistics:")
    print(f"  Total queries: {stats['total_queries']}")
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.1%}")
    print(f"  Average retrieval time: {stats['avg_retrieval_time_ms']:.1f}ms")
    
    return True


async def demo_integration():
    """Demonstrate full integration."""
    print("\n🔗 Integration Demo")
    print("=" * 25)
    
    # This would show how to use RAG-enhanced messaging
    # In a real scenario with actual providers
    print("\n📝 RAG-Enhanced Messaging Example:")
    print("  User query: 'What's the best way to learn Python?'")
    print("  → RAG system retrieves relevant memories")
    print("  → Prompt enhanced with context")
    print("  → LLM generates informed response")
    print("  → Response includes source citations")
    print("  → New knowledge stored in memory")
    
    return True


async def main():
    """Run the complete demo."""
    try:
        # Model management demo
        chat_manager = await demo_model_management()
        
        # RAG system demo
        await demo_rag_system(chat_manager)
        
        # Integration demo
        await demo_integration()
        
        print("\n🎉 Demo completed successfully!")
        print("\nKey Features Demonstrated:")
        print("✓ Automatic model discovery across providers")
        print("✓ Intelligent model selection strategies")
        print("✓ Provider health monitoring")
        print("✓ RAG system with memory integration")
        print("✓ Configurable retrieval and enhancement")
        print("✓ Performance tracking and statistics")
        
        print("\n📱 UI Features Available:")
        print("✓ Model management screen with visual cards")
        print("✓ RAG configuration panel")
        print("✓ Provider health dashboard")
        print("✓ One-click model switching")
        print("✓ Real-time statistics display")
        
    except Exception as e:
        print(f"\n💥 Demo failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    print("Starting Neuromancer Model Management & RAG Demo...")
    exit_code = asyncio.run(main())
    sys.exit(exit_code)