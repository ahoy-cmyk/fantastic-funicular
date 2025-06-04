#!/usr/bin/env python3
"""
Comprehensive test suite for model management and RAG functionality.

This test file validates:
1. Model discovery and management
2. RAG system functionality
3. Integration with existing chat system
4. UI components for model selection
"""

import asyncio
import pytest
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.model_manager import ModelManager, ModelInfo, ProviderInfo, ModelSelectionStrategy, ModelStatus, ProviderStatus
from src.core.rag_system import RAGSystem, RAGConfig, RetrievalContext
from src.core.chat_manager import ChatManager
from src.memory.manager import MemoryManager
from src.memory import Memory, MemoryType
from src.providers import Message, CompletionResponse
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class MockLLMProvider:
    """Mock LLM provider for testing."""
    
    def __init__(self, name: str, models: list[str] = None, healthy: bool = True):
        self.name = name
        self.models = models or ["test-model-1", "test-model-2"]
        self.healthy = healthy
        self.response_time = 100.0  # ms
    
    async def complete(self, messages, model, **kwargs):
        """Mock completion."""
        return CompletionResponse(
            content="Test response",
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


class TestModelManager:
    """Test suite for ModelManager."""
    
    @pytest.fixture
    def model_manager(self):
        """Create a test model manager."""
        return ModelManager()
    
    @pytest.fixture
    def mock_chat_manager(self):
        """Create a mock chat manager with providers."""
        chat_manager = MagicMock()
        chat_manager.providers = {
            "test_provider_1": MockLLMProvider("test_provider_1", ["model-1a", "model-1b"]),
            "test_provider_2": MockLLMProvider("test_provider_2", ["model-2a", "model-2b"]),
            "unhealthy_provider": MockLLMProvider("unhealthy_provider", [], False)
        }
        return chat_manager
    
    def test_model_manager_initialization(self, model_manager):
        """Test model manager initialization."""
        assert model_manager is not None
        assert model_manager._models == {}
        assert model_manager._providers == {}
        assert model_manager.selection_strategy == ModelSelectionStrategy.MANUAL
    
    @pytest.mark.asyncio
    async def test_model_discovery(self, model_manager, mock_chat_manager):
        """Test model discovery functionality."""
        model_manager.set_chat_manager(mock_chat_manager)
        
        # Test discovery
        success = await model_manager.discover_models(force_refresh=True)
        assert success
        
        # Check discovered models
        models = model_manager.get_available_models()
        assert len(models) >= 4  # 2 providers x 2 models each
        
        # Check model info
        model_names = [m.name for m in models]
        assert "model-1a" in model_names
        assert "model-2b" in model_names
        
        # Check provider health
        provider_info = model_manager.get_provider_info("test_provider_1")
        assert provider_info is not None
        assert provider_info.status == ProviderStatus.HEALTHY
        
        unhealthy_provider = model_manager.get_provider_info("unhealthy_provider")
        assert unhealthy_provider is not None
        assert unhealthy_provider.status == ProviderStatus.ERROR
    
    @pytest.mark.asyncio
    async def test_model_selection_strategies(self, model_manager, mock_chat_manager):
        """Test different model selection strategies."""
        model_manager.set_chat_manager(mock_chat_manager)
        await model_manager.discover_models(force_refresh=True)
        
        # Test manual selection (fallback to first available)
        model = await model_manager.select_best_model(ModelSelectionStrategy.MANUAL)
        assert model is not None
        assert model.is_available
        
        # Test fastest selection
        model = await model_manager.select_best_model(ModelSelectionStrategy.FASTEST)
        assert model is not None
        
        # Test cheapest selection (should prefer free models)
        model = await model_manager.select_best_model(ModelSelectionStrategy.CHEAPEST)
        assert model is not None
        assert model.cost_per_token is None  # Free model
        
        # Test balanced selection
        model = await model_manager.select_best_model(ModelSelectionStrategy.BALANCED)
        assert model is not None
    
    def test_model_info_properties(self):
        """Test ModelInfo properties and methods."""
        model_info = ModelInfo(
            name="test-model",
            provider="test_provider",
            display_name="Test Model",
            status=ModelStatus.AVAILABLE,
            parameters="7B",
            context_length=4096,
            cost_per_token=0.001
        )
        
        assert model_info.full_name == "test_provider:test-model"
        assert model_info.is_available
        assert model_info.size_human == "Unknown"  # No size_bytes set
        
        # Test with size
        model_info.size_bytes = 7_000_000_000
        assert "GB" in model_info.size_human
    
    def test_provider_info_properties(self):
        """Test ProviderInfo properties."""
        provider_info = ProviderInfo(
            name="test_provider",
            status=ProviderStatus.HEALTHY,
            response_time_ms=150.0,
            available_models=["model1", "model2"]
        )
        
        assert provider_info.is_healthy
        assert len(provider_info.available_models) == 2
        
        # Test unhealthy provider
        provider_info.status = ProviderStatus.ERROR
        assert not provider_info.is_healthy


class TestRAGSystem:
    """Test suite for RAG system."""
    
    @pytest.fixture
    def memory_manager(self):
        """Create a test memory manager."""
        return MemoryManager()
    
    @pytest.fixture
    def rag_system(self, memory_manager):
        """Create a test RAG system."""
        config = RAGConfig(
            max_memories=5,
            min_relevance_threshold=0.5,
            cite_sources=True,
            explain_reasoning=True
        )
        return RAGSystem(memory_manager, config)
    
    @pytest.fixture
    def sample_memories(self):
        """Create sample memories for testing."""
        return [
            Memory(
                id="mem1",
                content="Python is a programming language known for its simplicity.",
                memory_type=MemoryType.SEMANTIC,
                importance=0.8,
                created_at=datetime.now()
            ),
            Memory(
                id="mem2", 
                content="I helped the user with a Python script yesterday.",
                memory_type=MemoryType.EPISODIC,
                importance=0.6,
                created_at=datetime.now()
            ),
            Memory(
                id="mem3",
                content="The user prefers detailed explanations for programming concepts.",
                memory_type=MemoryType.LONG_TERM,
                importance=0.9,
                created_at=datetime.now()
            )
        ]
    
    def test_rag_config(self):
        """Test RAG configuration."""
        config = RAGConfig(
            max_memories=10,
            min_relevance_threshold=0.7,
            cite_sources=False
        )
        
        assert config.max_memories == 10
        assert config.min_relevance_threshold == 0.7
        assert not config.cite_sources
        assert config.max_context_tokens == 4000  # default
    
    @pytest.mark.asyncio
    async def test_retrieval_context_creation(self, rag_system, sample_memories):
        """Test retrieval context creation."""
        # Mock memory manager recall
        with patch.object(rag_system.memory_manager, 'recall', new_callable=AsyncMock) as mock_recall:
            mock_recall.return_value = sample_memories
            
            context = await rag_system.retrieve_context("python programming")
            
            assert context.query == "python programming"
            assert len(context.memories) == 3
            assert context.total_retrieved == 3
            assert context.retrieval_time_ms > 0
            assert "relevant memories" in context.reasoning.lower()
    
    @pytest.mark.asyncio
    async def test_prompt_enhancement(self, rag_system, sample_memories):
        """Test prompt enhancement with context."""
        context = RetrievalContext(
            memories=sample_memories,
            query="explain python",
            relevance_scores=[0.9, 0.7, 0.8],
            reasoning="Found relevant programming knowledge"
        )
        
        original_prompt = "What is Python?"
        enhanced_prompt = await rag_system.enhance_prompt(original_prompt, context)
        
        assert len(enhanced_prompt) > len(original_prompt)
        assert "Relevant Information from Memory" in enhanced_prompt
        assert "Python is a programming language" in enhanced_prompt
        assert "What is Python?" in enhanced_prompt
    
    @pytest.mark.asyncio 
    async def test_rag_response_generation(self, rag_system):
        """Test full RAG response generation."""
        # Mock components
        mock_provider = MockLLMProvider("test")
        
        with patch.object(rag_system, 'retrieve_context', new_callable=AsyncMock) as mock_retrieve:
            mock_context = RetrievalContext(
                memories=[],
                query="test query",
                reasoning="No memories found"
            )
            mock_retrieve.return_value = mock_context
            
            response, context = await rag_system.generate_rag_response(
                query="test query",
                llm_provider=mock_provider,
                model="test-model"
            )
            
            assert response == "Test response"
            assert context.query == "test query"
    
    def test_rag_stats(self, rag_system):
        """Test RAG statistics tracking."""
        stats = rag_system.get_stats()
        
        assert "total_queries" in stats
        assert "cache_hit_rate" in stats
        assert "config" in stats
        assert stats["total_queries"] == 0  # Fresh system
    
    def test_rag_config_update(self, rag_system):
        """Test RAG configuration updates."""
        new_config = RAGConfig(
            max_memories=20,
            min_relevance_threshold=0.8,
            cite_sources=False
        )
        
        rag_system.update_config(new_config)
        
        assert rag_system.config.max_memories == 20
        assert rag_system.config.min_relevance_threshold == 0.8
        assert not rag_system.config.cite_sources


class TestChatManagerIntegration:
    """Test integration of model management and RAG with chat manager."""
    
    @pytest.fixture
    def chat_manager(self):
        """Create a test chat manager."""
        # Mock the config to avoid file system dependencies
        with patch('src.core.chat_manager.settings') as mock_settings:
            mock_settings.DEFAULT_PROVIDER = "test"
            mock_settings.DEFAULT_MODEL = "test-model"
            
            # Mock the config manager
            with patch('src.core.chat_manager._config_manager') as mock_config:
                mock_config.get.return_value = False  # Disable providers
                
                chat_manager = ChatManager()
                
                # Add mock providers
                chat_manager.providers = {
                    "test": MockLLMProvider("test", ["test-model", "test-model-2"])
                }
                
                # Reset model manager with chat manager reference
                chat_manager.model_manager.set_chat_manager(chat_manager)
                
                return chat_manager
    
    @pytest.mark.asyncio
    async def test_chat_manager_model_discovery(self, chat_manager):
        """Test model discovery through chat manager."""
        success = await chat_manager.discover_models(force_refresh=True)
        assert success
        
        models = chat_manager.get_available_models()
        assert len(models) >= 2
        
        # Test model info retrieval
        model_info = chat_manager.get_model_info("test-model", "test")
        assert model_info is not None
        assert model_info.name == "test-model"
        assert model_info.provider == "test"
    
    @pytest.mark.asyncio
    async def test_model_switching(self, chat_manager):
        """Test model switching functionality."""
        await chat_manager.discover_models(force_refresh=True)
        
        # Test manual model selection
        success = chat_manager.set_model("test-model-2", "test")
        assert success
        assert chat_manager.current_model == "test-model-2"
        assert chat_manager.current_provider == "test"
        
        # Test automatic model selection
        best_model = await chat_manager.select_best_model(ModelSelectionStrategy.FASTEST)
        assert best_model is not None
    
    def test_rag_configuration(self, chat_manager):
        """Test RAG system configuration through chat manager."""
        # Test RAG enabling/disabling
        chat_manager.enable_rag(True)
        assert chat_manager.rag_enabled
        
        chat_manager.enable_rag(False) 
        assert not chat_manager.rag_enabled
        
        # Test RAG configuration
        new_config = RAGConfig(max_memories=15, cite_sources=True)
        chat_manager.configure_rag(new_config)
        
        assert chat_manager.rag_system.config.max_memories == 15
        assert chat_manager.rag_system.config.cite_sources
    
    @pytest.mark.asyncio
    async def test_rag_enhanced_messaging(self, chat_manager):
        """Test RAG-enhanced message sending."""
        # Enable RAG
        chat_manager.enable_rag(True)
        
        # Mock session for message storage
        chat_manager.current_session = MagicMock()
        chat_manager.current_session.add_message = AsyncMock()
        chat_manager.current_session.get_messages = AsyncMock(return_value=[])
        
        # Mock memory storage
        with patch.object(chat_manager, '_store_conversation_memory', new_callable=AsyncMock):
            response = await chat_manager.send_message_with_rag(
                content="What is Python?",
                provider="test",
                model="test-model"
            )
            
            assert response == "Test response"
    
    def test_provider_health_monitoring(self, chat_manager):
        """Test provider health monitoring."""
        health_status = chat_manager.get_provider_health()
        
        assert "test" in health_status
        # Provider info might be None before discovery
    
    def test_rag_stats_retrieval(self, chat_manager):
        """Test RAG statistics retrieval."""
        stats = chat_manager.get_rag_stats()
        
        assert isinstance(stats, dict)
        assert "total_queries" in stats


class TestUIComponents:
    """Test UI component functionality (mock testing)."""
    
    def test_model_info_card_creation(self):
        """Test ModelCard creation with mock data."""
        model_info = ModelInfo(
            name="test-model",
            provider="test_provider",
            display_name="Test Model",
            description="A test model for unit testing",
            status=ModelStatus.AVAILABLE,
            parameters="7B",
            capabilities=["chat", "code", "reasoning"],
            context_length=4096
        )
        
        # In a real test, we would create the actual widget
        # For now, just test the data structure
        assert model_info.display_name == "Test Model"
        assert len(model_info.capabilities) == 3
        assert model_info.is_available
    
    def test_rag_config_validation(self):
        """Test RAG configuration validation."""
        # Test valid config
        config = RAGConfig(
            max_memories=10,
            min_relevance_threshold=0.7,
            max_context_tokens=4000,
            retrieval_timeout_ms=5000
        )
        
        assert config.max_memories == 10
        assert 0.0 <= config.min_relevance_threshold <= 1.0
        assert config.max_context_tokens > 0
        assert config.retrieval_timeout_ms > 0
    
    def test_provider_health_display(self):
        """Test provider health information display."""
        provider_info = ProviderInfo(
            name="test_provider",
            status=ProviderStatus.HEALTHY,
            response_time_ms=150.0,
            available_models=["model1", "model2"],
            last_health_check=datetime.now()
        )
        
        # Test display formatting
        assert provider_info.name == "test_provider"
        assert provider_info.is_healthy
        assert len(provider_info.available_models) == 2


def run_performance_tests():
    """Run performance tests for critical paths."""
    print("\n=== Performance Tests ===")
    
    async def test_model_discovery_performance():
        """Test model discovery performance."""
        model_manager = ModelManager()
        
        # Mock chat manager with multiple providers
        mock_chat_manager = MagicMock()
        mock_chat_manager.providers = {
            f"provider_{i}": MockLLMProvider(f"provider_{i}", [f"model_{i}_{j}" for j in range(5)])
            for i in range(10)  # 10 providers, 5 models each
        }
        
        model_manager.set_chat_manager(mock_chat_manager)
        
        import time
        start_time = time.time()
        
        success = await model_manager.discover_models(force_refresh=True)
        
        end_time = time.time()
        discovery_time = end_time - start_time
        
        print(f"Model discovery time: {discovery_time:.3f}s")
        print(f"Models discovered: {len(model_manager.get_available_models())}")
        
        assert success
        assert discovery_time < 2.0  # Should complete within 2 seconds
    
    # Run async performance test
    asyncio.run(test_model_discovery_performance())


def run_integration_tests():
    """Run integration tests that require more complex setup."""
    print("\n=== Integration Tests ===")
    
    async def test_full_rag_workflow():
        """Test complete RAG workflow integration."""
        # Create temporary directory for memory storage
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize components
            memory_manager = MemoryManager()
            rag_system = RAGSystem(memory_manager)
            
            # Store some test memories
            await memory_manager.remember(
                "Python is a high-level programming language",
                MemoryType.SEMANTIC,
                importance=0.8
            )
            
            await memory_manager.remember(
                "User asked about Python yesterday and liked detailed explanations",
                MemoryType.EPISODIC,
                importance=0.7
            )
            
            # Test retrieval
            context = await rag_system.retrieve_context("what is python programming")
            
            assert len(context.memories) >= 1
            assert context.retrieval_time_ms > 0
            
            # Test prompt enhancement
            enhanced = await rag_system.enhance_prompt("Explain Python", context)
            assert "programming language" in enhanced.lower()
            
            print("✓ Full RAG workflow test passed")
    
    # Run integration test
    asyncio.run(test_full_rag_workflow())


if __name__ == "__main__":
    """Run tests when executed directly."""
    print("=== Model Management and RAG Test Suite ===")
    
    # Run pytest tests
    print("\nRunning unit tests...")
    pytest.main([__file__, "-v"])
    
    # Run custom performance tests
    run_performance_tests()
    
    # Run integration tests
    run_integration_tests()
    
    print("\n=== All Tests Complete ===")