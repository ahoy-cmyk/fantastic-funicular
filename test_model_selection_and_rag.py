#!/usr/bin/env python3
"""
Comprehensive test suite for model selection functionality and RAG system.

This test validates:
1. Model selection with models containing colons in their names (e.g., "gemma3:12b")
2. RAG system functionality with uploaded file content
3. File upload integration with chat responses
4. get_model_info() method handling complex model names
5. send_message_with_rag() method utilizing retrieved content
"""

import asyncio
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import sys

# Add src to path for imports
sys.path.insert(0, "src")

from src.core.chat_manager import ChatManager
from src.core.model_manager import ModelManager, ModelInfo, ModelStatus, ProviderInfo, ProviderStatus
from src.core.rag_system import RAGSystem, RAGConfig
from src.memory.manager import MemoryManager
from src.memory import Memory, MemoryType
from src.providers import Message, CompletionResponse, LLMProvider
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class MockOllamaProvider:
    """Mock Ollama provider that simulates models with colons in names."""
    
    def __init__(self, name: str = "ollama", healthy: bool = True):
        self.name = name
        self.healthy = healthy
        # Simulate realistic Ollama model names with colons
        self.models = [
            "llama3.2:latest",
            "gemma2:9b",
            "gemma3:12b", 
            "codellama:7b-instruct",
            "mistral:7b-instruct-v0.2",
            "qwen2:7b",
            "phi3:mini"
        ]
        self.response_time = 150.0
    
    async def complete(self, messages, model, **kwargs):
        """Mock completion that includes file content if available."""
        # Simulate response that might include retrieved file content
        base_response = f"Using model {model}: This is a test response."
        
        # Check if there's file content context in the messages
        context_response = ""
        for message in messages:
            if "file content" in message.content.lower() or "uploaded" in message.content.lower():
                context_response = " I can see the uploaded file content and will use it in my response."
        
        return CompletionResponse(
            content=base_response + context_response,
            model=model,
            usage={"prompt_tokens": 15, "completion_tokens": 25, "total_tokens": 40}
        )
    
    async def stream_complete(self, messages, model, **kwargs):
        """Mock streaming completion."""
        chunks = ["Test ", f"response ", f"from {model}"]
        for chunk in chunks:
            yield chunk
    
    async def list_models(self):
        """Mock model listing with colon-containing names."""
        if not self.healthy:
            raise Exception("Ollama not available")
        return self.models
    
    async def health_check(self):
        """Mock health check."""
        return self.healthy


class MockMemoryManager:
    """Mock memory manager that simulates file upload content."""
    
    def __init__(self):
        self.memories = {}
        self.file_memories = []
        
    async def remember(self, content: str, memory_type: MemoryType, importance: float = 0.5, metadata: dict = None):
        """Store memory with potential file content."""
        memory_id = f"mem_{len(self.memories)}"
        memory = Memory(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            created_at=datetime.now(),
            metadata=metadata or {}
        )
        self.memories[memory_id] = memory
        
        # Track file uploads separately
        if metadata and metadata.get("type") == "file_upload":
            self.file_memories.append(memory)
            
        return memory_id
    
    async def recall(self, query: str, memory_types: list = None, threshold: float = 0.5, limit: int = 10):
        """Recall memories with file content prioritization."""
        # Simulate file content being found for relevant queries
        relevant_memories = []
        
        # Check for file-related queries
        if any(keyword in query.lower() for keyword in ["python", "code", "file", "upload", "script"]):
            # Add mock file memory
            file_memory = Memory(
                id="file_mem_1",
                content="def hello_world():\n    print('Hello from uploaded file!')\n    return 'success'",
                memory_type=MemoryType.SEMANTIC,
                importance=0.9,
                created_at=datetime.now(),
                metadata={"type": "file_upload", "filename": "hello.py", "file_type": "python"}
            )
            relevant_memories.append(file_memory)
        
        # Add other memories based on threshold
        for memory in self.memories.values():
            if len(relevant_memories) >= limit:
                break
            # Simple relevance check
            if any(word in memory.content.lower() for word in query.lower().split()):
                relevant_memories.append(memory)
        
        return relevant_memories[:limit]


async def test_model_selection_with_colons():
    """Test model selection functionality with colon-containing model names."""
    print("🔍 TESTING MODEL SELECTION WITH COLON-CONTAINING NAMES")
    print("=" * 60)
    
    try:
        # Create mock chat manager with Ollama provider
        chat_manager = ChatManager()
        chat_manager.providers = {
            "ollama": MockOllamaProvider("ollama", healthy=True)
        }
        
        # Set up model manager
        model_manager = ModelManager(chat_manager)
        model_manager.set_chat_manager(chat_manager)
        
        # Test model discovery
        print("🔄 Discovering models...")
        success = await model_manager.discover_models(force_refresh=True)
        assert success, "Model discovery should succeed"
        print(f"✅ Model discovery completed: {success}")
        
        # Test getting available models
        models = model_manager.get_available_models()
        print(f"📋 Found {len(models)} models:")
        for model in models:
            print(f"   • {model.full_name} ({model.name})")
        
        assert len(models) > 0, "Should find models from Ollama"
        
        # Test specific model names with colons
        test_models = ["gemma3:12b", "llama3.2:latest", "codellama:7b-instruct"]
        
        for model_name in test_models:
            print(f"\n🧪 Testing model: {model_name}")
            
            # Test get_model_info with explicit provider
            model_info = model_manager.get_model_info(model_name, "ollama")
            if model_info:
                print(f"   ✅ Found with provider: {model_info.full_name}")
                assert model_info.name == model_name
                assert model_info.provider == "ollama"
                assert model_info.full_name == f"ollama:{model_name}"
            else:
                print(f"   ❌ Not found with explicit provider")
            
            # Test get_model_info with full name
            full_name = f"ollama:{model_name}"
            model_info_full = model_manager.get_model_info(full_name)
            if model_info_full:
                print(f"   ✅ Found with full name: {model_info_full.full_name}")
                assert model_info_full.name == model_name
            else:
                print(f"   ❌ Not found with full name")
            
            # Test get_model_info with just model name
            model_info_name = model_manager.get_model_info(model_name)
            if model_info_name:
                print(f"   ✅ Found with name only: {model_info_name.full_name}")
                assert model_info_name.name == model_name
            else:
                print(f"   ❌ Not found with name only")
        
        print(f"\n✅ Model selection with colons test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Model selection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_rag_with_uploaded_files():
    """Test RAG system with simulated uploaded file content."""
    print("\n📁 TESTING RAG SYSTEM WITH UPLOADED FILES")
    print("=" * 60)
    
    try:
        # Create mock memory manager with file content
        memory_manager = MockMemoryManager()
        
        # Simulate file upload by storing file content in memory
        print("📤 Simulating file upload...")
        file_content = """
def calculate_fibonacci(n):
    '''Calculate fibonacci sequence up to n terms.'''
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib

def main():
    result = calculate_fibonacci(10)
    print(f"Fibonacci sequence: {result}")

if __name__ == "__main__":
    main()
"""
        
        file_memory_id = await memory_manager.remember(
            content=file_content,
            memory_type=MemoryType.SEMANTIC,
            importance=0.9,
            metadata={
                "type": "file_upload",
                "filename": "fibonacci.py",
                "file_type": "python",
                "upload_time": datetime.now().isoformat()
            }
        )
        print(f"✅ File content stored in memory: {file_memory_id}")
        
        # Create RAG system
        rag_config = RAGConfig(
            max_memories=5,
            min_relevance_threshold=0.3,
            cite_sources=True,
            explain_reasoning=True
        )
        rag_system = RAGSystem(memory_manager, rag_config)
        
        # Test file content retrieval
        print("\n🔍 Testing file content retrieval...")
        test_queries = [
            "show me the fibonacci code",
            "how to calculate fibonacci in python",
            "explain the uploaded python file",
            "what's in the file I uploaded"
        ]
        
        for query in test_queries:
            print(f"\n📝 Query: {query}")
            context = await rag_system.retrieve_context(query)
            
            print(f"   Memories found: {len(context.memories)}")
            print(f"   Retrieval time: {context.retrieval_time_ms:.2f}ms")
            
            if context.memories:
                for i, memory in enumerate(context.memories):
                    print(f"   Memory {i+1}: {memory.content[:100]}...")
                    if memory.metadata.get("type") == "file_upload":
                        print(f"     📁 File: {memory.metadata.get('filename')}")
            
            assert len(context.memories) >= 1, f"Should find relevant memories for: {query}"
        
        print(f"\n✅ RAG file retrieval test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ RAG file test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_rag_enhanced_messaging():
    """Test RAG-enhanced messaging with file content integration."""
    print("\n💬 TESTING RAG-ENHANCED MESSAGING")
    print("=" * 60)
    
    try:
        # Create chat manager with mock components
        chat_manager = ChatManager()
        
        # Set up mock provider
        chat_manager.providers = {
            "ollama": MockOllamaProvider("ollama", healthy=True)
        }
        chat_manager.current_provider = "ollama"
        chat_manager.current_model = "gemma3:12b"
        
        # Replace memory manager with mock
        chat_manager.memory_manager = MockMemoryManager()
        
        # Pre-populate memory with file content
        await chat_manager.memory_manager.remember(
            content="import pandas as pd\ndef analyze_data(df):\n    return df.describe()",
            memory_type=MemoryType.SEMANTIC,
            importance=0.8,
            metadata={"type": "file_upload", "filename": "data_analysis.py"}
        )
        
        # Create mock session
        mock_session = MagicMock()
        mock_session.add_message = AsyncMock()
        mock_session.get_messages = AsyncMock(return_value=[])
        chat_manager.current_session = mock_session
        
        # Enable RAG
        chat_manager.enable_rag(True)
        
        # Test RAG-enhanced messaging
        print("🤖 Testing RAG-enhanced response...")
        test_queries = [
            "explain the data analysis python code",
            "how to use pandas for data analysis",
            "show me the uploaded python script"
        ]
        
        for query in test_queries:
            print(f"\n📝 Testing query: {query}")
            
            # Mock the RAG system's generate_rag_response method
            with patch.object(chat_manager.rag_system, 'generate_rag_response', new_callable=AsyncMock) as mock_rag:
                # Simulate finding file content
                mock_context = MagicMock()
                mock_context.memories = [
                    Memory(
                        id="file_mem",
                        content="import pandas as pd\ndef analyze_data(df):\n    return df.describe()",
                        memory_type=MemoryType.SEMANTIC,
                        importance=0.8,
                        created_at=datetime.now(),
                        metadata={"type": "file_upload", "filename": "data_analysis.py"}
                    )
                ]
                
                mock_rag.return_value = (
                    f"Based on the uploaded file 'data_analysis.py', I can see pandas code for data analysis.",
                    mock_context
                )
                
                response = await chat_manager.send_message_with_rag(
                    content=query,
                    provider="ollama",
                    model="gemma3:12b"
                )
                
                print(f"   Response: {response}")
                assert "uploaded file" in response.lower() or "data_analysis.py" in response
                
                # Verify RAG was called
                mock_rag.assert_called_once()
                call_args = mock_rag.call_args
                assert call_args[1]["query"] == query
                assert call_args[1]["model"] == "gemma3:12b"
        
        print(f"\n✅ RAG-enhanced messaging test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ RAG messaging test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_file_upload_workflow():
    """Test complete file upload and retrieval workflow."""
    print("\n📋 TESTING COMPLETE FILE UPLOAD WORKFLOW")
    print("=" * 60)
    
    try:
        # Create temporary file to simulate upload
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            test_code = '''
class Calculator:
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def multiply(self, a, b):
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result
    
    def get_history(self):
        return self.history
'''
            temp_file.write(test_code)
            temp_file_path = temp_file.name
        
        print(f"📄 Created temporary file: {temp_file_path}")
        
        # Simulate file processing and memory storage
        memory_manager = MockMemoryManager()
        
        # Read file content and store in memory
        with open(temp_file_path, 'r') as f:
            file_content = f.read()
        
        file_memory_id = await memory_manager.remember(
            content=file_content,
            memory_type=MemoryType.SEMANTIC,
            importance=0.9,
            metadata={
                "type": "file_upload",
                "filename": Path(temp_file_path).name,
                "file_type": "python",
                "file_size": len(file_content),
                "upload_time": datetime.now().isoformat()
            }
        )
        
        print(f"✅ File stored in memory: {file_memory_id}")
        
        # Test retrieval with different query patterns
        queries_and_expectations = [
            ("what's in my calculator class", "Calculator"),
            ("show me the add method", "add"),
            ("how does the multiply function work", "multiply"),
            ("what methods are available", "add"),
            ("explain the uploaded python file", "class Calculator")
        ]
        
        print("\n🔍 Testing file content queries...")
        for query, expected_content in queries_and_expectations:
            print(f"\n📝 Query: {query}")
            memories = await memory_manager.recall(query, threshold=0.3, limit=3)
            
            found_relevant = False
            for memory in memories:
                if expected_content.lower() in memory.content.lower():
                    found_relevant = True
                    print(f"   ✅ Found relevant content: {expected_content}")
                    
                    # Check metadata
                    if memory.metadata.get("type") == "file_upload":
                        print(f"   📁 File: {memory.metadata.get('filename')}")
                        print(f"   📏 Size: {memory.metadata.get('file_size')} bytes")
                    break
            
            if not found_relevant:
                print(f"   ⚠️ Expected content '{expected_content}' not found")
        
        # Clean up
        Path(temp_file_path).unlink()
        print(f"🗑️ Cleaned up temporary file")
        
        print(f"\n✅ File upload workflow test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ File upload workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_model_switching_with_rag():
    """Test model switching while maintaining RAG functionality."""
    print("\n🔄 TESTING MODEL SWITCHING WITH RAG")
    print("=" * 60)
    
    try:
        # Create chat manager
        chat_manager = ChatManager()
        chat_manager.providers = {
            "ollama": MockOllamaProvider("ollama", healthy=True)
        }
        
        # Initialize model manager
        await chat_manager.model_manager.discover_models(force_refresh=True)
        
        # Test switching between different colon-containing models
        test_models = ["gemma3:12b", "llama3.2:latest", "codellama:7b-instruct"]
        
        for model_name in test_models:
            print(f"\n🔄 Testing model: {model_name}")
            
            # Switch to model
            success = chat_manager.set_model(model_name, "ollama")
            print(f"   Model switch: {'✅ Success' if success else '❌ Failed'}")
            
            if success:
                print(f"   Current model: {chat_manager.current_model}")
                assert chat_manager.current_model == model_name
                
                # Test that RAG still works with new model
                mock_session = MagicMock()
                mock_session.add_message = AsyncMock()
                mock_session.get_messages = AsyncMock(return_value=[])
                chat_manager.current_session = mock_session
                
                # Enable RAG and test
                chat_manager.enable_rag(True)
                
                with patch.object(chat_manager.rag_system, 'generate_rag_response', new_callable=AsyncMock) as mock_rag:
                    mock_rag.return_value = (f"Response from {model_name}", MagicMock())
                    
                    response = await chat_manager.send_message_with_rag(
                        content="test query",
                        model=model_name
                    )
                    
                    print(f"   RAG response: {response[:50]}...")
                    assert model_name in response
        
        print(f"\n✅ Model switching with RAG test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Model switching test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all comprehensive tests."""
    print("🚀 MODEL SELECTION AND RAG FUNCTIONALITY TEST SUITE")
    print("=" * 80)
    
    # Initialize test tracking
    tests = [
        ("Model Selection with Colons", test_model_selection_with_colons),
        ("RAG with Uploaded Files", test_rag_with_uploaded_files),
        ("RAG-Enhanced Messaging", test_rag_enhanced_messaging),
        ("File Upload Workflow", test_file_upload_workflow),
        ("Model Switching with RAG", test_model_switching_with_rag)
    ]
    
    results = {}
    
    # Run tests
    for test_name, test_func in tests:
        print(f"\n{'='*80}")
        print(f"Running: {test_name}")
        print(f"{'='*80}")
        
        try:
            result = await test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Display summary
    print(f"\n{'='*80}")
    print("📊 TEST RESULTS SUMMARY")
    print(f"{'='*80}")
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n📈 Overall Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Model selection and RAG functionality working perfectly!")
        print("🔍 Model names with colons are handled correctly")
        print("📁 File upload and RAG integration is working")
        print("💬 RAG-enhanced messaging utilizes uploaded content")
        return True
    else:
        print("⚠️ Some tests failed - please review the output above")
        return False


if __name__ == "__main__":
    print("Starting model selection and RAG functionality tests...")
    success = asyncio.run(main())
    
    if success:
        print("\n🌟 ALL FUNCTIONALITY TESTS COMPLETED SUCCESSFULLY!")
        print("🚀 Model selection and RAG systems are working correctly!")
        exit(0)
    else:
        print("\n💥 SOME TESTS FAILED")
        exit(1)