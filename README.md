# 🧠 Neuromancer - AI Chat with Memory

A cross-platform Python application that provides an LLM interface with persistent memory capabilities and Model Context Protocol (MCP) support.

## ✨ Features

🎯 **Persistent Memory System**: ChromaDB-powered vector storage with four memory types (Short-term, Long-term, Episodic, Semantic) that remembers conversations and user information across sessions.

🔄 **Multiple LLM Providers**: Switch between Ollama, OpenAI, and LM Studio without losing conversation context.

🧠 **Retrieval-Augmented Generation (RAG)**: Enhanced responses using relevant memories with smart prioritization for personal information.

🔌 **MCP Integration**: WebSocket-based client for connecting to Model Context Protocol servers.

🎨 **Cross-Platform GUI**: Kivy-based interface with Material Design, supporting dark/light themes.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Ollama installed (for local models)

### Setup
```bash
git clone <repo-url>
cd fantastic-funicular
./scripts/setup.sh
```

### Configuration
Edit `.env` for your API keys:
```bash
OPENAI_API_KEY=your_key_here  # Optional
LMSTUDIO_HOST=http://localhost:1234  # Optional
```

### Run
```bash
./scripts/run.sh
```

## 🏗️ Architecture

```
src/
├── core/                    # Core functionality
│   ├── chat_manager.py      # Chat orchestration and RAG integration
│   ├── model_manager.py     # Provider switching and model management
│   ├── rag_system.py        # Retrieval-augmented generation
│   └── config.py           # Configuration management
├── providers/               # LLM provider implementations
│   ├── ollama_provider.py   # Ollama local models
│   ├── openai_provider.py   # OpenAI API integration
│   └── lmstudio_provider.py # LM Studio support
├── memory/                  # Memory system
│   ├── manager.py          # Memory operations and consolidation
│   ├── vector_store.py     # ChromaDB vector storage
│   └── intelligent_analyzer.py # Content analysis for memory formation
├── mcp/                     # Model Context Protocol
│   ├── client.py           # WebSocket MCP client
│   └── manager.py          # MCP server management
├── gui/                     # Kivy user interface
│   ├── app.py              # Main application
│   └── screens/            # Chat, memory, and settings screens
└── utils/                   # Utilities
    ├── embeddings.py       # Text embedding generation
    └── logger.py           # Logging configuration
```

## 💾 Memory System Details

The memory system uses ChromaDB for vector storage with:

- **Short-term**: Recent conversation context (24-hour default retention)
- **Long-term**: Important facts and user preferences
- **Episodic**: Specific events and experiences with timestamps
- **Semantic**: General knowledge and concepts

Features include:
- Automatic memory consolidation from short to long-term based on importance scores
- Vector similarity search with configurable thresholds
- Personal information prioritization in RAG retrieval
- Memory deduplication and contradiction filtering

## 🔧 Development

### Install dev dependencies
```bash
pip install -e ".[dev]"
```

### Run tests
```bash
pytest --cov=src --cov-report=html
```

### Code formatting
```bash
black src tests
ruff src tests
mypy src
```

## 📋 Requirements

- Python 3.10+
- ChromaDB for vector storage
- Kivy for GUI
- sentence-transformers for embeddings
- Optional: Ollama, OpenAI API key, LM Studio

## 🔌 MCP Integration

Supports WebSocket connections to MCP servers for extended functionality. Configure servers in the settings screen or via configuration files.

## ⚙️ Configuration

The application uses Pydantic for configuration management with support for:
- Environment variables
- JSON configuration files
- Runtime configuration updates

## 📜 License

MIT License