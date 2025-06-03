# Neuromancer - AI Assistant with Exceptional Memory

A cross-platform LLM interface with advanced memory capabilities and Model Context Protocol (MCP) support.

## Features

- **Multi-Provider Support**: Seamlessly switch between Ollama, OpenAI, and LM Studio
- **Advanced Memory System**:
  - Short-term and long-term memory with ChromaDB
  - Semantic search and similarity matching
  - Episodic memory for conversation tracking
  - Memory consolidation and optimization
- **MCP Integration**: Connect to MCP servers for extended tool capabilities
- **Cross-Platform GUI**: Built with Kivy, runs on macOS, Linux, and Windows
- **Modular Architecture**: Easy to extend with new providers and features

## Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/fantastic-funicular.git
   cd fantastic-funicular
   ```

2. **Run setup script**:
   ```bash
   ./scripts/setup.sh
   ```

3. **Configure your environment**:
   - Edit `.env` file with your API keys
   - Ensure Ollama is running if using local models

4. **Launch the application**:
   ```bash
   ./scripts/run.sh
   ```

## Architecture

```
src/
├── core/          # Core functionality
├── providers/     # LLM provider implementations
├── memory/        # Memory system with vector storage
├── mcp/           # Model Context Protocol integration
├── gui/           # Kivy-based user interface
└── utils/         # Utility functions
```

## Development

- Run tests: `pytest`
- Format code: `black src tests`
- Lint: `ruff src tests`
- Type check: `mypy src`

## Requirements

- Python 3.10+
- Ollama (for local models)
- OpenAI API key (optional)
- LM Studio (optional)

## License

MIT License - see LICENSE file for details.
