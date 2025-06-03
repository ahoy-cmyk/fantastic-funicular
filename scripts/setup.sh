#!/bin/bash
# Setup script for Neuromancer development environment

set -e

echo "🚀 Setting up Neuromancer development environment..."

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.10"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo "❌ Error: Python 3.10 or higher is required. Found: $PYTHON_VERSION"
    exit 1
fi

echo "✅ Python version: $PYTHON_VERSION"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else
    echo "📦 Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install dependencies
echo "📦 Installing dependencies..."
pip install -e .
pip install -e ".[dev]"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data logs scripts tests

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file..."
    cat > .env << EOF
# Neuromancer Configuration
DEBUG=True

# LLM Providers
DEFAULT_PROVIDER=ollama
DEFAULT_MODEL=llama3.2

# Ollama
OLLAMA_HOST=http://localhost:11434

# OpenAI (optional)
# OPENAI_API_KEY=your-api-key-here

# LM Studio
LMSTUDIO_HOST=http://localhost:1234

# UI Settings
THEME_STYLE=Dark
PRIMARY_PALETTE=DeepPurple
ACCENT_PALETTE=Cyan
EOF
    echo "✅ Created .env file - please update with your API keys"
else
    echo "📝 .env file already exists"
fi

# Install pre-commit hooks
if command -v pre-commit &> /dev/null; then
    echo "🪝 Installing pre-commit hooks..."
    pre-commit install
fi

echo "
✨ Setup complete! ✨

To start developing:
1. Activate the virtual environment: source venv/bin/activate
2. Update .env with your API keys if needed
3. Run the application: python -m src.main

For development:
- Run tests: pytest
- Format code: black src tests
- Lint code: ruff src tests
- Type check: mypy src

Happy coding! 🎉
"
