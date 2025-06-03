#!/bin/bash
# Run script for Neuromancer

set -e

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run: ./scripts/setup.sh"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if dependencies are installed
if ! python -c "import kivy" 2>/dev/null; then
    echo "❌ Dependencies not installed. Please run: ./scripts/setup.sh"
    exit 1
fi

# Run the application
echo "🚀 Starting Neuromancer..."
python -m src.main "$@"
