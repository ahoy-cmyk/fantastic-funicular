#!/usr/bin/env python3
"""Verify that the development environment is properly set up."""

import subprocess
import sys


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version >= (3, 10):
        print("✓ Python version OK")
        return True
    else:
        print("❌ Python 3.10+ required")
        return False


def check_module(module_name, display_name=None):
    """Check if a module is installed."""
    display_name = display_name or module_name
    try:
        __import__(module_name)
        print(f"✓ {display_name} installed")
        return True
    except ImportError:
        print(f"❌ {display_name} not installed")
        return False


def check_kivy_deps():
    """Check Kivy dependencies."""
    print("\nChecking Kivy dependencies:")

    # SDL2 (needed for Kivy)
    try:
        import kivy

        print(f"✓ Kivy {kivy.__version__} installed")

        # Check providers
        from kivy.core import window

        print(f"✓ Kivy window provider: {window.Window}")
        return True
    except Exception as e:
        print(f"❌ Kivy error: {e}")
        return False


def check_services():
    """Check external services."""
    print("\nChecking external services:")

    # Check Ollama
    try:
        result = subprocess.run(["which", "ollama"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Ollama found")
        else:
            print("⚠️  Ollama not found (optional)")
    except:
        print("⚠️  Could not check for Ollama")


def main():
    """Run all checks."""
    print("=== Neuromancer Setup Verification ===\n")

    all_good = True

    # Python version
    if not check_python_version():
        all_good = False

    print("\nChecking core dependencies:")

    # Core modules
    modules = [
        ("kivy", "Kivy"),
        ("kivymd", "KivyMD"),
        ("openai", "OpenAI"),
        ("ollama", "Ollama Python"),
        ("sqlalchemy", "SQLAlchemy"),
        ("pydantic", "Pydantic"),
        ("chromadb", "ChromaDB"),
        ("sentence_transformers", "Sentence Transformers"),
    ]

    for module, name in modules:
        if not check_module(module, name):
            all_good = False

    # Kivy specific checks
    if not check_kivy_deps():
        all_good = False

    # External services
    check_services()

    print("\n" + "=" * 40)

    if all_good:
        print("✅ All core dependencies are installed!")
        print("\nYou can now run:")
        print("  python -m src.main")
        print("\nOr for testing:")
        print("  python src/main_simple.py")
    else:
        print("❌ Some dependencies are missing.")
        print("\nPlease run:")
        print("  ./scripts/setup.sh")
        print("\nOr manually:")
        print("  pip install -e .")

    return 0 if all_good else 1


if __name__ == "__main__":
    sys.exit(main())
