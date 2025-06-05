#!/usr/bin/env python3
"""Test runner script for Neuromancer test suite."""

import subprocess
import sys
from pathlib import Path


def run_tests():
    """Run the full test suite with different configurations."""
    test_dir = Path(__file__).parent

    # Basic test run
    print("🧪 Running basic test suite...")
    result = subprocess.run([sys.executable, "-m", "pytest", str(test_dir), "-v", "--tb=short"])

    if result.returncode != 0:
        print("❌ Basic tests failed!")
        return False

    print("✅ Basic tests passed!")

    # Unit tests only
    print("\n🔬 Running unit tests...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_dir / "unit"), "-v", "-m", "unit"]
    )

    if result.returncode != 0:
        print("❌ Unit tests failed!")
        return False

    print("✅ Unit tests passed!")

    # Integration tests only
    print("\n🔗 Running integration tests...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_dir / "integration"), "-v", "-m", "integration"]
    )

    if result.returncode != 0:
        print("❌ Integration tests failed!")
        return False

    print("✅ Integration tests passed!")

    print("\n🎉 All tests passed successfully!")
    return True


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
