"""Complete warning suppression for embedding models and ONNX."""

import logging
import os
import sys
import warnings

# Set environment variables before any ML library imports
os.environ["ONNX_VERBOSE"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "0"
os.environ["PYTHONWARNINGS"] = "ignore"
# Additional ONNX runtime suppression
os.environ["ORT_DISABLE_ALL_LOGS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"  # Can help with context leaks
os.environ["ONNXRUNTIME_LOG_SEVERITY_LEVEL"] = "4"  # Error level only

# Suppress all categories of warnings aggressively
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

# Additional specific suppressions
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=ImportWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)

# Set logging levels for all noisy libraries
loggers_to_silence = [
    "onnxruntime",
    "onnxruntime.capi",
    "onnxruntime.capi.onnxruntime_pybind11_state",
    "sentence_transformers",
    "transformers",
    "torch",
    "tensorflow",
    "httpx",
    "urllib3",
    "requests",
    "chromadb",
    "chromadb.telemetry",
    "chromadb.db",
    "sqlite3",
]

for logger_name in loggers_to_silence:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.CRITICAL)
    logger.disabled = True  # Completely disable these loggers

# Additional suppression for context leak messages


class ContextLeakSuppressor:
    """Suppress 'Context leak detected' messages by filtering stderr."""

    def __init__(self):
        self.original_stderr = sys.stderr
        self.filtered_stderr = FilteredStderr(self.original_stderr)

    def __enter__(self):
        sys.stderr = self.filtered_stderr
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr = self.original_stderr


class FilteredStderr:
    """Filter stderr to suppress context leak messages."""

    def __init__(self, original_stderr):
        self.original_stderr = original_stderr

    def write(self, text):
        # Filter out context leak messages
        if "Context leak detected" not in text and "msgtracer returned -1" not in text:
            self.original_stderr.write(text)

    def flush(self):
        self.original_stderr.flush()

    def __getattr__(self, name):
        return getattr(self.original_stderr, name)


# Install context leak suppressor globally
_context_suppressor = ContextLeakSuppressor()
_context_suppressor.__enter__()


# Redirect stderr to suppress C-level warnings
class SuppressStderr:
    def __enter__(self):
        self.stderr = sys.stderr
        sys.stderr = open(os.devnull, "w")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self.stderr


# Export suppression context manager
__all__ = ["SuppressStderr", "ContextLeakSuppressor"]
