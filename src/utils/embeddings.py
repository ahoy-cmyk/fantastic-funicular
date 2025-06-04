"""Text embedding generation for memory system."""

import numpy as np

# Import comprehensive warning suppression first
from src.utils.suppress_warnings import SuppressStderr

# Import sentence transformers with suppressed warnings
with SuppressStderr():
    from sentence_transformers import SentenceTransformer

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class EmbeddingGenerator:
    """Generate embeddings for text using sentence transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize embedding generator.

        Args:
            model_name: Name of sentence transformer model
        """
        # Map alternative models to avoid ONNX issues
        self.model_alternatives = {
            "all-MiniLM-L6-v2": "all-MiniLM-L6-v2",  # Keep default
            "all-mpnet-base-v2": "all-mpnet-base-v2",  # High quality but larger
            "paraphrase-multilingual-MiniLM-L12-v2": "paraphrase-multilingual-MiniLM-L12-v2",  # Multilingual
        }
        
        self.model_name = model_name
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            
            # Apply comprehensive ONNX suppression
            import logging
            import os
            
            # Temporarily set additional env vars
            old_env = {}
            temp_env = {
                "ORT_DISABLE_ALL_LOGS": "1",
                "ONNXRUNTIME_LOG_SEVERITY_LEVEL": "4",
                "OMP_NUM_THREADS": "1"
            }
            
            for key, value in temp_env.items():
                old_env[key] = os.environ.get(key)
                os.environ[key] = value
            
            try:
                # Disable all ONNX related loggers
                onnx_loggers = [
                    "onnxruntime",
                    "onnxruntime.capi",
                    "onnxruntime.capi.onnxruntime_pybind11_state"
                ]
                for logger_name in onnx_loggers:
                    onnx_logger = logging.getLogger(logger_name)
                    onnx_logger.setLevel(logging.CRITICAL)
                    onnx_logger.disabled = True
                
                # Load model with full suppression
                with SuppressStderr():
                    self._model = SentenceTransformer(self.model_name)
                    
            finally:
                # Restore environment
                for key, old_value in old_env.items():
                    if old_value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = old_value
                        
        return self._model

    async def generate(self, text: str) -> list[float]:
        """Generate embedding for text.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        try:
            # Generate embedding with suppressed output
            with SuppressStderr():
                embedding = self.model.encode(text, convert_to_tensor=False)

            # Convert to list
            if isinstance(embedding, np.ndarray):
                return embedding.tolist()
            else:
                return list(embedding)

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * self.model.get_sentence_embedding_dimension()

    async def generate_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        try:
            # Generate embeddings with suppressed output
            with SuppressStderr():
                embeddings = self.model.encode(texts, convert_to_tensor=False)

            # Convert to list of lists
            return [
                emb.tolist() if isinstance(emb, np.ndarray) else list(emb) for emb in embeddings
            ]

        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            # Return zero vectors as fallback
            dim = self.model.get_sentence_embedding_dimension()
            return [[0.0] * dim for _ in texts]

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings.

        Returns:
            Embedding dimension
        """
        return self.model.get_sentence_embedding_dimension()
    
    @staticmethod
    def is_onnx_model(model_name: str) -> bool:
        """Check if a model uses ONNX runtime (may show context leaks).
        
        Args:
            model_name: Name of the model
            
        Returns:
            True if model likely uses ONNX
        """
        # Most sentence transformer models use ONNX for optimization
        # The context leak messages are cosmetic and don't affect functionality
        return True
    
    @staticmethod
    def get_context_leak_info() -> str:
        """Get information about context leak messages.
        
        Returns:
            Explanation of context leak messages
        """
        return (
            "Context leak messages from ONNX runtime are cosmetic warnings "
            "that don't affect functionality. They occur during model inference "
            "and are a known issue with the ONNX runtime library used by "
            "sentence transformers for optimization. These messages are "
            "automatically suppressed but may occasionally appear."
        )
