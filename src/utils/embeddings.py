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
        self.model_name = model_name
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            # Suppress additional ONNX warnings during model loading
            import logging

            onnx_logger = logging.getLogger("onnxruntime")
            onnx_logger.setLevel(logging.ERROR)

            self._model = SentenceTransformer(self.model_name)
        return self._model

    async def generate(self, text: str) -> list[float]:
        """Generate embedding for text.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        try:
            # Generate embedding
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
            # Generate embeddings
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
