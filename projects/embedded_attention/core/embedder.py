"""Embedding model wrapper for encoding conversation chunks."""

from typing import Union
import numpy as np


class Embedder:
    """Wrapper for sentence embedding models.

    Uses sentence-transformers for efficient text embedding with support
    for batch processing and query-specific handling.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        """Initialize the embedder with a sentence-transformers model.

        Args:
            model_name: HuggingFace model identifier. Recommended options:
                - "BAAI/bge-small-en-v1.5": 33M params, 384 dims, fast
                - "BAAI/bge-base-en-v1.5": 110M params, 768 dims, better quality
                - "sentence-transformers/all-mpnet-base-v2": 110M params, 768 dims
        """
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string.

        Args:
            text: Text to embed

        Returns:
            Normalized embedding vector of shape (dim,)
        """
        return self.model.encode(text, normalize_embeddings=True)

    def embed_batch(
        self, texts: list[str], batch_size: int = 32, show_progress: bool = False
    ) -> np.ndarray:
        """Embed multiple texts efficiently.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar

        Returns:
            Array of shape (len(texts), dim) with normalized embeddings
        """
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=show_progress,
        )

    def embed_query(self, query: str, context: str = "") -> np.ndarray:
        """Embed a query, optionally with recent context for short queries.

        Short queries like "yes", "do that", or "what about X?" lack semantic
        content on their own. Including recent assistant context helps produce
        more meaningful embeddings for retrieval.

        Args:
            query: The user's query text
            context: Optional recent context (e.g., last assistant response)

        Returns:
            Normalized embedding vector
        """
        # Short queries benefit from context
        if len(query.split()) < 5 and context:
            # Truncate context to last ~500 chars to avoid dominating the embedding
            truncated_context = context[-500:] if len(context) > 500 else context
            augmented_query = f"{truncated_context} {query}"
            return self.embed(augmented_query)
        return self.embed(query)

    def similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score in [-1, 1]
        """
        # Embeddings are already normalized, so dot product = cosine similarity
        return float(np.dot(embedding1, embedding2))

    def similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute pairwise similarity matrix.

        Args:
            embeddings: Array of shape (n, dim)

        Returns:
            Similarity matrix of shape (n, n)
        """
        # With normalized vectors, similarity matrix is just the Gram matrix
        return embeddings @ embeddings.T

    def most_similar(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """Find most similar embeddings to query.

        Args:
            query_embedding: Query vector of shape (dim,)
            candidate_embeddings: Candidate vectors of shape (n, dim)
            top_k: Number of results to return

        Returns:
            List of (index, similarity) tuples, sorted by similarity descending
        """
        similarities = candidate_embeddings @ query_embedding
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(int(idx), float(similarities[idx])) for idx in top_indices]

    def get_model_info(self) -> dict:
        """Get information about the embedding model.

        Returns:
            Dict with model name, dimension, and other metadata
        """
        return {
            "model_name": self.model_name,
            "dimension": self.dim,
            "max_seq_length": self.model.max_seq_length,
        }
