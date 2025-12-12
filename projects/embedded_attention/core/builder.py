"""ConversationBuilder for wiring all conversation components.

This factory handles all component instantiation and dependency injection,
keeping the Conversation class thin and focused on orchestration.
"""

from dataclasses import dataclass
from typing import Optional

from .chunk import Chunk, ScoredChunk
from .chunk_store import ChunkStore
from .chunking import TextChunker, ChunkingConfig
from .embedder import Embedder
from .retriever import Retriever, RetrievalConfig
from .scoring import RelevanceScorer, ScoringConfig
from .context_assembler import ContextAssembler, ContextConfig
from .generator import Generator, DummyGenerator
from .conversation import Conversation


@dataclass
class ConversationConfig:
    """Unified configuration for all conversation components.

    Groups all configuration options in one place for convenience.
    Individual component configs can still be used for fine-grained control.

    Attributes:
        db_path: Path to DuckDB database (":memory:" for in-memory).
        embedding_model: HuggingFace model name for embeddings.
        max_chunk_tokens: Maximum tokens per chunk segment.
        semantic_top_k: Number of chunks from semantic search.
        recent_n: Number of recent chunks to always include.
        min_similarity: Minimum cosine similarity threshold.
        cross_conversation: Enable cross-conversation retrieval.
        recency_decay_rate: Decay rate per hour for recency scoring.
        total_budget: Total context window size in tokens.
        reserved_for_generation: Tokens reserved for model output.
    """

    # Storage
    db_path: str = ":memory:"

    # Embedding
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Chunking
    max_chunk_tokens: int = 512

    # Retrieval
    semantic_top_k: int = 5
    recent_n: int = 2
    min_similarity: float = 0.7
    cross_conversation: bool = False

    # Scoring
    recency_decay_rate: float = 0.1

    # Context
    total_budget: int = 8192
    reserved_for_generation: int = 1024


class ConversationBuilder:
    """Fluent builder for creating configured Conversation instances.

    Handles all component instantiation and wiring, providing a clean
    interface for configuration. The builder pattern allows for:
    - Fluent chained configuration
    - Sensible defaults that can be overridden
    - Proper dependency injection

    Example:
        >>> conv = (
        ...     ConversationBuilder()
        ...     .with_generator(my_generator)
        ...     .with_db_path("conversations.duckdb")
        ...     .with_cross_conversation(True)
        ...     .build(conversation_group_id="project-alpha")
        ... )

        >>> # Or use convenience function
        >>> from embedded_attention import create_conversation
        >>> conv = create_conversation(generator=my_generator)
    """

    def __init__(self, config: Optional[ConversationConfig] = None):
        """Initialize builder with optional unified configuration.

        Args:
            config: Unified configuration. Uses defaults if None.
        """
        self.config = config or ConversationConfig()
        self._generator: Optional[Generator] = None
        self._embedder: Optional[Embedder] = None
        self._store: Optional[ChunkStore] = None
        self._tokenizer = None

    def with_generator(self, generator: Generator) -> "ConversationBuilder":
        """Set the generator to use for response generation.

        Args:
            generator: Generator instance (HFGenerator, APIGenerator, etc.)

        Returns:
            Self for method chaining.
        """
        self._generator = generator
        return self

    def with_db_path(self, path: str) -> "ConversationBuilder":
        """Set the database path for chunk storage.

        Args:
            path: Path to DuckDB file, or ":memory:" for in-memory.

        Returns:
            Self for method chaining.
        """
        self.config.db_path = path
        return self

    def with_cross_conversation(self, enabled: bool = True) -> "ConversationBuilder":
        """Enable or disable cross-conversation retrieval.

        When enabled, retrieval can search across all conversations
        in the same conversation_group_id.

        Args:
            enabled: Whether to enable cross-conversation retrieval.

        Returns:
            Self for method chaining.
        """
        self.config.cross_conversation = enabled
        return self

    def with_embedding_model(self, model_name: str) -> "ConversationBuilder":
        """Set the embedding model to use.

        Args:
            model_name: HuggingFace model name (e.g., "BAAI/bge-small-en-v1.5").

        Returns:
            Self for method chaining.
        """
        self.config.embedding_model = model_name
        return self

    def with_retrieval_config(
        self,
        semantic_top_k: Optional[int] = None,
        recent_n: Optional[int] = None,
        min_similarity: Optional[float] = None,
    ) -> "ConversationBuilder":
        """Configure retrieval parameters.

        Args:
            semantic_top_k: Number of chunks from semantic search.
            recent_n: Number of recent chunks to always include.
            min_similarity: Minimum cosine similarity threshold.

        Returns:
            Self for method chaining.
        """
        if semantic_top_k is not None:
            self.config.semantic_top_k = semantic_top_k
        if recent_n is not None:
            self.config.recent_n = recent_n
        if min_similarity is not None:
            self.config.min_similarity = min_similarity
        return self

    def with_context_budget(
        self,
        total_budget: Optional[int] = None,
        reserved_for_generation: Optional[int] = None,
    ) -> "ConversationBuilder":
        """Configure token budget for context assembly.

        Args:
            total_budget: Total context window size in tokens.
            reserved_for_generation: Tokens reserved for model output.

        Returns:
            Self for method chaining.
        """
        if total_budget is not None:
            self.config.total_budget = total_budget
        if reserved_for_generation is not None:
            self.config.reserved_for_generation = reserved_for_generation
        return self

    def with_existing_store(self, store: ChunkStore) -> "ConversationBuilder":
        """Use an existing ChunkStore instance.

        Useful when sharing a store across multiple conversations.

        Args:
            store: Existing ChunkStore instance.

        Returns:
            Self for method chaining.
        """
        self._store = store
        return self

    def with_existing_embedder(self, embedder: Embedder) -> "ConversationBuilder":
        """Use an existing Embedder instance.

        Useful when sharing an embedder across multiple conversations
        to avoid loading the model multiple times.

        Args:
            embedder: Existing Embedder instance.

        Returns:
            Self for method chaining.
        """
        self._embedder = embedder
        return self

    def build(
        self,
        conversation_id: Optional[str] = None,
        conversation_group_id: Optional[str] = None,
    ) -> Conversation:
        """Build and return a fully configured Conversation.

        All components are instantiated and wired together.
        The scorer is shared between Retriever and ContextAssembler
        to ensure consistent scoring behavior.

        Args:
            conversation_id: Optional ID for this conversation (auto-generated if None).
            conversation_group_id: Optional group ID for cross-conversation retrieval.

        Returns:
            Configured Conversation instance.
        """
        from transformers import AutoTokenizer

        # Initialize or reuse shared components
        embedder = self._embedder or Embedder(self.config.embedding_model)
        store = self._store or ChunkStore(self.config.db_path, dim=embedder.dim)
        tokenizer = self._tokenizer or AutoTokenizer.from_pretrained("gpt2")

        # Build component configs
        chunking_config = ChunkingConfig(max_tokens=self.config.max_chunk_tokens)

        scoring_config = ScoringConfig(
            recency_decay_rate=self.config.recency_decay_rate
        )

        retrieval_config = RetrievalConfig(
            semantic_top_k=self.config.semantic_top_k,
            recent_n=self.config.recent_n,
            min_similarity=self.config.min_similarity,
            cross_conversation=self.config.cross_conversation,
        )

        context_config = ContextConfig(
            total_budget=self.config.total_budget,
            reserved_for_generation=self.config.reserved_for_generation,
        )

        # Wire components with shared scorer
        chunker = TextChunker(tokenizer, chunking_config)
        scorer = RelevanceScorer(scoring_config)
        retriever = Retriever(store, embedder, scorer, retrieval_config)
        assembler = ContextAssembler(scorer, context_config)
        generator = self._generator or DummyGenerator()

        return Conversation(
            store=store,
            chunker=chunker,
            embedder=embedder,
            retriever=retriever,
            assembler=assembler,
            generator=generator,
            conversation_id=conversation_id,
            conversation_group_id=conversation_group_id,
        )


def create_conversation(
    db_path: str = ":memory:",
    generator: Optional[Generator] = None,
    conversation_id: Optional[str] = None,
    conversation_group_id: Optional[str] = None,
    **kwargs,
) -> Conversation:
    """Convenience function to create a conversation with defaults.

    For more control, use ConversationBuilder directly.

    Args:
        db_path: Path to DuckDB database (":memory:" for in-memory).
        generator: Generator instance (uses DummyGenerator if None).
        conversation_id: Optional conversation ID.
        conversation_group_id: Optional group ID for cross-conversation retrieval.
        **kwargs: Additional config options passed to ConversationConfig.

    Returns:
        Configured Conversation instance.

    Example:
        >>> conv = create_conversation()
        >>> response = conv.chat("Hello!")

        >>> # With custom settings
        >>> conv = create_conversation(
        ...     db_path="memory.duckdb",
        ...     semantic_top_k=10,
        ...     min_similarity=0.6,
        ... )
    """
    config = ConversationConfig(db_path=db_path, **kwargs)
    builder = ConversationBuilder(config)

    if generator:
        builder.with_generator(generator)

    return builder.build(
        conversation_id=conversation_id,
        conversation_group_id=conversation_group_id,
    )
