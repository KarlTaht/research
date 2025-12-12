"""Embedded Attention: Chunk-based conversational memory with RAG.

This package implements a retrieval-augmented generation (RAG) system for
conversational AI, where each conversation turn is encoded, stored in a
vector database, and dynamically retrieved to augment the LLM's context window.

Key components:
- Chunk/ScoredChunk: Pure data classes for conversation chunks
- TextChunker: Splits long content into semantically coherent segments
- ChunkStore: DuckDB + HNSW vector storage for conversation chunks
- Embedder: Sentence embedding wrapper (default: BGE-small)
- RelevanceScorer: Unified scoring for retrieval and selection
- Retriever: Hybrid retrieval (semantic + recency + linked chunks)
- ContextAssembler: Token budget management and formatting
- Generator: HuggingFace-compatible generation interface
- Conversation: Thin orchestration layer
- ConversationBuilder: Factory for creating configured conversations

Example usage:
    from projects.embedded_attention import create_conversation
    from projects.embedded_attention.core.generator import HFGenerator

    # With a HuggingFace model
    generator = HFGenerator.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    conv = create_conversation(generator=generator)

    # Chat with memory
    response = conv.chat("Hello! My name is Alice.")
    response = conv.chat("What's my name?")  # Will retrieve the first message

    # Or use the builder for more control
    from projects.embedded_attention import ConversationBuilder

    conv = (
        ConversationBuilder()
        .with_generator(generator)
        .with_db_path("conversations.duckdb")
        .with_cross_conversation(True)
        .build(conversation_group_id="project-alpha")
    )
"""

# Re-export everything from core
from .core import (
    # Core data structures
    Chunk,
    ScoredChunk,
    # Chunking
    TextChunker,
    ChunkingConfig,
    # Storage
    ChunkStore,
    # Embedding
    Embedder,
    # Scoring
    RelevanceScorer,
    ScoringConfig,
    # Retrieval
    Retriever,
    RetrievalConfig,
    # Context assembly
    ContextAssembler,
    ContextConfig,
    # Generation
    Generator,
    HFGenerator,
    APIGenerator,
    DummyGenerator,
    # Orchestration
    Conversation,
    # Builder
    ConversationBuilder,
    ConversationConfig,
    create_conversation,
)

__all__ = [
    # Core data structures
    "Chunk",
    "ScoredChunk",
    # Chunking
    "TextChunker",
    "ChunkingConfig",
    # Storage
    "ChunkStore",
    # Embedding
    "Embedder",
    # Scoring
    "RelevanceScorer",
    "ScoringConfig",
    # Retrieval
    "Retriever",
    "RetrievalConfig",
    # Context assembly
    "ContextAssembler",
    "ContextConfig",
    # Generation
    "Generator",
    "HFGenerator",
    "APIGenerator",
    "DummyGenerator",
    # Orchestration
    "Conversation",
    # Builder
    "ConversationBuilder",
    "ConversationConfig",
    "create_conversation",
]

__version__ = "0.1.0"
