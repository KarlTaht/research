"""Core components for Embedded Attention.

This module contains all the core implementation classes for the
RAG-based conversational memory system.
"""

# Core data structures
from .chunk import Chunk, ScoredChunk

# Text chunking
from .chunking import TextChunker, ChunkingConfig

# Storage
from .chunk_store import ChunkStore

# Embedding
from .embedder import Embedder

# Scoring
from .scoring import RelevanceScorer, ScoringConfig

# Retrieval
from .retriever import Retriever, RetrievalConfig

# Context assembly
from .context_assembler import ContextAssembler, ContextConfig

# Generation
from .generator import Generator, HFGenerator, APIGenerator, DummyGenerator

# Orchestration
from .conversation import Conversation

# Builder
from .builder import ConversationBuilder, ConversationConfig, create_conversation

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
