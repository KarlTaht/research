"""Pure data classes for conversation chunks.

This module contains only data structures with no business logic.
All chunk manipulation logic lives in chunking.py and scoring.py.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional
from uuid import uuid4

import numpy as np


@dataclass
class Chunk:
    """Immutable chunk of conversation content.

    Attributes:
        id: Unique identifier for the chunk.
        role: Type of content (user, assistant, tool_call, tool_result).
        content: The text content of the chunk.
        embedding: Normalized float32 vector from embedding model.
        timestamp: Unix timestamp when chunk was created.
        token_count: Number of tokens in content.
        conversation_id: ID of the conversation this chunk belongs to.
        conversation_group_id: Optional group ID for cross-conversation retrieval.
        parent_chunk_id: Links tool_call to its tool_result (or vice versa).
        segment_index: Index if this chunk is part of a split turn.
        total_segments: Total segments if turn was split.
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    role: Literal["user", "assistant", "tool_call", "tool_result"] = "user"
    content: str = ""
    embedding: Optional[np.ndarray] = None
    timestamp: float = 0.0
    token_count: int = 0
    conversation_id: str = ""
    conversation_group_id: Optional[str] = None
    parent_chunk_id: Optional[str] = None
    segment_index: int = 0
    total_segments: int = 1

    def __post_init__(self):
        """Validate embedding dtype if present."""
        if self.embedding is not None and self.embedding.dtype != np.float32:
            self.embedding = self.embedding.astype(np.float32)


@dataclass
class ScoredChunk:
    """Chunk with computed relevance score.

    Used throughout retrieval and context assembly to track
    both raw similarity and final weighted score.

    Attributes:
        chunk: The underlying Chunk data.
        similarity: Raw semantic similarity score [0, 1].
        score: Final score after recency weighting and other adjustments.
    """

    chunk: Chunk
    similarity: float = 0.0
    score: float = 0.0
