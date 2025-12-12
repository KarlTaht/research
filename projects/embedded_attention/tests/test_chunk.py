"""Tests for Chunk and ScoredChunk data classes."""

import numpy as np
import pytest

from projects.embedded_attention.core.chunk import Chunk, ScoredChunk


class TestChunk:
    """Tests for Chunk dataclass."""

    def test_chunk_creation_defaults(self):
        """Test creating a chunk with default values."""
        chunk = Chunk()
        assert chunk.id is not None
        assert chunk.role == "user"
        assert chunk.content == ""
        assert chunk.embedding is None
        assert chunk.timestamp == 0.0
        assert chunk.token_count == 0
        assert chunk.conversation_id == ""
        assert chunk.segment_index == 0
        assert chunk.total_segments == 1

    def test_chunk_creation_with_values(self):
        """Test creating a chunk with custom values."""
        embedding = np.random.randn(384).astype(np.float32)
        chunk = Chunk(
            role="assistant",
            content="Hello, world!",
            embedding=embedding,
            timestamp=1234567890.0,
            token_count=3,
            conversation_id="conv-123",
            conversation_group_id="group-456",
        )

        assert chunk.role == "assistant"
        assert chunk.content == "Hello, world!"
        assert np.array_equal(chunk.embedding, embedding)
        assert chunk.timestamp == 1234567890.0
        assert chunk.token_count == 3
        assert chunk.conversation_id == "conv-123"
        assert chunk.conversation_group_id == "group-456"

    def test_chunk_unique_ids(self):
        """Test that each chunk gets a unique ID."""
        chunk1 = Chunk()
        chunk2 = Chunk()
        assert chunk1.id != chunk2.id

    def test_chunk_roles(self):
        """Test different valid roles."""
        for role in ["user", "assistant", "tool_call", "tool_result"]:
            chunk = Chunk(role=role)
            assert chunk.role == role

    def test_chunk_with_parent(self):
        """Test chunk with parent_chunk_id for tool linking."""
        parent = Chunk(role="tool_call", content='{"tool": "search"}')
        child = Chunk(
            role="tool_result",
            content="Search results...",
            parent_chunk_id=parent.id,
        )

        assert child.parent_chunk_id == parent.id

    def test_chunk_segments(self):
        """Test multi-segment chunks."""
        chunks = [
            Chunk(content=f"Part {i}", segment_index=i, total_segments=3)
            for i in range(3)
        ]

        assert chunks[0].segment_index == 0
        assert chunks[1].segment_index == 1
        assert chunks[2].segment_index == 2
        assert all(c.total_segments == 3 for c in chunks)


class TestScoredChunk:
    """Tests for ScoredChunk dataclass."""

    def test_scored_chunk_creation(self):
        """Test creating a ScoredChunk."""
        chunk = Chunk(content="Test content")
        scored = ScoredChunk(
            chunk=chunk,
            similarity=0.85,
            score=0.72,
        )

        assert scored.chunk is chunk
        assert scored.similarity == 0.85
        assert scored.score == 0.72

    def test_scored_chunk_defaults(self):
        """Test ScoredChunk default values."""
        chunk = Chunk(content="Test")
        scored = ScoredChunk(chunk=chunk)

        assert scored.similarity == 0.0
        assert scored.score == 0.0

    def test_scored_chunk_sorting(self):
        """Test that ScoredChunks can be sorted by score."""
        chunks = [
            ScoredChunk(chunk=Chunk(content=f"Chunk {i}"), score=score)
            for i, score in enumerate([0.3, 0.9, 0.1, 0.7])
        ]

        sorted_chunks = sorted(chunks, key=lambda sc: -sc.score)

        assert sorted_chunks[0].score == 0.9
        assert sorted_chunks[1].score == 0.7
        assert sorted_chunks[2].score == 0.3
        assert sorted_chunks[3].score == 0.1
