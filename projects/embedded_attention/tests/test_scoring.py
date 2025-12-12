"""Tests for RelevanceScorer."""

import time
import pytest

from projects.embedded_attention.core.chunk import Chunk, ScoredChunk
from projects.embedded_attention.core.scoring import RelevanceScorer, ScoringConfig


class TestScoringConfig:
    """Tests for ScoringConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ScoringConfig()
        assert config.recency_decay_rate == 0.1
        assert config.linked_chunk_penalty == 0.9
        assert config.recent_chunk_base_score == 0.5

    def test_custom_config(self):
        """Test custom configuration."""
        config = ScoringConfig(
            recency_decay_rate=0.2,
            linked_chunk_penalty=0.8,
            recent_chunk_base_score=0.6,
        )
        assert config.recency_decay_rate == 0.2


class TestRelevanceScorer:
    """Tests for RelevanceScorer."""

    @pytest.fixture
    def scorer(self):
        """Provide scorer with default config."""
        return RelevanceScorer()

    @pytest.fixture
    def recent_chunk(self):
        """Provide a chunk with current timestamp."""
        return Chunk(
            content="Recent content",
            timestamp=time.time(),
            token_count=10,
        )

    @pytest.fixture
    def old_chunk(self):
        """Provide a chunk from 1 hour ago."""
        return Chunk(
            content="Old content",
            timestamp=time.time() - 3600,  # 1 hour ago
            token_count=10,
        )

    def test_score_basic(self, scorer, recent_chunk):
        """Test basic scoring of a chunk."""
        scored = scorer.score(recent_chunk, similarity=0.8)

        assert isinstance(scored, ScoredChunk)
        assert scored.chunk is recent_chunk
        assert scored.similarity == 0.8
        assert scored.score > 0

    def test_score_recency_decay(self, scorer, recent_chunk, old_chunk):
        """Test that older chunks get lower scores."""
        now = time.time()
        recent_scored = scorer.score(recent_chunk, similarity=0.8, reference_time=now)
        old_scored = scorer.score(old_chunk, similarity=0.8, reference_time=now)

        # Same similarity, but old chunk should have lower score
        assert old_scored.score < recent_scored.score

    def test_score_zero_similarity(self, scorer, recent_chunk):
        """Test scoring with zero similarity."""
        scored = scorer.score(recent_chunk, similarity=0.0)
        assert scored.score == 0.0

    def test_score_recent(self, scorer, recent_chunk):
        """Test scoring by recency alone."""
        scored = scorer.score_recent(recent_chunk)

        # Should use base score as similarity
        assert scored.similarity == scorer.config.recent_chunk_base_score
        assert scored.score > 0

    def test_score_linked(self, scorer, recent_chunk):
        """Test scoring of linked chunks."""
        parent_similarity = 0.9
        scored = scorer.score_linked(recent_chunk, parent_similarity)

        # Linked chunks get penalized similarity
        expected_sim = parent_similarity * scorer.config.linked_chunk_penalty
        assert scored.similarity == pytest.approx(expected_sim)

    def test_merge_scored_deduplicates(self, scorer):
        """Test that merge keeps highest score per chunk ID."""
        chunk = Chunk(content="Test")

        list1 = [ScoredChunk(chunk=chunk, score=0.5)]
        list2 = [ScoredChunk(chunk=chunk, score=0.8)]

        merged = scorer.merge_scored(list1, list2)

        assert len(merged) == 1
        assert merged[0].score == 0.8

    def test_merge_scored_combines_different(self, scorer):
        """Test merging lists with different chunks."""
        chunk1 = Chunk(content="Chunk 1")
        chunk2 = Chunk(content="Chunk 2")

        list1 = [ScoredChunk(chunk=chunk1, score=0.5)]
        list2 = [ScoredChunk(chunk=chunk2, score=0.8)]

        merged = scorer.merge_scored(list1, list2)

        assert len(merged) == 2

    def test_select_top_k(self, scorer):
        """Test selecting top-k by score."""
        chunks = [
            ScoredChunk(chunk=Chunk(content=f"Chunk {i}"), score=score)
            for i, score in enumerate([0.3, 0.9, 0.1, 0.7, 0.5])
        ]

        top3 = scorer.select_top_k(chunks, k=3)

        assert len(top3) == 3
        assert top3[0].score == 0.9
        assert top3[1].score == 0.7
        assert top3[2].score == 0.5

    def test_select_by_budget(self, scorer):
        """Test selecting chunks within token budget."""
        chunks = [
            ScoredChunk(
                chunk=Chunk(content=f"Chunk {i}", token_count=30),
                score=score,
            )
            for i, score in enumerate([0.9, 0.8, 0.7, 0.6])
        ]

        # Budget of 65 tokens should fit 2 chunks (30 each = 60)
        selected = scorer.select_by_budget(chunks, token_budget=65)

        assert len(selected) == 2
        # Should select highest scoring chunks
        total_tokens = sum(sc.chunk.token_count for sc in selected)
        assert total_tokens <= 65

    def test_select_by_budget_returns_chronological(self, scorer):
        """Test that select_by_budget returns in chronological order."""
        now = time.time()
        chunks = [
            ScoredChunk(
                chunk=Chunk(content="New", token_count=10, timestamp=now),
                score=0.5,
            ),
            ScoredChunk(
                chunk=Chunk(content="Old", token_count=10, timestamp=now - 1000),
                score=0.9,
            ),
        ]

        selected = scorer.select_by_budget(chunks, token_budget=100)

        # Should be sorted by timestamp (old first)
        assert selected[0].chunk.content == "Old"
        assert selected[1].chunk.content == "New"
