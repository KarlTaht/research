"""Tests for TextChunker."""

import pytest

from projects.embedded_attention.core.chunking import TextChunker, ChunkingConfig


class MockTokenizer:
    """Simple tokenizer mock that splits on whitespace."""

    def encode(self, text: str) -> list[int]:
        """Count words as tokens (simple approximation)."""
        return list(range(len(text.split())))


class TestChunkingConfig:
    """Tests for ChunkingConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ChunkingConfig()
        assert config.max_tokens == 512
        assert config.split_on_paragraphs is True
        assert config.split_on_sentences is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = ChunkingConfig(
            max_tokens=128,
            split_on_paragraphs=False,
            split_on_sentences=True,
        )
        assert config.max_tokens == 128
        assert config.split_on_paragraphs is False


class TestTextChunker:
    """Tests for TextChunker."""

    @pytest.fixture
    def tokenizer(self):
        """Provide mock tokenizer."""
        return MockTokenizer()

    @pytest.fixture
    def chunker(self, tokenizer):
        """Provide chunker with default config."""
        return TextChunker(tokenizer, ChunkingConfig(max_tokens=10))

    def test_short_content_single_chunk(self, chunker):
        """Test that short content stays as single chunk."""
        content = "Hello world"  # 2 tokens
        segments = chunker.split(content)

        assert len(segments) == 1
        assert segments[0]["content"] == content
        assert segments[0]["segment_index"] == 0
        assert segments[0]["total_segments"] == 1

    def test_long_content_splits(self, chunker):
        """Test that long content is split into multiple chunks."""
        # Content with sentence boundaries so chunker can split
        # 15+ words with periods, max is 10 tokens
        content = "One two three four five. Six seven eight nine ten. Eleven twelve thirteen fourteen fifteen."
        segments = chunker.split(content)

        assert len(segments) > 1
        # Each segment should have segment_index and total_segments
        for i, seg in enumerate(segments):
            assert seg["segment_index"] == i
            assert seg["total_segments"] == len(segments)

    def test_paragraph_splitting(self, tokenizer):
        """Test splitting on paragraph boundaries."""
        chunker = TextChunker(
            tokenizer,
            ChunkingConfig(max_tokens=10, split_on_paragraphs=True),
        )

        content = "First paragraph here.\n\nSecond paragraph follows.\n\nThird one too."
        segments = chunker.split(content)

        # Should respect paragraph boundaries
        assert len(segments) >= 1

    def test_sentence_splitting(self, tokenizer):
        """Test splitting on sentence boundaries."""
        chunker = TextChunker(
            tokenizer,
            ChunkingConfig(max_tokens=5, split_on_sentences=True),
        )

        content = "First sentence here. Second sentence follows. Third sentence too."
        segments = chunker.split(content)

        # Should split at sentence boundaries
        assert len(segments) >= 2

    def test_count_tokens(self, chunker):
        """Test token counting."""
        text = "one two three"
        count = chunker.count_tokens(text)
        assert count == 3

    def test_empty_content(self, chunker):
        """Test handling of empty content."""
        segments = chunker.split("")

        assert len(segments) == 1
        assert segments[0]["content"] == ""

    def test_whitespace_handling(self, chunker):
        """Test that whitespace is handled correctly."""
        content = "   Hello   world   "
        segments = chunker.split(content)

        assert len(segments) == 1
        # Content should be preserved (chunker doesn't strip)
        assert segments[0]["content"] == content
