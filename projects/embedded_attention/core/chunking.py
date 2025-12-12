"""Text chunking strategies for splitting long content.

Separates chunking logic from storage - ChunkStore handles persistence,
this module handles text splitting decisions.
"""

from dataclasses import dataclass
from typing import Protocol


class Tokenizer(Protocol):
    """Protocol for any tokenizer with encode()."""

    def encode(self, text: str) -> list[int]:
        ...


@dataclass
class ChunkingConfig:
    """Configuration for text chunking.

    Attributes:
        max_tokens: Maximum tokens per chunk segment.
        split_on_paragraphs: Whether to split on paragraph boundaries first.
        split_on_sentences: Whether to split on sentence boundaries if needed.
    """

    max_tokens: int = 512
    split_on_paragraphs: bool = True
    split_on_sentences: bool = True


class TextChunker:
    """Splits long text into semantically coherent segments.

    Uses a two-pass approach:
    1. Split on paragraph boundaries (\n\n)
    2. If segments still exceed budget, split on sentences

    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> chunker = TextChunker(tokenizer)
        >>> segments = chunker.split("Very long text...")
        >>> for seg in segments:
        ...     print(seg["segment_index"], len(seg["content"]))
    """

    def __init__(self, tokenizer: Tokenizer, config: ChunkingConfig | None = None):
        """Initialize chunker with tokenizer.

        Args:
            tokenizer: Any object with encode(text) -> list[int] method.
            config: Chunking configuration. Uses defaults if None.
        """
        self.tokenizer = tokenizer
        self.config = config or ChunkingConfig()

    def split(self, content: str) -> list[dict]:
        """Split content into segments that fit within token budget.

        Args:
            content: Text to split.

        Returns:
            List of dicts with keys:
                - content: The segment text
                - segment_index: 0-based index
                - total_segments: Total number of segments
        """
        if not content.strip():
            return [{"content": "", "segment_index": 0, "total_segments": 1}]

        tokens = self.tokenizer.encode(content)
        if len(tokens) <= self.config.max_tokens:
            return [{"content": content, "segment_index": 0, "total_segments": 1}]

        # First pass: split by paragraphs
        if self.config.split_on_paragraphs:
            segments = self._split_by_paragraphs(content)
        else:
            segments = [content]

        # Second pass: split oversized segments by sentences
        if self.config.split_on_sentences:
            segments = self._refine_by_sentences(segments)

        return [
            {"content": seg, "segment_index": i, "total_segments": len(segments)}
            for i, seg in enumerate(segments)
        ]

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count tokens for.

        Returns:
            Number of tokens.
        """
        return len(self.tokenizer.encode(text))

    def _split_by_paragraphs(self, content: str) -> list[str]:
        """First pass: split on paragraph boundaries.

        Greedily combines paragraphs until adding another would
        exceed the token budget.

        Args:
            content: Full text to split.

        Returns:
            List of paragraph-based segments.
        """
        paragraphs = content.split("\n\n")
        segments = []
        current = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            test = f"{current}\n\n{para}" if current else para
            if len(self.tokenizer.encode(test)) > self.config.max_tokens and current:
                segments.append(current.strip())
                current = para
            else:
                current = test

        if current.strip():
            segments.append(current.strip())

        return segments if segments else [content]

    def _refine_by_sentences(self, segments: list[str]) -> list[str]:
        """Second pass: split oversized segments by sentence.

        Args:
            segments: List of paragraph-based segments.

        Returns:
            Refined list where all segments fit within budget.
        """
        refined = []
        for seg in segments:
            if len(self.tokenizer.encode(seg)) <= self.config.max_tokens:
                refined.append(seg)
            else:
                refined.extend(self._split_by_sentences(seg))
        return refined

    def _split_by_sentences(self, text: str) -> list[str]:
        """Split text by sentence boundaries.

        Uses simple heuristics (period + space). For production use,
        consider nltk.sent_tokenize or spacy for better accuracy.

        Args:
            text: Text to split by sentences.

        Returns:
            List of sentence-based segments.
        """
        # Simple sentence splitting - handles common cases
        # Replace common sentence endings with markers
        for end in [". ", "! ", "? "]:
            text = text.replace(end, end[0] + "\n")

        sentences = [s.strip() for s in text.split("\n") if s.strip()]
        segments = []
        current = ""

        for sent in sentences:
            test = f"{current} {sent}" if current else sent
            if len(self.tokenizer.encode(test)) > self.config.max_tokens and current:
                segments.append(current.strip())
                current = sent
            else:
                current = test

        if current.strip():
            segments.append(current.strip())

        return segments if segments else [text]
