"""Unified relevance scoring for retrieval and context assembly.

This is the single source of truth for all relevance scoring logic.
Both Retriever and ContextAssembler use this module to ensure
consistent scoring behavior.
"""

import time
from dataclasses import dataclass

from .chunk import Chunk, ScoredChunk


@dataclass
class ScoringConfig:
    """Configuration for relevance scoring.

    Attributes:
        recency_decay_rate: Decay rate per hour for recency weighting.
            Higher values penalize older chunks more strongly.
        linked_chunk_penalty: Multiplier for linked chunks (tool_call <-> tool_result).
            Applied to parent's similarity before computing score.
        recent_chunk_base_score: Base similarity for recency-only retrieved chunks.
            Used when a chunk is retrieved by recency, not semantic search.
    """

    recency_decay_rate: float = 0.1
    linked_chunk_penalty: float = 0.9
    recent_chunk_base_score: float = 0.5


class RelevanceScorer:
    """Unified relevance scoring for retrieval and selection.

    Combines semantic similarity with recency weighting to produce
    final relevance scores. Used by both Retriever (to score search results)
    and ContextAssembler (to select chunks within token budget).

    Example:
        >>> scorer = RelevanceScorer()
        >>> scored = scorer.score(chunk, similarity=0.85)
        >>> print(f"Final score: {scored.score:.3f}")

        >>> # Select chunks within budget
        >>> selected = scorer.select_by_budget(scored_chunks, token_budget=4000)
    """

    def __init__(self, config: ScoringConfig | None = None):
        """Initialize scorer with configuration.

        Args:
            config: Scoring configuration. Uses defaults if None.
        """
        self.config = config or ScoringConfig()

    def score(
        self, chunk: Chunk, similarity: float, reference_time: float | None = None
    ) -> ScoredChunk:
        """Compute final relevance score combining similarity and recency.

        Score = similarity * recency_weight
        where recency_weight = 1 / (1 + age_hours * decay_rate)

        Args:
            chunk: The chunk to score.
            similarity: Raw semantic similarity [0, 1].
            reference_time: Unix timestamp for age calculation. Uses current time if None.

        Returns:
            ScoredChunk with computed score.
        """
        now = reference_time or time.time()
        age_hours = (now - chunk.timestamp) / 3600
        recency_weight = 1.0 / (1.0 + age_hours * self.config.recency_decay_rate)

        return ScoredChunk(
            chunk=chunk,
            similarity=similarity,
            score=similarity * recency_weight,
        )

    def score_recent(
        self, chunk: Chunk, reference_time: float | None = None
    ) -> ScoredChunk:
        """Score a chunk retrieved by recency (no semantic similarity).

        Uses a base similarity score since no query comparison was made.

        Args:
            chunk: The chunk to score.
            reference_time: Unix timestamp for age calculation.

        Returns:
            ScoredChunk with score based on recency only.
        """
        return self.score(chunk, self.config.recent_chunk_base_score, reference_time)

    def score_linked(
        self,
        chunk: Chunk,
        parent_similarity: float,
        reference_time: float | None = None,
    ) -> ScoredChunk:
        """Score a linked chunk (tool_call <-> tool_result).

        Applies a penalty to the parent's similarity since the linked
        chunk wasn't directly matched by the query.

        Args:
            chunk: The linked chunk to score.
            parent_similarity: Similarity of the parent chunk that was matched.
            reference_time: Unix timestamp for age calculation.

        Returns:
            ScoredChunk with adjusted score.
        """
        adjusted_sim = parent_similarity * self.config.linked_chunk_penalty
        return self.score(chunk, adjusted_sim, reference_time)

    def merge_scored(self, *scored_lists: list[ScoredChunk]) -> list[ScoredChunk]:
        """Merge multiple scored chunk lists, keeping highest score per chunk.

        When the same chunk appears in multiple lists (e.g., both semantic
        and recency results), keeps only the highest-scoring instance.

        Args:
            *scored_lists: Variable number of ScoredChunk lists to merge.

        Returns:
            Merged list with one entry per unique chunk ID.
        """
        seen: dict[str, ScoredChunk] = {}
        for scored_list in scored_lists:
            for sc in scored_list:
                if sc.chunk.id not in seen or sc.score > seen[sc.chunk.id].score:
                    seen[sc.chunk.id] = sc
        return list(seen.values())

    def select_top_k(self, scored: list[ScoredChunk], k: int) -> list[ScoredChunk]:
        """Select top-k chunks by score.

        Args:
            scored: List of scored chunks.
            k: Number of chunks to select.

        Returns:
            Top-k chunks sorted by descending score.
        """
        return sorted(scored, key=lambda sc: -sc.score)[:k]

    def select_by_budget(
        self, scored: list[ScoredChunk], token_budget: int
    ) -> list[ScoredChunk]:
        """Select chunks greedily by score until budget exhausted.

        Greedily selects highest-scoring chunks that fit within the
        token budget, then returns them in chronological order for
        coherent context presentation.

        Args:
            scored: List of scored chunks to select from.
            token_budget: Maximum total tokens to include.

        Returns:
            Selected chunks sorted by timestamp (chronological order).
        """
        sorted_chunks = sorted(scored, key=lambda sc: -sc.score)
        selected = []
        used_tokens = 0

        for sc in sorted_chunks:
            if used_tokens + sc.chunk.token_count <= token_budget:
                selected.append(sc)
                used_tokens += sc.chunk.token_count

        # Return in chronological order for coherence
        return sorted(selected, key=lambda sc: sc.chunk.timestamp)
