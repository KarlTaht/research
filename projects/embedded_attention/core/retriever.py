"""Hybrid retrieval strategy combining semantic search, recency, and linked chunks.

Uses RelevanceScorer (from scoring.py) for all scoring decisions to ensure
consistent scoring behavior across retrieval and context assembly.
"""

import time
from dataclasses import dataclass
from typing import Optional

from .chunk import Chunk, ScoredChunk
from .chunk_store import ChunkStore
from .embedder import Embedder
from .scoring import RelevanceScorer


@dataclass
class RetrievalConfig:
    """Configuration for the retrieval strategy."""

    semantic_top_k: int = 5  # Number of chunks from semantic search
    recent_n: int = 2  # Number of recent chunks to always include
    min_similarity: float = 0.7  # Minimum cosine similarity threshold
    include_linked: bool = True  # Include linked tool_call/tool_result pairs
    cross_conversation: bool = False  # Enable cross-conversation retrieval


class Retriever:
    """Hybrid retriever combining multiple retrieval strategies.

    Retrieval strategy:
    1. Semantic search: Find top-k most similar chunks by embedding
    2. Recency: Always include the n most recent chunks from current conversation
    3. Linked chunks: Include tool_call/tool_result pairs when one is retrieved

    Results are deduplicated and merged via the scorer, which keeps the highest
    score for each unique chunk.
    """

    def __init__(
        self,
        store: ChunkStore,
        embedder: Embedder,
        scorer: RelevanceScorer,
        config: Optional[RetrievalConfig] = None,
    ):
        """Initialize the retriever.

        Args:
            store: ChunkStore for vector search and retrieval.
            embedder: Embedder for encoding queries.
            scorer: RelevanceScorer for scoring chunks (shared with ContextAssembler).
            config: Retrieval configuration (uses defaults if not provided).
        """
        self.store = store
        self.embedder = embedder
        self.scorer = scorer
        self.config = config or RetrievalConfig()

    def retrieve(
        self,
        query: str,
        conversation_id: str,
        conversation_group_id: Optional[str] = None,
        recent_context: str = "",
        reference_time: Optional[float] = None,
    ) -> list[ScoredChunk]:
        """Retrieve relevant chunks using hybrid strategy.

        Args:
            query: The user's query text.
            conversation_id: Current conversation ID.
            conversation_group_id: Optional group ID for cross-conversation retrieval.
            recent_context: Recent assistant response for query augmentation.
            reference_time: Unix timestamp for scoring (uses current time if None).

        Returns:
            List of ScoredChunks, deduplicated with highest score per chunk.
        """
        now = reference_time or time.time()

        # 1. Embed query (with context augmentation for short queries)
        query_emb = self.embedder.embed_query(query, context=recent_context)

        # 2. Semantic search -> score results
        semantic_results = self._semantic_search(
            query_emb, conversation_id, conversation_group_id
        )
        semantic_scored = [
            self.scorer.score(chunk, sim, now) for chunk, sim in semantic_results
        ]

        # 3. Recent chunks -> score with recency-only scoring
        recent_chunks = self.store.get_recent(self.config.recent_n, conversation_id)
        recent_scored = [self.scorer.score_recent(c, now) for c in recent_chunks]

        # 4. Linked chunks -> score with penalty
        linked_scored = []
        if self.config.include_linked:
            for sc in semantic_scored:
                for linked in self.store.get_linked(sc.chunk.id):
                    linked_scored.append(
                        self.scorer.score_linked(linked, sc.similarity, now)
                    )

        # 5. Merge (keeps highest score per chunk)
        return self.scorer.merge_scored(semantic_scored, recent_scored, linked_scored)

    def _semantic_search(
        self,
        query_emb,
        conversation_id: str,
        conversation_group_id: Optional[str],
    ) -> list[tuple[Chunk, float]]:
        """Perform semantic similarity search."""
        if self.config.cross_conversation and conversation_group_id:
            # Search across conversation group
            return self.store.query(
                query_emb,
                top_k=self.config.semantic_top_k,
                min_similarity=self.config.min_similarity,
                conversation_group_id=conversation_group_id,
            )
        else:
            # Search within current conversation only
            return self.store.query(
                query_emb,
                top_k=self.config.semantic_top_k,
                min_similarity=self.config.min_similarity,
                conversation_id=conversation_id,
            )

    def retrieve_with_breakdown(
        self,
        query: str,
        conversation_id: str,
        conversation_group_id: Optional[str] = None,
        recent_context: str = "",
        reference_time: Optional[float] = None,
    ) -> dict[str, list[ScoredChunk]]:
        """Retrieve with detailed breakdown by source.

        Useful for debugging and analysis.

        Returns:
            Dict with keys 'semantic', 'recent', 'linked', 'merged'.
        """
        now = reference_time or time.time()
        query_emb = self.embedder.embed_query(query, context=recent_context)

        # Semantic
        semantic_results = self._semantic_search(
            query_emb, conversation_id, conversation_group_id
        )
        semantic_scored = [
            self.scorer.score(chunk, sim, now) for chunk, sim in semantic_results
        ]

        # Recent
        recent_chunks = self.store.get_recent(self.config.recent_n, conversation_id)
        recent_scored = [self.scorer.score_recent(c, now) for c in recent_chunks]

        # Linked
        linked_scored = []
        if self.config.include_linked:
            for sc in semantic_scored:
                for linked in self.store.get_linked(sc.chunk.id):
                    linked_scored.append(
                        self.scorer.score_linked(linked, sc.similarity, now)
                    )

        # Merged
        merged = self.scorer.merge_scored(semantic_scored, recent_scored, linked_scored)

        return {
            "semantic": semantic_scored,
            "recent": recent_scored,
            "linked": linked_scored,
            "merged": merged,
        }
