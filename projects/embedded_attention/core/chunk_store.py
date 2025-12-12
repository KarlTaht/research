"""Chunk storage with DuckDB and HNSW vector indexing.

This module handles persistence and vector search for conversation chunks.
The Chunk dataclass is imported from chunk.py (pure data, no logic here).
Text splitting logic is in chunking.py.
"""

from pathlib import Path
from typing import Optional

import duckdb
import numpy as np

from .chunk import Chunk


class ChunkStore:
    """Vector store for conversation chunks using DuckDB with HNSW indexing."""

    def __init__(self, db_path: str = ":memory:", dim: int = 384):
        """Initialize the chunk store.

        Args:
            db_path: Path to DuckDB database file, or ":memory:" for in-memory
            dim: Dimensionality of embedding vectors (384 for bge-small)
        """
        self.dim = dim
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self._init_schema()

    def _init_schema(self):
        """Initialize database schema with HNSW vector index."""
        # Install and load VSS extension for vector similarity search
        self.conn.execute("INSTALL vss; LOAD vss;")

        # Create chunks table with dynamic embedding dimension
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id VARCHAR PRIMARY KEY,
                role VARCHAR NOT NULL,
                content TEXT NOT NULL,
                embedding FLOAT[{self.dim}],
                timestamp DOUBLE NOT NULL,
                token_count INTEGER NOT NULL,
                conversation_id VARCHAR NOT NULL,
                conversation_group_id VARCHAR,
                parent_chunk_id VARCHAR,
                segment_index INTEGER DEFAULT 0,
                total_segments INTEGER DEFAULT 1
            )
        """)

        # Create HNSW index for fast approximate nearest neighbor search
        # Note: HNSW index creation may need to happen after data is inserted
        # depending on DuckDB VSS version
        try:
            self.conn.execute(f"""
                CREATE INDEX IF NOT EXISTS chunks_embedding_idx
                ON chunks
                USING HNSW (embedding)
                WITH (metric = 'cosine')
            """)
        except Exception:
            # Index creation might fail on empty table in some versions
            pass

        # Create indexes for common filters
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conv ON chunks(conversation_id)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_group ON chunks(conversation_group_id)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON chunks(timestamp DESC)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_parent ON chunks(parent_chunk_id)"
        )

    def add(self, chunk: Chunk) -> None:
        """Insert a chunk into the store.

        Args:
            chunk: Chunk to insert (must have embedding set)

        Raises:
            ValueError: If chunk has no embedding
        """
        if chunk.embedding is None:
            raise ValueError("Chunk must have embedding")

        self.conn.execute(
            """
            INSERT INTO chunks (
                id, role, content, embedding, timestamp, token_count,
                conversation_id, conversation_group_id, parent_chunk_id,
                segment_index, total_segments
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            [
                chunk.id,
                chunk.role,
                chunk.content,
                chunk.embedding.tolist(),
                chunk.timestamp,
                chunk.token_count,
                chunk.conversation_id,
                chunk.conversation_group_id,
                chunk.parent_chunk_id,
                chunk.segment_index,
                chunk.total_segments,
            ],
        )

    def add_batch(self, chunks: list[Chunk]) -> None:
        """Insert multiple chunks efficiently.

        Args:
            chunks: List of chunks to insert (all must have embeddings)
        """
        for chunk in chunks:
            self.add(chunk)

    def query(
        self,
        embedding: np.ndarray,
        top_k: int = 10,
        min_similarity: float = 0.0,
        conversation_id: Optional[str] = None,
        conversation_group_id: Optional[str] = None,
    ) -> list[tuple[Chunk, float]]:
        """Vector similarity search with optional filtering.

        Args:
            embedding: Query embedding vector
            top_k: Maximum number of results to return
            min_similarity: Minimum cosine similarity threshold
            conversation_id: Filter to specific conversation (if no group_id)
            conversation_group_id: Filter to conversation group (overrides conversation_id)

        Returns:
            List of (Chunk, similarity) tuples, sorted by similarity descending
        """
        # Build WHERE clause for filtering
        conditions = []
        params = []

        if conversation_group_id:
            conditions.append("conversation_group_id = ?")
            params.append(conversation_group_id)
        elif conversation_id:
            conditions.append("conversation_id = ?")
            params.append(conversation_id)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        # Cosine similarity search
        # Note: array_cosine_similarity returns cosine similarity (1 = identical)
        query_sql = f"""
            SELECT
                id, role, content, embedding, timestamp, token_count,
                conversation_id, conversation_group_id, parent_chunk_id,
                segment_index, total_segments,
                array_cosine_similarity(embedding, ?::FLOAT[{self.dim}]) as similarity
            FROM chunks
            {where_clause}
            ORDER BY similarity DESC
            LIMIT ?
        """

        result = self.conn.execute(
            query_sql, [embedding.tolist()] + params + [top_k]
        ).fetchall()

        chunks = []
        for row in result:
            similarity = row[-1]  # Last column is similarity
            if similarity is None or similarity < min_similarity:
                continue
            chunk = self._row_to_chunk(row[:-1])  # Exclude similarity from chunk
            chunks.append((chunk, float(similarity)))

        return chunks

    def get_recent(self, n: int, conversation_id: str) -> list[Chunk]:
        """Get n most recent chunks from a conversation.

        Args:
            n: Number of chunks to return
            conversation_id: Conversation to query

        Returns:
            List of most recent chunks, ordered by timestamp descending
        """
        result = self.conn.execute(
            """
            SELECT id, role, content, embedding, timestamp, token_count,
                   conversation_id, conversation_group_id, parent_chunk_id,
                   segment_index, total_segments
            FROM chunks
            WHERE conversation_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """,
            [conversation_id, n],
        ).fetchall()

        return [self._row_to_chunk(row) for row in result]

    def get_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """Get a specific chunk by ID.

        Args:
            chunk_id: The chunk's unique identifier

        Returns:
            The Chunk if found, None otherwise
        """
        result = self.conn.execute(
            """
            SELECT id, role, content, embedding, timestamp, token_count,
                   conversation_id, conversation_group_id, parent_chunk_id,
                   segment_index, total_segments
            FROM chunks WHERE id = ?
        """,
            [chunk_id],
        ).fetchone()
        return self._row_to_chunk(result) if result else None

    def get_linked(self, chunk_id: str) -> list[Chunk]:
        """Get chunks linked to this one (tool_call <-> tool_result).

        Args:
            chunk_id: The chunk to find links for

        Returns:
            List of linked chunks (parent and/or children)
        """
        result = self.conn.execute(
            """
            SELECT id, role, content, embedding, timestamp, token_count,
                   conversation_id, conversation_group_id, parent_chunk_id,
                   segment_index, total_segments
            FROM chunks
            WHERE id = (SELECT parent_chunk_id FROM chunks WHERE id = ?)
               OR parent_chunk_id = ?
        """,
            [chunk_id, chunk_id],
        ).fetchall()

        return [self._row_to_chunk(row) for row in result]

    def get_conversation_chunks(self, conversation_id: str) -> list[Chunk]:
        """Get all chunks for a conversation, ordered by timestamp.

        Args:
            conversation_id: The conversation to query

        Returns:
            List of all chunks in chronological order
        """
        result = self.conn.execute(
            """
            SELECT id, role, content, embedding, timestamp, token_count,
                   conversation_id, conversation_group_id, parent_chunk_id,
                   segment_index, total_segments
            FROM chunks
            WHERE conversation_id = ?
            ORDER BY timestamp ASC
        """,
            [conversation_id],
        ).fetchall()

        return [self._row_to_chunk(row) for row in result]

    def count(self, conversation_id: Optional[str] = None) -> int:
        """Count chunks in store, optionally filtered by conversation.

        Args:
            conversation_id: Optional conversation to count

        Returns:
            Number of chunks
        """
        if conversation_id:
            result = self.conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE conversation_id = ?",
                [conversation_id],
            ).fetchone()
        else:
            result = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return result[0] if result else 0

    def _row_to_chunk(self, row: tuple) -> Chunk:
        """Convert database row to Chunk object.

        Args:
            row: Tuple of column values from SELECT

        Returns:
            Chunk object
        """
        return Chunk(
            id=row[0],
            role=row[1],
            content=row[2],
            embedding=np.array(row[3], dtype=np.float32) if row[3] else None,
            timestamp=row[4],
            token_count=row[5],
            conversation_id=row[6],
            conversation_group_id=row[7],
            parent_chunk_id=row[8],
            segment_index=row[9],
            total_segments=row[10],
        )

    def delete_conversation(self, conversation_id: str) -> int:
        """Delete all chunks for a conversation.

        Args:
            conversation_id: The conversation to delete

        Returns:
            Number of deleted chunks
        """
        count = self.count(conversation_id)
        self.conn.execute(
            "DELETE FROM chunks WHERE conversation_id = ?", [conversation_id]
        )
        return count

    def export_parquet(self, path: Path) -> None:
        """Export database to Parquet format.

        Args:
            path: Directory to export to
        """
        path.mkdir(parents=True, exist_ok=True)
        self.conn.execute(f"EXPORT DATABASE '{path}' (FORMAT PARQUET)")

    def close(self):
        """Close database connection."""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
