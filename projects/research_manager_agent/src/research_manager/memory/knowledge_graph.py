"""Placeholder for future knowledge graph memory backend.

This module defines the interface for a knowledge graph-based memory
system with semantic retrieval. Implementation is deferred to a future phase.
"""

from typing import Any, Optional

from research_manager.memory.base import MemoryBackend, MemoryEntry


class KnowledgeGraphMemory(MemoryBackend):
    """Knowledge graph memory with semantic retrieval.

    Future implementation will include:
    - NetworkX graph for entity relationships
    - sentence-transformers embeddings for semantic search
    - Entity types: Projects, Experiments, Configs, Scripts, Errors
    - Relationships: uses_config, produced_checkpoint, related_to, etc.

    For now, this raises NotImplementedError to indicate it's a placeholder.
    """

    def __init__(self):
        raise NotImplementedError(
            "KnowledgeGraphMemory is a placeholder for future implementation. "
            "Use SessionMemory or PersistentMemory instead."
        )

    async def store(self, entry: MemoryEntry) -> None:
        raise NotImplementedError

    async def retrieve(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        raise NotImplementedError

    async def get_context(self) -> dict:
        raise NotImplementedError

    async def set_context(self, key: str, value: Any) -> None:
        raise NotImplementedError

    async def clear_session(self) -> None:
        raise NotImplementedError

    async def clear_all(self) -> None:
        raise NotImplementedError
