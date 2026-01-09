"""Abstract interface for memory backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class MemoryEntry:
    """A single memory entry."""

    key: str
    value: Any
    category: str  # "project", "experiment", "config", "conversation"
    timestamp: datetime = field(default_factory=datetime.now)
    relevance_score: float = 1.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "key": self.key,
            "value": self.value,
            "category": self.category,
            "timestamp": self.timestamp.isoformat(),
            "relevance_score": self.relevance_score,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryEntry":
        """Create from dictionary."""
        return cls(
            key=data["key"],
            value=data["value"],
            category=data["category"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            relevance_score=data.get("relevance_score", 1.0),
            metadata=data.get("metadata", {}),
        )


class MemoryBackend(ABC):
    """Abstract interface for memory backends.

    This interface supports both simple session memory and more advanced
    implementations like knowledge graphs with semantic retrieval.
    """

    @abstractmethod
    async def store(self, entry: MemoryEntry) -> None:
        """Store a memory entry.

        Args:
            entry: The memory entry to store.
        """
        pass

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """Retrieve relevant memories.

        Args:
            query: Search query (interpreted by backend).
            category: Optional category filter.
            limit: Maximum number of entries to return.

        Returns:
            List of relevant memory entries, sorted by relevance.
        """
        pass

    @abstractmethod
    async def get_context(self) -> dict:
        """Get current context.

        Returns:
            Dictionary containing current project, recent items, etc.
        """
        pass

    @abstractmethod
    async def set_context(self, key: str, value: Any) -> None:
        """Set a context value.

        Args:
            key: Context key (e.g., "current_project").
            value: Value to set.
        """
        pass

    @abstractmethod
    async def clear_session(self) -> None:
        """Clear session-level memory (keeps persistent data)."""
        pass

    @abstractmethod
    async def clear_all(self) -> None:
        """Clear all memory including persistent data."""
        pass
