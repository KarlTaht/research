"""In-memory session storage for the research manager agent."""

from typing import Any, Optional

from research_manager.memory.base import MemoryBackend, MemoryEntry


class SessionMemory(MemoryBackend):
    """In-memory session storage.

    This is the simplest memory backend - stores everything in memory
    and loses state when the process exits. Useful for development
    and as a base for composition with persistent backends.
    """

    def __init__(self):
        self._entries: list[MemoryEntry] = []
        self._context: dict[str, Any] = {
            "current_project": None,
            "recent_experiments": [],
            "recent_configs": [],
            "conversation_turns": 0,
        }

    async def store(self, entry: MemoryEntry) -> None:
        """Store a memory entry in session."""
        self._entries.append(entry)

        # Update context based on category
        if entry.category == "experiment":
            recent = self._context["recent_experiments"]
            if entry.key not in recent:
                recent.insert(0, entry.key)
                self._context["recent_experiments"] = recent[:10]  # Keep last 10

        elif entry.category == "config":
            recent = self._context["recent_configs"]
            if entry.key not in recent:
                recent.insert(0, entry.key)
                self._context["recent_configs"] = recent[:10]

        elif entry.category == "project":
            self._context["current_project"] = entry.value

    async def retrieve(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """Retrieve memories matching query.

        For session memory, this does simple substring matching.
        More advanced backends can do semantic search.
        """
        query_lower = query.lower()
        matches = []

        for entry in reversed(self._entries):  # Most recent first
            if category and entry.category != category:
                continue

            # Simple substring match on key and string values
            key_match = query_lower in entry.key.lower()
            value_match = False
            if isinstance(entry.value, str):
                value_match = query_lower in entry.value.lower()
            elif isinstance(entry.value, dict):
                value_str = str(entry.value).lower()
                value_match = query_lower in value_str

            if key_match or value_match:
                matches.append(entry)
                if len(matches) >= limit:
                    break

        return matches

    async def get_context(self) -> dict:
        """Get current session context."""
        return self._context.copy()

    async def set_context(self, key: str, value: Any) -> None:
        """Set a context value."""
        self._context[key] = value

    async def clear_session(self) -> None:
        """Clear all session memory."""
        self._entries.clear()
        self._context = {
            "current_project": None,
            "recent_experiments": [],
            "recent_configs": [],
            "conversation_turns": 0,
        }

    async def clear_all(self) -> None:
        """Clear all memory (same as clear_session for this backend)."""
        await self.clear_session()

    def get_entries_count(self) -> int:
        """Get number of stored entries."""
        return len(self._entries)

    def get_entries_by_category(self, category: str) -> list[MemoryEntry]:
        """Get all entries in a category."""
        return [e for e in self._entries if e.category == category]
