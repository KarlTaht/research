"""Persistent memory storage using JSON files."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from research_manager.memory.base import MemoryBackend, MemoryEntry
from research_manager.memory.session import SessionMemory


class PersistentMemory(MemoryBackend):
    """Hybrid memory: session memory + JSON file persistence.

    Composes SessionMemory for fast access with JSON file storage
    for cross-session persistence. Automatically loads on init
    and saves on modifications.
    """

    def __init__(self, storage_path: Path):
        """Initialize persistent memory.

        Args:
            storage_path: Path to JSON file for persistent storage.
        """
        self.storage_path = storage_path
        self._session = SessionMemory()
        self._persistent_context: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        """Load persistent data from file."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path) as f:
                    data = json.load(f)

                self._persistent_context = data.get("context", {})

                # Restore entries to session
                for entry_data in data.get("entries", []):
                    entry = MemoryEntry.from_dict(entry_data)
                    self._session._entries.append(entry)

                # Merge persistent context with session defaults
                session_ctx = self._session._context
                for key, value in self._persistent_context.items():
                    if key in session_ctx:
                        session_ctx[key] = value

            except (json.JSONDecodeError, KeyError) as e:
                # Corrupted file - start fresh
                print(f"Warning: Could not load memory from {self.storage_path}: {e}")

    def _save(self) -> None:
        """Save persistent data to file."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Only persist certain context keys
        persistent_keys = ["current_project", "recent_experiments", "recent_configs"]
        context_to_save = {k: v for k, v in self._session._context.items() if k in persistent_keys}

        data = {
            "context": context_to_save,
            "entries": [e.to_dict() for e in self._session._entries[-100:]],  # Last 100
            "saved_at": datetime.now().isoformat(),
        }

        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    async def store(self, entry: MemoryEntry) -> None:
        """Store entry and persist to disk."""
        await self._session.store(entry)
        self._save()

    async def retrieve(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """Retrieve from session memory."""
        return await self._session.retrieve(query, category, limit)

    async def get_context(self) -> dict:
        """Get current context."""
        return await self._session.get_context()

    async def set_context(self, key: str, value: Any) -> None:
        """Set context and persist if it's a persistent key."""
        await self._session.set_context(key, value)
        self._save()

    async def clear_session(self) -> None:
        """Clear session memory but keep persistent context."""
        # Save current persistent context
        ctx = await self._session.get_context()
        persistent_keys = ["current_project", "recent_experiments", "recent_configs"]
        saved_ctx = {k: ctx.get(k) for k in persistent_keys}

        # Clear session
        await self._session.clear_session()

        # Restore persistent context
        for key, value in saved_ctx.items():
            if value is not None:
                await self._session.set_context(key, value)

    async def clear_all(self) -> None:
        """Clear all memory including persistent storage."""
        await self._session.clear_all()
        if self.storage_path.exists():
            self.storage_path.unlink()

    def get_storage_size(self) -> int:
        """Get size of persistent storage in bytes."""
        if self.storage_path.exists():
            return self.storage_path.stat().st_size
        return 0
