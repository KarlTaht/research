"""Tests for memory backends."""

from datetime import datetime
from pathlib import Path

import pytest

from research_repository.memory import SessionMemory, PersistentMemory, MemoryEntry


class TestMemoryEntry:
    """Tests for MemoryEntry."""

    def test_create_entry(self):
        """Test creating a memory entry."""
        entry = MemoryEntry(
            key="test_key",
            value={"data": 123},
            category="experiment",
        )
        assert entry.key == "test_key"
        assert entry.category == "experiment"
        assert entry.relevance_score == 1.0

    def test_entry_to_dict(self):
        """Test converting entry to dict."""
        entry = MemoryEntry(
            key="test_key",
            value="test_value",
            category="config",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
        )
        d = entry.to_dict()
        assert d["key"] == "test_key"
        assert d["value"] == "test_value"
        assert d["timestamp"] == "2024-01-01T12:00:00"

    def test_entry_from_dict(self):
        """Test creating entry from dict."""
        data = {
            "key": "test_key",
            "value": {"nested": "data"},
            "category": "experiment",
            "timestamp": "2024-01-01T12:00:00",
            "relevance_score": 0.8,
        }
        entry = MemoryEntry.from_dict(data)
        assert entry.key == "test_key"
        assert entry.value["nested"] == "data"
        assert entry.relevance_score == 0.8


class TestSessionMemory:
    """Tests for SessionMemory."""

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, session_memory: SessionMemory):
        """Test storing and retrieving entries."""
        entry = MemoryEntry(
            key="exp_001",
            value={"perplexity": 15.2},
            category="experiment",
        )
        await session_memory.store(entry)

        results = await session_memory.retrieve("exp_001")
        assert len(results) == 1
        assert results[0].key == "exp_001"

    @pytest.mark.asyncio
    async def test_retrieve_by_category(self, session_memory: SessionMemory):
        """Test retrieving by category."""
        await session_memory.store(MemoryEntry("exp_001", {"perplexity": 15}, "experiment"))
        await session_memory.store(MemoryEntry("config.yaml", {"model": {}}, "config"))

        exp_results = await session_memory.retrieve("exp", category="experiment")
        assert len(exp_results) == 1

        config_results = await session_memory.retrieve("config", category="config")
        assert len(config_results) == 1

    @pytest.mark.asyncio
    async def test_context_management(self, session_memory: SessionMemory):
        """Test context get/set."""
        await session_memory.set_context("current_project", "test_project")

        context = await session_memory.get_context()
        assert context["current_project"] == "test_project"

    @pytest.mark.asyncio
    async def test_recent_experiments_tracking(self, session_memory: SessionMemory):
        """Test that recent experiments are tracked."""
        await session_memory.store(MemoryEntry("exp_001", {}, "experiment"))
        await session_memory.store(MemoryEntry("exp_002", {}, "experiment"))

        context = await session_memory.get_context()
        assert "exp_002" in context["recent_experiments"]
        assert "exp_001" in context["recent_experiments"]

    @pytest.mark.asyncio
    async def test_clear_session(self, session_memory: SessionMemory):
        """Test clearing session."""
        await session_memory.store(MemoryEntry("exp_001", {}, "experiment"))
        await session_memory.clear_session()

        assert session_memory.get_entries_count() == 0

    @pytest.mark.asyncio
    async def test_retrieve_limit(self, session_memory: SessionMemory):
        """Test retrieve respects limit."""
        for i in range(20):
            await session_memory.store(MemoryEntry(f"exp_{i:03d}", {}, "experiment"))

        results = await session_memory.retrieve("exp", limit=5)
        assert len(results) == 5


class TestPersistentMemory:
    """Tests for PersistentMemory."""

    @pytest.mark.asyncio
    async def test_persistence(self, temp_dir: Path):
        """Test that data persists across instances."""
        memory_path = temp_dir / "memory.json"

        # First instance
        mem1 = PersistentMemory(memory_path)
        await mem1.store(MemoryEntry("exp_001", {"perplexity": 15.2}, "experiment"))
        await mem1.set_context("current_project", "test_project")

        # Second instance - should load from file
        mem2 = PersistentMemory(memory_path)

        results = await mem2.retrieve("exp_001")
        assert len(results) == 1

        context = await mem2.get_context()
        assert context["current_project"] == "test_project"

    @pytest.mark.asyncio
    async def test_clear_session_keeps_persistent(self, persistent_memory: PersistentMemory):
        """Test clear_session keeps persistent context."""
        await persistent_memory.set_context("current_project", "test_project")
        await persistent_memory.store(MemoryEntry("exp_001", {}, "experiment"))

        await persistent_memory.clear_session()

        context = await persistent_memory.get_context()
        assert context["current_project"] == "test_project"

    @pytest.mark.asyncio
    async def test_clear_all(self, temp_dir: Path):
        """Test clear_all removes everything."""
        memory_path = temp_dir / "memory.json"
        memory = PersistentMemory(memory_path)

        await memory.store(MemoryEntry("exp_001", {}, "experiment"))
        await memory.clear_all()

        assert not memory_path.exists()
