"""Pytest fixtures for research manager agent tests."""

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Generator

import pytest

from research_repository.indexer import RepoIndexer
from research_repository.memory import SessionMemory, PersistentMemory, MemoryEntry
from research_repository.safety import SafetyHooks, AuditLog
from research_repository.tools import ToolRegistry, reset_global_registry


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_repo(temp_dir: Path) -> Path:
    """Create a mock repository structure for testing."""
    # Create common directory structure
    (temp_dir / "common" / "models").mkdir(parents=True)
    (temp_dir / "common" / "training").mkdir(parents=True)
    (temp_dir / "common" / "utils").mkdir(parents=True)

    # Create projects
    project1 = temp_dir / "projects" / "test_project"
    project1.mkdir(parents=True)
    (project1 / "train.py").write_text(
        "# Training script\nimport argparse\nparser = argparse.ArgumentParser()\nparser.add_argument('--config')"
    )
    (project1 / "evaluate.py").write_text("# Eval script")
    (project1 / "README.md").write_text("# Test Project\n\nA test project for unit tests.")

    # Create configs
    configs_dir = project1 / "configs"
    configs_dir.mkdir()
    (configs_dir / "default.yaml").write_text(
        """
model:
  d_model: 256
  n_heads: 4
training:
  batch_size: 32
  learning_rate: 0.001
data:
  dataset: test_dataset
"""
    )

    # Create another project
    project2 = temp_dir / "projects" / "another_project"
    project2.mkdir(parents=True)
    (project2 / "train.py").write_text("# Another training script")

    # Create archive
    archive = temp_dir / "projects" / "archive" / "old_project"
    archive.mkdir(parents=True)
    (archive / "train.py").write_text("# Old training script")

    # Create tools directory
    tools_dir = temp_dir / "tools"
    tools_dir.mkdir()
    (tools_dir / "download_dataset.py").write_text("# Download script")

    # Create CLAUDE.md at root
    (temp_dir / "CLAUDE.md").write_text("# Test Repo\n\nTest repository for research manager.")

    return temp_dir


@pytest.fixture
def indexer(mock_repo: Path) -> RepoIndexer:
    """Create an indexer with mock repository."""
    idx = RepoIndexer(mock_repo)
    idx.refresh()
    return idx


@pytest.fixture
def session_memory() -> SessionMemory:
    """Create a fresh session memory."""
    return SessionMemory()


@pytest.fixture
def persistent_memory(temp_dir: Path) -> PersistentMemory:
    """Create a persistent memory with temp storage."""
    return PersistentMemory(temp_dir / "memory.json")


@pytest.fixture
def sample_memory_entries() -> list[MemoryEntry]:
    """Create sample memory entries for testing."""
    return [
        MemoryEntry(
            key="exp_001",
            value={"perplexity": 15.2, "epochs": 10},
            category="experiment",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
        ),
        MemoryEntry(
            key="exp_002",
            value={"perplexity": 18.7, "epochs": 5},
            category="experiment",
            timestamp=datetime(2024, 1, 2, 12, 0, 0),
        ),
        MemoryEntry(
            key="default.yaml",
            value={"model": {"d_model": 256}},
            category="config",
            timestamp=datetime(2024, 1, 3, 12, 0, 0),
        ),
    ]


@pytest.fixture
def safety_hooks() -> SafetyHooks:
    """Create safety hooks for testing."""
    return SafetyHooks()


@pytest.fixture
def audit_log(temp_dir: Path) -> AuditLog:
    """Create an audit log with temp storage."""
    return AuditLog(temp_dir / "audit.json")


@pytest.fixture
def tool_registry() -> Generator[ToolRegistry, None, None]:
    """Create a fresh tool registry for testing."""
    reset_global_registry()
    registry = ToolRegistry()
    yield registry
    reset_global_registry()


@pytest.fixture
def sample_experiments() -> list[dict]:
    """Create sample experiment data."""
    return [
        {
            "name": "exp_001",
            "project": "test_project",
            "perplexity": 15.2,
            "loss": 2.1,
            "epochs": 10,
            "timestamp": "2024-01-01T12:00:00",
        },
        {
            "name": "exp_002",
            "project": "test_project",
            "perplexity": 18.7,
            "loss": 2.5,
            "epochs": 5,
            "timestamp": "2024-01-02T12:00:00",
        },
        {
            "name": "exp_003",
            "project": "another_project",
            "perplexity": 12.3,
            "loss": 1.8,
            "epochs": 20,
            "timestamp": "2024-01-03T12:00:00",
        },
    ]


@pytest.fixture
def dangerous_commands() -> list[str]:
    """List of dangerous commands that should be blocked."""
    return [
        "rm -rf /",
        "rm -rf ~",
        "rm -rf *",
        "sudo rm -rf /tmp",
        "git push --force origin main",
        "git push -f origin main",
        "chmod 777 /etc/passwd",
        "> /dev/sda",
    ]


@pytest.fixture
def safe_commands() -> list[str]:
    """List of safe commands that should be allowed."""
    return [
        "ls -la",
        "pwd",
        "cat README.md",
        "git status",
        "python --version",
        "nvidia-smi",
    ]


@pytest.fixture
def confirmation_commands() -> list[str]:
    """List of commands that should require confirmation."""
    return [
        "python train.py --config config.yaml",
        "python tools/download_dataset.py --name squad",
        "git reset --hard HEAD~1",
        "pip install numpy",
        "rm test.txt",
    ]
