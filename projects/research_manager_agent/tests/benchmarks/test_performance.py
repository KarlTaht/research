"""Performance benchmarks for research manager agent.

These benchmarks measure:
- Tool execution latency
- Repository indexing time
- Memory operations
- Query performance
"""

from pathlib import Path
from datetime import datetime

import pytest
import pandas as pd

from research_manager.indexer import RepoIndexer
from research_manager.memory import SessionMemory, PersistentMemory, MemoryEntry
from research_manager.tools.explore import explore_repo
from research_manager.tools.projects import list_projects
from research_manager.tools.scripts import find_script
from research_manager.tools.configs import read_config
from research_manager.tools.experiments import query_experiments, compare_runs
from research_manager.tools.assistant import explain_error, suggest_cleanup


@pytest.fixture
def large_repo(temp_dir: Path) -> Path:
    """Create a large repository for benchmarking."""
    # Create many projects
    for i in range(20):
        project_dir = temp_dir / "projects" / f"project_{i:03d}"
        project_dir.mkdir(parents=True)
        (project_dir / "train.py").write_text(f"# Training script for project {i}")
        (project_dir / "evaluate.py").write_text(f"# Eval script for project {i}")

        # Create configs
        configs_dir = project_dir / "configs"
        configs_dir.mkdir()
        for j in range(5):
            (configs_dir / f"config_{j}.yaml").write_text(
                f"""
model:
  d_model: {128 * (j + 1)}
  n_heads: {4 * (j + 1)}
training:
  batch_size: {32 * (j + 1)}
"""
            )

    # Create common directory
    (temp_dir / "common" / "models").mkdir(parents=True)
    (temp_dir / "common" / "training").mkdir(parents=True)

    # Create tools directory
    tools_dir = temp_dir / "tools"
    tools_dir.mkdir()
    for i in range(10):
        (tools_dir / f"tool_{i}.py").write_text(f"# Tool script {i}")

    (temp_dir / "CLAUDE.md").write_text("# Test Repo")

    return temp_dir


@pytest.fixture
def large_experiments(temp_dir: Path) -> Path:
    """Create many experiment files for benchmarking."""
    exp_dir = temp_dir / "experiments"
    exp_dir.mkdir(parents=True)

    # Create 50 experiments
    for i in range(50):
        df = pd.DataFrame(
            {
                "epoch": list(range(100)),
                "loss": [2.0 - (e * 0.01) for e in range(100)],
                "perplexity": [30.0 - (e * 0.2) for e in range(100)],
                "accuracy": [0.1 + (e * 0.008) for e in range(100)],
                "experiment_name": [f"exp_{i:03d}"] * 100,
                "saved_at": [datetime.now().isoformat()] * 100,
            }
        )
        df.to_parquet(exp_dir / f"exp_{i:03d}.parquet")

    return exp_dir


class TestIndexingBenchmarks:
    """Benchmarks for repository indexing."""

    def test_index_large_repo(self, benchmark, large_repo: Path):
        """Benchmark indexing a large repository."""
        indexer = RepoIndexer(large_repo)

        benchmark(indexer.refresh)

        # Verify indexing worked
        stats = indexer.get_stats()
        assert stats["projects"] >= 20
        assert stats["configs"] >= 100

    def test_index_refresh_cached(self, benchmark, large_repo: Path):
        """Benchmark re-indexing (should be fast if cached)."""
        indexer = RepoIndexer(large_repo)
        indexer.refresh()  # Initial index

        # Benchmark refresh
        benchmark(indexer.refresh)

        stats = indexer.get_stats()
        assert stats["projects"] >= 20


class TestToolLatencyBenchmarks:
    """Benchmarks for individual tool execution latency."""

    @pytest.fixture
    def indexed_repo(self, large_repo: Path) -> tuple[Path, RepoIndexer]:
        """Create an indexed repository."""
        indexer = RepoIndexer(large_repo)
        indexer.refresh()
        return large_repo, indexer

    def test_explore_repo_latency(self, benchmark, indexed_repo):
        """Benchmark explore_repo tool."""
        repo_root, indexer = indexed_repo

        async def run():
            return await explore_repo(path=".", depth=2, repo_root=repo_root)

        import asyncio

        result = benchmark(lambda: asyncio.run(run()))
        assert "error" not in result

    def test_list_projects_latency(self, benchmark, indexed_repo):
        """Benchmark list_projects tool."""
        repo_root, indexer = indexed_repo

        async def run():
            return await list_projects(indexer=indexer)

        import asyncio

        result = benchmark(lambda: asyncio.run(run()))
        assert result["summary"]["total"] >= 20

    def test_find_script_latency(self, benchmark, indexed_repo):
        """Benchmark find_script tool."""
        repo_root, indexer = indexed_repo

        async def run():
            return await find_script(task="train a model", indexer=indexer)

        import asyncio

        result = benchmark(lambda: asyncio.run(run()))
        assert "error" not in result

    def test_read_config_latency(self, benchmark, indexed_repo):
        """Benchmark read_config tool."""
        repo_root, indexer = indexed_repo

        async def run():
            return await read_config(
                path="projects/project_000/configs/config_0.yaml",
                repo_root=repo_root,
            )

        import asyncio

        result = benchmark(lambda: asyncio.run(run()))
        assert "error" not in result


class TestQueryBenchmarks:
    """Benchmarks for experiment queries."""

    def test_query_all_experiments(self, benchmark, large_experiments: Path):
        """Benchmark querying all experiments."""

        async def run():
            return await query_experiments(
                query="all",
                experiments_dir=large_experiments,
            )

        import asyncio

        result = benchmark(lambda: asyncio.run(run()))
        assert result["total_experiments"] >= 50

    def test_query_with_filter(self, benchmark, large_experiments: Path):
        """Benchmark filtered query."""

        async def run():
            return await query_experiments(
                query="perplexity < 20",
                experiments_dir=large_experiments,
            )

        import asyncio

        result = benchmark(lambda: asyncio.run(run()))
        assert "error" not in result

    def test_query_best_metric(self, benchmark, large_experiments: Path):
        """Benchmark best metric query."""

        async def run():
            return await query_experiments(
                query="best perplexity",
                experiments_dir=large_experiments,
            )

        import asyncio

        result = benchmark(lambda: asyncio.run(run()))
        assert "error" not in result

    def test_compare_experiments(self, benchmark, large_experiments: Path):
        """Benchmark comparing experiments."""

        async def run():
            return await compare_runs(
                experiments=["exp_000", "exp_001", "exp_002"],
                experiments_dir=large_experiments,
            )

        import asyncio

        result = benchmark(lambda: asyncio.run(run()))
        assert "error" not in result


class TestMemoryBenchmarks:
    """Benchmarks for memory operations."""

    def test_session_memory_store(self, benchmark):
        """Benchmark storing in session memory."""
        memory = SessionMemory()

        async def run():
            entry = MemoryEntry(
                key="test_key",
                value={"data": "test"},
                category="test",
                timestamp=datetime.now(),
            )
            await memory.store(entry)

        import asyncio

        benchmark(lambda: asyncio.run(run()))

    def test_session_memory_retrieve(self, benchmark):
        """Benchmark retrieving from session memory."""
        memory = SessionMemory()

        # Pre-populate
        import asyncio

        for i in range(100):
            asyncio.run(
                memory.store(
                    MemoryEntry(
                        key=f"key_{i}",
                        value={"data": f"value_{i}"},
                        category="test",
                        timestamp=datetime.now(),
                    )
                )
            )

        async def run():
            return await memory.retrieve("test", limit=10)

        benchmark(lambda: asyncio.run(run()))

    def test_persistent_memory_store(self, benchmark, temp_dir: Path):
        """Benchmark storing in persistent memory."""
        memory = PersistentMemory(temp_dir / "memory.json")

        async def run():
            entry = MemoryEntry(
                key="test_key",
                value={"data": "test"},
                category="test",
                timestamp=datetime.now(),
            )
            await memory.store(entry)

        import asyncio

        benchmark(lambda: asyncio.run(run()))


class TestAssistantToolBenchmarks:
    """Benchmarks for assistant tools."""

    def test_explain_error_latency(self, benchmark):
        """Benchmark explain_error tool."""
        error_message = """
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 15.78 GiB total capacity; 12.50 GiB already allocated; 1.50 GiB free; 13.00 GiB reserved in total by PyTorch)
"""

        async def run():
            return await explain_error(
                error=error_message,
                context="training transformer",
            )

        import asyncio

        result = benchmark(lambda: asyncio.run(run()))
        assert result["category"] == "GPU Memory"

    def test_suggest_cleanup_latency(self, benchmark, large_repo: Path):
        """Benchmark suggest_cleanup tool."""

        async def run():
            return await suggest_cleanup(
                scope="all",
                max_age_days=30,
                repo_root=large_repo,
            )

        import asyncio

        result = benchmark(lambda: asyncio.run(run()))
        assert "summary" in result


# Performance targets (for documentation, not enforced in CI)
PERFORMANCE_TARGETS = {
    "index_large_repo": {"max_seconds": 2.0},
    "list_projects": {"max_seconds": 0.1},
    "find_script": {"max_seconds": 0.1},
    "query_experiments": {"max_seconds": 0.5},
    "explain_error": {"max_seconds": 0.05},
}
