"""Tests for experiment tools."""

from pathlib import Path
from datetime import datetime

import pytest
import pandas as pd

from research_repository.tools.experiments import (
    query_experiments,
    list_experiments_tool,
    analyze_logs,
    find_checkpoint,
    compare_runs,
    _natural_language_to_sql,
)


class TestNaturalLanguageToSQL:
    """Tests for natural language to SQL conversion."""

    def test_best_metric(self):
        """Test 'best' query conversion."""
        sql = _natural_language_to_sql("best perplexity")
        assert sql is not None
        assert "ORDER BY" in sql.upper()
        assert "perplexity" in sql.lower()

    def test_worst_metric(self):
        """Test 'worst' query conversion."""
        sql = _natural_language_to_sql("worst loss")
        assert sql is not None
        assert "ORDER BY" in sql.upper()
        assert "DESC" in sql.upper()

    def test_top_n(self):
        """Test 'top N' query conversion."""
        sql = _natural_language_to_sql("top 5 by loss")
        assert sql is not None
        assert "LIMIT 5" in sql.upper()

    def test_filter_less_than(self):
        """Test '<' filter conversion."""
        sql = _natural_language_to_sql("perplexity < 20")
        assert sql is not None
        assert "WHERE" in sql.upper()
        assert "< 20" in sql

    def test_filter_greater_than(self):
        """Test '>' filter conversion."""
        sql = _natural_language_to_sql("loss > 1.5")
        assert sql is not None
        assert "WHERE" in sql.upper()
        assert "> 1.5" in sql

    def test_recent_query(self):
        """Test 'recent' query conversion."""
        sql = _natural_language_to_sql("recent")
        assert sql is not None
        assert "ORDER BY" in sql.upper()
        assert "DESC" in sql.upper()

    def test_passthrough_sql(self):
        """Test that SQL queries pass through unchanged."""
        original = "SELECT * FROM experiments WHERE perplexity < 15"
        sql = _natural_language_to_sql(original)
        assert sql == original

    def test_unknown_query_returns_none(self):
        """Test that unknown queries return None."""
        sql = _natural_language_to_sql("xyz unknown query abc")
        assert sql is None


class TestQueryExperiments:
    """Tests for query_experiments tool."""

    @pytest.fixture
    def experiments_dir(self, temp_dir: Path) -> Path:
        """Create a mock experiments directory with Parquet files."""
        exp_dir = temp_dir / "experiments"
        exp_dir.mkdir(parents=True)

        # Create sample experiment data
        exp1 = pd.DataFrame(
            {
                "epoch": [1, 2, 3, 4, 5],
                "perplexity": [25.0, 20.0, 18.0, 16.0, 15.0],
                "loss": [3.0, 2.5, 2.2, 2.0, 1.8],
                "experiment_name": ["exp_001"] * 5,
                "saved_at": [datetime.now().isoformat()] * 5,
            }
        )
        exp1.to_parquet(exp_dir / "exp_001.parquet")

        exp2 = pd.DataFrame(
            {
                "epoch": [1, 2, 3],
                "perplexity": [30.0, 25.0, 22.0],
                "loss": [3.5, 3.0, 2.8],
                "experiment_name": ["exp_002"] * 3,
                "saved_at": [datetime.now().isoformat()] * 3,
            }
        )
        exp2.to_parquet(exp_dir / "exp_002.parquet")

        return exp_dir

    @pytest.mark.asyncio
    async def test_query_all_experiments(self, experiments_dir: Path):
        """Test querying all experiments."""
        result = await query_experiments(
            query="all",
            experiments_dir=experiments_dir,
        )

        assert "error" not in result
        assert "results" in result
        assert result["count"] > 0

    @pytest.mark.asyncio
    async def test_query_best_perplexity(self, experiments_dir: Path):
        """Test querying best perplexity."""
        result = await query_experiments(
            query="best perplexity",
            experiments_dir=experiments_dir,
        )

        assert "error" not in result
        assert "results" in result
        assert "sql" in result

    @pytest.mark.asyncio
    async def test_query_with_filter(self, experiments_dir: Path):
        """Test querying with filter."""
        result = await query_experiments(
            query="perplexity < 20",
            experiments_dir=experiments_dir,
        )

        assert "error" not in result
        assert "results" in result
        # All results should have perplexity < 20
        for r in result["results"]:
            assert r["perplexity"] < 20

    @pytest.mark.asyncio
    async def test_query_with_sql(self, experiments_dir: Path):
        """Test querying with raw SQL."""
        result = await query_experiments(
            query="SELECT DISTINCT experiment_name FROM experiments",
            experiments_dir=experiments_dir,
        )

        assert "error" not in result
        assert "results" in result
        names = [r["experiment_name"] for r in result["results"]]
        assert "exp_001" in names
        assert "exp_002" in names

    @pytest.mark.asyncio
    async def test_query_empty_directory(self, temp_dir: Path):
        """Test querying empty experiments directory."""
        empty_dir = temp_dir / "empty_experiments"
        empty_dir.mkdir()

        result = await query_experiments(
            query="all",
            experiments_dir=empty_dir,
        )

        assert "error" in result

    @pytest.mark.asyncio
    async def test_query_invalid_sql(self, experiments_dir: Path):
        """Test handling invalid SQL."""
        result = await query_experiments(
            query="SELECT * FROM nonexistent_table",
            experiments_dir=experiments_dir,
        )

        assert "error" in result


class TestListExperiments:
    """Tests for list_experiments_tool."""

    @pytest.fixture
    def experiments_dir(self, temp_dir: Path) -> Path:
        """Create mock experiments."""
        exp_dir = temp_dir / "experiments"
        exp_dir.mkdir(parents=True)

        # Create empty parquet files
        df = pd.DataFrame({"experiment_name": ["test"], "value": [1]})
        df.to_parquet(exp_dir / "exp_001.parquet")
        df.to_parquet(exp_dir / "exp_002.parquet")
        df.to_parquet(exp_dir / "transformer_exp.parquet")

        return exp_dir

    @pytest.mark.asyncio
    async def test_list_all_experiments(self, experiments_dir: Path):
        """Test listing all experiments."""
        result = await list_experiments_tool(experiments_dir=experiments_dir)

        assert "error" not in result
        assert "experiments" in result
        assert result["count"] == 3

    @pytest.mark.asyncio
    async def test_list_filtered_by_project(self, experiments_dir: Path):
        """Test listing experiments filtered by project."""
        result = await list_experiments_tool(
            project="transformer",
            experiments_dir=experiments_dir,
        )

        assert "error" not in result
        assert result["count"] == 1
        assert "transformer" in result["experiments"][0].lower()

    @pytest.mark.asyncio
    async def test_list_empty_directory(self, temp_dir: Path):
        """Test listing from empty directory."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()

        result = await list_experiments_tool(experiments_dir=empty_dir)

        assert "error" not in result
        assert result["count"] == 0


class TestAnalyzeLogs:
    """Tests for analyze_logs tool."""

    @pytest.fixture
    def mock_logs(self, temp_dir: Path) -> Path:
        """Create mock log files."""
        logs_dir = temp_dir / "logs"
        logs_dir.mkdir(parents=True)

        # Create a sample log file
        log_content = """
2024-01-01 12:00:00 - Starting training
2024-01-01 12:00:01 - Epoch 1
2024-01-01 12:00:02 - loss: 3.5, perplexity: 25.0
2024-01-01 12:00:03 - Epoch 2
2024-01-01 12:00:04 - loss: 2.8, perplexity: 20.0
2024-01-01 12:00:05 - WARNING: Learning rate is high
2024-01-01 12:00:06 - Epoch 3
2024-01-01 12:00:07 - loss: 2.2, perplexity: 15.0
2024-01-01 12:00:08 - ERROR: CUDA out of memory
2024-01-01 12:00:09 - Training complete
"""
        (logs_dir / "train.log").write_text(log_content)

        return temp_dir

    @pytest.mark.asyncio
    async def test_analyze_log_file(self, mock_logs: Path):
        """Test analyzing a log file."""
        result = await analyze_logs(
            log_path="logs/train.log",
            repo_root=mock_logs,
        )

        assert "error" not in result
        assert "files_analyzed" in result
        assert len(result["files_analyzed"]) == 1

    @pytest.mark.asyncio
    async def test_extract_metrics(self, mock_logs: Path):
        """Test extracting metrics from logs."""
        result = await analyze_logs(
            log_path="logs/train.log",
            repo_root=mock_logs,
        )

        assert "metrics" in result
        # Should extract loss and perplexity
        assert "loss" in result["metrics"] or "perplexity" in result["metrics"]

    @pytest.mark.asyncio
    async def test_find_errors(self, mock_logs: Path):
        """Test finding errors in logs."""
        result = await analyze_logs(
            log_path="logs/train.log",
            repo_root=mock_logs,
        )

        assert "errors" in result
        assert len(result["errors"]) > 0
        assert "CUDA" in result["errors"][0]["content"]

    @pytest.mark.asyncio
    async def test_find_warnings(self, mock_logs: Path):
        """Test finding warnings in logs."""
        result = await analyze_logs(
            log_path="logs/train.log",
            repo_root=mock_logs,
        )

        assert "warnings" in result
        assert len(result["warnings"]) > 0

    @pytest.mark.asyncio
    async def test_custom_pattern(self, mock_logs: Path):
        """Test searching with custom pattern."""
        result = await analyze_logs(
            log_path="logs/train.log",
            pattern=r"Epoch \d+",
            repo_root=mock_logs,
        )

        assert "pattern_matches" in result
        assert len(result["pattern_matches"]) == 3  # 3 epochs

    @pytest.mark.asyncio
    async def test_no_logs_found(self, temp_dir: Path):
        """Test when no logs are found."""
        result = await analyze_logs(
            log_path="nonexistent/logs",
            repo_root=temp_dir,
        )

        assert "error" in result


class TestFindCheckpoint:
    """Tests for find_checkpoint tool."""

    @pytest.fixture
    def mock_checkpoints(self, temp_dir: Path) -> Path:
        """Create mock checkpoint files."""
        # Create checkpoint directories
        ckpt_dir = temp_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True)

        # Create some checkpoint files
        (ckpt_dir / "model_epoch_5.pt").write_bytes(b"fake checkpoint")
        (ckpt_dir / "model_epoch_10.pt").write_bytes(b"fake checkpoint")
        (ckpt_dir / "optimizer.pt").write_bytes(b"fake optimizer")

        # Create project-specific checkpoints
        proj_ckpt = temp_dir / "projects" / "test_project" / "checkpoints"
        proj_ckpt.mkdir(parents=True)
        (proj_ckpt / "best_model.pth").write_bytes(b"fake checkpoint")

        return temp_dir

    @pytest.mark.asyncio
    async def test_find_all_checkpoints(self, mock_checkpoints: Path):
        """Test finding all checkpoints."""
        result = await find_checkpoint(repo_root=mock_checkpoints)

        assert "error" not in result
        assert "checkpoints" in result
        assert result["total_found"] >= 3

    @pytest.mark.asyncio
    async def test_find_checkpoint_by_experiment(self, mock_checkpoints: Path):
        """Test finding checkpoints by experiment name."""
        result = await find_checkpoint(
            experiment="epoch_5",
            repo_root=mock_checkpoints,
        )

        assert "error" not in result
        assert "checkpoints" in result
        assert any("epoch_5" in c["name"] for c in result["checkpoints"])

    @pytest.mark.asyncio
    async def test_find_checkpoint_by_project(self, mock_checkpoints: Path):
        """Test finding checkpoints by project."""
        result = await find_checkpoint(
            project="test_project",
            repo_root=mock_checkpoints,
        )

        assert "error" not in result
        assert "checkpoints" in result

    @pytest.mark.asyncio
    async def test_find_latest_checkpoint(self, mock_checkpoints: Path):
        """Test finding latest checkpoint only."""
        result = await find_checkpoint(
            latest_only=True,
            repo_root=mock_checkpoints,
        )

        assert "error" not in result
        assert "checkpoint" in result
        assert "total_found" in result

    @pytest.mark.asyncio
    async def test_checkpoint_metadata(self, mock_checkpoints: Path):
        """Test that checkpoint metadata is included."""
        result = await find_checkpoint(repo_root=mock_checkpoints)

        assert "checkpoints" in result
        for ckpt in result["checkpoints"]:
            assert "name" in ckpt
            assert "path" in ckpt
            assert "size_mb" in ckpt
            assert "modified" in ckpt

    @pytest.mark.asyncio
    async def test_no_checkpoints_found(self, temp_dir: Path):
        """Test when no checkpoints are found."""
        result = await find_checkpoint(
            experiment="nonexistent",
            repo_root=temp_dir,
        )

        assert "error" in result


class TestCompareRuns:
    """Tests for compare_runs tool."""

    @pytest.fixture
    def experiments_dir(self, temp_dir: Path) -> Path:
        """Create mock experiments for comparison."""
        exp_dir = temp_dir / "experiments"
        exp_dir.mkdir(parents=True)

        # Create first experiment
        exp1 = pd.DataFrame(
            {
                "epoch": [1, 2, 3],
                "perplexity": [25.0, 20.0, 15.0],
                "loss": [3.0, 2.5, 2.0],
                "accuracy": [0.6, 0.7, 0.8],
                "experiment_name": ["exp_001"] * 3,
                "saved_at": [datetime.now().isoformat()] * 3,
            }
        )
        exp1.to_parquet(exp_dir / "exp_001.parquet")

        # Create second experiment
        exp2 = pd.DataFrame(
            {
                "epoch": [1, 2, 3, 4],
                "perplexity": [30.0, 22.0, 18.0, 14.0],
                "loss": [3.5, 2.8, 2.3, 1.9],
                "accuracy": [0.5, 0.65, 0.75, 0.82],
                "experiment_name": ["exp_002"] * 4,
                "saved_at": [datetime.now().isoformat()] * 4,
            }
        )
        exp2.to_parquet(exp_dir / "exp_002.parquet")

        return exp_dir

    @pytest.mark.asyncio
    async def test_compare_two_experiments(self, experiments_dir: Path):
        """Test comparing two experiments."""
        result = await compare_runs(
            experiments=["exp_001", "exp_002"],
            experiments_dir=experiments_dir,
        )

        assert "error" not in result
        assert "experiments" in result
        assert "metrics_comparison" in result

    @pytest.mark.asyncio
    async def test_compare_shows_all_metrics(self, experiments_dir: Path):
        """Test that comparison shows all metrics."""
        result = await compare_runs(
            experiments=["exp_001", "exp_002"],
            experiments_dir=experiments_dir,
        )

        assert "metrics_comparison" in result
        metrics = result["metrics_comparison"]
        assert "perplexity" in metrics or "loss" in metrics or "accuracy" in metrics

    @pytest.mark.asyncio
    async def test_compare_specific_metrics(self, experiments_dir: Path):
        """Test comparing specific metrics only."""
        result = await compare_runs(
            experiments=["exp_001", "exp_002"],
            metrics=["perplexity", "loss"],
            experiments_dir=experiments_dir,
        )

        assert "error" not in result
        assert "available_metrics" in result

    @pytest.mark.asyncio
    async def test_compare_identifies_best(self, experiments_dir: Path):
        """Test that comparison identifies best performer."""
        result = await compare_runs(
            experiments=["exp_001", "exp_002"],
            experiments_dir=experiments_dir,
        )

        assert "best_by_metric" in result
        # exp_002 has lower final perplexity (14.0 vs 15.0)
        if "perplexity" in result["best_by_metric"]:
            assert result["best_by_metric"]["perplexity"]["experiment"] == "exp_002"

    @pytest.mark.asyncio
    async def test_compare_single_experiment_error(self):
        """Test that comparing single experiment returns error."""
        result = await compare_runs(experiments=["exp_001"])

        assert "error" in result

    @pytest.mark.asyncio
    async def test_compare_nonexistent_experiment(self, experiments_dir: Path):
        """Test comparing with nonexistent experiment."""
        result = await compare_runs(
            experiments=["exp_001", "nonexistent"],
            experiments_dir=experiments_dir,
        )

        assert "error" in result

    @pytest.mark.asyncio
    async def test_compare_row_counts(self, experiments_dir: Path):
        """Test that comparison shows row counts."""
        result = await compare_runs(
            experiments=["exp_001", "exp_002"],
            experiments_dir=experiments_dir,
        )

        assert "row_counts" in result
        assert result["row_counts"]["exp_001"] == 3
        assert result["row_counts"]["exp_002"] == 4
