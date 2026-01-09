"""Integration tests for tool combinations.

These tests verify that multiple tools work together correctly in common workflows.
"""

from pathlib import Path
from datetime import datetime

import pytest
import pandas as pd

from research_manager.indexer import RepoIndexer
from research_manager.tools.explore import explore_repo
from research_manager.tools.projects import list_projects, get_project
from research_manager.tools.scripts import find_script, list_scripts
from research_manager.tools.configs import read_config, list_configs
from research_manager.tools.experiments import (
    query_experiments,
    list_experiments_tool,
    find_checkpoint,
    compare_runs,
)
from research_manager.tools.assistant import (
    generate_train_command,
    explain_error,
    suggest_cleanup,
)


@pytest.fixture
def full_mock_repo(temp_dir: Path) -> Path:
    """Create a comprehensive mock repository for integration tests."""
    # Common directory
    (temp_dir / "common" / "models").mkdir(parents=True)
    (temp_dir / "common" / "training").mkdir(parents=True)
    (temp_dir / "common" / "utils").mkdir(parents=True)

    # Project 1: custom_transformer
    project1 = temp_dir / "projects" / "custom_transformer"
    project1.mkdir(parents=True)
    (project1 / "train.py").write_text(
        """# Training script
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config')
parser.add_argument('--device', default='cuda')
parser.add_argument('--resume')
"""
    )
    (project1 / "evaluate.py").write_text("# Evaluation script")
    (project1 / "README.md").write_text("# Custom Transformer\n\nA transformer implementation.")

    # Configs for project 1
    configs1 = project1 / "configs"
    configs1.mkdir()
    (configs1 / "default.yaml").write_text(
        """
model:
  d_model: 256
  n_heads: 4
  n_layers: 6
  vocab_size: 50257
training:
  batch_size: 32
  learning_rate: 0.0003
  max_steps: 10000
"""
    )
    (configs1 / "large.yaml").write_text(
        """
model:
  d_model: 512
  n_heads: 8
  n_layers: 12
  vocab_size: 50257
training:
  batch_size: 16
  learning_rate: 0.0001
  max_steps: 50000
"""
    )

    # Checkpoints for project 1
    ckpt1 = temp_dir / "checkpoints"
    ckpt1.mkdir(parents=True)
    (ckpt1 / "custom_transformer_epoch_10.pt").write_bytes(b"x" * 1024 * 1024 * 50)
    (ckpt1 / "custom_transformer_best.pt").write_bytes(b"x" * 1024 * 1024 * 50)

    # Project 2: embedded_attention
    project2 = temp_dir / "projects" / "embedded_attention"
    project2.mkdir(parents=True)
    (project2 / "train.py").write_text("# Another training script")

    # Experiments directory
    exp_dir = temp_dir / "experiments"
    exp_dir.mkdir(parents=True)

    # Create experiment data
    exp1 = pd.DataFrame(
        {
            "epoch": [1, 2, 3, 4, 5],
            "perplexity": [30.0, 25.0, 20.0, 18.0, 15.0],
            "loss": [3.5, 3.0, 2.5, 2.2, 2.0],
            "experiment_name": ["custom_transformer_v1"] * 5,
            "meta_project": ["custom_transformer"] * 5,
            "saved_at": [datetime.now().isoformat()] * 5,
        }
    )
    exp1.to_parquet(exp_dir / "custom_transformer_v1.parquet")

    exp2 = pd.DataFrame(
        {
            "epoch": [1, 2, 3, 4, 5, 6],
            "perplexity": [28.0, 22.0, 18.0, 15.0, 13.0, 12.0],
            "loss": [3.3, 2.7, 2.3, 2.0, 1.8, 1.7],
            "experiment_name": ["custom_transformer_v2"] * 6,
            "meta_project": ["custom_transformer"] * 6,
            "saved_at": [datetime.now().isoformat()] * 6,
        }
    )
    exp2.to_parquet(exp_dir / "custom_transformer_v2.parquet")

    # Logs directory
    log_dir = temp_dir / "logs"
    log_dir.mkdir(parents=True)
    (log_dir / "train_v1.log").write_text(
        """
2024-01-01 12:00:00 - Starting training
2024-01-01 12:00:01 - loss: 3.5, perplexity: 30.0
2024-01-01 12:01:00 - loss: 3.0, perplexity: 25.0
2024-01-01 12:02:00 - ERROR: CUDA out of memory
2024-01-01 12:02:01 - Reducing batch size
2024-01-01 12:03:00 - loss: 2.0, perplexity: 15.0
2024-01-01 12:04:00 - Training complete
"""
    )

    # CLAUDE.md at root
    (temp_dir / "CLAUDE.md").write_text("# Test Repo\n\nML research repository.")

    return temp_dir


@pytest.fixture
def full_indexer(full_mock_repo: Path) -> RepoIndexer:
    """Create indexer with full mock repo."""
    indexer = RepoIndexer(full_mock_repo)
    indexer.refresh()
    return indexer


class TestExploreToTrainWorkflow:
    """Test workflow: explore repo -> find project -> find config -> generate train command."""

    @pytest.mark.asyncio
    async def test_explore_find_generate(self, full_mock_repo: Path, full_indexer: RepoIndexer):
        """Test the full explore-to-train workflow."""
        # Step 1: Explore the repo
        explore_result = await explore_repo(
            path=".",
            depth=2,
            repo_root=full_mock_repo,
        )

        assert "error" not in explore_result
        assert "projects" in str(explore_result["contents"])

        # Step 2: List projects
        projects_result = await list_projects(indexer=full_indexer)

        assert "error" not in projects_result
        assert projects_result["summary"]["total"] >= 2

        # Step 3: Get project details
        project_result = await get_project(name="custom_transformer", indexer=full_indexer)

        assert "error" not in project_result
        assert project_result["has_train_script"]

        # Step 4: List configs for project
        configs_result = await list_configs(project="custom_transformer", indexer=full_indexer)

        assert "error" not in configs_result
        assert configs_result["total"] >= 1

        # Step 5: Read a config
        config_result = await read_config(
            path="projects/custom_transformer/configs/default.yaml",
            repo_root=full_mock_repo,
        )

        assert "error" not in config_result
        assert "model" in config_result["content"]

        # Step 6: Generate training command
        command_result = await generate_train_command(
            project="custom_transformer",
            config="configs/default.yaml",
            device="cuda",
            repo_root=full_mock_repo,
        )

        assert "error" not in command_result
        assert "train.py" in command_result["command"]
        assert "--config" in command_result["command"]
        assert "--device cuda" in command_result["command"]


class TestExperimentAnalysisWorkflow:
    """Test workflow: query experiments -> compare runs -> find checkpoints."""

    @pytest.mark.asyncio
    async def test_query_compare_checkpoint(self, full_mock_repo: Path):
        """Test the full experiment analysis workflow."""
        exp_dir = full_mock_repo / "experiments"

        # Step 1: List all experiments
        list_result = await list_experiments_tool(experiments_dir=exp_dir)

        assert "error" not in list_result
        assert list_result["count"] == 2

        # Step 2: Query for best perplexity
        query_result = await query_experiments(
            query="best perplexity",
            experiments_dir=exp_dir,
        )

        assert "error" not in query_result
        assert query_result["count"] > 0

        # Step 3: Compare the two experiments
        compare_result = await compare_runs(
            experiments=["custom_transformer_v1", "custom_transformer_v2"],
            experiments_dir=exp_dir,
        )

        assert "error" not in compare_result
        assert "metrics_comparison" in compare_result
        # v2 should have better final perplexity
        if "perplexity" in compare_result["best_by_metric"]:
            assert (
                compare_result["best_by_metric"]["perplexity"]["experiment"]
                == "custom_transformer_v2"
            )

        # Step 4: Find checkpoints
        checkpoint_result = await find_checkpoint(
            experiment="custom_transformer",
            repo_root=full_mock_repo,
        )

        assert "error" not in checkpoint_result
        assert checkpoint_result["total_found"] >= 1


class TestErrorDebugWorkflow:
    """Test workflow: run into error -> explain error -> fix and retry."""

    @pytest.mark.asyncio
    async def test_cuda_oom_workflow(self, full_mock_repo: Path):
        """Test CUDA OOM error handling workflow."""
        # Step 1: Simulate encountering an error
        error_message = """
Traceback (most recent call last):
  File "train.py", line 100, in main
    loss = model(batch)
  File "model.py", line 50, in forward
    return self.transformer(x)
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
"""

        # Step 2: Explain the error
        explain_result = await explain_error(
            error=error_message,
            context="training custom transformer",
        )

        assert "error" not in explain_result
        assert explain_result["category"] == "GPU Memory"
        assert "batch" in " ".join(explain_result["suggestions"]).lower()

        # Step 3: Check traceback parsing
        assert "traceback_locations" in explain_result
        assert len(explain_result["traceback_locations"]) >= 2
        assert explain_result["traceback_locations"][0]["file"] == "train.py"

        # Step 4: Read config to find batch_size to reduce
        config_result = await read_config(
            path="projects/custom_transformer/configs/default.yaml",
            repo_root=full_mock_repo,
        )

        assert "error" not in config_result
        assert config_result["content"]["training"]["batch_size"] == 32

        # Step 5: Generate new command with smaller batch size
        command_result = await generate_train_command(
            project="custom_transformer",
            overrides={"batch_size": 16},
            repo_root=full_mock_repo,
        )

        assert "error" not in command_result
        assert "--batch_size 16" in command_result["command"]


class TestCleanupWorkflow:
    """Test workflow: analyze disk usage -> suggest cleanup -> identify old files."""

    @pytest.mark.asyncio
    async def test_cleanup_workflow(self, full_mock_repo: Path):
        """Test cleanup analysis workflow."""
        # Step 1: List experiments to understand what exists
        exp_dir = full_mock_repo / "experiments"
        list_result = await list_experiments_tool(experiments_dir=exp_dir)

        assert list_result["count"] == 2

        # Step 2: Find checkpoints
        checkpoint_result = await find_checkpoint(repo_root=full_mock_repo)

        assert "error" not in checkpoint_result
        assert checkpoint_result["total_found"] >= 1

        # Step 3: Suggest cleanup
        cleanup_result = await suggest_cleanup(
            scope="all",
            max_age_days=0,  # All files count as old
            min_size_mb=10,
            repo_root=full_mock_repo,
        )

        assert "error" not in cleanup_result
        assert "summary" in cleanup_result

        # Should find the large checkpoint files
        assert (
            cleanup_result["summary"]["large_files_count"] > 0
            or cleanup_result["summary"]["old_checkpoints_count"] > 0
        )


class TestScriptDiscoveryWorkflow:
    """Test workflow: describe task -> find script -> get project details."""

    @pytest.mark.asyncio
    async def test_find_script_workflow(self, full_indexer: RepoIndexer):
        """Test script discovery workflow."""
        # Step 1: User wants to train a model
        find_result = await find_script(task="train a model", indexer=full_indexer)

        assert "error" not in find_result
        assert len(find_result["matches"]) > 0

        # Should find training scripts
        train_scripts = [m for m in find_result["matches"] if "train" in m["name"].lower()]
        assert len(train_scripts) > 0

        # Step 2: Get details about the project
        first_script = find_result["matches"][0]
        project_name = first_script["project"]

        project_result = await get_project(name=project_name, indexer=full_indexer)

        assert "error" not in project_result
        assert project_result["has_train_script"]

        # Step 3: List all scripts for the project
        scripts_result = await list_scripts(project=project_name, indexer=full_indexer)

        assert "error" not in scripts_result


class TestConfigComparisonWorkflow:
    """Test workflow: list configs -> read them -> compare."""

    @pytest.mark.asyncio
    async def test_config_comparison(self, full_mock_repo: Path, full_indexer: RepoIndexer):
        """Test config comparison workflow."""
        # Step 1: List configs
        list_result = await list_configs(project="custom_transformer", indexer=full_indexer)

        assert "error" not in list_result
        assert list_result["total"] >= 2

        # Step 2: Read default config
        default_result = await read_config(
            path="projects/custom_transformer/configs/default.yaml",
            repo_root=full_mock_repo,
        )

        assert "error" not in default_result
        assert default_result["content"]["model"]["d_model"] == 256

        # Step 3: Read large config
        large_result = await read_config(
            path="projects/custom_transformer/configs/large.yaml",
            repo_root=full_mock_repo,
        )

        assert "error" not in large_result
        assert large_result["content"]["model"]["d_model"] == 512

        # Step 4: Analyze the differences
        # Both should have computed values
        if default_result.get("computed") and large_result.get("computed"):
            default_params = default_result["computed"].get("estimated_params", 0)
            large_params = large_result["computed"].get("estimated_params", 0)
            # Large config should have more parameters
            if default_params and large_params:
                assert large_params > default_params
